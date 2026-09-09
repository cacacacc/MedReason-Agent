"""运行 Phase 5 Verification + Self-Reflection 实验。

Phase 5 先生成 candidate reasoning + answer，然后只在 candidate 之后做
process-level verification / self-reflection。它不会在 reflection 阶段重新调用
Vision、Retrieval 或 Supervisor，避免把 self-reflection 和 replanning 混在一起。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

from medreason_agent.agents.memory import (
    PersistentAgentMemoryStore,
    PersistentMemorySettings,
)
from medreason_agent.agents.phase5_verification import (
    Phase5VerificationAgent,
    Phase5VerificationResult,
)
from medreason_agent.agents.rag_agents import RetrievalPipeline, RetrievalSettings
from medreason_agent.data.vqa_rad import VQARADSample, load_vqa_rad_split
from medreason_agent.evaluation.agent_metrics import (
    has_potential_hallucination,
    summarize_agent_metrics,
)
from medreason_agent.evaluation.answer_metrics import exact_match, summarize_answer_metrics
from medreason_agent.evaluation.claim_status import summarize_claim_statuses
from medreason_agent.evaluation.error_attribution import (
    attribute_error,
    summarize_error_attribution,
)
from medreason_agent.evaluation.evidence_metrics import (
    score_evidence_quality,
    summarize_evidence_quality,
)
from medreason_agent.evaluation.phase5_metrics import summarize_phase5_metrics
from medreason_agent.experiments.resume import (
    append_jsonl_record,
    load_records_by_sample_id,
    ordered_records_for_sample_ids,
)
from medreason_agent.models.vlm import create_vlm_backend
from medreason_agent.paths import resolve_project_path
from medreason_agent.retrieval.faiss_retriever import FAISSRetriever
from medreason_agent.retrieval.keyword import KeywordRetriever, load_chunks
from medreason_agent.retrieval.rerank import KeywordReranker


def load_config(path: Path) -> dict[str, Any]:
    """读取 Phase 5 YAML 配置。"""
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def create_retrieval_pipeline(retrieval_config: dict[str, Any]) -> RetrievalPipeline | None:
    """根据配置创建 candidate route 使用的 Retrieval Agent。"""
    if not retrieval_config.get("enabled", False):
        return None

    retriever_name = str(retrieval_config.get("retriever", "keyword"))
    if retriever_name == "keyword":
        retriever = KeywordRetriever(
            load_chunks(resolve_project_path(retrieval_config["corpus_path"]))
        )
    elif retriever_name == "faiss":
        retriever = FAISSRetriever(
            index_path=resolve_project_path(retrieval_config["index_path"]),
            metadata_path=resolve_project_path(retrieval_config["metadata_path"]),
            embedding_model=str(
                retrieval_config.get("embedding_model", "BAAI/bge-small-en-v1.5")
            ),
            device=retrieval_config.get("embedding_device"),
        )
    else:
        raise ValueError(f"Unsupported retriever: {retriever_name}")

    return RetrievalPipeline(
        retriever=retriever,
        reranker=KeywordReranker(),
        settings=RetrievalSettings(
            candidate_top_k=int(retrieval_config.get("candidate_top_k", 10)),
            top_k=int(retrieval_config.get("top_k", 5)),
            min_rerank_score=float(retrieval_config.get("min_rerank_score", 0.0)),
            max_chars_per_evidence=int(retrieval_config.get("max_chars_per_evidence", 500)),
            retriever_name=retriever_name,
            reranker_name=str(retrieval_config.get("reranker", "keyword_reranker")),
        ),
    )


def create_memory_store(memory_config: dict[str, Any]) -> PersistentAgentMemoryStore | None:
    """根据配置创建 persistent memory store。"""
    if not memory_config.get("enabled", False):
        return None

    return PersistentAgentMemoryStore(
        PersistentMemorySettings(
            enabled=True,
            path=resolve_project_path(memory_config["path"]),
            namespace=str(memory_config.get("namespace", "default")),
            top_k=int(memory_config.get("top_k", 3)),
            min_score=float(memory_config.get("min_score", 0.05)),
            read_enabled=bool(memory_config.get("read_enabled", True)),
            write_enabled=bool(memory_config.get("write_enabled", True)),
            include_prediction=bool(memory_config.get("include_prediction", False)),
        )
    )


def build_prediction_record(
    sample: VQARADSample,
    result: Phase5VerificationResult,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """构造 Phase 5 prediction record。"""
    candidate = result.candidate
    candidate_correct = exact_match(
        candidate.prediction,
        sample.answer,
        question=sample.question,
    )
    final_correct = exact_match(result.prediction, sample.answer, question=sample.question)
    evidence_quality = score_evidence_quality(
        question=sample.question,
        ground_truth=sample.answer,
        evidence_records=candidate.retrieved_evidence,
    )
    phase5_stages = [
        call["stage"] for call in result.tool_calls if call.get("tool") == "vlm_generate"
    ]
    process_feedback = result.process_feedback

    candidate_record = {
        "prediction": candidate.prediction,
        "ground_truth": sample.answer,
        "correct": candidate_correct,
        "agent_outputs": candidate.agent_outputs,
        "agent_route": candidate.agent_route,
        "expected_agent_route": candidate.expected_agent_route,
        "selected_tools": candidate.selected_tools,
        "expected_selected_tools": candidate.expected_selected_tools,
        "retrieved_evidence": candidate.retrieved_evidence,
        "evidence_quality": evidence_quality,
        "claim_statuses": candidate.claim_statuses,
        "claim_verification_status": candidate.claim_verification_status,
        "shared_state": candidate.shared_state,
        "state_compression": candidate.shared_state.get("state_compression", {}),
    }
    candidate_error_attribution = attribute_error(candidate_record)

    agent_outputs = dict(candidate.agent_outputs)
    agent_outputs["process_feedback"] = result.process_feedback_output
    agent_outputs["revision"] = result.revision_output
    final_record_for_hallucination = dict(
        candidate_record,
        prediction=result.prediction,
        correct=final_correct,
        agent_outputs=agent_outputs,
    )

    record = {
        "experiment_id": metadata["experiment_id"],
        "model": metadata["model"],
        "backend": metadata["backend"],
        "prompt_version": metadata["prompt_version"],
        "prompt_contract": metadata["prompt_contract"],
        "record_protocol_version": "phase5_candidate_post_verification_v3",
        "verification_mode": result.verification_mode,
        "selective_verification": metadata["selective_verification"],
        "rag_mode": metadata.get("rag_mode", ""),
        "agent_mode": metadata["agent_mode"],
        "question_routing": metadata.get("question_routing", "none"),
        "dataset": sample.dataset,
        "split": sample.split,
        "sample_id": sample.sample_id,
        "image_id": sample.image_id,
        "image_path": sample.image_path,
        "question": sample.question,
        "candidate_prediction": candidate.prediction,
        "candidate_correct": candidate_correct,
        "prediction": result.prediction,
        "ground_truth": sample.answer,
        "answer_type": sample.answer_type,
        "question_type": sample.question_type,
        "image_organ": sample.image_organ,
        "candidate_reasoning_output": candidate.reasoning_output,
        "reasoning_output": result.revision_output or candidate.reasoning_output,
        "candidate_raw_output": candidate.raw_output,
        "raw_output": result.raw_output,
        "process_feedback_output": result.process_feedback_output,
        "process_feedback": process_feedback,
        "process_verdict": process_feedback.get("verdict", ""),
        "process_recommended_action": process_feedback.get("recommended_action", ""),
        "process_step_results": process_feedback.get("step_results", []),
        "process_error_types": process_feedback.get("error_types", []),
        "process_grounding_score": process_feedback.get("grounding_score"),
        "process_logical_consistency": process_feedback.get("logical_consistency"),
        "process_confidence_alignment": process_feedback.get("confidence_alignment"),
        "phase5_process_stages": phase5_stages,
        "revision_output": result.revision_output,
        "revision_applied": result.revision_applied,
        "selective_verification_applied": result.selective_verification_applied,
        "candidate_hallucination": has_potential_hallucination(candidate_record),
        "final_hallucination": has_potential_hallucination(final_record_for_hallucination),
        "agent_outputs": agent_outputs,
        "shared_state": candidate.shared_state,
        "state_compression": candidate.shared_state.get("state_compression", {}),
        "memory_records": candidate.memory_records or [],
        "memory_write_record": candidate.memory_write_record,
        "retrieved_evidence": candidate.retrieved_evidence,
        "evidence_query": candidate.evidence_query,
        "generated_claims": candidate.generated_claims,
        "claim_statuses": candidate.claim_statuses,
        "claim_verification_status": candidate.claim_verification_status,
        "answer_gate": candidate.answer_gate,
        "evidence_quality_score": evidence_quality["evidence_quality_score"],
        "evidence_quality": evidence_quality,
        "selected_tools": candidate.selected_tools,
        "expected_selected_tools": candidate.expected_selected_tools,
        "expected_agent_route": [*candidate.expected_agent_route, *phase5_stages],
        "tool_calls": [*candidate.tool_calls, *result.tool_calls],
        "agent_route": [*candidate.agent_route, *phase5_stages],
        "critic_decision": str(process_feedback.get("decision", "")),
        "candidate_error_attribution": candidate_error_attribution,
        "confidence": candidate.confidence,
        "latency_ms": metadata["latency_ms"],
        "candidate_input_tokens": candidate.input_tokens,
        "candidate_output_tokens": candidate.output_tokens,
        "verification_input_tokens": result.input_tokens,
        "verification_output_tokens": result.output_tokens,
        "input_tokens": _sum_optional(candidate.input_tokens, result.input_tokens),
        "output_tokens": _sum_optional(candidate.output_tokens, result.output_tokens),
        "correct": final_correct,
        "error_type": "",
    }
    error_attribution = attribute_error(record)
    record["error_attribution"] = error_attribution
    record["error_type"] = error_attribution["primary_error_type"]
    return record


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """写出 Phase 5 predictions JSONL。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def is_current_record(record: dict[str, Any], method: dict[str, Any]) -> bool:
    """判断已有记录是否符合当前 Phase 5 输出协议。"""
    return (
        record.get("prompt_version") == method["prompt_version"]
        and record.get("prompt_contract") == method["prompt_contract"]
        and record.get("question_routing", "none") == method.get(
            "question_routing",
            "none",
        )
        and record.get("record_protocol_version") == "phase5_candidate_post_verification_v3"
        and record.get("verification_mode") == method["verification_mode"]
        and "candidate_prediction" in record
        and "candidate_correct" in record
        and "process_feedback" in record
        and "process_error_types" in record
        and "process_step_results" in record
        and "phase5_process_stages" in record
        and "revision_applied" in record
        and "selective_verification_applied" in record
        and "error_attribution" in record
    )


def run(config_path: Path, backend_name: str | None = None) -> dict[str, Any]:
    """运行 Phase 5 实验，并返回 summary metrics。"""
    config = load_config(config_path)
    method = config["method"]
    dataset_config = config["dataset"]
    generation_config = config["generation"]
    output_config = config["outputs"]
    retrieval_config = config.get("retrieval", {"enabled": False})
    memory_config = config.get("memory", {"enabled": False})

    backend = create_vlm_backend(
        backend=backend_name or method.get("backend", "mock"),
        model_id=method.get("model_id"),
        torch_dtype=method.get("torch_dtype", "auto"),
        device_map=method.get("device_map", "auto"),
    )
    agent = Phase5VerificationAgent(
        backend=backend,
        retrieval_pipeline=create_retrieval_pipeline(retrieval_config),
        memory_store=create_memory_store(memory_config),
        verification_mode=method["verification_mode"],
        dynamic_routing=bool(method.get("dynamic_routing", False)),
        deterministic_answer_gate=bool(method.get("deterministic_answer_gate", False)),
        selective_verification=str(method.get("selective_verification", "all")),
        question_routing=str(method.get("question_routing", "none")),
    )
    samples = load_vqa_rad_split(
        split=dataset_config["split"],
        max_samples=dataset_config.get("max_samples"),
    )

    result_dir = resolve_project_path(output_config["result_dir"])
    prediction_path = result_dir / output_config["prediction_file"]
    metrics_path = result_dir / output_config["metrics_file"]

    sample_ids = [sample.sample_id for sample in samples]
    existing_records_by_sample_id = load_records_by_sample_id(prediction_path)
    records_by_sample_id = {
        sample_id: record
        for sample_id, record in existing_records_by_sample_id.items()
        if is_current_record(record, method)
    }
    missing_samples = [sample for sample in samples if sample.sample_id not in records_by_sample_id]
    print(
        f"RESUME existing={len(samples) - len(missing_samples)} "
        f"missing={len(missing_samples)} total={len(samples)}"
    )

    for sample in tqdm(missing_samples, desc=config["experiment_id"], ascii=True):
        start = time.perf_counter()
        result = agent.run(
            sample=sample,
            max_new_tokens=int(generation_config["max_new_tokens"]),
            experiment_id=config["experiment_id"],
        )
        record = build_prediction_record(
            sample=sample,
            result=result,
            metadata={
                "experiment_id": config["experiment_id"],
                "model": backend.model_name,
                "backend": backend.backend_name,
                "prompt_version": method["prompt_version"],
                "prompt_contract": method["prompt_contract"],
                "rag_mode": method.get("rag_mode", ""),
                "agent_mode": method["agent_mode"],
                "question_routing": method.get("question_routing", "none"),
                "selective_verification": method.get("selective_verification", "all"),
                "latency_ms": round((time.perf_counter() - start) * 1000, 3),
            },
        )
        append_jsonl_record(prediction_path, record)
        records_by_sample_id[sample.sample_id] = record

    records = ordered_records_for_sample_ids(records_by_sample_id, sample_ids)
    write_jsonl(prediction_path, records)
    metrics = summarize_answer_metrics(records)
    metrics.update(summarize_evidence_quality(records))
    metrics.update(summarize_claim_statuses(records))
    metrics.update(summarize_agent_metrics(records))
    metrics.update(summarize_error_attribution(records))
    metrics.update(summarize_phase5_metrics(records))
    metrics.update(
        {
            "experiment_id": config["experiment_id"],
            "backend": backend.backend_name,
            "model": backend.model_name,
            "split": dataset_config["split"],
            "scale": dataset_config["scale"],
            "prompt_version": method["prompt_version"],
            "prompt_contract": method["prompt_contract"],
            "agent_mode": method["agent_mode"],
            "question_routing": method.get("question_routing", "none"),
            "verification_mode": method["verification_mode"],
            "selective_verification": method.get("selective_verification", "all"),
            "rag_mode": method.get("rag_mode", ""),
            "retriever": retrieval_config.get("retriever"),
            "top_k": retrieval_config.get("top_k"),
            "is_valid_main_result": backend.backend_name != "mock",
            "prediction_file": str(prediction_path.relative_to(resolve_project_path("."))),
        }
    )
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 Phase 5 Verification 实验。")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/exp05_supervisor_no_critic.yaml"),
    )
    parser.add_argument("--backend", choices=["mock", "qwen2_5_vl"], default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run(resolve_project_path(args.config), backend_name=args.backend)
    return 0


def _sum_optional(*values: int | None) -> int | None:
    """汇总 token；任一真实调用缺 token 时返回 None。"""
    if any(value is None for value in values):
        return None
    return sum(int(value) for value in values)


if __name__ == "__main__":
    raise SystemExit(main())
