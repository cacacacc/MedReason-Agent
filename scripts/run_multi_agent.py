"""运行 Phase 4 Fixed / Supervisor Multi-Agent experiments。"""

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
from medreason_agent.agents.multi_agent import MultiAgentResult, create_multi_agent
from medreason_agent.agents.rag_agents import RetrievalPipeline, RetrievalSettings
from medreason_agent.data.vqa_rad import VQARADSample, load_vqa_rad_split
from medreason_agent.evaluation.agent_metrics import summarize_agent_metrics
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
    """读取 Phase 4 YAML 配置。"""
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def create_retrieval_pipeline(retrieval_config: dict[str, Any]) -> RetrievalPipeline | None:
    """根据配置创建 Retrieval Agent 使用的检索链路。"""
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
    """根据配置创建跨运行 persistent memory。"""
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
    result: MultiAgentResult,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """构造 Phase 4 prediction record。"""
    correct = exact_match(result.prediction, sample.answer)
    evidence_quality = score_evidence_quality(
        question=sample.question,
        ground_truth=sample.answer,
        evidence_records=result.retrieved_evidence,
    )
    record = {
        "experiment_id": metadata["experiment_id"],
        "model": metadata["model"],
        "backend": metadata["backend"],
        "prompt_version": metadata["prompt_version"],
        "prompt_contract": metadata["prompt_contract"],
        "rag_mode": metadata.get("rag_mode", ""),
        "agent_mode": metadata["agent_mode"],
        "dataset": sample.dataset,
        "split": sample.split,
        "sample_id": sample.sample_id,
        "image_id": sample.image_id,
        "image_path": sample.image_path,
        "question": sample.question,
        "prediction": result.prediction,
        "ground_truth": sample.answer,
        "answer_type": sample.answer_type,
        "question_type": sample.question_type,
        "image_organ": sample.image_organ,
        "reasoning_output": result.reasoning_output,
        "raw_output": result.raw_output,
        "agent_outputs": result.agent_outputs,
        "shared_state": result.shared_state,
        "state_compression": result.shared_state.get("state_compression", {}),
        "memory_records": result.memory_records or [],
        "memory_write_record": result.memory_write_record,
        "retrieved_evidence": result.retrieved_evidence,
        "evidence_query": result.evidence_query,
        "generated_claims": result.generated_claims,
        "claim_statuses": result.claim_statuses,
        "claim_verification_status": result.claim_verification_status,
        "evidence_quality_score": evidence_quality["evidence_quality_score"],
        "evidence_quality": evidence_quality,
        "selected_tools": result.selected_tools,
        "expected_selected_tools": result.expected_selected_tools,
        "expected_agent_route": result.expected_agent_route,
        "tool_calls": result.tool_calls,
        "agent_route": result.agent_route,
        "critic_decision": result.critic_decision,
        "confidence": result.confidence,
        "latency_ms": metadata["latency_ms"],
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "correct": correct,
        "error_type": "",
    }
    error_attribution = attribute_error(record)
    record["error_attribution"] = error_attribution
    record["error_type"] = error_attribution["primary_error_type"]
    return record


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """把 Phase 4 prediction records 写成 JSONL。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def is_current_record(record: dict[str, Any], method: dict[str, Any]) -> bool:
    """判断已有记录是否符合当前 Phase 4 配置。"""
    return (
        record.get("prompt_version") == method["prompt_version"]
        and record.get("prompt_contract") == method["prompt_contract"]
        and record.get("agent_mode") == method["agent_mode"]
        and "agent_outputs" in record
        and "selected_tools" in record
        and "expected_selected_tools" in record
        and "expected_agent_route" in record
        and "claim_statuses" in record
        and "shared_state" in record
        and "state_compression" in record
        and "memory_records" in record
        and "memory_write_record" in record
        and "error_attribution" in record
    )


def run(config_path: Path, backend_name: str | None = None) -> dict[str, Any]:
    """运行 Phase 4 实验，并返回 summary metrics。"""
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
    retrieval_pipeline = create_retrieval_pipeline(retrieval_config)
    memory_store = create_memory_store(memory_config)
    multi_agent = create_multi_agent(
        mode=method["agent_mode"],
        backend=backend,
        retrieval_pipeline=retrieval_pipeline,
        memory_store=memory_store,
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
        result = multi_agent.run(
            sample=sample,
            max_new_tokens=int(generation_config["max_new_tokens"]),
            experiment_id=config["experiment_id"],
        )
        latency_ms = round((time.perf_counter() - start) * 1000, 3)
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
                "latency_ms": latency_ms,
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
            "rag_mode": method.get("rag_mode", ""),
            "retriever": retrieval_config.get("retriever"),
            "top_k": retrieval_config.get("top_k"),
            "memory_enabled": bool(memory_config.get("enabled", False)),
            "memory_namespace": memory_config.get("namespace"),
            "memory_top_k": memory_config.get("top_k"),
            "memory_path": memory_config.get("path"),
            "is_valid_main_result": backend.backend_name != "mock",
            "prediction_file": str(prediction_path.relative_to(resolve_project_path("."))),
        }
    )
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return metrics


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="运行 Phase 4 Multi-Agent 实验。")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/exp04_fixed_multi_agent.yaml"),
    )
    parser.add_argument("--backend", choices=["mock", "qwen2_5_vl"], default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run(resolve_project_path(args.config), backend_name=args.backend)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
