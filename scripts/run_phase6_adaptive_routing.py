"""运行 Phase 6 Rule-Based Supervisor Adaptive Routing 实验。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

from medreason_agent.agents.multi_agent import MultiAgentResult, create_multi_agent
from medreason_agent.agents.phase6_routing import (
    HIGH_ROUTE,
    LOW_ROUTE,
    MEDIUM_ROUTE,
    RoutingDecision,
    decide_rule_based_route,
    fallback_route,
    should_fallback,
)
from medreason_agent.agents.rag_agents import RetrievalPipeline, RetrievalSettings
from medreason_agent.answer_normalization import canonical_short_answer
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
from medreason_agent.evaluation.phase6_metrics import summarize_phase6_metrics
from medreason_agent.experiments.resume import (
    append_jsonl_record,
    load_records_by_sample_id,
    ordered_records_for_sample_ids,
)
from medreason_agent.models.vlm import VLMBackend, VLMRequest, VLMResponse, create_vlm_backend
from medreason_agent.paths import resolve_project_path
from medreason_agent.prompts.cot import build_cot_prompt
from medreason_agent.prompts.cot import extract_final_answer as extract_cot_answer
from medreason_agent.prompts.direct import build_direct_prompt
from medreason_agent.retrieval.faiss_retriever import FAISSRetriever
from medreason_agent.retrieval.keyword import KeywordRetriever, load_chunks
from medreason_agent.retrieval.rerank import KeywordReranker


def load_config(path: Path) -> dict[str, Any]:
    """读取 Phase 6 YAML 配置。"""
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def create_retrieval_pipeline(retrieval_config: dict[str, Any]) -> RetrievalPipeline | None:
    """根据配置创建高复杂度路线使用的 retrieval pipeline。"""
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


def run(config_path: Path, backend_name: str | None = None) -> dict[str, Any]:
    """运行 Phase 6 adaptive routing，并返回汇总指标。"""
    config = load_config(config_path)
    method = config["method"]
    dataset_config = config["dataset"]
    generation_config = config["generation"]
    output_config = config["outputs"]
    retrieval_config = config.get("retrieval", {"enabled": False})
    fallback_config = method.get("fallback", {"enabled": False})

    backend = create_vlm_backend(
        backend=backend_name or method.get("backend", "mock"),
        model_id=method.get("model_id"),
        torch_dtype=method.get("torch_dtype", "auto"),
        device_map=method.get("device_map", "auto"),
    )
    retrieval_pipeline = create_retrieval_pipeline(retrieval_config)
    multi_agent = create_multi_agent(
        mode="supervisor_multi_agent",
        backend=backend,
        retrieval_pipeline=retrieval_pipeline,
        dynamic_routing=False,
        deterministic_answer_gate=bool(method.get("deterministic_answer_gate", False)),
        question_routing="none",
    )

    samples = load_vqa_rad_split(
        split=dataset_config["split"],
        max_samples=dataset_config.get("max_samples"),
    )
    result_dir = resolve_project_path(output_config["result_dir"])
    prediction_path = result_dir / output_config["prediction_file"]
    metrics_path = result_dir / output_config["metrics_file"]

    sample_ids = [sample.sample_id for sample in samples]
    records_by_sample_id = {
        sample_id: record
        for sample_id, record in load_records_by_sample_id(prediction_path).items()
        if is_current_record(record, method)
    }
    missing_samples = [sample for sample in samples if sample.sample_id not in records_by_sample_id]
    print(
        f"RESUME existing={len(samples) - len(missing_samples)} "
        f"missing={len(missing_samples)} total={len(samples)}"
    )

    for sample in tqdm(missing_samples, desc=config["experiment_id"], ascii=True):
        start = time.perf_counter()
        routing_decision = decide_rule_based_route(sample)
        route_result = execute_route(
            sample=sample,
            route=routing_decision.route,
            backend=backend,
            multi_agent=multi_agent,
            max_new_tokens=int(generation_config["max_new_tokens"]),
            experiment_id=config["experiment_id"],
        )
        fallback_applied = False
        fallback_reason = ""
        original_route = routing_decision.route
        if bool(fallback_config.get("enabled", False)):
            should_step_up, fallback_reason = should_fallback(
                route_result["prediction"],
                route_result["raw_output"],
                routing_decision.route,
            )
            next_route = fallback_route(routing_decision.route)
            if should_step_up and next_route is not None:
                fallback_applied = True
                route_result = execute_route(
                    sample=sample,
                    route=next_route,
                    backend=backend,
                    multi_agent=multi_agent,
                    max_new_tokens=int(generation_config["max_new_tokens"]),
                    experiment_id=config["experiment_id"],
                )
                routing_decision = RoutingDecision(
                    route=next_route,
                    complexity=routing_decision.complexity,
                    confidence_signal=routing_decision.confidence_signal,
                    uncertainty_signal="FALLBACK_TRIGGERED",
                    evidence_support_signal=routing_decision.evidence_support_signal,
                    agent_agreement_signal=routing_decision.agent_agreement_signal,
                    reason=f"{routing_decision.reason}; fallback={fallback_reason}",
                    signals={
                        **routing_decision.signals,
                        "original_route": original_route,
                        "fallback_reason": fallback_reason,
                    },
                )

        latency_ms = round((time.perf_counter() - start) * 1000, 3)
        record = build_prediction_record(
            sample=sample,
            route_result=route_result,
            routing_decision=routing_decision,
            fallback_applied=fallback_applied,
            fallback_reason=fallback_reason,
            latency_ms=latency_ms,
            metadata={
                "experiment_id": config["experiment_id"],
                "model": backend.model_name,
                "backend": backend.backend_name,
                "prompt_version": method["prompt_version"],
                "prompt_contract": method["prompt_contract"],
                "routing_policy": method["routing_policy"],
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
    metrics.update(summarize_phase6_metrics(records))
    metrics.update(
        {
            "experiment_id": config["experiment_id"],
            "backend": backend.backend_name,
            "model": backend.model_name,
            "split": dataset_config["split"],
            "scale": dataset_config["scale"],
            "prompt_version": method["prompt_version"],
            "prompt_contract": method["prompt_contract"],
            "routing_policy": method["routing_policy"],
            "is_valid_main_result": backend.backend_name != "mock",
            "prediction_file": str(prediction_path.relative_to(resolve_project_path("."))),
        }
    )
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return metrics


def execute_route(
    sample: VQARADSample,
    route: str,
    backend: VLMBackend,
    multi_agent,
    max_new_tokens: int,
    experiment_id: str,
) -> dict[str, Any]:
    """执行 Phase6 选择的一条路线。"""
    if route == LOW_ROUTE:
        return execute_direct_route(sample, backend, max_new_tokens)
    if route == MEDIUM_ROUTE:
        return execute_structured_cot_route(sample, backend, max_new_tokens)
    if route == HIGH_ROUTE:
        return execute_multi_agent_route(sample, multi_agent, max_new_tokens, experiment_id)
    raise ValueError(f"Unsupported Phase 6 route: {route}")


def execute_direct_route(
    sample: VQARADSample,
    backend: VLMBackend,
    max_new_tokens: int,
) -> dict[str, Any]:
    """执行 LOW route：Direct VLM。"""
    response = backend.generate(
        VLMRequest(
            image_path=str(sample.absolute_image_path),
            question=sample.question,
            prompt=build_direct_prompt(sample.question),
            max_new_tokens=min(max_new_tokens, 64),
        )
    )
    prediction = canonical_short_answer(response.answer, question=sample.question)
    return route_result_from_response(
        route=LOW_ROUTE,
        response=response,
        prediction=prediction,
        agent_route=["direct_vlm"],
        tool_calls=[{"tool": "vlm_generate", "stage": "direct_vlm"}],
    )


def execute_structured_cot_route(
    sample: VQARADSample,
    backend: VLMBackend,
    max_new_tokens: int,
) -> dict[str, Any]:
    """执行 MEDIUM route：Structured CoT。"""
    response = backend.generate(
        VLMRequest(
            image_path=str(sample.absolute_image_path),
            question=sample.question,
            prompt=build_cot_prompt(sample.question, prompt_style="structured"),
            max_new_tokens=max_new_tokens,
        )
    )
    prediction = extract_cot_answer(response.raw_output, question=sample.question)
    return route_result_from_response(
        route=MEDIUM_ROUTE,
        response=response,
        prediction=prediction,
        agent_route=["cot_vlm"],
        tool_calls=[{"tool": "vlm_generate", "stage": "cot_vlm"}],
        reasoning_output=response.raw_output,
    )


def execute_multi_agent_route(
    sample: VQARADSample,
    multi_agent,
    max_new_tokens: int,
    experiment_id: str,
) -> dict[str, Any]:
    """执行 HIGH route：完整 Supervisor Multi-Agent。"""
    result: MultiAgentResult = multi_agent.run(
        sample=sample,
        max_new_tokens=max_new_tokens,
        experiment_id=experiment_id,
    )
    return {
        "route": HIGH_ROUTE,
        "prediction": result.prediction,
        "raw_output": result.raw_output,
        "reasoning_output": result.reasoning_output,
        "retrieved_evidence": result.retrieved_evidence,
        "evidence_query": result.evidence_query,
        "generated_claims": result.generated_claims,
        "claim_statuses": result.claim_statuses,
        "claim_verification_status": result.claim_verification_status,
        "agent_outputs": result.agent_outputs,
        "shared_state": result.shared_state,
        "state_compression": result.shared_state.get("state_compression", {}),
        "memory_records": result.memory_records or [],
        "memory_write_record": result.memory_write_record,
        "selected_tools": result.selected_tools,
        "expected_selected_tools": result.expected_selected_tools,
        "expected_agent_route": result.expected_agent_route,
        "tool_calls": result.tool_calls,
        "agent_route": result.agent_route,
        "critic_decision": result.critic_decision,
        "answer_gate": result.answer_gate,
        "confidence": result.confidence,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
    }


def route_result_from_response(
    route: str,
    response: VLMResponse,
    prediction: str,
    agent_route: list[str],
    tool_calls: list[dict],
    reasoning_output: str = "",
) -> dict[str, Any]:
    """把 Direct / CoT 输出规整为和 Multi-Agent 兼容的 record 字段。"""
    return {
        "route": route,
        "prediction": prediction,
        "raw_output": response.raw_output,
        "reasoning_output": reasoning_output,
        "retrieved_evidence": [],
        "evidence_query": "",
        "generated_claims": [],
        "claim_statuses": [],
        "claim_verification_status": "",
        "agent_outputs": {},
        "shared_state": {},
        "state_compression": {},
        "memory_records": [],
        "memory_write_record": None,
        "selected_tools": [],
        "expected_selected_tools": [],
        "expected_agent_route": agent_route,
        "tool_calls": tool_calls,
        "agent_route": agent_route,
        "critic_decision": "",
        "answer_gate": {"decision": "NOT_APPLICABLE"},
        "confidence": response.confidence,
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
    }


def build_prediction_record(
    sample: VQARADSample,
    route_result: dict[str, Any],
    routing_decision: RoutingDecision,
    fallback_applied: bool,
    fallback_reason: str,
    latency_ms: float,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """构造 Phase 6 prediction record。"""
    prediction = canonical_short_answer(route_result["prediction"], question=sample.question)
    correct = exact_match(prediction, sample.answer, question=sample.question)
    evidence_quality = score_evidence_quality(
        question=sample.question,
        ground_truth=sample.answer,
        evidence_records=route_result["retrieved_evidence"],
    )
    record = {
        "experiment_id": metadata["experiment_id"],
        "model": metadata["model"],
        "backend": metadata["backend"],
        "prompt_version": metadata["prompt_version"],
        "prompt_contract": metadata["prompt_contract"],
        "routing_policy": metadata["routing_policy"],
        "phase6_route": route_result["route"],
        "phase6_routing_decision": routing_decision.to_record(),
        "phase6_fallback_applied": fallback_applied,
        "phase6_fallback_reason": fallback_reason,
        "dataset": sample.dataset,
        "split": sample.split,
        "sample_id": sample.sample_id,
        "image_id": sample.image_id,
        "image_path": sample.image_path,
        "question": sample.question,
        "prediction": prediction,
        "ground_truth": sample.answer,
        "answer_type": sample.answer_type,
        "question_type": sample.question_type,
        "image_organ": sample.image_organ,
        "reasoning_output": route_result["reasoning_output"],
        "raw_output": route_result["raw_output"],
        "agent_outputs": route_result["agent_outputs"],
        "shared_state": route_result["shared_state"],
        "state_compression": route_result["state_compression"],
        "memory_records": route_result["memory_records"],
        "memory_write_record": route_result["memory_write_record"],
        "retrieved_evidence": route_result["retrieved_evidence"],
        "evidence_query": route_result["evidence_query"],
        "generated_claims": route_result["generated_claims"],
        "claim_statuses": route_result["claim_statuses"],
        "claim_verification_status": route_result["claim_verification_status"],
        "answer_gate": route_result["answer_gate"],
        "evidence_quality_score": evidence_quality["evidence_quality_score"],
        "evidence_quality": evidence_quality,
        "selected_tools": route_result["selected_tools"],
        "expected_selected_tools": route_result["expected_selected_tools"],
        "expected_agent_route": route_result["expected_agent_route"],
        "tool_calls": route_result["tool_calls"],
        "agent_route": route_result["agent_route"],
        "critic_decision": route_result["critic_decision"],
        "confidence": route_result["confidence"],
        "latency_ms": latency_ms,
        "input_tokens": route_result["input_tokens"],
        "output_tokens": route_result["output_tokens"],
        "correct": correct,
        "error_type": "",
    }
    error_attribution = attribute_error(record)
    record["error_attribution"] = error_attribution
    record["error_type"] = error_attribution["primary_error_type"]
    return record


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """写出 Phase6 JSONL records。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def is_current_record(record: dict[str, Any], method: dict[str, Any]) -> bool:
    """判断断点续跑记录是否匹配当前 Phase6 配置。"""
    return (
        record.get("prompt_version") == method["prompt_version"]
        and record.get("prompt_contract") == method["prompt_contract"]
        and record.get("routing_policy") == method["routing_policy"]
        and "phase6_routing_decision" in record
        and "phase6_route" in record
        and "phase6_fallback_applied" in record
        and "error_attribution" in record
    )


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="运行 Phase 6 adaptive routing 实验。")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/exp06_rule_based_adaptive_routing_100.yaml"),
    )
    parser.add_argument("--backend", choices=["mock", "qwen2_5_vl"], default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run(resolve_project_path(args.config), backend_name=args.backend)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
