"""运行 Phase 6 v9 学习型 adaptive routing。

流程是 Direct-first：先跑 Direct 得到低成本答案和可见信号，再由 oracle
蒸馏得到的轻量 router 决定是否升级到 Structured CoT 或 Full Multi-Agent。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

from medreason_agent.agents.learned_router import (
    COT_ROUTE,
    DIRECT_ROUTE,
    FULL_ROUTE,
    extract_router_features,
    load_router_model,
    predict_route,
)
from medreason_agent.agents.multi_agent import create_multi_agent
from medreason_agent.agents.phase6_routing import RoutingDecision
from medreason_agent.data.vqa_rad import load_vqa_rad_split
from medreason_agent.evaluation.agent_metrics import summarize_agent_metrics
from medreason_agent.evaluation.answer_metrics import summarize_answer_metrics
from medreason_agent.evaluation.claim_status import summarize_claim_statuses
from medreason_agent.evaluation.error_attribution import summarize_error_attribution
from medreason_agent.evaluation.evidence_metrics import summarize_evidence_quality
from medreason_agent.evaluation.phase6_metrics import summarize_phase6_metrics
from medreason_agent.experiments.resume import (
    append_jsonl_record,
    load_records_by_sample_id,
    ordered_records_for_sample_ids,
)
from medreason_agent.models.vlm import create_vlm_backend
from medreason_agent.paths import resolve_project_path
from scripts.run_phase6_adaptive_routing import (
    build_prediction_record,
    create_retrieval_pipeline,
    execute_direct_route,
    execute_multi_agent_route,
    execute_structured_cot_route,
    write_jsonl,
)


def load_config(path: Path) -> dict[str, Any]:
    """读取 learned-router 实验 YAML。"""
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def run(config_path: Path, backend_name: str | None = None) -> dict[str, Any]:
    """执行 Direct-first learned routing 实验。"""
    config = load_config(config_path)
    method = config["method"]
    dataset_config = config["dataset"]
    generation_config = config["generation"]
    output_config = config["outputs"]
    retrieval_config = config.get("retrieval", {"enabled": False})
    learned_router_config = config["learned_router"]

    backend = create_vlm_backend(
        backend=backend_name or method.get("backend", "mock"),
        model_id=method.get("model_id"),
        torch_dtype=method.get("torch_dtype", "auto"),
        device_map=method.get("device_map", "auto"),
    )
    router_model = load_router_model(resolve_project_path(learned_router_config["model_path"]))
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
        direct_result = execute_direct_route(
            sample=sample,
            backend=backend,
            max_new_tokens=int(generation_config["max_new_tokens"]),
        )
        features = extract_router_features(
            {
                "question": sample.question,
                "answer_type": sample.answer_type,
                "question_type": sample.question_type,
                "image_organ": sample.image_organ,
                "direct_prediction": direct_result["prediction"],
                "direct_raw_output": direct_result["raw_output"],
            }
        )
        router_prediction = predict_route(router_model, features)
        route = apply_confidence_gate(
            route=router_prediction.route,
            confidence=router_prediction.confidence,
            min_confidence=float(learned_router_config.get("min_confidence", 0.0)),
        )
        if route == DIRECT_ROUTE:
            route_result = direct_result
        elif route == COT_ROUTE:
            route_result = execute_structured_cot_route(
                sample=sample,
                backend=backend,
                max_new_tokens=int(generation_config["max_new_tokens"]),
            )
        elif route == FULL_ROUTE:
            route_result = execute_multi_agent_route(
                sample=sample,
                multi_agent=multi_agent,
                max_new_tokens=int(generation_config["max_new_tokens"]),
                experiment_id=config["experiment_id"],
            )
        else:
            raise ValueError(f"Unsupported learned route: {route}")

        routing_decision = RoutingDecision(
            route=route_result["route"],
            complexity=_route_complexity(route_result["route"]),
            confidence_signal=f"LEARNED_ROUTER_CONFIDENCE={router_prediction.confidence:.4f}",
            uncertainty_signal="DIRECT_FIRST_FEATURES",
            evidence_support_signal=(
                "OPTIONAL" if route_result["route"] != DIRECT_ROUTE else "NOT_REQUIRED"
            ),
            agent_agreement_signal="UNKNOWN_PRE_ROUTE",
            reason="phase6 v9 oracle-distilled learned router",
            signals={
                "learned_route": router_prediction.route,
                "learned_route_after_confidence_gate": route,
                "learned_route_confidence": router_prediction.confidence,
                "learned_route_probabilities": router_prediction.probabilities,
                "learned_router_features": router_prediction.features,
                "direct_prediction": direct_result["prediction"],
            },
        )
        latency_ms = round((time.perf_counter() - start) * 1000, 3)
        record = build_prediction_record(
            sample=sample,
            route_result=route_result,
            routing_decision=routing_decision,
            fallback_applied=False,
            fallback_reason="",
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
            "learned_router_model": learned_router_config["model_path"],
            "is_valid_main_result": backend.backend_name != "mock",
            "prediction_file": str(prediction_path.relative_to(resolve_project_path("."))),
        }
    )
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return metrics


def apply_confidence_gate(route: str, confidence: float, min_confidence: float) -> str:
    """当 learned router 置信度不足时回退到 Direct。"""
    if confidence < min_confidence:
        return DIRECT_ROUTE
    return route


def is_current_record(record: dict[str, Any], method: dict[str, Any]) -> bool:
    """判断断点续跑记录是否匹配当前 learned-router 配置。"""
    return (
        record.get("prompt_version") == method["prompt_version"]
        and record.get("prompt_contract") == method["prompt_contract"]
        and record.get("routing_policy") == method["routing_policy"]
        and "phase6_routing_decision" in record
        and "phase6_route" in record
        and "error_attribution" in record
    )


def _route_complexity(route: str) -> str:
    if route == DIRECT_ROUTE:
        return "LOW"
    if route == COT_ROUTE:
        return "MEDIUM"
    return "HIGH"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 Phase6 learned router。")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/exp06_learned_router.yaml"),
    )
    parser.add_argument("--backend", choices=["mock", "qwen2_5_vl"], default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run(resolve_project_path(args.config), backend_name=args.backend)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
