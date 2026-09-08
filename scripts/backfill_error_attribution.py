"""离线补全 prediction records 的错误归因字段。

用途：旧实验已经在 GPU 上跑完，但当时 `predictions.jsonl` 里还没有
`error_attribution` / `error_type`。这个脚本只做 JSONL 后处理，不加载 VLM，
因此可以在 CPU 上安全运行。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

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
from medreason_agent.paths import resolve_project_path


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL prediction 文件。"""
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """覆盖写回补全后的 JSONL。"""
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def backfill_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """补齐单条记录里的派生评估字段。"""
    updated_records: list[dict[str, Any]] = []
    for record in records:
        updated = dict(record)
        if "correct" not in updated:
            updated["correct"] = exact_match(
                str(updated.get("prediction", "")),
                str(updated.get("ground_truth", "")),
            )
        if "evidence_quality" not in updated:
            evidence_quality = score_evidence_quality(
                question=str(updated.get("question", "")),
                ground_truth=str(updated.get("ground_truth", "")),
                evidence_records=list(updated.get("retrieved_evidence", [])),
            )
            updated["evidence_quality"] = evidence_quality
            updated["evidence_quality_score"] = evidence_quality["evidence_quality_score"]
        if updated.get("agent_mode") == "supervisor_multi_agent" and "answer_gate" not in updated:
            updated["answer_gate"] = {
                "decision": "DISABLED",
                "forced_prediction": None,
                "blocked_claims": [],
                "reason": "Backfilled legacy Phase 4 record.",
            }

        error_attribution = attribute_error(updated)
        updated["error_attribution"] = error_attribution
        updated["error_type"] = error_attribution["primary_error_type"]
        updated_records.append(updated)
    return updated_records


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """重新汇总 metrics，并保留常用实验元信息。"""
    metrics: dict[str, Any] = {}
    metrics.update(summarize_answer_metrics(records))
    metrics.update(summarize_evidence_quality(records))
    metrics.update(summarize_claim_statuses(records))
    metrics.update(summarize_agent_metrics(records))
    metrics.update(summarize_error_attribution(records))

    if records:
        first = records[0]
        for key in (
            "experiment_id",
            "backend",
            "model",
            "split",
            "prompt_version",
            "prompt_contract",
            "agent_mode",
            "rag_mode",
        ):
            if key in first:
                metrics[key] = first[key]
    metrics["num_samples"] = len(records)
    return metrics


def backfill_predictions(
    predictions_path: Path,
    metrics_path: Path | None = None,
) -> dict[str, Any]:
    """补全 predictions.jsonl，并写出新的 metrics.json。"""
    records = load_jsonl(predictions_path)
    updated_records = backfill_records(records)
    write_jsonl(predictions_path, updated_records)

    metrics = summarize_records(updated_records)
    target_metrics_path = metrics_path or predictions_path.with_name("metrics.json")
    with target_metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="离线补全旧实验的错误归因字段。")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metrics = backfill_predictions(
        predictions_path=resolve_project_path(args.predictions),
        metrics_path=resolve_project_path(args.metrics) if args.metrics else None,
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
