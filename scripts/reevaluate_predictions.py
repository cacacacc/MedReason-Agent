"""Re-evaluate existing prediction JSONL files with current answer normalization.

用途：
已经跑完的 GPU 实验不需要重跑模型。这个脚本只读取 predictions.jsonl，
用当前 `answer_metrics.py` 重新计算 correct、accuracy、token F1、BLEU-1，
并写出新的 predictions/metrics。
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
from medreason_agent.evaluation.evidence_metrics import summarize_evidence_quality
from medreason_agent.paths import resolve_project_path


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read prediction records from JSONL."""
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
    """Write prediction records to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def reevaluate_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Recompute correctness and error attribution with current normalization."""
    updated_records: list[dict[str, Any]] = []
    for record in records:
        updated = dict(record)
        updated["correct"] = exact_match(
            str(updated.get("prediction", "")),
            str(updated.get("ground_truth", "")),
            question=str(updated.get("question", "")),
        )
        error_attribution = attribute_error(updated)
        updated["error_attribution"] = error_attribution
        updated["error_type"] = error_attribution["primary_error_type"]
        updated_records.append(updated)
    return updated_records


def summarize_records(records: list[dict[str, Any]], source_predictions: Path) -> dict[str, Any]:
    """Build metrics summary while preserving common experiment metadata."""
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
            "routing_policy",
            "agent_mode",
            "rag_mode",
        ):
            if key in first:
                metrics[key] = first[key]
    metrics["num_samples"] = len(records)
    metrics["source_prediction_file"] = str(source_predictions)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-evaluate predictions with current answer normalization.",
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for re-evaluated predictions.jsonl and metrics.json.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    predictions_path = resolve_project_path(args.predictions)
    output_dir = resolve_project_path(args.output_dir)
    output_predictions = output_dir / "predictions.jsonl"
    output_metrics = output_dir / "metrics.json"

    records = load_jsonl(predictions_path)
    updated_records = reevaluate_records(records)
    write_jsonl(output_predictions, updated_records)
    metrics = summarize_records(updated_records, source_predictions=args.predictions)
    metrics["prediction_file"] = str(output_predictions.relative_to(resolve_project_path(".")))
    output_dir.mkdir(parents=True, exist_ok=True)
    with output_metrics.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
