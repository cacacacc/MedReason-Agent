"""Compare controlled Phase 3 RAG ablations from metrics.json files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from medreason_agent.paths import resolve_project_path

FIELDS = (
    "experiment_id",
    "orchestrator",
    "retriever",
    "num_samples",
    "accuracy",
    "mean_token_f1",
    "mean_evidence_quality_score",
    "hallucination_rate",
    "mean_latency_ms",
    "top_k",
    "candidate_top_k",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare controlled Phase 3 RAG ablations.")
    parser.add_argument(
        "--metrics",
        type=Path,
        nargs="+",
        required=True,
        help="One or more Results/**/metrics.json files.",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = [load_metrics(path) for path in args.metrics]
    markdown = build_markdown(records)
    if args.output:
        output_path = resolve_project_path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


def load_metrics(path: Path) -> dict[str, Any]:
    metrics_path = resolve_project_path(path)
    record = json.loads(metrics_path.read_text(encoding="utf-8"))
    record["metrics_path"] = str(metrics_path)
    return record


def build_markdown(records: list[dict[str, Any]]) -> str:
    normalized_records = [_normalize_record(record) for record in records]
    sorted_records = sorted(
        normalized_records,
        key=lambda item: (_number(item.get("accuracy")), _number(item.get("mean_token_f1"))),
        reverse=True,
    )
    lines = ["# Phase 3 RAG Controlled Ablation", ""]
    lines.append("| " + " | ".join(FIELDS) + " |")
    lines.append("| " + " | ".join("---" for _field in FIELDS) + " |")
    for record in sorted_records:
        lines.append("| " + " | ".join(_format_value(record.get(field)) for field in FIELDS) + " |")

    best = sorted_records[0] if sorted_records else {}
    best_accuracy = _number(best.get("accuracy")) if best else 0.0
    best_records = [
        record
        for record in sorted_records
        if abs(_number(record.get("accuracy")) - best_accuracy) < 1e-12
    ]
    lines.extend(["", "## Current Best", ""])
    if len(best_records) > 1:
        tied = ", ".join(
            f"`{record.get('experiment_id')}` "
            f"({record.get('orchestrator')} + {record.get('retriever')})"
            for record in best_records
        )
        lines.append(f"- Best by accuracy is tied at {_format_value(best_accuracy)}: {tied}.")
    elif best:
        lines.append(
            "- Best by accuracy: "
            f"`{best.get('experiment_id')}` "
            f"({best.get('orchestrator')} + {best.get('retriever')})."
        )
    else:
        lines.append("- No metrics were provided.")

    lines.extend(["", "## Control Check", ""])
    lines.append(
        "- Keep dataset split, max_samples, model_id, prompt_version, rag_mode, "
        "embedding_model, top_k, and candidate_top_k identical when judging one variable."
    )
    lines.append(
        "- Compare Python vs LangGraph with the same retriever; compare FAISS vs Qdrant "
        "with the same orchestrator."
    )
    return "\n".join(lines)


def _normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    """Fill defaults for older metrics emitted before ablation fields existed."""
    normalized = dict(record)
    if not normalized.get("orchestrator"):
        normalized["orchestrator"] = "python"
    return normalized


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
