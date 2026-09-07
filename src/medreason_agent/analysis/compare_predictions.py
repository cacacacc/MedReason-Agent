"""Compare Direct VLM and CoT prediction files."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL records."""
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))
    return records


def outcome_label(direct_correct: bool, cot_correct: bool) -> str:
    """Label how CoT changed the result relative to direct inference."""
    if direct_correct and cot_correct:
        return "both_correct"
    if not direct_correct and cot_correct:
        return "cot_helped"
    if direct_correct and not cot_correct:
        return "cot_hurt"
    return "both_wrong"


def compare_records(
    direct_records: list[dict[str, Any]],
    cot_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare two prediction record lists by sample_id."""
    direct_by_id = {record["sample_id"]: record for record in direct_records}
    cot_by_id = {record["sample_id"]: record for record in cot_records}
    shared_ids = sorted(direct_by_id.keys() & cot_by_id.keys())

    comparisons = []
    outcome_counts: Counter[str] = Counter()
    by_question_type: dict[str, Counter[str]] = defaultdict(Counter)
    by_answer_type: dict[str, Counter[str]] = defaultdict(Counter)

    for sample_id in shared_ids:
        direct = direct_by_id[sample_id]
        cot = cot_by_id[sample_id]
        label = outcome_label(bool(direct["correct"]), bool(cot["correct"]))
        outcome_counts[label] += 1
        by_question_type[str(direct.get("question_type", ""))][label] += 1
        by_answer_type[str(direct.get("answer_type", ""))][label] += 1
        comparisons.append(
            {
                "sample_id": sample_id,
                "question": direct["question"],
                "ground_truth": direct["ground_truth"],
                "direct_prediction": direct["prediction"],
                "cot_prediction": cot["prediction"],
                "direct_correct": direct["correct"],
                "cot_correct": cot["correct"],
                "outcome": label,
                "answer_type": direct.get("answer_type", ""),
                "question_type": direct.get("question_type", ""),
            }
        )

    return {
        "num_direct_records": len(direct_records),
        "num_cot_records": len(cot_records),
        "num_shared_records": len(shared_ids),
        "outcome_counts": dict(outcome_counts),
        "by_question_type": {key: dict(value) for key, value in by_question_type.items()},
        "by_answer_type": {key: dict(value) for key, value in by_answer_type.items()},
        "comparisons": comparisons,
    }


def write_comparison_outputs(output_dir: Path, comparison: dict[str, Any]) -> None:
    """Write summary and per-sample comparison outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {key: value for key, value in comparison.items() if key != "comparisons"}
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    with (output_dir / "comparisons.jsonl").open("w", encoding="utf-8") as file:
        for record in comparison["comparisons"]:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
