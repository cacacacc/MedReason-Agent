"""分析 Phase 6 路由错误来源。

这个脚本读取 oracle routing analysis 生成的 per_sample 文件，按 selected route、
oracle route、routing reason 和 marker 统计错误来源。用途是定位哪些规则造成
over-reasoning / under-reasoning，避免后续 v7 调参只能凭整体指标猜。
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from medreason_agent.paths import resolve_project_path


def load_records(path: Path) -> list[dict[str, Any]]:
    """读取 oracle analysis 输出的 JSONL 或 CSV 明细文件。"""
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as file:
            return list(csv.DictReader(file))

    records = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))
    return records


def summarize(records: list[dict[str, Any]], selected: str, oracle: str) -> dict[str, Any]:
    """统计某一种 selected-vs-oracle 组合的 route reason 和 marker 分布。"""
    subset = [
        record
        for record in records
        if _field(record, "selected_route", "phase6_route") == selected
        and _field(record, "oracle_route") == oracle
    ]
    reason_counts = Counter(
        _field(record, "phase6_routing_reason", "routing_reason") for record in subset
    )
    question_type_counts = Counter(_field(record, "question_type") for record in subset)
    answer_type_counts = Counter(_field(record, "answer_type") for record in subset)
    marker_counts: Counter[str] = Counter()
    examples_by_reason: dict[str, list[dict[str, str]]] = defaultdict(list)

    for record in subset:
        for marker in _markers(record):
            marker_counts[marker] += 1
        reason = _field(record, "phase6_routing_reason", "routing_reason")
        if len(examples_by_reason[reason]) < 5:
            examples_by_reason[reason].append(
                {
                    "sample_id": _field(record, "sample_id"),
                    "question": _field(record, "question"),
                    "ground_truth": _field(record, "ground_truth"),
                    "selected_prediction": _field(
                        record,
                        "routing_prediction",
                        "selected_prediction",
                    ),
                    "direct_prediction": _field(record, "direct_prediction"),
                    "cot_prediction": _field(record, "cot_prediction"),
                }
            )

    return {
        "selected_route": selected,
        "oracle_route": oracle,
        "num_samples": len(subset),
        "reason_counts": dict(reason_counts.most_common()),
        "question_type_counts": dict(question_type_counts.most_common()),
        "answer_type_counts": dict(answer_type_counts.most_common()),
        "marker_counts": dict(marker_counts.most_common()),
        "examples_by_reason": dict(examples_by_reason),
    }


def _field(record: dict[str, Any], *names: str) -> str:
    """按多个候选字段名读取字符串字段。"""
    for name in names:
        value = record.get(name)
        if value is not None and value != "":
            return str(value)
    return ""


def _markers(record: dict[str, Any]) -> list[str]:
    """从 oracle 明细中抽取 routing signals 里的 marker。"""
    signals = record.get("phase6_routing_signals") or record.get("routing_signals") or {}
    if isinstance(signals, str):
        try:
            signals = json.loads(signals)
        except json.JSONDecodeError:
            signals = {}
    if not isinstance(signals, dict):
        return []

    markers: list[str] = []
    for key in (
        "matched_markers",
        "strong_medical_markers",
        "weak_visual_markers",
    ):
        value = signals.get(key, [])
        if isinstance(value, list):
            markers.extend(str(item) for item in value if str(item))
    return markers


def write_report(output: Path, sections: list[dict[str, Any]]) -> None:
    """写出 JSON 和 Markdown 两种摘要。"""
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "route_error_breakdown.json"
    md_path = output / "route_error_breakdown.md"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(sections, file, indent=2, ensure_ascii=False)

    lines = ["# Phase 6 Route Error Breakdown", ""]
    for section in sections:
        lines.extend(
            [
                f"## selected={section['selected_route']} oracle={section['oracle_route']}",
                "",
                f"Samples: {section['num_samples']}",
                "",
                "### Routing Reasons",
                "",
                *[
                    f"- {reason or '(empty)'}: {count}"
                    for reason, count in section["reason_counts"].items()
                ],
                "",
                "### Markers",
                "",
                *[
                    f"- {marker}: {count}"
                    for marker, count in section["marker_counts"].items()
                ],
                "",
                "### Question Types",
                "",
                *[
                    f"- {question_type or '(empty)'}: {count}"
                    for question_type, count in section["question_type_counts"].items()
                ],
                "",
            ]
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(md_path)}, indent=2))


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="分析 Phase 6 routing error 来源。")
    parser.add_argument("--oracle-details", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = load_records(resolve_project_path(args.oracle_details))
    sections = [
        summarize(records, selected="structured_cot", oracle="direct"),
        summarize(records, selected="direct", oracle="structured_cot"),
        summarize(records, selected="structured_cot", oracle="none"),
        summarize(records, selected="direct", oracle="none"),
    ]
    write_report(resolve_project_path(args.output), sections)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
