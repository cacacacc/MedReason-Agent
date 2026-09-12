"""Analyze Phase 6 oracle routing upper bound.

Oracle Routing 是离线分析：它会偷看 ground truth，因此不能作为真实部署策略。
它的作用是衡量当前 adaptive router 离“最低成本正确路线”还有多远。
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from medreason_agent.evaluation.answer_metrics import exact_match
from medreason_agent.paths import resolve_project_path

DIRECT_ROUTE = "direct"
COT_ROUTE = "structured_cot"
FULL_ROUTE = "full_multi_agent"
NONE_ROUTE = "none"

ROUTE_ORDER = [DIRECT_ROUTE, COT_ROUTE, FULL_ROUTE]
ROUTE_COST_RANK = {
    DIRECT_ROUTE: 0,
    COT_ROUTE: 1,
    FULL_ROUTE: 2,
    NONE_ROUTE: 3,
}


def load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    """读取 JSONL predictions，并按 sample_id 建索引。"""
    records: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            sample_id = str(record["sample_id"])
            if sample_id in records:
                raise ValueError(f"Duplicate sample_id={sample_id} in {path}:{line_number}")
            records[sample_id] = record
    return records


def is_correct(record: dict[str, Any]) -> bool:
    """优先使用记录里的 correct 字段；缺失时重新计算 exact match。"""
    if "correct" in record:
        return bool(record["correct"])
    return exact_match(
        str(record.get("prediction", "")),
        str(record.get("ground_truth", "")),
        question=str(record.get("question", "")),
    )


def oracle_route(route_correctness: dict[str, bool]) -> str:
    """选择最低成本的正确路线；三条路线全错时返回 none。"""
    for route in ROUTE_ORDER:
        if route_correctness[route]:
            return route
    return NONE_ROUTE


def selected_route(record: dict[str, Any]) -> str:
    """读取 Phase6 实际选择的 route，并兼容旧字段。"""
    route = str(record.get("phase6_route", "")).strip()
    if route:
        return route
    route = str(record.get("route", "")).strip()
    if route:
        return route
    agent_route = record.get("agent_route", [])
    if agent_route == ["direct_vlm"]:
        return DIRECT_ROUTE
    if agent_route == ["cot_vlm"]:
        return COT_ROUTE
    return FULL_ROUTE


def compare_depth(selected: str, oracle: str) -> str:
    """判断当前路由相对 oracle 是刚好、过度推理、还是推理不足。"""
    if oracle == NONE_ROUTE:
        return "unrecoverable"
    if selected == oracle:
        return "matched"
    selected_rank = ROUTE_COST_RANK.get(selected, ROUTE_COST_RANK[FULL_ROUTE])
    oracle_rank = ROUTE_COST_RANK[oracle]
    if selected_rank > oracle_rank:
        return "over_reasoning"
    return "under_reasoning"


def build_analysis_rows(
    direct_records: dict[str, dict[str, Any]],
    cot_records: dict[str, dict[str, Any]],
    full_records: dict[str, dict[str, Any]],
    routing_records: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """按共同 sample_id 对齐四组结果，并生成逐样本 oracle 分析行。"""
    common_ids = sorted(
        set(direct_records)
        & set(cot_records)
        & set(full_records)
        & set(routing_records),
        key=_sample_sort_key,
    )
    if not common_ids:
        raise ValueError("No overlapping sample_id found across the four prediction files.")

    rows: list[dict[str, Any]] = []
    for sample_id in common_ids:
        direct = direct_records[sample_id]
        cot = cot_records[sample_id]
        full = full_records[sample_id]
        routing = routing_records[sample_id]
        route_correctness = {
            DIRECT_ROUTE: is_correct(direct),
            COT_ROUTE: is_correct(cot),
            FULL_ROUTE: is_correct(full),
        }
        oracle = oracle_route(route_correctness)
        selected = selected_route(routing)
        depth_error = compare_depth(selected, oracle)
        rows.append(
            {
                "sample_id": sample_id,
                "question": routing.get("question", direct.get("question", "")),
                "ground_truth": routing.get("ground_truth", direct.get("ground_truth", "")),
                "answer_type": routing.get("answer_type", direct.get("answer_type", "")),
                "question_type": routing.get("question_type", direct.get("question_type", "")),
                "image_organ": routing.get("image_organ", direct.get("image_organ", "")),
                "direct_prediction": direct.get("prediction", ""),
                "cot_prediction": cot.get("prediction", ""),
                "full_agent_prediction": full.get("prediction", ""),
                "routing_prediction": routing.get("prediction", ""),
                "direct_correct": route_correctness[DIRECT_ROUTE],
                "cot_correct": route_correctness[COT_ROUTE],
                "full_agent_correct": route_correctness[FULL_ROUTE],
                "routing_correct": is_correct(routing),
                "oracle_route": oracle,
                "oracle_correct": oracle != NONE_ROUTE,
                "selected_route": selected,
                "routing_matches_oracle": selected == oracle,
                "depth_error": depth_error,
                "phase6_fallback_applied": bool(routing.get("phase6_fallback_applied", False)),
                "phase6_fallback_reason": routing.get("phase6_fallback_reason", ""),
            }
        )
    return rows


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总 Oracle Routing 指标。"""
    total = len(rows)
    oracle_correct = sum(bool(row["oracle_correct"]) for row in rows)
    routing_correct = sum(bool(row["routing_correct"]) for row in rows)
    routing_matches = sum(bool(row["routing_matches_oracle"]) for row in rows)
    depth_counts = Counter(str(row["depth_error"]) for row in rows)
    oracle_counts = Counter(str(row["oracle_route"]) for row in rows)
    selected_counts = Counter(str(row["selected_route"]) for row in rows)

    by_question_type = _group_distribution(rows, "question_type", "oracle_route")
    by_answer_type = _group_distribution(rows, "answer_type", "oracle_route")
    confusion = _confusion_matrix(rows)

    return {
        "num_samples": total,
        "oracle_accuracy": oracle_correct / total,
        "routing_accuracy": routing_correct / total,
        "routing_to_oracle_match_rate": routing_matches / total,
        "over_reasoning_rate": depth_counts["over_reasoning"] / total,
        "under_reasoning_rate": depth_counts["under_reasoning"] / total,
        "unrecoverable_error_rate": depth_counts["unrecoverable"] / total,
        "matched_route_rate": depth_counts["matched"] / total,
        "oracle_route_distribution": dict(sorted(oracle_counts.items())),
        "selected_route_distribution": dict(sorted(selected_counts.items())),
        "depth_error_counts": dict(sorted(depth_counts.items())),
        "confusion_matrix_selected_vs_oracle": confusion,
        "oracle_route_by_question_type": by_question_type,
        "oracle_route_by_answer_type": by_answer_type,
    }


def write_outputs(rows: list[dict[str, Any]], metrics: dict[str, Any], output_dir: Path) -> None:
    """写出 oracle_metrics.json、oracle_analysis.csv 和 oracle_summary.md。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "oracle_metrics.json"
    csv_path = output_dir / "oracle_analysis.csv"
    summary_path = output_dir / "oracle_summary.md"

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)

    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with summary_path.open("w", encoding="utf-8") as file:
        file.write(format_markdown_summary(metrics))


def format_markdown_summary(metrics: dict[str, Any]) -> str:
    """生成论文分析可直接引用的 Markdown summary。"""
    lines = [
        "# Phase 6 Oracle Routing Analysis",
        "",
        "Oracle Routing 是离线上界分析，会使用 ground truth，因此不是可部署方法。",
        "",
        "## Main Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Samples | {metrics['num_samples']} |",
        f"| Oracle Accuracy | {_pct(metrics['oracle_accuracy'])} |",
        f"| Phase6 Routing Accuracy | {_pct(metrics['routing_accuracy'])} |",
        f"| Routing-to-Oracle Match | {_pct(metrics['routing_to_oracle_match_rate'])} |",
        f"| Over-Reasoning Rate | {_pct(metrics['over_reasoning_rate'])} |",
        f"| Under-Reasoning Rate | {_pct(metrics['under_reasoning_rate'])} |",
        f"| Unrecoverable Error Rate | {_pct(metrics['unrecoverable_error_rate'])} |",
        "",
        "## Route Distribution",
        "",
        "### Oracle Route",
        "",
        _format_counter_table(metrics["oracle_route_distribution"]),
        "",
        "### Selected Route",
        "",
        _format_counter_table(metrics["selected_route_distribution"]),
        "",
        "## Selected vs Oracle",
        "",
        _format_confusion_table(metrics["confusion_matrix_selected_vs_oracle"]),
        "",
        "## Interpretation",
        "",
        "- Oracle Accuracy 表示 Direct / CoT / Full Multi-Agent 三条路线的理论上界。",
        "- Routing-to-Oracle Match 衡量当前 rule-based supervisor 是否选中了最低成本正确路线。",
        "- Over-Reasoning 表示实际选择的路线比 oracle 更复杂。",
        "- Under-Reasoning 表示实际选择的路线不够复杂，错过了更深推理路线的正确答案。",
        "- Unrecoverable 表示三条候选路线都答错，当前 route search space 无法修复。",
        "",
    ]
    return "\n".join(lines)


def _group_distribution(rows: list[dict[str, Any]], group_key: str, value_key: str) -> dict:
    grouped: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        grouped[str(row[group_key])][str(row[value_key])] += 1
    return {key: dict(sorted(counter.items())) for key, counter in sorted(grouped.items())}


def _confusion_matrix(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    matrix: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        matrix[str(row["selected_route"])][str(row["oracle_route"])] += 1
    return {key: dict(sorted(counter.items())) for key, counter in sorted(matrix.items())}


def _format_counter_table(counter: dict[str, int]) -> str:
    lines = ["| Route | Count |", "|---|---:|"]
    for key, value in sorted(counter.items()):
        lines.append(f"| {key} | {value} |")
    return "\n".join(lines)


def _format_confusion_table(matrix: dict[str, dict[str, int]]) -> str:
    routes = [DIRECT_ROUTE, COT_ROUTE, FULL_ROUTE, NONE_ROUTE]
    lines = [
        "| Selected \\ Oracle | direct | structured_cot | full_multi_agent | none |",
        "|---|---:|---:|---:|---:|",
    ]
    for selected in routes:
        row = matrix.get(selected, {})
        values = [str(row.get(route, 0)) for route in routes]
        lines.append(f"| {selected} | " + " | ".join(values) + " |")
    return "\n".join(lines)


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _sample_sort_key(sample_id: str) -> tuple[int, str]:
    return (int(sample_id), sample_id) if sample_id.isdigit() else (10**9, sample_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Phase 6 oracle routing.")
    parser.add_argument("--direct", type=Path, required=True)
    parser.add_argument("--cot", type=Path, required=True)
    parser.add_argument("--full-agent", type=Path, required=True)
    parser.add_argument("--routing", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    direct_records = load_predictions(resolve_project_path(args.direct))
    cot_records = load_predictions(resolve_project_path(args.cot))
    full_records = load_predictions(resolve_project_path(args.full_agent))
    routing_records = load_predictions(resolve_project_path(args.routing))

    rows = build_analysis_rows(
        direct_records=direct_records,
        cot_records=cot_records,
        full_records=full_records,
        routing_records=routing_records,
    )
    metrics = summarize_rows(rows)
    write_outputs(rows, metrics, resolve_project_path(args.output))
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
