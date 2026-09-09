"""汇总 Phase 4 / Phase 5 调参结果。

这个脚本只读取 `Results/**/metrics.json`，不会重新跑模型。适合在 AutoDL 上跑完
100 条调参实验后快速生成表格，判断哪个配置值得扩展到 full。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from medreason_agent.paths import resolve_project_path

PHASE4_FIELDS = (
    "experiment_id",
    "num_samples",
    "accuracy",
    "mean_token_f1",
    "mean_evidence_quality_score",
    "hallucination_rate",
    "mean_tool_selection_accuracy",
    "top_k",
    "question_routing",
    "memory_enabled",
)

PHASE5_FIELDS = (
    "experiment_id",
    "num_samples",
    "candidate_accuracy",
    "final_accuracy",
    "error_correction_rate",
    "error_regression_rate",
    "correct_answer_preservation_rate",
    "net_correction",
    "revision_rate",
    "selective_verification_rate",
    "verifier_call_count",
    "question_routing",
)


def parse_args() -> argparse.Namespace:
    """解析汇总脚本参数。"""
    parser = argparse.ArgumentParser(description="汇总 Phase 4 / Phase 5 调参 metrics。")
    parser.add_argument("--results-dir", type=Path, default=Path("Results"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Results/tuning_summary"),
    )
    parser.add_argument(
        "--include-mock",
        action="store_true",
        help="默认跳过 mock metrics；调试时可打开。",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results_dir = resolve_project_path(args.results_dir)
    output_dir = resolve_project_path(args.output_dir)
    records = load_metrics(results_dir, include_mock=args.include_mock)

    phase4 = [record for record in records if _phase(record) == 4]
    phase5 = [record for record in records if _phase(record) == 5]
    summary = {
        "phase4": phase4,
        "phase5": phase5,
        "recommendation": build_recommendation(phase4, phase5),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    markdown = build_markdown(summary)
    (output_dir / "summary.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


def load_metrics(results_dir: Path, include_mock: bool) -> list[dict[str, Any]]:
    """读取 Results 下所有 metrics.json。"""
    records = []
    for path in sorted(results_dir.glob("**/metrics.json")):
        try:
            metrics = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not include_mock and metrics.get("backend") == "mock":
            continue
        metrics["metrics_path"] = str(path)
        records.append(metrics)
    return records


def build_recommendation(
    phase4: list[dict[str, Any]],
    phase5: list[dict[str, Any]],
) -> dict[str, Any]:
    """基于当前 metrics 给出机械但可解释的推荐。"""
    best_phase4 = _best_by_accuracy(phase4, key="accuracy")
    best_phase5 = _best_by_accuracy(phase5, key="final_accuracy")
    phase5_positive = [
        record
        for record in phase5
        if _number(record.get("net_correction")) > 0
        and _number(record.get("error_regression_rate")) <= 0.05
    ]
    return {
        "best_phase4_by_accuracy": best_phase4.get("experiment_id") if best_phase4 else "",
        "best_phase5_by_final_accuracy": best_phase5.get("experiment_id") if best_phase5 else "",
        "phase5_positive_net_correction": [
            record.get("experiment_id") for record in phase5_positive
        ],
        "full_run_advice": _full_run_advice(best_phase4, best_phase5, phase5_positive),
    }


def build_markdown(summary: dict[str, Any]) -> str:
    """生成可直接贴进中文实验记录的 Markdown 表格。"""
    lines = ["# Phase 4 / Phase 5 Tuning Summary", ""]
    lines.extend(_table("Phase 4", summary["phase4"], PHASE4_FIELDS))
    lines.extend([""])
    lines.extend(_table("Phase 5", summary["phase5"], PHASE5_FIELDS))
    lines.extend(["", "## Recommendation", ""])
    recommendation = summary["recommendation"]
    for key, value in recommendation.items():
        lines.append(f"- `{key}`: {value}")
    return "\n".join(lines)


def _table(title: str, records: list[dict[str, Any]], fields: tuple[str, ...]) -> list[str]:
    """把 metrics records 转成 Markdown 表。"""
    lines = [f"## {title}", ""]
    if not records:
        return [*lines, "No non-mock metrics found."]

    sorted_records = sorted(
        records,
        key=lambda item: (
            _number(item.get("final_accuracy", item.get("accuracy"))),
            -_number(item.get("mean_latency_ms")),
        ),
        reverse=True,
    )
    lines.append("| " + " | ".join(fields) + " |")
    lines.append("| " + " | ".join("---" for _ in fields) + " |")
    for record in sorted_records:
        values = [_format_value(record.get(field)) for field in fields]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _full_run_advice(
    best_phase4: dict[str, Any],
    best_phase5: dict[str, Any],
    phase5_positive: list[dict[str, Any]],
) -> str:
    """生成 full run 选择建议。"""
    if not best_phase4 and not best_phase5:
        return "No valid non-mock tuning results yet."
    if best_phase5 and phase5_positive:
        return (
            "Run full for the strongest Phase 4 route and the Phase 5 verifier "
            "configuration with positive net correction."
        )
    return (
        "Run full for the best Phase 4 route first. Keep Phase 5 as analysis-only "
        "unless it improves final_accuracy or net_correction."
    )


def _best_by_accuracy(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    """按 accuracy 类指标选择最佳记录。"""
    if not records:
        return {}
    return max(records, key=lambda item: _number(item.get(key)))


def _phase(record: dict[str, Any]) -> int | None:
    """从 experiment_id 判断 phase。"""
    experiment_id = str(record.get("experiment_id", ""))
    if experiment_id.startswith("exp04"):
        return 4
    if experiment_id.startswith("exp05"):
        return 5
    return None


def _number(value: Any) -> float:
    """把可选指标转成排序用数字。"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _format_value(value: Any) -> str:
    """格式化表格值。"""
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
