"""对比 Direct VLM 和 CoT 的 prediction 文件。

这个模块用于回答 Phase 2 的核心问题：在同一批样本上，显式推理相对 Phase 1
直接回答 baseline 是帮助了结果，还是伤害了结果？
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """从 JSONL 文件读取项目 prediction records。"""
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))
    return records


def outcome_label(direct_correct: bool, cot_correct: bool) -> str:
    """标记 CoT 相对 direct inference 如何改变结果。

    四种标签能区分真正改进和退步，比只比较整体 accuracy 更有信息量。
    """
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
    """根据 sample_id 对比两组 prediction records。

    按 `sample_id` 匹配非常关键：Direct 和 CoT 必须在同一批 VQA-RAD 样本上
    比较，结果才是 controlled comparison。
    """
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
        # 分组统计可以看到 CoT 在哪里有帮助：例如 open questions 和 closed
        # yes/no questions 可能表现完全不同。
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
    """写出 summary 和逐样本 comparison 输出。"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # summary 文件较小，适合快速查看或写入实验报告。
    summary = {key: value for key, value in comparison.items() if key != "comparisons"}
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    # JSONL 每行保存一个样本的对比，方便后续错误分析和人工检查。
    with (output_dir / "comparisons.jsonl").open("w", encoding="utf-8") as file:
        for record in comparison["comparisons"]:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
