"""实验断点续跑工具。

VLM / CoT / RAG 这类实验通常运行时间较长，中断后如果从头开始会浪费大量 GPU 时间。
本模块统一处理 `predictions.jsonl` 的读取、追加、去重和排序，让不同 phase 的 runner
都可以安全复用同一套续跑逻辑。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def load_records_by_sample_id(path: Path) -> dict[str, dict[str, Any]]:
    """读取已有 prediction 文件，并按 `sample_id` 建立索引。

    如果同一个 `sample_id` 在文件里出现多次，保留最后一次记录。这样即使之前的实验
    中途追加过重复行，最终也会以最新结果为准，并在本轮结束时被整理成无重复 JSONL。
    """
    if not path.exists():
        return {}

    records: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc

            sample_id = record.get("sample_id")
            if sample_id is None:
                continue
            records[str(sample_id)] = record
    return records


def append_jsonl_record(path: Path, record: dict[str, Any]) -> None:
    """把单条 prediction 立即追加到 JSONL 文件。

    runner 每完成一个样本就调用一次这个函数。即使进程随后被关闭，已经完成的样本也会
    留在磁盘上，下次运行时可以被 `load_records_by_sample_id` 识别并跳过。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def ordered_records_for_sample_ids(
    records_by_sample_id: dict[str, dict[str, Any]],
    sample_ids: Iterable[str],
) -> list[dict[str, Any]]:
    """按照当前实验 split 的样本顺序导出 records。

    这样最终的 `predictions.jsonl` 顺序稳定，便于人工检查、错误分析和不同方法之间对齐。
    不属于当前配置的旧记录会被过滤掉，避免同一个输出目录被误复用时污染指标。
    """
    return [
        records_by_sample_id[sample_id]
        for sample_id in sample_ids
        if sample_id in records_by_sample_id
    ]

