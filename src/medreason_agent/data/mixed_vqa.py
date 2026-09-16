"""混合 Medical VQA 数据读取工具。

这个模块用于 Phase 8 之后的数据增强 SFT。它读取已经标准化好的 JSONL 文件，
让训练脚本可以统一处理 VQA-RAD、PathVQA、SLAKE 等不同来源的短答案样本。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from medreason_agent.paths import resolve_project_path


@dataclass(frozen=True)
class MixedVQASample:
    """一条统一格式的 Medical VQA 样本。"""

    dataset: str
    sample_id: str
    image_id: str
    image_path: str
    question: str
    answer: str
    answer_type: str
    question_type: str
    image_organ: str
    split: str

    @classmethod
    def from_record(cls, record: dict) -> MixedVQASample:
        """从标准化 JSONL record 构造样本对象。"""
        return cls(
            dataset=str(record.get("dataset", "mixed_vqa")),
            sample_id=str(record["sample_id"]),
            image_id=str(record.get("image_id", record["sample_id"])),
            image_path=str(record["image_path"]),
            question=str(record["question"]),
            answer=str(record["answer"]),
            answer_type=str(record.get("answer_type", "OPEN")),
            question_type=str(record.get("question_type", "OTHER")),
            image_organ=str(record.get("image_organ", "UNKNOWN")),
            split=str(record.get("split", "train")),
        )

    @property
    def absolute_image_path(self) -> Path:
        """返回图片绝对路径；兼容项目相对路径和外部绝对路径。"""
        path = Path(self.image_path)
        if path.is_absolute():
            return path
        return resolve_project_path(path)


def load_mixed_vqa_jsonl(
    path: str | Path,
    max_samples: int | None = None,
) -> list[MixedVQASample]:
    """读取标准化混合 VQA JSONL 文件。"""
    resolved_path = resolve_project_path(path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Missing mixed VQA file: {resolved_path}")

    samples: list[MixedVQASample] = []
    with resolved_path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                samples.append(MixedVQASample.from_record(json.loads(line)))
            if max_samples is not None and len(samples) >= max_samples:
                break
    return samples
