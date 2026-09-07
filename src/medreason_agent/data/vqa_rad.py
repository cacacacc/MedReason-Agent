"""VQA-RAD 数据读取器。

这个模块只读取已经处理好的 `Data/Processed/vqa_rad/*.jsonl` 文件。原始 OSF 数据
下载和清洗由 `scripts/import_vqa_rad.py` 负责。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from medreason_agent.paths import resolve_project_path


@dataclass(frozen=True)
class VQARADSample:
    """一条归一化后的 VQA-RAD 样本。

    实验脚本不直接操作原始 JSON dict，而是使用这个结构化对象。这样字段名更稳定，
    也方便以后扩展到其他 Medical VQA 数据集。
    """

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
    def from_record(cls, record: dict) -> VQARADSample:
        """从处理后的 JSONL record 构造 `VQARADSample`。"""
        return cls(
            dataset=str(record["dataset"]),
            sample_id=str(record["sample_id"]),
            image_id=str(record["image_id"]),
            image_path=str(record["image_path"]),
            question=str(record["question"]),
            answer=str(record["answer"]),
            answer_type=str(record["answer_type"]),
            question_type=str(record["question_type"]),
            image_organ=str(record["image_organ"]),
            split=str(record["split"]),
        )

    @property
    def absolute_image_path(self) -> Path:
        """返回图像文件的项目内绝对路径，供 VLM backend 读取图片。"""
        return resolve_project_path(self.image_path)


def load_vqa_rad_split(split: str, max_samples: int | None = None) -> list[VQARADSample]:
    """读取处理后的 VQA-RAD split。

    `max_samples` 用于 smoke -> dev -> full 的逐步实验协议。设为 `None` 时读取
    整个 split，例如 full test 的 451 条样本。
    """
    if split not in {"train", "validation", "test", "all"}:
        raise ValueError(f"Unsupported VQA-RAD split: {split}")

    path = resolve_project_path(f"Data/Processed/vqa_rad/{split}.jsonl")
    if not path.exists():
        raise FileNotFoundError(
            f"Missing processed VQA-RAD split: {path}. "
            "Run scripts/import_vqa_rad.py before running experiments."
        )

    samples: list[VQARADSample] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                samples.append(VQARADSample.from_record(json.loads(line)))
            # 达到 `max_samples` 后提前停止，避免 smoke/dev 实验浪费算力。
            if max_samples is not None and len(samples) >= max_samples:
                break

    return samples
