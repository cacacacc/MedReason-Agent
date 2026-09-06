"""VQA-RAD dataset loader."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from medreason_agent.paths import resolve_project_path


@dataclass(frozen=True)
class VQARADSample:
    """One normalized VQA-RAD sample."""

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
        return resolve_project_path(self.image_path)


def load_vqa_rad_split(split: str, max_samples: int | None = None) -> list[VQARADSample]:
    """Load a processed VQA-RAD split."""
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
            if max_samples is not None and len(samples) >= max_samples:
                break

    return samples
