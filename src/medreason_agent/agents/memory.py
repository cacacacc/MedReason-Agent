"""跨运行持久化 Agent Memory。

这个 memory 和 `SharedAgentState` 不同：Shared State 只在单个样本内部存在；
Persistent Memory 会写入磁盘 JSONL，下次运行或下次对话仍可读取。

为了避免 benchmark 泄漏，memory record 默认不保存 ground truth，也不把历史
final answer 当成事实。主实验应默认关闭，只有 memory ablation 才开启。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from medreason_agent.data.vqa_rad import VQARADSample
from medreason_agent.evaluation.evidence_metrics import content_tokens

PERSISTENT_MEMORY_VERSION = "persistent_memory_v1"


@dataclass(frozen=True)
class PersistentMemorySettings:
    """持久 memory 的运行配置。"""

    enabled: bool = False
    path: Path | None = None
    namespace: str = "default"
    top_k: int = 3
    min_score: float = 0.05
    read_enabled: bool = True
    write_enabled: bool = True
    include_prediction: bool = False


class PersistentAgentMemoryStore:
    """基于 JSONL 的轻量持久 memory store。

    第一版使用可解释的关键词重叠检索，不引入额外 embedding/index 依赖。后续如果要做
    长期 memory 专项实验，可以把这里替换为 BGE + FAISS memory index。
    """

    def __init__(self, settings: PersistentMemorySettings) -> None:
        if settings.path is None:
            raise ValueError("Persistent memory path is required when memory is enabled.")
        self.settings = settings
        self.path = settings.path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def retrieve(self, sample: VQARADSample) -> list[dict[str, Any]]:
        """按问题文本从历史 memory 中检索相似经验。"""
        if not self.settings.read_enabled or not self.path.exists():
            return []

        query_tokens = content_tokens(
            " ".join(
                [
                    sample.question,
                    sample.image_organ,
                    sample.question_type,
                    sample.answer_type,
                ]
            )
        )
        if not query_tokens:
            return []

        scored: list[tuple[float, dict[str, Any]]] = []
        for record in self._iter_records():
            if record.get("namespace") != self.settings.namespace:
                continue
            if record.get("sample_id") == sample.sample_id:
                continue
            memory_tokens = content_tokens(str(record.get("memory_text", "")))
            if not memory_tokens:
                continue
            score = len(query_tokens & memory_tokens) / len(query_tokens)
            if score >= self.settings.min_score:
                scored.append((score, dict(record, score=round(score, 6))))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [record for _, record in scored[: self.settings.top_k]]

    def append_from_state(
        self,
        sample: VQARADSample,
        shared_state: dict[str, Any],
        prediction: str,
        experiment_id: str,
    ) -> dict[str, Any] | None:
        """把当前样本的压缩经验追加写入持久 memory。"""
        if not self.settings.write_enabled:
            return None

        record = build_persistent_memory_record(
            sample=sample,
            shared_state=shared_state,
            prediction=prediction,
            experiment_id=experiment_id,
            namespace=self.settings.namespace,
            include_prediction=self.settings.include_prediction,
        )
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def _iter_records(self) -> list[dict[str, Any]]:
        """读取 JSONL memory；坏行直接跳过，避免单条损坏影响实验。"""
        records: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records


def build_persistent_memory_record(
    sample: VQARADSample,
    shared_state: dict[str, Any],
    prediction: str,
    experiment_id: str,
    namespace: str,
    include_prediction: bool = False,
) -> dict[str, Any]:
    """从当前 shared state 生成一条可持久化 memory record。"""
    claim_statuses = list(shared_state.get("claim_statuses", []))
    evidence = list(shared_state.get("retrieved_evidence", []))
    memory_text = format_memory_text(
        sample=sample,
        claim_statuses=claim_statuses,
        evidence=evidence,
        prediction=prediction if include_prediction else "",
    )
    memory_id = _memory_id(namespace=namespace, sample_id=sample.sample_id, text=memory_text)
    return {
        "memory_id": memory_id,
        "version": PERSISTENT_MEMORY_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "namespace": namespace,
        "experiment_id": experiment_id,
        "dataset": sample.dataset,
        "split": sample.split,
        "sample_id": sample.sample_id,
        "image_id": sample.image_id,
        "image_organ": sample.image_organ,
        "question_type": sample.question_type,
        "answer_type": sample.answer_type,
        "question": sample.question,
        "claim_statuses": claim_statuses,
        "evidence_titles": [str(item.get("title", "")) for item in evidence[:3]],
        "memory_text": memory_text,
        "stores_ground_truth": False,
        "stores_prediction": include_prediction,
    }


def format_memory_text(
    sample: VQARADSample,
    claim_statuses: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    prediction: str = "",
) -> str:
    """生成用于下次检索和 prompt 注入的 compact memory 文本。"""
    parts = [
        f"Question: {sample.question}",
        f"Image Organ: {sample.image_organ}",
        f"Question Type: {sample.question_type}",
        f"Answer Type: {sample.answer_type}",
    ]
    if claim_statuses:
        parts.append(
            "Claim Statuses:\n"
            + "\n".join(
                f"- [{item.get('status', '')}] {item.get('claim', '')}"
                for item in claim_statuses
            )
        )
    if evidence:
        parts.append(
            "Evidence Titles:\n"
            + "\n".join(f"- {item.get('title', '')}" for item in evidence[:3])
        )
    if prediction:
        parts.append(f"Previous Prediction: {prediction}")
    return "\n".join(parts)


def format_persistent_memories(memories: list[dict[str, Any]]) -> str:
    """把检索到的持久 memory 格式化给 SharedAgentState。"""
    if not memories:
        return ""

    lines: list[str] = []
    for index, memory in enumerate(memories, start=1):
        memory_text = " ".join(str(memory.get("memory_text", "")).split())
        lines.append(
            f"[Memory {index}] score={memory.get('score', '')}; "
            f"source_sample={memory.get('sample_id', '')}\n{memory_text}"
        )
    return "\n\n".join(lines)


def _memory_id(namespace: str, sample_id: str, text: str) -> str:
    """为 memory record 生成稳定 ID。"""
    digest = hashlib.sha1(f"{namespace}|{sample_id}|{text}".encode()).hexdigest()
    return digest[:16]
