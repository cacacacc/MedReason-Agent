"""轻量级关键词检索器。

这是 Phase 3 的第一版检索器，用来先跑通 RAG 实验闭环。它不是最终的 BGE+Qdrant
向量检索方案，但输出接口已经和后续 dense retriever 保持一致。
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from medreason_agent.retrieval.chunking import KnowledgeChunk


@dataclass(frozen=True)
class RetrievedEvidence:
    """一次检索返回的一条证据。"""

    chunk_id: str
    doc_id: str
    title: str
    text: str
    score: float
    source: str = ""


_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """把英文医学文本切成小写 token。

    这里先用正则 tokenizer，保证不依赖额外包。后续 BGE embedding 版本不会复用这个
    tokenization，但可以复用 `RetrievedEvidence` 输出结构。
    """
    return _TOKEN_PATTERN.findall(text.lower())


def load_chunks(path: Path) -> list[KnowledgeChunk]:
    """从 JSONL 文件读取知识库 chunks。"""
    chunks: list[KnowledgeChunk] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            chunks.append(
                KnowledgeChunk(
                    chunk_id=str(record["chunk_id"]),
                    doc_id=str(record["doc_id"]),
                    title=str(record.get("title", "")),
                    text=str(record["text"]),
                    source=str(record.get("source", "")),
                )
            )
    return chunks


def write_chunks(path: Path, chunks: list[KnowledgeChunk]) -> None:
    """把知识库 chunks 写成 JSONL，供 RAG runner 加载。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")


class KeywordRetriever:
    """基于 TF-IDF 风格打分的本地关键词检索器。

    这个类的输入是一组 `KnowledgeChunk`。查询时，它会按 query token 和 chunk token
    的重叠程度打分，返回 top-k evidence。
    """

    def __init__(self, chunks: list[KnowledgeChunk]) -> None:
        self.chunks = chunks
        self._chunk_tokens = [tokenize(f"{chunk.title} {chunk.text}") for chunk in chunks]
        self._doc_freq = self._build_doc_freq(self._chunk_tokens)

    @staticmethod
    def _build_doc_freq(chunk_tokens: list[list[str]]) -> Counter[str]:
        """统计每个 token 出现在多少个 chunk 中，用于计算 IDF。"""
        doc_freq: Counter[str] = Counter()
        for tokens in chunk_tokens:
            doc_freq.update(set(tokens))
        return doc_freq

    def _idf(self, token: str) -> float:
        """计算平滑 IDF，让少见医学词获得更高权重。"""
        total = max(len(self.chunks), 1)
        return math.log((1 + total) / (1 + self._doc_freq.get(token, 0))) + 1.0

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedEvidence]:
        """检索和问题最相关的 top-k 证据。"""
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        query_counts = Counter(query_tokens)
        scored: list[tuple[float, KnowledgeChunk]] = []
        for chunk, tokens in zip(self.chunks, self._chunk_tokens, strict=True):
            token_counts = Counter(tokens)
            score = 0.0
            for token, query_count in query_counts.items():
                if token in token_counts:
                    score += query_count * token_counts[token] * self._idf(token)
            if score > 0:
                scored.append((score, chunk))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            RetrievedEvidence(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                title=chunk.title,
                text=chunk.text,
                score=round(score, 6),
                source=chunk.source,
            )
            for score, chunk in scored[:top_k]
        ]


def evidence_to_record(evidence: RetrievedEvidence) -> dict[str, Any]:
    """把检索结果转成可写入 prediction record 的普通 dict。"""
    return asdict(evidence)
