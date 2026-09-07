"""医学文本切分工具。

RAG 的第一步不是直接把整篇摘要塞给模型，而是先把文档切成较短 chunk。这样检索器
可以返回更聚焦的证据，也能控制 VLM prompt 的长度。
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class KnowledgeDocument:
    """一篇医学知识文档。

    第一版主要面向 PubMed/PubMed Central 摘要。`doc_id` 用于追踪来源，
    `title` 和 `text` 用于生成可检索文本。
    """

    doc_id: str
    title: str
    text: str
    source: str = ""


@dataclass(frozen=True)
class KnowledgeChunk:
    """一段可被检索的医学证据 chunk。"""

    chunk_id: str
    doc_id: str
    title: str
    text: str
    source: str = ""


def normalize_whitespace(text: str) -> str:
    """压缩多余空白，避免换行和制表符影响 chunk 与检索。"""
    return re.sub(r"\s+", " ", text).strip()


def chunk_document(
    document: KnowledgeDocument,
    chunk_size: int = 120,
    overlap: int = 20,
) -> list[KnowledgeChunk]:
    """把一篇文档按词切成多个 chunk。

    当前实现用 word-level chunk，简单、可复现、无额外依赖。后续如果接入 tokenizer，
    可以把这里升级成 token-level chunk，但外部接口保持不变。
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if overlap < 0:
        raise ValueError("overlap must be non-negative.")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    text = normalize_whitespace(document.text)
    words = text.split()
    if not words:
        return []

    chunks: list[KnowledgeChunk] = []
    step = chunk_size - overlap
    for start in range(0, len(words), step):
        chunk_words = words[start : start + chunk_size]
        if not chunk_words:
            break

        chunk_index = len(chunks)
        chunks.append(
            KnowledgeChunk(
                chunk_id=f"{document.doc_id}::chunk_{chunk_index:04d}",
                doc_id=document.doc_id,
                title=document.title,
                text=" ".join(chunk_words),
                source=document.source,
            )
        )

        if start + chunk_size >= len(words):
            break

    return chunks


def chunk_documents(
    documents: list[KnowledgeDocument],
    chunk_size: int = 120,
    overlap: int = 20,
) -> list[KnowledgeChunk]:
    """批量切分文档，输出一个平铺的 chunk 列表。"""
    chunks: list[KnowledgeChunk] = []
    for document in documents:
        chunks.extend(chunk_document(document, chunk_size=chunk_size, overlap=overlap))
    return chunks
