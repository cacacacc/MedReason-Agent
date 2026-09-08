"""医学文本切分工具。

RAG 的第一步不是直接把整篇摘要塞给模型，而是先把文档切成较短 chunk。这样检索器
可以返回更聚焦的证据，也能控制 VLM prompt 的长度。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol


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


class TextTokenizer(Protocol):
    """用于 token-level chunking 的最小 tokenizer 接口。"""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """把文本转成 token ids。"""

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        """把 token ids 还原成文本。"""


def normalize_whitespace(text: str) -> str:
    """压缩多余空白，避免换行和制表符影响 chunk 与检索。"""
    return re.sub(r"\s+", " ", text).strip()


def validate_chunking_args(chunk_size: int, overlap: int) -> None:
    """校验 chunk 参数，避免 step 为 0 或负数。"""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if overlap < 0:
        raise ValueError("overlap must be non-negative.")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")


def chunk_document(
    document: KnowledgeDocument,
    chunk_size: int = 120,
    overlap: int = 20,
) -> list[KnowledgeChunk]:
    """把一篇文档按词切成多个 chunk。

    当前实现用 word-level chunk，简单、可复现、无额外依赖。后续如果接入 tokenizer，
    可以把这里升级成 token-level chunk，但外部接口保持不变。
    """
    validate_chunking_args(chunk_size, overlap)

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


def chunk_document_by_tokens(
    document: KnowledgeDocument,
    tokenizer: TextTokenizer,
    chunk_size: int = 256,
    overlap: int = 50,
) -> list[KnowledgeChunk]:
    """把一篇文档按 tokenizer token 切成多个 chunk。

    这是 Phase 3 的主实验切分方式。相比 word-level chunk，token-level chunk 更接近
    模型真实上下文长度，适合后续接 Qwen / BGE / Qdrant。
    """
    validate_chunking_args(chunk_size, overlap)

    text = normalize_whitespace(document.text)
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        return []

    chunks: list[KnowledgeChunk] = []
    step = chunk_size - overlap
    for start in range(0, len(token_ids), step):
        chunk_token_ids = token_ids[start : start + chunk_size]
        if not chunk_token_ids:
            break

        chunk_index = len(chunks)
        chunk_text = normalize_whitespace(
            tokenizer.decode(chunk_token_ids, skip_special_tokens=True)
        )
        chunks.append(
            KnowledgeChunk(
                chunk_id=f"{document.doc_id}::chunk_{chunk_index:04d}",
                doc_id=document.doc_id,
                title=document.title,
                text=chunk_text,
                source=document.source,
            )
        )

        if start + chunk_size >= len(token_ids):
            break

    return chunks


def chunk_documents(
    documents: list[KnowledgeDocument],
    chunk_size: int = 120,
    overlap: int = 20,
    tokenizer: TextTokenizer | None = None,
) -> list[KnowledgeChunk]:
    """批量切分文档，输出一个平铺的 chunk 列表。"""
    chunks: list[KnowledgeChunk] = []
    for document in documents:
        if tokenizer is None:
            chunks.extend(chunk_document(document, chunk_size=chunk_size, overlap=overlap))
        else:
            chunks.extend(
                chunk_document_by_tokens(
                    document,
                    tokenizer=tokenizer,
                    chunk_size=chunk_size,
                    overlap=overlap,
                )
            )
    return chunks
