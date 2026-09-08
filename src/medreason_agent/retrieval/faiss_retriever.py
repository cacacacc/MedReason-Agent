"""BGE-small + FAISS 向量检索器。

Phase 3 主 RAG 实验使用 `BAAI/bge-small-en-v1.5` 把医学摘要 chunks 编成向量，
再用 FAISS 做近邻检索。这里保留和 `KeywordRetriever` 一致的输出结构，方便
Knowledge RAG、Evidence RAG 和组合 verifier 共用同一个 agent 接口。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from medreason_agent.retrieval.keyword import RetrievedEvidence


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """把 embedding 归一化，使内积等价于 cosine similarity。"""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return embeddings / norms


def load_vector_metadata(path: Path) -> list[dict[str, Any]]:
    """读取 FAISS index 对应的 chunk metadata。"""
    metadata: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                metadata.append(json.loads(line))
    return metadata


class SentenceTransformerEmbedder:
    """sentence-transformers embedding backend。"""

    def __init__(self, model_name_or_path: str, device: str | None = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Vector retrieval requires sentence-transformers. Install with: "
                "pip install -r requirements/rag.txt"
            ) from exc

        self.model = SentenceTransformer(model_name_or_path, device=device)

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """把文本编码成 float32 numpy array。"""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)


class FAISSRetriever:
    """基于 FAISS IndexFlatIP 的向量检索器。"""

    def __init__(
        self,
        index_path: Path,
        metadata_path: Path,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        device: str | None = None,
    ) -> None:
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError(
                "FAISS retrieval requires faiss-cpu. Install with: "
                "pip install -r requirements/rag.txt"
            ) from exc

        self.embedding_model = embedding_model
        self.embedder = SentenceTransformerEmbedder(embedding_model, device=device)
        self.index = faiss.read_index(str(index_path))
        self.metadata = load_vector_metadata(metadata_path)
        if self.index.ntotal != len(self.metadata):
            raise ValueError(
                f"FAISS index size ({self.index.ntotal}) does not match metadata "
                f"records ({len(self.metadata)})."
            )

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedEvidence]:
        """检索和 query 最相近的 top-k chunks。"""
        if top_k <= 0 or not query.strip():
            return []

        query_embedding = self.embedder.encode([query], batch_size=1)
        scores, indices = self.index.search(query_embedding, top_k)

        evidence: list[RetrievedEvidence] = []
        for score, index_id in zip(scores[0], indices[0], strict=True):
            if index_id < 0:
                continue
            record = self.metadata[int(index_id)]
            evidence.append(
                RetrievedEvidence(
                    chunk_id=str(record["chunk_id"]),
                    doc_id=str(record["doc_id"]),
                    title=str(record.get("title", "")),
                    text=str(record["text"]),
                    score=round(float(score), 6),
                    source=str(record.get("source", "")),
                )
            )
        return evidence

