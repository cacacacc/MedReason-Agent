"""BGE-small + Qdrant vector retriever.

This retriever intentionally mirrors ``FAISSRetriever``: it uses the same
SentenceTransformer embedding model, the same cosine-style normalized vectors,
and returns the same ``RetrievedEvidence`` records. That makes FAISS vs Qdrant
an infrastructure ablation instead of a different retrieval-method experiment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from medreason_agent.retrieval.faiss_retriever import SentenceTransformerEmbedder
from medreason_agent.retrieval.keyword import RetrievedEvidence


class QdrantRetriever:
    """Vector retriever backed by a local or remote Qdrant collection."""

    def __init__(
        self,
        collection_name: str,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        device: str | None = None,
        path: Path | None = None,
        url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            raise RuntimeError(
                "Qdrant retrieval requires qdrant-client. Install with: "
                "pip install -r requirements/rag.txt"
            ) from exc

        if path is None and url is None:
            raise ValueError("QdrantRetriever requires either a local path or a URL.")

        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.embedder = SentenceTransformerEmbedder(embedding_model, device=device)
        if url:
            self.client = QdrantClient(url=url, api_key=api_key)
        else:
            self.client = QdrantClient(path=str(path))

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedEvidence]:
        """Retrieve the nearest chunks for a query."""
        if top_k <= 0 or not query.strip():
            return []

        query_embedding = self.embedder.encode([query], batch_size=1)[0]
        points = _query_points(
            client=self.client,
            collection_name=self.collection_name,
            query_vector=query_embedding,
            top_k=top_k,
        )
        return [_point_to_evidence(point) for point in points]


def _query_points(
    client: Any,
    collection_name: str,
    query_vector: np.ndarray,
    top_k: int,
) -> list[Any]:
    """Run a Qdrant nearest-neighbor query across client API versions."""
    vector = query_vector.astype(np.float32).tolist()
    if hasattr(client, "query_points"):
        response = client.query_points(
            collection_name=collection_name,
            query=vector,
            limit=top_k,
            with_payload=True,
        )
        return list(getattr(response, "points", response))
    return list(
        client.search(
            collection_name=collection_name,
            query_vector=vector,
            limit=top_k,
            with_payload=True,
        )
    )


def _point_to_evidence(point: Any) -> RetrievedEvidence:
    """Convert a Qdrant scored point into the shared evidence dataclass."""
    payload = dict(getattr(point, "payload", None) or {})
    return RetrievedEvidence(
        chunk_id=str(payload.get("chunk_id", getattr(point, "id", ""))),
        doc_id=str(payload.get("doc_id", "")),
        title=str(payload.get("title", "")),
        text=str(payload.get("text", "")),
        score=round(float(getattr(point, "score", 0.0)), 6),
        source=str(payload.get("source", "")),
    )
