"""Build a BGE-small + Qdrant collection for Phase 3 controlled retrieval runs."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from medreason_agent.paths import resolve_project_path
from medreason_agent.retrieval.faiss_retriever import SentenceTransformerEmbedder
from medreason_agent.retrieval.keyword import load_chunks


def parse_args() -> argparse.Namespace:
    """Parse Qdrant collection build arguments."""
    parser = argparse.ArgumentParser(description="Build a BGE-small + Qdrant collection.")
    parser.add_argument(
        "--chunks",
        type=Path,
        default=Path("Data/Processed/medical_kb/pmc_10k_chunks.jsonl"),
    )
    parser.add_argument(
        "--qdrant-path",
        type=Path,
        default=Path("Data/Processed/medical_kb/qdrant_pmc_10k_bge_small"),
    )
    parser.add_argument("--collection-name", default="pmc_10k_bge_small")
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--upsert-batch-size", type=int, default=256)
    parser.add_argument(
        "--force-recreate",
        action="store_true",
        help="Drop and recreate the collection if it already exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, PointStruct, VectorParams
    except ImportError as exc:
        raise RuntimeError(
            "Qdrant collection build requires qdrant-client. Install with: "
            "pip install -r requirements/rag.txt"
        ) from exc

    chunks = load_chunks(resolve_project_path(args.chunks))
    if not chunks:
        raise ValueError(f"No chunks found in {args.chunks}.")

    embedder = SentenceTransformerEmbedder(args.embedding_model, device=args.device)
    texts = [f"{chunk.title}\n{chunk.text}" for chunk in chunks]
    all_embeddings: list[np.ndarray] = []
    for start in tqdm(range(0, len(texts), args.batch_size), desc="embed", ascii=True):
        batch = texts[start : start + args.batch_size]
        all_embeddings.append(embedder.encode(batch, batch_size=args.batch_size))

    embeddings = np.vstack(all_embeddings).astype(np.float32)
    dimension = int(embeddings.shape[1])
    qdrant_path = resolve_project_path(args.qdrant_path)
    qdrant_path.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(qdrant_path))

    if args.force_recreate:
        client.recreate_collection(
            collection_name=args.collection_name,
            vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
        )
    else:
        existing = {collection.name for collection in client.get_collections().collections}
        if args.collection_name not in existing:
            client.create_collection(
                collection_name=args.collection_name,
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
            )

    for start in tqdm(
        range(0, len(chunks), args.upsert_batch_size),
        desc="upsert",
        ascii=True,
    ):
        batch_chunks = chunks[start : start + args.upsert_batch_size]
        batch_embeddings = embeddings[start : start + len(batch_chunks)]
        points = [
            PointStruct(
                id=start + offset,
                vector=batch_embeddings[offset].tolist(),
                payload=asdict(chunk),
            )
            for offset, chunk in enumerate(batch_chunks)
        ]
        client.upsert(collection_name=args.collection_name, points=points)

    print(
        json.dumps(
            {
                "embedding_model": args.embedding_model,
                "num_chunks": len(chunks),
                "embedding_dim": dimension,
                "qdrant_path": str(qdrant_path.relative_to(resolve_project_path("."))),
                "collection_name": args.collection_name,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
