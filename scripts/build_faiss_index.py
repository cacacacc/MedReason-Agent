"""为 Phase 3 PMC 10k chunks 构建 BGE-small + FAISS 向量索引。"""

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
    """解析 FAISS 索引构建参数。"""
    parser = argparse.ArgumentParser(description="构建 BGE-small + FAISS 检索索引。")
    parser.add_argument(
        "--chunks",
        type=Path,
        default=Path("Data/Processed/medical_kb/pmc_10k_chunks.jsonl"),
    )
    parser.add_argument(
        "--index-output",
        type=Path,
        default=Path("Data/Processed/medical_kb/pmc_10k_bge_small.faiss"),
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=Path("Data/Processed/medical_kb/pmc_10k_bge_small_metadata.jsonl"),
    )
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError(
            "FAISS index build requires faiss-cpu. Install with: "
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
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    index_path = resolve_project_path(args.index_output)
    metadata_path = resolve_project_path(args.metadata_output)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    with metadata_path.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "embedding_model": args.embedding_model,
                "num_chunks": len(chunks),
                "embedding_dim": dimension,
                "index_output": str(index_path.relative_to(resolve_project_path("."))),
                "metadata_output": str(metadata_path.relative_to(resolve_project_path("."))),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
