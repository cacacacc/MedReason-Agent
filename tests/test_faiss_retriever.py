import numpy as np

from medreason_agent.retrieval.faiss_retriever import normalize_embeddings


def test_normalize_embeddings_returns_unit_vectors() -> None:
    embeddings = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)

    normalized = normalize_embeddings(embeddings)

    assert np.allclose(normalized[0], np.array([0.6, 0.8], dtype=np.float32))
    assert np.allclose(normalized[1], np.array([0.0, 0.0], dtype=np.float32))

