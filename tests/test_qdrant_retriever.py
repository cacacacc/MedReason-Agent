from types import SimpleNamespace

from medreason_agent.retrieval.qdrant_retriever import _point_to_evidence


def test_point_to_evidence_uses_payload_fields() -> None:
    point = SimpleNamespace(
        id=7,
        score=0.875,
        payload={
            "chunk_id": "chunk-1",
            "doc_id": "doc-1",
            "title": "Opacity",
            "text": "Right lung opacity can indicate consolidation.",
            "source": "pmc",
        },
    )

    evidence = _point_to_evidence(point)

    assert evidence.chunk_id == "chunk-1"
    assert evidence.doc_id == "doc-1"
    assert evidence.title == "Opacity"
    assert evidence.text == "Right lung opacity can indicate consolidation."
    assert evidence.score == 0.875
    assert evidence.source == "pmc"
