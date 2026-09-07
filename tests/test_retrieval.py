from medreason_agent.retrieval.chunking import KnowledgeDocument, chunk_document
from medreason_agent.retrieval.keyword import KeywordRetriever
from medreason_agent.retrieval.rerank import KeywordReranker, filter_evidence


def test_chunk_document_creates_overlapping_chunks() -> None:
    document = KnowledgeDocument(
        doc_id="doc1",
        title="Test",
        text="one two three four five six",
    )

    chunks = chunk_document(document, chunk_size=4, overlap=2)

    assert [chunk.text for chunk in chunks] == ["one two three four", "three four five six"]


def test_keyword_retriever_returns_relevant_chunk_first() -> None:
    chunks = [
        *chunk_document(
            KnowledgeDocument(
                doc_id="a",
                title="Aortic aneurysm",
                text="Aortic aneurysm can show widened aortic contour.",
            )
        ),
        *chunk_document(
            KnowledgeDocument(
                doc_id="b",
                title="Brain lesion",
                text="Brain lesions can appear on head imaging.",
            )
        ),
    ]

    retriever = KeywordRetriever(chunks)
    evidence = retriever.retrieve("Is there evidence of an aortic aneurysm?", top_k=1)

    assert len(evidence) == 1
    assert evidence[0].doc_id == "a"


def test_keyword_reranker_promotes_query_matched_title() -> None:
    chunks = [
        *chunk_document(
            KnowledgeDocument(
                doc_id="a",
                title="General imaging",
                text="Aneurysm can be seen as a widened contour.",
            )
        ),
        *chunk_document(
            KnowledgeDocument(
                doc_id="b",
                title="Aortic aneurysm",
                text="Vascular abnormality on radiology.",
            )
        ),
    ]
    evidence = KeywordRetriever(chunks).retrieve("Is there an aortic aneurysm?", top_k=2)

    reranked = KeywordReranker().rerank("Is there an aortic aneurysm?", evidence)

    assert reranked[0].doc_id == "b"


def test_filter_evidence_keeps_only_high_quality_top_k() -> None:
    chunks = chunk_document(
        KnowledgeDocument(
            doc_id="a",
            title="Aortic aneurysm",
            text="Aortic aneurysm can show widened aortic contour.",
        )
    )
    evidence = KeywordRetriever(chunks).retrieve("aortic aneurysm", top_k=1)
    reranked = KeywordReranker().rerank("aortic aneurysm", evidence)

    assert filter_evidence(reranked, top_k=1, min_score=0.1)
    assert filter_evidence(reranked, top_k=1, min_score=999.0) == []
