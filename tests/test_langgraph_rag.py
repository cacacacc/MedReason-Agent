import pytest

from medreason_agent.agents.langgraph_rag import create_langgraph_rag_agent
from medreason_agent.agents.rag_agents import RetrievalPipeline, RetrievalSettings
from medreason_agent.data.vqa_rad import VQARADSample
from medreason_agent.models.vlm import MockVLMBackend
from medreason_agent.retrieval.chunking import KnowledgeDocument, chunk_document
from medreason_agent.retrieval.keyword import KeywordRetriever
from medreason_agent.retrieval.rerank import KeywordReranker


def _sample() -> VQARADSample:
    return VQARADSample(
        dataset="VQA-RAD",
        sample_id="s1",
        image_id="image.jpg",
        image_path="Data/Raw/vqa_rad/images/image.jpg",
        question="Is there right lung opacity?",
        answer="yes",
        answer_type="CLOSED",
        question_type="PRES",
        image_organ="CHEST",
        split="test",
    )


def _retrieval_pipeline() -> RetrievalPipeline:
    chunks = chunk_document(
        KnowledgeDocument(
            doc_id="opacity",
            title="Right lung opacity and pneumonia",
            text="Right lung opacity can be associated with pneumonia or consolidation.",
        )
    )
    return RetrievalPipeline(
        retriever=KeywordRetriever(chunks),
        reranker=KeywordReranker(),
        settings=RetrievalSettings(
            candidate_top_k=3,
            top_k=2,
            min_rerank_score=0.0,
            max_chars_per_evidence=300,
        ),
    )


def test_langgraph_knowledge_rag_matches_baseline_contract() -> None:
    pytest.importorskip("langgraph")
    agent = create_langgraph_rag_agent(
        rag_mode="knowledge_acquisition",
        backend=MockVLMBackend(),
        retrieval_pipeline=_retrieval_pipeline(),
    )

    result = agent.run(_sample(), max_new_tokens=64)

    assert result.prediction == "yes"
    assert result.agent_route[0] == "langgraph"
    assert result.evidence_query == "Is there right lung opacity?"
    assert result.tool_calls[0] == {
        "tool": "langgraph_state_graph",
        "stage": "orchestration",
    }
