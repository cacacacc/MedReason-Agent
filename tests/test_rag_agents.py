from medreason_agent.agents.rag_agents import (
    EvidenceRAGAgent,
    KnowledgeRAGAgent,
    KnowledgeThenEvidenceRAGAgent,
    RetrievalPipeline,
    RetrievalSettings,
)
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


def test_knowledge_rag_retrieves_before_reasoning() -> None:
    agent = KnowledgeRAGAgent(
        backend=MockVLMBackend(),
        retrieval_pipeline=_retrieval_pipeline(),
    )

    result = agent.run(_sample(), max_new_tokens=64)

    assert result.prediction == "yes"
    assert result.evidence_query == "Is there right lung opacity?"
    assert result.initial_prediction == ""
    assert {
        "claim": "mock visual observation",
        "status": "OBSERVED",
        "source_agent": "clinical_reasoning_agent",
    } in result.claim_statuses
    assert result.agent_route == [
        "knowledge_agent",
        "retriever",
        "reranker",
        "evidence_filter",
        "clinical_reasoning_agent",
    ]
    assert result.tool_calls[0]["stage"] == "pre_reasoning"


def test_evidence_rag_retrieves_after_initial_claim() -> None:
    agent = EvidenceRAGAgent(
        backend=MockVLMBackend(),
        retrieval_pipeline=_retrieval_pipeline(),
    )

    result = agent.run(_sample(), max_new_tokens=64)

    assert result.prediction == "yes"
    assert result.initial_prediction == "yes"
    assert result.verified_claim == "yes"
    assert result.claim_verification_status == "SUPPORTED"
    assert result.critic_decision == "SUPPORTED"
    assert {
        "claim": "yes",
        "status": "HYPOTHESIS",
        "source_agent": "reasoning_agent",
    } in result.claim_statuses
    assert {
        "claim": "yes",
        "status": "SUPPORTED",
        "source_agent": "verifier_agent",
    } in result.claim_statuses
    assert "Claim: yes" in result.evidence_query
    assert result.agent_route == [
        "reasoning_agent",
        "claim_extractor",
        "evidence_agent",
        "retriever",
        "reranker",
        "evidence_filter",
        "verifier_agent",
    ]
    assert [call["stage"] for call in result.tool_calls] == [
        "initial_reasoning",
        "post_reasoning",
        "claim_verification",
        "claim_verification",
        "claim_verification",
        "verification",
    ]


def test_knowledge_then_evidence_rag_uses_both_rag_modes() -> None:
    agent = KnowledgeThenEvidenceRAGAgent(
        backend=MockVLMBackend(),
        retrieval_pipeline=_retrieval_pipeline(),
    )

    result = agent.run(_sample(), max_new_tokens=64)

    assert result.prediction == "yes"
    assert result.knowledge_query == "Is there right lung opacity?"
    assert "Claim: yes" in result.evidence_query
    assert result.generated_claims == ["yes"]
    assert result.claim_verification_status == "SUPPORTED"
    assert result.critic_decision == "SUPPORTED"
    assert {
        "claim": "yes",
        "status": "HYPOTHESIS",
        "source_agent": "reasoning_agent",
    } in result.claim_statuses
    assert {
        "claim": "yes",
        "status": "SUPPORTED",
        "source_agent": "verifier_agent",
    } in result.claim_statuses
    assert {item["evidence_stage"] for item in result.retrieved_evidence} == {
        "knowledge_acquisition",
        "claim_verification",
    }
    assert result.agent_route == [
        "knowledge_agent",
        "retriever",
        "reranker",
        "evidence_filter",
        "reasoning_agent",
        "claim_extractor",
        "evidence_agent",
        "retriever",
        "reranker",
        "evidence_filter",
        "verifier_agent",
    ]
