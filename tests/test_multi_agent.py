from medreason_agent.agents.memory import PersistentAgentMemoryStore, PersistentMemorySettings
from medreason_agent.agents.multi_agent import FixedMultiAgent, SupervisorMultiAgent
from medreason_agent.agents.rag_agents import RetrievalPipeline, RetrievalSettings
from medreason_agent.data.vqa_rad import VQARADSample
from medreason_agent.evaluation.agent_metrics import summarize_agent_metrics
from medreason_agent.models.vlm import MockVLMBackend, VLMRequest, VLMResponse
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


def _sample_with_id(sample_id: str) -> VQARADSample:
    sample = _sample()
    return VQARADSample(
        dataset=sample.dataset,
        sample_id=sample_id,
        image_id=sample.image_id,
        image_path=sample.image_path,
        question=sample.question,
        answer=sample.answer,
        answer_type=sample.answer_type,
        question_type=sample.question_type,
        image_organ=sample.image_organ,
        split=sample.split,
    )


def _retrieval_pipeline() -> RetrievalPipeline:
    chunks = chunk_document(
        KnowledgeDocument(
            doc_id="opacity",
            title="Right lung opacity",
            text="Right lung opacity can be associated with consolidation.",
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


class AnswerOnlySupervisorBackend(MockVLMBackend):
    def generate(self, request: VLMRequest) -> VLMResponse:
        if "Act as a supervisor for a multimodal medical reasoning system" in request.prompt:
            raw_output = (
                "Selected Tools: Answer Agent\n"
                "Rationale: the question can be answered directly."
            )
            return VLMResponse(
                answer=raw_output,
                raw_output=raw_output,
                input_tokens=len(request.prompt.split()),
                output_tokens=len(raw_output.split()),
            )
        return super().generate(request)


class UnsupportedVerifierBackend(MockVLMBackend):
    def generate(self, request: VLMRequest) -> VLMResponse:
        if "Verify the reasoning claims against retrieved evidence" in request.prompt:
            raw_output = (
                "Verification: the retrieved evidence does not support the claim.\n"
                "Verification Status: UNSUPPORTED.\n"
                'Claim Statuses:\n{"claim": "yes", "status": "UNSUPPORTED"}\n'
                "Unsupported Medical Claims: yes."
            )
            return VLMResponse(
                answer=raw_output,
                raw_output=raw_output,
                input_tokens=len(request.prompt.split()),
                output_tokens=len(raw_output.split()),
            )
        return super().generate(request)


def test_fixed_multi_agent_runs_fixed_route() -> None:
    result = FixedMultiAgent(MockVLMBackend()).run(_sample(), max_new_tokens=64)

    assert result.prediction == "yes"
    assert result.agent_route == ["vision_agent", "reasoning_agent", "critic_agent", "answer_agent"]
    assert result.critic_decision == "ACCEPT"
    assert result.generated_claims == ["yes"]
    assert {
        "claim": "yes",
        "status": "HYPOTHESIS",
        "source_agent": "reasoning_agent",
    } in result.claim_statuses
    assert result.retrieved_evidence == []
    assert result.expected_selected_tools == []
    assert result.shared_state["state_compression"]["method"] == "section_extract_v1"
    assert "Agent Memory:" in result.shared_state["compressed_context"]


def test_supervisor_multi_agent_runs_retrieval_and_verifier_route() -> None:
    result = SupervisorMultiAgent(
        backend=MockVLMBackend(),
        retrieval_pipeline=_retrieval_pipeline(),
    ).run(_sample(), max_new_tokens=64)

    assert result.prediction == "yes"
    assert result.agent_route == [
        "supervisor_agent",
        "vision_agent",
        "retrieval_agent",
        "reasoning_agent",
        "verifier_agent",
        "answer_agent",
    ]
    assert result.claim_verification_status == "SUPPORTED"
    assert result.critic_decision == "SUPPORTED"
    assert {
        "claim": "yes",
        "status": "SUPPORTED",
        "source_agent": "verifier_agent",
    } in result.claim_statuses
    assert result.retrieved_evidence
    assert "Retrieval Agent" in result.selected_tools
    assert result.selected_tools == result.expected_selected_tools
    assert result.shared_state["retrieved_evidence"]
    assert "Claim Statuses:" in result.shared_state["compressed_context"]
    assert result.answer_gate["decision"] == "DISABLED"


def test_supervisor_selected_tools_control_actual_route() -> None:
    result = SupervisorMultiAgent(
        backend=AnswerOnlySupervisorBackend(),
        retrieval_pipeline=_retrieval_pipeline(),
        dynamic_routing=True,
    ).run(_sample(), max_new_tokens=64)

    assert result.selected_tools == ["Answer Agent"]
    assert result.agent_route == ["supervisor_agent", "answer_agent"]
    assert result.expected_agent_route == ["supervisor_agent", "answer_agent"]
    assert result.agent_outputs["vision"] == ""
    assert result.agent_outputs["reasoning"] == ""
    assert result.agent_outputs["verifier"] == ""
    assert result.retrieved_evidence == []
    assert result.generated_claims == []
    assert result.prediction == "yes"


def test_heuristic_question_routing_skips_rag_for_simple_closed_question() -> None:
    result = SupervisorMultiAgent(
        backend=MockVLMBackend(),
        retrieval_pipeline=_retrieval_pipeline(),
        dynamic_routing=True,
        question_routing="heuristic",
    ).run(_sample(), max_new_tokens=64)

    assert result.selected_tools == ["Vision Agent", "Answer Agent"]
    assert result.agent_route == ["supervisor_agent", "vision_agent", "answer_agent"]
    assert result.retrieved_evidence == []
    assert result.agent_outputs["reasoning"] == ""
    assert result.agent_outputs["verifier"] == ""


def test_answer_gate_forces_uncertain_for_unsupported_claim() -> None:
    result = SupervisorMultiAgent(
        backend=UnsupportedVerifierBackend(),
        retrieval_pipeline=_retrieval_pipeline(),
        deterministic_answer_gate=True,
    ).run(_sample(), max_new_tokens=64)

    assert result.claim_verification_status == "UNSUPPORTED"
    assert result.answer_gate["decision"] == "RESTRICT_UNSUPPORTED"
    assert result.answer_gate["forced_prediction"] == "uncertain"
    assert result.prediction == "uncertain"
    assert "Answer Gate: RESTRICT_UNSUPPORTED" in result.shared_state["compressed_context"]


def test_supervisor_multi_agent_uses_persistent_memory(tmp_path) -> None:
    memory_store = PersistentAgentMemoryStore(
        PersistentMemorySettings(
            enabled=True,
            path=tmp_path / "memory.jsonl",
            namespace="unit",
            top_k=2,
            min_score=0.01,
        )
    )
    first = SupervisorMultiAgent(
        backend=MockVLMBackend(),
        retrieval_pipeline=_retrieval_pipeline(),
        memory_store=memory_store,
    ).run(_sample(), max_new_tokens=64, experiment_id="exp_memory")

    second = SupervisorMultiAgent(
        backend=MockVLMBackend(),
        retrieval_pipeline=_retrieval_pipeline(),
        memory_store=memory_store,
    ).run(_sample_with_id("s2"), max_new_tokens=64, experiment_id="exp_memory")

    assert first.memory_write_record is not None
    assert second.memory_records
    assert "Persistent Memory:" in second.shared_state["compressed_context"]


def test_summarize_agent_metrics() -> None:
    metrics = summarize_agent_metrics(
        [
            {
                "prediction": "yes",
                "agent_outputs": {
                    "vision": "Observation: finding",
                    "reasoning": "Reasoning: ok\nClaims:\n- yes",
                    "critic": "Unsupported Medical Claims: None",
                },
                "agent_route": ["a"],
                "expected_agent_route": ["a"],
                "selected_tools": ["Vision Agent", "Answer Agent"],
                "expected_selected_tools": ["Vision Agent", "Answer Agent"],
                "critic_decision": "ACCEPT",
                "claim_verification_status": "",
                "claim_statuses": [{"claim": "yes", "status": "HYPOTHESIS"}],
                "state_compression": {"compression_ratio": 0.5},
                "memory_records": [{"memory_id": "m1"}],
                "error_attribution": {
                    "primary_error_type": "",
                    "candidate_error_types": [],
                    "requires_human_review": False,
                },
                "answer_gate": {"decision": "ALLOW"},
            }
        ]
    )

    assert metrics["mean_reasoning_score"] == 1.0
    assert metrics["hallucination_rate"] == 0.0
    assert metrics["mean_tool_selection_accuracy"] == 1.0
    assert metrics["mean_state_compression_ratio"] == 0.5
    assert metrics["mean_persistent_memory_hits"] == 1.0
    assert metrics["answer_gate_counts"] == {"ALLOW": 1}
