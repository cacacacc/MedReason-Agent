from medreason_agent.agents.memory import (
    PersistentAgentMemoryStore,
    PersistentMemorySettings,
    build_persistent_memory_record,
)
from medreason_agent.agents.state import SharedAgentState
from medreason_agent.data.vqa_rad import VQARADSample


def _sample(sample_id: str = "s1") -> VQARADSample:
    return VQARADSample(
        dataset="VQA-RAD",
        sample_id=sample_id,
        image_id=f"{sample_id}.jpg",
        image_path=f"Data/Raw/vqa_rad/images/{sample_id}.jpg",
        question="Is there right lung opacity?",
        answer="yes",
        answer_type="CLOSED",
        question_type="PRES",
        image_organ="CHEST",
        split="test",
    )


def test_persistent_memory_writes_and_reads_across_store_instances(tmp_path) -> None:
    memory_path = tmp_path / "agent_memory.jsonl"
    settings = PersistentMemorySettings(
        enabled=True,
        path=memory_path,
        namespace="unit",
        top_k=2,
        min_score=0.01,
    )
    state = SharedAgentState(question="Is there right lung opacity?")
    state.extend_claim_statuses(
        [
            {
                "claim": "right lung opacity",
                "status": "OBSERVED",
                "source_agent": "vision_agent",
            }
        ]
    )

    writer = PersistentAgentMemoryStore(settings)
    written = writer.append_from_state(
        sample=_sample("past"),
        shared_state=state.to_record(),
        prediction="yes",
        experiment_id="exp_memory",
    )
    reader = PersistentAgentMemoryStore(settings)
    memories = reader.retrieve(_sample("current"))

    assert written is not None
    assert memory_path.exists()
    assert memories
    assert memories[0]["sample_id"] == "past"
    assert memories[0]["stores_ground_truth"] is False
    assert memories[0]["stores_prediction"] is False


def test_persistent_memory_record_can_optionally_store_prediction() -> None:
    state = SharedAgentState(question="Is there opacity?")
    record = build_persistent_memory_record(
        sample=_sample(),
        shared_state=state.to_record(),
        prediction="yes",
        experiment_id="exp_memory",
        namespace="unit",
        include_prediction=True,
    )

    assert record["stores_ground_truth"] is False
    assert record["stores_prediction"] is True
    assert "Previous Prediction: yes" in record["memory_text"]
