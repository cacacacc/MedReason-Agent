from medreason_agent.evaluation.error_attribution import (
    attribute_error,
    summarize_error_attribution,
)


def test_correct_record_has_no_error_type() -> None:
    attribution = attribute_error({"correct": True})

    assert attribution["primary_error_type"] == ""
    assert attribution["candidate_error_types"] == []


def test_retrieval_error_for_low_quality_evidence() -> None:
    attribution = attribute_error(
        {
            "correct": False,
            "agent_route": ["retrieval_agent", "reasoning_agent"],
            "retrieved_evidence": [{"title": "irrelevant", "text": "none"}],
            "evidence_quality_score": 0.0,
        }
    )

    assert attribution["primary_error_type"] == "Retrieval Error"
    assert "Retrieval Error" in attribution["candidate_error_types"]


def test_routing_error_has_priority() -> None:
    attribution = attribute_error(
        {
            "correct": False,
            "selected_tools": ["Vision Agent", "Answer Agent"],
            "expected_selected_tools": ["Vision Agent", "Retrieval Agent", "Answer Agent"],
            "retrieved_evidence": [],
        }
    )

    assert attribution["primary_error_type"] == "Routing Error"


def test_state_error_for_missing_compressed_context() -> None:
    attribution = attribute_error(
        {
            "correct": False,
            "shared_state": {"messages": [{"agent_name": "vision_agent"}]},
            "state_compression": {"compression_ratio": 0.5},
        }
    )

    assert attribution["primary_error_type"] == "State Error"


def test_verification_error_when_verifier_supports_wrong_answer() -> None:
    attribution = attribute_error(
        {
            "correct": False,
            "claim_verification_status": "SUPPORTED",
            "retrieved_evidence": [{"title": "evidence", "text": "support"}],
            "evidence_quality_score": 0.8,
        }
    )

    assert attribution["primary_error_type"] == "Verification Error"
    assert attribution["requires_human_review"]


def test_reasoning_error_is_default_for_unexplained_wrong_answer() -> None:
    attribution = attribute_error(
        {
            "correct": False,
            "agent_outputs": {
                "vision": "Observation: finding",
                "reasoning": "Reasoning: wrong but structured\nClaims:\n- finding",
            },
            "agent_route": ["vision_agent", "reasoning_agent", "answer_agent"],
            "claim_statuses": [{"claim": "finding", "status": "OBSERVED"}],
        }
    )

    assert attribution["primary_error_type"] == "Reasoning Error"


def test_summarize_error_attribution_counts_primary_and_candidates() -> None:
    summary = summarize_error_attribution(
        [
            {"correct": True},
            {
                "correct": False,
                "error_attribution": {
                    "primary_error_type": "Retrieval Error",
                    "candidate_error_types": ["Retrieval Error", "Reasoning Error"],
                    "requires_human_review": True,
                },
            },
        ]
    )

    assert summary["num_incorrect"] == 1
    assert summary["error_type_counts"]["Retrieval Error"] == 1
    assert summary["candidate_error_type_counts"]["Reasoning Error"] == 1
    assert summary["human_review_error_count"] == 1
