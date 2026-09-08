from medreason_agent.prompts.rag import (
    build_claim_verification_query,
    build_evidence_verification_prompt,
    build_rag_prompt,
    extract_claims,
    extract_final_answer,
    extract_verification_status,
)


def test_build_rag_prompt_contains_evidence_and_sections() -> None:
    prompt = build_rag_prompt(
        question="Is there opacity?",
        evidence_records=[
            {
                "title": "Chest opacity",
                "text": "Opacity can indicate consolidation.",
                "score": 2.5,
            }
        ],
    )

    assert "Retrieved Evidence:" in prompt
    assert "Use retrieved evidence explicitly" in prompt
    assert "Do not introduce unsupported medical claims" in prompt
    assert "Separate image observation from external knowledge" in prompt
    assert "State uncertainty if the evidence is insufficient" in prompt
    assert "Unsupported Assumptions:" in prompt
    assert "Claim Statuses:" in prompt
    assert "HYPOTHESIS" in prompt
    assert "Uncertainty:" in prompt
    assert "Conclusion:" in prompt
    assert "Final Answer:" in prompt
    assert "Opacity can indicate consolidation." in prompt


def test_extract_final_answer_from_rag_output() -> None:
    output = "Observation: finding.\nRetrieved Evidence: relevant.\nFinal Answer: yes"

    assert extract_final_answer(output) == "yes"


def test_extract_final_answer_can_fall_back_to_conclusion() -> None:
    output = "Observation: finding.\nReasoning: supported.\nConclusion: yes"

    assert extract_final_answer(output) == "yes"


def test_build_evidence_verification_prompt_contains_claim_contract() -> None:
    prompt = build_evidence_verification_prompt(
        question="Is there opacity?",
        initial_claim="yes",
        initial_reasoning_output="Observation: opacity.\nFinal Answer: yes",
        evidence_records=[
            {
                "title": "Opacity",
                "text": "Opacity can indicate consolidation.",
                "score": 2.0,
            }
        ],
    )

    assert "Verify the initial medical claim" in prompt
    assert "Initial Claim:" in prompt
    assert "Verification:" in prompt
    assert "Claim Statuses:" in prompt
    assert "Compare the initial claim against the retrieved evidence" in prompt


def test_build_claim_verification_query_uses_claim_not_only_question() -> None:
    query = build_claim_verification_query(
        question="Is there opacity?",
        claim="pneumonia",
        reasoning_output="The image may show right lower lung opacity.",
    )

    assert "Claim: pneumonia" in query
    assert "supporting or contradicting" in query


def test_extract_claims_from_claims_section() -> None:
    output = (
        "Observation: finding.\n"
        "Claims:\n"
        "- right lung opacity is present\n"
        "- opacity may support pneumonia\n"
        "Unsupported Assumptions: None"
    )

    assert extract_claims(output) == [
        "right lung opacity is present",
        "opacity may support pneumonia",
    ]


def test_extract_verification_status() -> None:
    assert extract_verification_status("Verification Status: SUPPORTED") == "SUPPORTED"
    assert extract_verification_status("Verification Status: contradicted") == "CONTRADICTED"
    assert extract_verification_status("No explicit status") == "UNSUPPORTED"
