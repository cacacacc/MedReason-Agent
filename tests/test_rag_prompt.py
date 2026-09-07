from medreason_agent.prompts.rag import build_rag_prompt, extract_final_answer


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
    assert "Evidence Use:" in prompt
    assert "Final Answer:" in prompt
    assert "Opacity can indicate consolidation." in prompt


def test_extract_final_answer_from_rag_output() -> None:
    output = "Observation: finding.\nEvidence Use: relevant.\nFinal Answer: yes"

    assert extract_final_answer(output) == "yes"
