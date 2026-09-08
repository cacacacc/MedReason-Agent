from medreason_agent.agents.state import SharedAgentState, compress_agent_output


def test_shared_state_compresses_agent_outputs_and_keeps_claim_statuses() -> None:
    state = SharedAgentState(question="Is there opacity?", max_chars_per_agent=120)
    state.set_selected_tools(["Vision Agent", "Reasoning Agent"])
    state.add_agent_output(
        "vision_agent",
        (
            "Observation: right lung opacity is visible.\n"
            'Claim Statuses:\n{"claim": "right lung opacity", "status": "OBSERVED"}\n'
            "Uncertainty: mild."
        ),
        claim_statuses=[
            {
                "claim": "right lung opacity",
                "status": "OBSERVED",
                "source_agent": "vision_agent",
            }
        ],
    )
    state.add_retrieved_evidence(
        [
            {
                "title": "Opacity",
                "text": "Opacity can be associated with consolidation.",
                "score": 1.0,
            }
        ],
        stage="retrieval_agent",
    )

    context = state.compressed_context()
    record = state.to_record()

    assert "Selected Tools: Vision Agent, Reasoning Agent" in context
    assert "[OBSERVED] right lung opacity" in context
    assert "stage=retrieval_agent" in context
    assert record["state_compression"]["method"] == "section_extract_v1"
    assert record["messages"][0]["compressed_chars"] <= record["messages"][0]["output_chars"]


def test_compress_agent_output_prefers_structured_sections() -> None:
    compressed = compress_agent_output(
        "reasoning_agent",
        (
            "Irrelevant preface.\n"
            "Reasoning: use image and evidence.\n"
            "Claims:\n- possible pneumonia\n"
            "Preliminary Answer: yes"
        ),
    )

    assert "Irrelevant preface" not in compressed
    assert "Reasoning: use image and evidence." in compressed
    assert "Claims: - possible pneumonia" in compressed
