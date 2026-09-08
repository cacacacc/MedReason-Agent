from medreason_agent.prompts.multi_agent import (
    build_supervisor_prompt,
    extract_claims,
    extract_critic_decision,
    extract_final_answer,
    extract_selected_tools,
    extract_verification_status,
)


def test_build_supervisor_prompt_contains_tool_contract() -> None:
    prompt = build_supervisor_prompt("Is there opacity?")

    assert "Act as a supervisor" in prompt
    assert "Selected Tools:" in prompt
    assert "Vision Agent" in prompt
    assert "Verifier Agent" in prompt


def test_reasoning_prompt_contains_claim_status_contract() -> None:
    from medreason_agent.prompts.multi_agent import build_reasoning_prompt

    prompt = build_reasoning_prompt("Is there opacity?", "Observation: opacity")

    assert "Claim Statuses:" in prompt
    assert "Do not treat HYPOTHESIS as fact" in prompt


def test_extract_multi_agent_outputs() -> None:
    assert extract_final_answer("Conclusion: yes\nFinal Answer: yes") == "yes"
    assert extract_critic_decision("Critic Decision: revise") == "REVISE"
    assert extract_verification_status("Verification Status: contradicted") == "CONTRADICTED"
    assert extract_selected_tools(
        "Selected Tools: Vision Agent, Retrieval Agent, Answer Agent"
    ) == ["Vision Agent", "Retrieval Agent", "Answer Agent"]
    claims_output = "Claims:\n- opacity is present\n- pneumonia is possible\nFinal Answer: yes"
    assert extract_claims(claims_output) == [
        "opacity is present",
        "pneumonia is possible",
    ]
