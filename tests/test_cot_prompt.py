from medreason_agent.prompts.cot import build_cot_prompt, extract_final_answer


def test_build_structured_cot_prompt_contains_required_sections() -> None:
    prompt = build_cot_prompt("Is there opacity?", prompt_style="structured")

    assert "Observation:" in prompt
    assert "Reasoning:" in prompt
    assert "Final Answer:" in prompt
    assert "Is there opacity?" in prompt


def test_build_vanilla_cot_prompt_is_not_structured() -> None:
    prompt = build_cot_prompt("Is there opacity?", prompt_style="vanilla")

    assert "Think step by step" in prompt
    assert "Final Answer:" in prompt
    assert "Observation:" not in prompt
    assert "Reasoning:" not in prompt


def test_extract_final_answer_from_structured_output() -> None:
    output = "Observation: finding.\nReasoning: because.\nFinal Answer: yes"

    assert extract_final_answer(output) == "yes"


def test_extract_final_answer_falls_back_to_raw_output() -> None:
    assert extract_final_answer("yes") == "yes"
