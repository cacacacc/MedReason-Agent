from medreason_agent.models.vlm import VLMRequest, create_vlm_backend
from medreason_agent.prompts.direct import build_direct_prompt


def test_build_direct_prompt_contains_question() -> None:
    prompt = build_direct_prompt("What organ is shown?")
    assert "What organ is shown?" in prompt
    assert "Return only the short final answer" in prompt


def test_mock_vlm_backend_is_deterministic() -> None:
    backend = create_vlm_backend("mock")
    response = backend.generate(
        VLMRequest(
            image_path="image.jpg",
            question="Is there opacity?",
            prompt="Question: Is there opacity?",
            max_new_tokens=16,
        )
    )

    assert response.answer == "yes"
    assert response.raw_output == "yes"
