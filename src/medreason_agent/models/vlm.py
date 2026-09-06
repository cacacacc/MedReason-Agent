"""Vision-language model backends for baseline experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class VLMRequest:
    """Input to a VLM backend."""

    image_path: str
    question: str
    prompt: str
    max_new_tokens: int


@dataclass(frozen=True)
class VLMResponse:
    """Output from a VLM backend."""

    answer: str
    raw_output: str
    confidence: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


class VLMBackend(Protocol):
    """Minimal interface implemented by all VLM backends."""

    model_name: str
    backend_name: str

    def generate(self, request: VLMRequest) -> VLMResponse:
        """Generate an answer for one image-question pair."""


class MockVLMBackend:
    """Deterministic backend used only to smoke-test the experiment pipeline."""

    model_name = "mock-vlm"
    backend_name = "mock"

    def generate(self, request: VLMRequest) -> VLMResponse:
        question = request.question.lower()
        if question.startswith(("is ", "are ", "does ", "do ", "can ", "has ", "have ")):
            answer = "yes"
        else:
            answer = "unknown"

        return VLMResponse(
            answer=answer,
            raw_output=answer,
            confidence=None,
            input_tokens=len(request.prompt.split()),
            output_tokens=len(answer.split()),
        )


def create_vlm_backend(backend: str) -> VLMBackend:
    """Create a VLM backend by name."""
    if backend == "mock":
        return MockVLMBackend()

    if backend == "qwen2_5_vl":
        raise NotImplementedError(
            "The qwen2_5_vl backend will be added after installing requirements/vlm.txt "
            "and confirming local/API inference resources."
        )

    raise ValueError(f"Unsupported VLM backend: {backend}")
