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
        if "Final Answer:" in request.prompt:
            raw_output = (
                "Observation: mock visual observation.\n"
                "Reasoning: mock reasoning path for pipeline validation.\n"
                f"Final Answer: {answer}"
            )
        else:
            raw_output = answer

        return VLMResponse(
            answer=answer,
            raw_output=raw_output,
            confidence=None,
            input_tokens=len(request.prompt.split()),
            output_tokens=len(raw_output.split()),
        )


class Qwen25VLBackend:
    """Local Hugging Face backend for Qwen2.5-VL models."""

    backend_name = "qwen2_5_vl"

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        torch_dtype: str = "auto",
        device_map: str = "auto",
    ) -> None:
        self.model_name = model_id
        self.model_id = model_id
        self.torch_dtype = torch_dtype
        self.device_map = device_map
        self._model = None
        self._processor = None
        self._process_vision_info = None

    @staticmethod
    def _resolve_torch_dtype(torch_module, torch_dtype: str):
        """Convert config strings into torch dtype objects."""
        if torch_dtype in {"auto", ""}:
            return "auto"

        dtype_map = {
            "float32": torch_module.float32,
            "fp32": torch_module.float32,
            "float16": torch_module.float16,
            "fp16": torch_module.float16,
            "bfloat16": torch_module.bfloat16,
            "bf16": torch_module.bfloat16,
        }
        try:
            return dtype_map[torch_dtype.lower()]
        except KeyError as exc:
            supported = ", ".join(sorted(dtype_map | {"auto": "auto"}))
            raise ValueError(
                f"Unsupported torch_dtype={torch_dtype!r}. Use one of: {supported}"
            ) from exc

    def _load(self) -> None:
        """Load model dependencies lazily so normal tests do not require them."""
        if self._model is not None:
            return

        try:
            import torch
            from qwen_vl_utils import process_vision_info
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError(
                "Qwen2.5-VL backend dependencies are missing. Install them with: "
                ".\\.venv\\Scripts\\python.exe -m pip install -r requirements\\vlm.txt"
            ) from exc

        torch_dtype = (
            torch.float32
            if self.device_map == "cpu"
            else self._resolve_torch_dtype(torch, self.torch_dtype)
        )
        model_kwargs = {"torch_dtype": torch_dtype}
        if self.device_map != "cpu":
            model_kwargs["device_map"] = self.device_map

        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            **model_kwargs,
        )
        if self.device_map == "cpu":
            self._model.to("cpu")

        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._process_vision_info = process_vision_info

    def generate(self, request: VLMRequest) -> VLMResponse:
        self._load()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": request.image_path},
                    {"type": "text", "text": request.prompt},
                ],
            }
        ]

        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = self._process_vision_info(messages)
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        device = next(self._model.parameters()).device
        inputs = inputs.to(device)

        generated_ids = self._model.generate(
            **inputs,
            max_new_tokens=request.max_new_tokens,
            do_sample=False,
        )
        generated_ids_trimmed = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids, strict=True)
        ]
        output_text = self._processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        return VLMResponse(
            answer=output_text,
            raw_output=output_text,
            confidence=None,
            input_tokens=int(inputs.input_ids.shape[-1]),
            output_tokens=int(generated_ids_trimmed[0].shape[-1]),
        )


def create_vlm_backend(
    backend: str,
    model_id: str | None = None,
    torch_dtype: str = "auto",
    device_map: str = "auto",
) -> VLMBackend:
    """Create a VLM backend by name."""
    if backend == "mock":
        return MockVLMBackend()

    if backend == "qwen2_5_vl":
        return Qwen25VLBackend(
            model_id=model_id or "Qwen/Qwen2.5-VL-3B-Instruct",
            torch_dtype=torch_dtype,
            device_map=device_map,
        )

    raise ValueError(f"Unsupported VLM backend: {backend}")
