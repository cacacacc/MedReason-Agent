"""baseline 实验使用的视觉语言模型 backend。

其他实验代码都通过一个小接口和这里交互：`VLMRequest -> VLMResponse`。
这样无论底层是 mock 还是真实 Qwen，Phase 1 direct inference 和 Phase 2 CoT
inference 都能保持可比。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class VLMRequest:
    """传给 VLM backend 的输入。

    `prompt` 包含实验条件。Phase 1 要求短答案；Phase 2 要求
    Observation / Reasoning / Final Answer。
    """

    image_path: str
    question: str
    prompt: str
    max_new_tokens: int


@dataclass(frozen=True)
class VLMResponse:
    """VLM backend 的输出。

    `answer` 是 backend 的主要答案字段。`raw_output` 保留模型完整输出，
    方便 CoT 运行后继续分析推理过程。
    """

    answer: str
    raw_output: str
    confidence: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


class VLMBackend(Protocol):
    """所有 VLM backend 都要实现的最小接口。

    每个实验 runner 都依赖这个协议，而不是依赖某个具体模型类。
    这样我们可以在不重写评估 pipeline 的情况下切换 `mock` 和 `qwen2_5_vl`。
    """

    model_name: str
    backend_name: str

    def generate(self, request: VLMRequest) -> VLMResponse:
        """Generate an answer for one image-question pair."""


class MockVLMBackend:
    """只用于 smoke test 的确定性 backend。

    Mock 结果永远不能作为正式科研结果。它的作用是在不加载大模型的情况下，
    检查文件写入、指标计算和 CoT 解析流程是否正常。
    """

    model_name = "mock-vlm"
    backend_name = "mock"

    def generate(self, request: VLMRequest) -> VLMResponse:
        # 简单的 yes/no 规则让测试输出保持确定性。它故意很弱，
        # 因为它只负责验证 pipeline，不负责提供真实模型能力。
        question = request.question.lower()
        if question.startswith(("is ", "are ", "does ", "do ", "can ", "has ", "have ")):
            answer = "yes"
        else:
            answer = "unknown"
        # 如果 prompt 要求 CoT 段落，就返回结构化 mock 输出，
        # 让 Phase 2 测试能走到同一个 final-answer 抽取逻辑。
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
    """本地 Hugging Face Qwen2.5-VL backend。

    这是主实验使用的真实 frozen backbone。模型采用懒加载，
    这样普通单元测试不会下载或初始化大模型。
    """

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
        """懒加载模型依赖，避免普通单元测试必须初始化大模型。"""
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

        # CPU 推理固定使用 float32，兼容性更稳。CUDA 实验使用 config 指定的 dtype，
        # 例如 Qwen 7B 使用 bfloat16。
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

        # Qwen2.5-VL 需要 chat-style 多模态消息：一个 image item 加一个 text prompt。
        # Direct 和 CoT 实验都使用同一种消息结构。
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": request.image_path},
                    {"type": "text", "text": request.prompt},
                ],
            }
        ]

        # processor 会把 chat message 转成文本 tokens 和视觉 tensors。
        # 这一步放在 backend 内部，runner 就不需要理解具体模型细节。
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
        # `generate` 返回的是 prompt tokens + 新生成 tokens。
        # 这里裁掉 prompt 部分，让 `raw_output` 只保留模型生成的答案。
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
    """根据实验配置里的 backend 名称创建 VLM backend。"""
    if backend == "mock":
        return MockVLMBackend()

    if backend == "qwen2_5_vl":
        return Qwen25VLBackend(
            model_id=model_id or "Qwen/Qwen2.5-VL-3B-Instruct",
            torch_dtype=torch_dtype,
            device_map=device_map,
        )

    raise ValueError(f"Unsupported VLM backend: {backend}")
