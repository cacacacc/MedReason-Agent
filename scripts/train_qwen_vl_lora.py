"""Phase 8：Qwen2.5-VL LoRA 微调脚本。

训练目标是最小化的 Medical VQA 指令微调：

Image + Question -> short benchmark answer

这里不训练长 CoT，也不加入 RAG/Verifier。原因是当前项目 accuracy 的主要瓶颈在
VLM 对 VQA-RAD 图像和短答案格式的适配；先把 Direct VQA 能力拉高，再让 Phase 6
Adaptive Routing 使用微调后的 backbone。
"""

from __future__ import annotations

import argparse
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from medreason_agent.data.vqa_rad import VQARADSample, load_vqa_rad_split
from medreason_agent.paths import resolve_project_path

USER_PROMPT = """Answer the medical question based on the image.

Question: {question}

Return only the short final answer.
For yes/no questions, answer exactly yes or no.
For open questions, answer with one short phrase, not a sentence."""


@dataclass(frozen=True)
class TrainingExample:
    """LoRA 训练用的单条样本。"""

    sample: VQARADSample

    @property
    def answer(self) -> str:
        """返回 benchmark-style 短答案。"""
        return self.sample.answer.strip()

    def user_messages(self) -> list[dict[str, Any]]:
        """构造只有 user turn 的 Qwen 多模态 chat message。"""
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(self.sample.absolute_image_path)},
                    {"type": "text", "text": USER_PROMPT.format(question=self.sample.question)},
                ],
            }
        ]

    def full_messages(self) -> list[dict[str, Any]]:
        """构造 user + assistant answer 的监督微调 message。"""
        return [
            *self.user_messages(),
            {
                "role": "assistant",
                "content": [{"type": "text", "text": self.answer}],
            },
        ]


class VQARADLoRADataset:
    """轻量 Dataset，避免强依赖 datasets 包。"""

    def __init__(self, split: str, max_samples: int | None = None) -> None:
        self.examples = [
            TrainingExample(sample) for sample in load_vqa_rad_split(split, max_samples=max_samples)
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> TrainingExample:
        return self.examples[index]


class QwenVLDataCollator:
    """把 Qwen chat messages 转成训练 batch，并 mask 掉 prompt tokens。"""

    def __init__(self, processor, process_vision_info) -> None:
        self.processor = processor
        self.process_vision_info = process_vision_info

    def __call__(self, examples: list[TrainingExample]) -> dict[str, Any]:
        import torch

        full_texts = []
        prompt_lengths = []
        images = []
        videos = []

        for example in examples:
            prompt_messages = example.user_messages()
            full_messages = example.full_messages()

            prompt_text = self.processor.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            full_text = self.processor.apply_chat_template(
                full_messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            prompt_image_inputs, prompt_video_inputs = self.process_vision_info(prompt_messages)
            prompt_inputs = self.processor(
                text=[prompt_text],
                images=prompt_image_inputs,
                videos=prompt_video_inputs,
                padding=False,
                return_tensors="pt",
            )
            prompt_lengths.append(int(prompt_inputs["input_ids"].shape[-1]))

            image_inputs, video_inputs = self.process_vision_info(full_messages)
            full_texts.append(full_text)
            images.extend(image_inputs or [])
            videos.extend(video_inputs or [])

        batch = self.processor(
            text=full_texts,
            images=images or None,
            videos=videos or None,
            padding=True,
            return_tensors="pt",
        )
        labels = batch["input_ids"].clone()

        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100
        for row_index, prompt_length in enumerate(prompt_lengths):
            labels[row_index, :prompt_length] = -100

        # image token 本身不是要预测的文本答案，也 mask 掉。
        image_token_id = getattr(self.processor.tokenizer, "image_token_id", None)
        if image_token_id is not None:
            labels[labels == image_token_id] = -100

        batch["labels"] = labels.to(dtype=torch.long)
        return batch


def load_config(path: Path) -> dict[str, Any]:
    """读取 Phase8 YAML 配置。"""
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_torch_dtype(torch_module, dtype_name: str):
    """把 YAML 里的 dtype 字符串转成 torch dtype。"""
    if dtype_name in {"auto", ""}:
        return "auto"
    mapping = {
        "float32": torch_module.float32,
        "fp32": torch_module.float32,
        "float16": torch_module.float16,
        "fp16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
        "bf16": torch_module.bfloat16,
    }
    return mapping[dtype_name.lower()]


def build_lora_config(config: dict[str, Any]):
    """根据 YAML 创建 PEFT LoRA 配置。"""
    from peft import LoraConfig

    lora = config["lora"]
    return LoraConfig(
        r=int(lora.get("r", 16)),
        lora_alpha=int(lora.get("lora_alpha", 32)),
        lora_dropout=float(lora.get("lora_dropout", 0.05)),
        bias=str(lora.get("bias", "none")),
        task_type=str(lora.get("task_type", "CAUSAL_LM")),
        target_modules=list(lora.get("target_modules", ["q_proj", "v_proj"])),
    )


def build_training_arguments(training_arguments_cls, output_dir: Path, config: dict[str, Any]):
    """构造兼容不同 transformers 版本的 TrainingArguments。

    AutoDL 镜像里的 transformers 版本可能偏旧。这里先准备完整参数，再根据当前
    `TrainingArguments.__init__` 的签名过滤不支持的字段，并处理
    `eval_strategy` / `evaluation_strategy` 这类版本命名差异。
    """
    training_config = config["training"]
    signature = inspect.signature(training_arguments_cls.__init__)
    supported = set(signature.parameters)
    warmup_ratio = float(training_config.get("warmup_ratio", 0.03))

    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "num_train_epochs": float(training_config.get("num_train_epochs", 3)),
        "per_device_train_batch_size": int(
            training_config.get("per_device_train_batch_size", 1)
        ),
        "per_device_eval_batch_size": int(
            training_config.get("per_device_eval_batch_size", 1)
        ),
        "gradient_accumulation_steps": int(
            training_config.get("gradient_accumulation_steps", 16)
        ),
        "learning_rate": float(training_config.get("learning_rate", 2e-4)),
        "weight_decay": float(training_config.get("weight_decay", 0.0)),
        "logging_steps": int(training_config.get("logging_steps", 10)),
        "save_steps": int(training_config.get("save_steps", 100)),
        "eval_steps": int(training_config.get("eval_steps", 100)),
        "save_total_limit": int(training_config.get("save_total_limit", 2)),
        "bf16": bool(training_config.get("bf16", True)),
        "fp16": bool(training_config.get("fp16", False)),
        "remove_unused_columns": False,
        "report_to": list(training_config.get("report_to", [])),
        "dataloader_num_workers": int(training_config.get("dataloader_num_workers", 0)),
    }
    if "warmup_ratio" in supported:
        kwargs["warmup_ratio"] = warmup_ratio
    elif "warmup_steps" in supported:
        # 旧版 transformers 没有 warmup_ratio，就用 0 表示不显式 warmup。
        kwargs["warmup_steps"] = int(training_config.get("warmup_steps", 0))

    eval_strategy = str(training_config.get("eval_strategy", "steps"))
    if "eval_strategy" in supported:
        kwargs["eval_strategy"] = eval_strategy
    elif "evaluation_strategy" in supported:
        kwargs["evaluation_strategy"] = eval_strategy

    save_strategy = str(training_config.get("save_strategy", "steps"))
    if "save_strategy" in supported:
        kwargs["save_strategy"] = save_strategy

    filtered_kwargs = {
        key: value for key, value in kwargs.items() if key in supported
    }
    return training_arguments_cls(**filtered_kwargs)


def run(config_path: Path) -> dict[str, Any]:
    """执行 LoRA 训练并保存 adapter / metrics。"""
    import torch
    from peft import get_peft_model
    from qwen_vl_utils import process_vision_info
    from transformers import (
        AutoProcessor,
        Qwen2_5_VLForConditionalGeneration,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    config = load_config(config_path)
    method = config["method"]
    dataset_config = config["dataset"]
    training_config = config["training"]
    output_config = config["outputs"]

    set_seed(int(training_config.get("seed", 42)))

    output_dir = resolve_project_path(output_config["output_dir"])
    adapter_dir = resolve_project_path(output_config["adapter_dir"])
    merged_model_dir = resolve_project_path(output_config["merged_model_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(method["model_id"])
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        method["model_id"],
        torch_dtype=resolve_torch_dtype(torch, str(method.get("torch_dtype", "bfloat16"))),
        device_map=method.get("device_map", "auto"),
    )
    if bool(training_config.get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    model = get_peft_model(model, build_lora_config(config))
    model.print_trainable_parameters()

    train_dataset = VQARADLoRADataset(
        split=str(dataset_config.get("train_split", "train")),
        max_samples=dataset_config.get("max_train_samples"),
    )
    eval_dataset = VQARADLoRADataset(
        split=str(dataset_config.get("validation_split", "validation")),
        max_samples=dataset_config.get("max_validation_samples"),
    )

    args = build_training_arguments(TrainingArguments, output_dir, config)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=QwenVLDataCollator(processor, process_vision_info),
    )
    train_result = trainer.train(
        resume_from_checkpoint=training_config.get("resume_from_checkpoint"),
    )
    metrics = train_result.metrics
    trainer.save_model(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))

    if bool(output_config.get("merge_adapter", False)):
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(str(merged_model_dir), safe_serialization=True)
        processor.save_pretrained(str(merged_model_dir))

    summary = {
        "experiment_id": config["experiment_id"],
        "base_model": method["model_id"],
        "adapter_dir": str(adapter_dir),
        "merged_model_dir": str(merged_model_dir),
        "merge_adapter": bool(output_config.get("merge_adapter", False)),
        "num_train_samples": len(train_dataset),
        "num_validation_samples": len(eval_dataset),
        "train_metrics": metrics,
    }
    metrics_path = output_dir / output_config.get("metrics_file", "train_metrics.json")
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="训练 Qwen2.5-VL LoRA Medical VQA adapter。")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/training/phase8_qwen_vl_lora_vqa_rad_4090d.yaml"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run(resolve_project_path(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
