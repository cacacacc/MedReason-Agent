"""运行 Chain-of-Thought VLM baseline。

Phase 2 保持和 Phase 1 相同的 frozen VLM 与数据集，只把 prompt 改成要求
Observation、Reasoning 和 Final Answer。完整推理文本用于后续分析，
指标只基于抽取出的最终短答案计算。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

from medreason_agent.data.vqa_rad import VQARADSample, load_vqa_rad_split
from medreason_agent.evaluation.answer_metrics import exact_match, summarize_answer_metrics
from medreason_agent.evaluation.error_attribution import (
    attribute_error,
    summarize_error_attribution,
)
from medreason_agent.experiments.resume import (
    append_jsonl_record,
    load_records_by_sample_id,
    ordered_records_for_sample_ids,
)
from medreason_agent.models.vlm import VLMRequest, create_vlm_backend
from medreason_agent.paths import resolve_project_path
from medreason_agent.prompts.cot import build_cot_prompt, extract_final_answer


def load_config(path: Path) -> dict[str, Any]:
    """读取定义 CoT 实验条件的 YAML 配置文件。"""
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def build_prediction_record(
    sample: VQARADSample,
    prediction: str,
    reasoning_output: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """构造一条项目标准 CoT prediction record。

    `prediction` 是抽取出的最终短答案。`reasoning_output` 保存模型完整回答，
    方便后续检查 CoT 到底帮助了还是伤害了结果。
    """
    correct = exact_match(prediction, sample.answer)
    record = {
        "experiment_id": metadata["experiment_id"],
        "model": metadata["model"],
        "backend": metadata["backend"],
        "prompt_version": metadata["prompt_version"],
        "prompt_style": metadata["prompt_style"],
        "prompt_contract": metadata["prompt_contract"],
        "dataset": sample.dataset,
        "split": sample.split,
        "sample_id": sample.sample_id,
        "image_id": sample.image_id,
        "image_path": sample.image_path,
        "question": sample.question,
        "prediction": prediction,
        "ground_truth": sample.answer,
        "answer_type": sample.answer_type,
        "question_type": sample.question_type,
        "image_organ": sample.image_organ,
        "reasoning_output": reasoning_output,
        "raw_output": reasoning_output,
        "retrieved_evidence": [],
        "tool_calls": [],
        "agent_route": ["cot_vlm"],
        "critic_decision": "",
        "confidence": metadata.get("confidence"),
        "latency_ms": metadata["latency_ms"],
        "input_tokens": metadata.get("input_tokens"),
        "output_tokens": metadata.get("output_tokens"),
        "correct": correct,
        "error_type": "",
    }
    error_attribution = attribute_error(record)
    record["error_attribution"] = error_attribution
    record["error_type"] = error_attribution["primary_error_type"]
    return record


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """把 prediction records 写成 JSONL，每行对应一个样本。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def is_current_record(record: dict[str, Any], method: dict[str, Any]) -> bool:
    """判断已有 CoT record 是否符合当前输出协议。"""
    return (
        record.get("prompt_version") == method["prompt_version"]
        and record.get("prompt_style", "structured") == method.get(
            "prompt_style",
            "structured",
        )
        and record.get("prompt_contract", "") == method.get("prompt_contract", "")
        and "error_attribution" in record
    )


def run(config_path: Path, backend_name: str | None = None) -> dict[str, Any]:
    """运行 CoT baseline，并返回整体指标。"""
    config = load_config(config_path)
    method = config["method"]
    dataset_config = config["dataset"]
    output_config = config["outputs"]
    generation_config = config["generation"]

    # 使用和 Phase 1 相同的 backend factory，确保唯一有意改变的是
    # prompt 和 final-answer 抽取方式。
    backend = create_vlm_backend(
        backend=backend_name or method.get("backend", "mock"),
        model_id=method.get("model_id"),
        torch_dtype=method.get("torch_dtype", "auto"),
        device_map=method.get("device_map", "auto"),
    )
    # `max_samples` 支持 smoke、dev、full，不需要改代码。
    samples = load_vqa_rad_split(
        split=dataset_config["split"],
        max_samples=dataset_config.get("max_samples"),
    )

    result_dir = resolve_project_path(output_config["result_dir"])
    prediction_path = result_dir / output_config["prediction_file"]
    metrics_path = result_dir / output_config["metrics_file"]

    # 断点续跑：已经写入 predictions.jsonl 的 sample_id 会被跳过。
    # 新结果每条即时追加，避免长时间 CoT 推理中断后丢失进度。
    sample_ids = [sample.sample_id for sample in samples]
    records_by_sample_id = {
        sample_id: record
        for sample_id, record in load_records_by_sample_id(prediction_path).items()
        if is_current_record(record, method)
    }
    missing_samples = [sample for sample in samples if sample.sample_id not in records_by_sample_id]
    print(
        f"RESUME existing={len(samples) - len(missing_samples)} "
        f"missing={len(missing_samples)} total={len(samples)}"
    )

    for sample in tqdm(missing_samples, desc=config["experiment_id"], ascii=True):
        # Phase 2 要求模型在最终答案前暴露中间推理。
        # 这就是本阶段要研究的实验变量。
        prompt_style = str(method.get("prompt_style", "structured"))
        prompt = build_cot_prompt(sample.question, prompt_style=prompt_style)
        request = VLMRequest(
            image_path=str(sample.absolute_image_path),
            question=sample.question,
            prompt=prompt,
            max_new_tokens=int(generation_config["max_new_tokens"]),
        )

        # CoT 通常比直接回答生成更多 token，因此 Phase 2 尤其要记录
        # latency 和 output tokens。
        start = time.perf_counter()
        response = backend.generate(request)
        latency_ms = round((time.perf_counter() - start) * 1000, 3)
        # 指标只比较 `Final Answer:` 后面的短答案。
        prediction = extract_final_answer(response.raw_output)

        record = build_prediction_record(
            sample=sample,
            prediction=prediction,
            reasoning_output=response.raw_output,
            metadata={
                "experiment_id": config["experiment_id"],
                "model": backend.model_name,
                "backend": backend.backend_name,
                "prompt_version": method["prompt_version"],
                "prompt_style": prompt_style,
                "prompt_contract": method.get("prompt_contract", ""),
                "confidence": response.confidence,
                "latency_ms": latency_ms,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
            },
        )
        append_jsonl_record(prediction_path, record)
        records_by_sample_id[sample.sample_id] = record

    # JSONL 输出后续会和 Phase 1 对比，用于统计 cot_helped 和 cot_hurt。
    records = ordered_records_for_sample_ids(records_by_sample_id, sample_ids)
    write_jsonl(prediction_path, records)
    metrics = summarize_answer_metrics(records)
    metrics.update(summarize_error_attribution(records))
    metrics.update(
        {
            "experiment_id": config["experiment_id"],
            "backend": backend.backend_name,
            "model": backend.model_name,
            "split": dataset_config["split"],
            "scale": dataset_config["scale"],
            "prompt_version": method["prompt_version"],
            "prompt_style": method.get("prompt_style", "structured"),
            "prompt_contract": method.get("prompt_contract", ""),
            "is_valid_main_result": backend.backend_name != "mock",
            "prediction_file": str(prediction_path.relative_to(resolve_project_path("."))),
        }
    )
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 Chain-of-Thought VLM baseline。")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/exp02_cot.yaml"),
    )
    parser.add_argument("--backend", choices=["mock", "qwen2_5_vl"], default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run(resolve_project_path(args.config), backend_name=args.backend)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
