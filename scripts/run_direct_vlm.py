"""运行 Direct VLM baseline。

Phase 1 实验流程：

1. 读取一个 YAML 实验配置。
2. 从 `Data/Processed` 读取 VQA-RAD split。
3. 为每个 image-question pair 构造 direct-answer prompt。
4. 调用配置中选择的 VLM backend。
5. 保存逐样本 predictions 和整体 metrics。
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
from medreason_agent.prompts.direct import build_direct_prompt


def load_config(path: Path) -> dict[str, Any]:
    """读取定义实验条件的 YAML 配置文件。"""
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def build_prediction_record(
    sample: VQARADSample,
    response_answer: str,
    raw_output: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """构造一条项目标准 prediction record。

    这个 schema 故意比 Phase 1 当前需要的字段更宽。后续阶段会填入
    `reasoning_output`、`retrieved_evidence`、`agent_route` 等字段，
    这样所有实验输出都能保持可比。
    """
    correct = exact_match(response_answer, sample.answer)
    record = {
        "experiment_id": metadata["experiment_id"],
        "model": metadata["model"],
        "backend": metadata["backend"],
        "prompt_version": metadata["prompt_version"],
        "dataset": sample.dataset,
        "split": sample.split,
        "sample_id": sample.sample_id,
        "image_id": sample.image_id,
        "image_path": sample.image_path,
        "question": sample.question,
        "prediction": response_answer,
        "ground_truth": sample.answer,
        "answer_type": sample.answer_type,
        "question_type": sample.question_type,
        "image_organ": sample.image_organ,
        "reasoning_output": "",
        "raw_output": raw_output,
        "retrieved_evidence": [],
        "tool_calls": [],
        "agent_route": ["direct_vlm"],
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
    """判断已有 Direct VLM record 是否符合当前输出协议。"""
    return (
        record.get("prompt_version") == method["prompt_version"]
        and "error_attribution" in record
    )


def run(config_path: Path, backend_name: str | None = None) -> dict[str, Any]:
    """运行 direct baseline，并返回整体指标。"""
    config = load_config(config_path)
    method = config["method"]
    dataset_config = config["dataset"]
    output_config = config["outputs"]
    generation_config = config["generation"]

    # config 决定这是便宜的 mock smoke test 还是真实 Qwen 实验。
    # runner 本身不关心具体模型细节。
    backend = create_vlm_backend(
        backend=backend_name or method.get("backend", "mock"),
        model_id=method.get("model_id"),
        torch_dtype=method.get("torch_dtype", "auto"),
        device_map=method.get("device_map", "auto"),
    )
    # `max_samples` 实现 smoke -> dev -> full 的逐步扩展协议。
    samples = load_vqa_rad_split(
        split=dataset_config["split"],
        max_samples=dataset_config.get("max_samples"),
    )

    result_dir = resolve_project_path(output_config["result_dir"])
    prediction_path = result_dir / output_config["prediction_file"]
    metrics_path = result_dir / output_config["metrics_file"]

    # 断点续跑：先读取已有 predictions，只对当前配置里还没完成的样本继续推理。
    # 注意这里只按 sample_id 判断是否完成，因此同一个输出目录不要混跑不同方法。
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
        # Phase 1 只使用这个 prompt，不要求模型写推理过程。
        prompt = build_direct_prompt(sample.question)
        request = VLMRequest(
            image_path=str(sample.absolute_image_path),
            question=sample.question,
            prompt=prompt,
            max_new_tokens=int(generation_config["max_new_tokens"]),
        )

        # 逐样本记录 latency，方便后续比较准确率提升是否值得额外运行成本。
        start = time.perf_counter()
        response = backend.generate(request)
        latency_ms = round((time.perf_counter() - start) * 1000, 3)

        record = build_prediction_record(
            sample=sample,
            response_answer=response.answer,
            raw_output=response.raw_output,
            metadata={
                "experiment_id": config["experiment_id"],
                "model": backend.model_name,
                "backend": backend.backend_name,
                "prompt_version": method["prompt_version"],
                "confidence": response.confidence,
                "latency_ms": latency_ms,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
            },
        )
        append_jsonl_record(prediction_path, record)
        records_by_sample_id[sample.sample_id] = record

    # 先保存详细记录，再写 summary metrics。JSONL 是后续错误分析和
    # Direct-vs-CoT 对比的数据来源。
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
            "is_valid_main_result": backend.backend_name != "mock",
            "prediction_file": str(prediction_path.relative_to(resolve_project_path("."))),
        }
    )
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 Direct VLM baseline。")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/exp01_direct_vlm.yaml"),
    )
    parser.add_argument("--backend", choices=["mock", "qwen2_5_vl"], default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run(resolve_project_path(args.config), backend_name=args.backend)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
