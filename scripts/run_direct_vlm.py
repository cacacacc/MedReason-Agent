"""Run the Direct VLM baseline."""

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
from medreason_agent.models.vlm import VLMRequest, create_vlm_backend
from medreason_agent.paths import resolve_project_path
from medreason_agent.prompts.direct import build_direct_prompt


def load_config(path: Path) -> dict[str, Any]:
    """Load one YAML config file."""
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def build_prediction_record(
    sample: VQARADSample,
    response_answer: str,
    raw_output: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Build one project-standard prediction record."""
    correct = exact_match(response_answer, sample.answer)
    return {
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
        "error_type": "" if correct else "UNKNOWN",
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write prediction records as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def run(config_path: Path, backend_name: str | None = None) -> dict[str, Any]:
    """Run the baseline and return metrics."""
    config = load_config(config_path)
    method = config["method"]
    dataset_config = config["dataset"]
    output_config = config["outputs"]
    generation_config = config["generation"]

    backend = create_vlm_backend(backend_name or method.get("backend", "mock"))
    samples = load_vqa_rad_split(
        split=dataset_config["split"],
        max_samples=dataset_config.get("max_samples"),
    )

    records: list[dict[str, Any]] = []
    for sample in tqdm(samples, desc=config["experiment_id"], ascii=True):
        prompt = build_direct_prompt(sample.question)
        request = VLMRequest(
            image_path=str(sample.absolute_image_path),
            question=sample.question,
            prompt=prompt,
            max_new_tokens=int(generation_config["max_new_tokens"]),
        )

        start = time.perf_counter()
        response = backend.generate(request)
        latency_ms = round((time.perf_counter() - start) * 1000, 3)

        records.append(
            build_prediction_record(
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
        )

    result_dir = resolve_project_path(output_config["result_dir"])
    prediction_path = result_dir / output_config["prediction_file"]
    metrics_path = result_dir / output_config["metrics_file"]

    write_jsonl(prediction_path, records)
    metrics = summarize_answer_metrics(records)
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
    parser = argparse.ArgumentParser(description="Run Direct VLM baseline.")
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
