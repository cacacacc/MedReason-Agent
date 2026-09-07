"""运行 Phase 3 Medical RAG baseline。

RAG 流程：

1. 读取 VQA-RAD 样本。
2. 用问题检索本地医学知识库。
3. 把图像、问题和 retrieved evidence 一起交给 VLM。
4. 保存最终答案、完整 reasoning、检索证据和指标。
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
from medreason_agent.experiments.resume import (
    append_jsonl_record,
    load_records_by_sample_id,
    ordered_records_for_sample_ids,
)
from medreason_agent.models.vlm import VLMRequest, create_vlm_backend
from medreason_agent.paths import resolve_project_path
from medreason_agent.prompts.rag import build_rag_prompt, extract_final_answer
from medreason_agent.retrieval.keyword import KeywordRetriever, evidence_to_record, load_chunks
from medreason_agent.retrieval.rerank import KeywordReranker, filter_evidence


def load_config(path: Path) -> dict[str, Any]:
    """读取定义 RAG 实验条件的 YAML 配置。"""
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def build_prediction_record(
    sample: VQARADSample,
    prediction: str,
    reasoning_output: str,
    retrieved_evidence: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """构造一条项目标准 RAG prediction record。"""
    correct = exact_match(prediction, sample.answer)
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
        "prediction": prediction,
        "ground_truth": sample.answer,
        "answer_type": sample.answer_type,
        "question_type": sample.question_type,
        "image_organ": sample.image_organ,
        "reasoning_output": reasoning_output,
        "raw_output": reasoning_output,
        "retrieved_evidence": retrieved_evidence,
        "tool_calls": [
            {
                "tool": "keyword_retriever",
                "top_k": metadata["candidate_top_k"],
                "num_candidates": metadata["num_candidates"],
            },
            {
                "tool": metadata["reranker"],
                "num_candidates": metadata["num_candidates"],
                "num_reranked": metadata["num_reranked"],
            },
            {
                "tool": "evidence_filter",
                "top_k": metadata["top_k"],
                "min_score": metadata["min_rerank_score"],
                "num_evidence": len(retrieved_evidence),
            }
        ],
        "agent_route": ["retriever", "reranker", "evidence_filter", "rag_vlm"],
        "critic_decision": "",
        "confidence": metadata.get("confidence"),
        "latency_ms": metadata["latency_ms"],
        "input_tokens": metadata.get("input_tokens"),
        "output_tokens": metadata.get("output_tokens"),
        "correct": correct,
        "error_type": "" if correct else "UNKNOWN",
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """把 RAG prediction records 写成 JSONL。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def run(config_path: Path, backend_name: str | None = None) -> dict[str, Any]:
    """运行 RAG baseline，并返回整体指标。"""
    config = load_config(config_path)
    method = config["method"]
    dataset_config = config["dataset"]
    generation_config = config["generation"]
    output_config = config["outputs"]
    retrieval_config = config["retrieval"]

    corpus_path = resolve_project_path(retrieval_config["corpus_path"])
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Missing RAG corpus: {corpus_path}. "
            "Run scripts/build_medical_kb.py before running Phase 3."
        )

    # 第一版使用关键词检索器，先验证 RAG 实验格式。后续 BGE+Qdrant 可以替换这里，
    # 但 prediction record 的 retrieved_evidence 字段保持一致。
    retriever = KeywordRetriever(load_chunks(corpus_path))
    reranker = KeywordReranker()
    backend = create_vlm_backend(
        backend=backend_name or method.get("backend", "mock"),
        model_id=method.get("model_id"),
        torch_dtype=method.get("torch_dtype", "auto"),
        device_map=method.get("device_map", "auto"),
    )
    samples = load_vqa_rad_split(
        split=dataset_config["split"],
        max_samples=dataset_config.get("max_samples"),
    )

    top_k = int(retrieval_config.get("top_k", 5))
    candidate_top_k = int(retrieval_config.get("candidate_top_k", max(top_k, 5)))
    min_rerank_score = float(retrieval_config.get("min_rerank_score", 0.0))
    max_chars_per_evidence = int(retrieval_config.get("max_chars_per_evidence", 700))

    result_dir = resolve_project_path(output_config["result_dir"])
    prediction_path = result_dir / output_config["prediction_file"]
    metrics_path = result_dir / output_config["metrics_file"]

    # 断点续跑：RAG 每个样本都包含检索证据和 VLM 回答，跑得更慢，所以必须边跑边落盘。
    sample_ids = [sample.sample_id for sample in samples]
    existing_records_by_sample_id = load_records_by_sample_id(prediction_path)
    expected_route = ["retriever", "reranker", "evidence_filter", "rag_vlm"]
    records_by_sample_id = {
        sample_id: record
        for sample_id, record in existing_records_by_sample_id.items()
        if record.get("prompt_version") == method["prompt_version"]
        and record.get("agent_route") == expected_route
    }
    missing_samples = [sample for sample in samples if sample.sample_id not in records_by_sample_id]
    print(
        f"RESUME existing={len(samples) - len(missing_samples)} "
        f"missing={len(missing_samples)} total={len(samples)}"
    )

    for sample in tqdm(missing_samples, desc=config["experiment_id"], ascii=True):
        # RAG latency 包含检索和 VLM 生成两部分，因为用户最终感受到的是端到端耗时。
        start = time.perf_counter()
        # Retriever 先召回候选 chunk，Reranker 再按问题相关性重排，Evidence Filter 只保留
        # 最终传给 Reasoning Agent 的高质量证据。
        candidate_evidence = retriever.retrieve(sample.question, candidate_top_k)
        reranked_evidence = reranker.rerank(sample.question, candidate_evidence)
        filtered_evidence = filter_evidence(
            reranked_evidence,
            top_k=top_k,
            min_score=min_rerank_score,
        )
        retrieved = [evidence_to_record(item) for item in filtered_evidence]
        prompt = build_rag_prompt(
            question=sample.question,
            evidence_records=retrieved,
            max_chars_per_evidence=max_chars_per_evidence,
        )
        request = VLMRequest(
            image_path=str(sample.absolute_image_path),
            question=sample.question,
            prompt=prompt,
            max_new_tokens=int(generation_config["max_new_tokens"]),
        )

        response = backend.generate(request)
        latency_ms = round((time.perf_counter() - start) * 1000, 3)
        prediction = extract_final_answer(response.raw_output)

        record = build_prediction_record(
            sample=sample,
            prediction=prediction,
            reasoning_output=response.raw_output,
            retrieved_evidence=retrieved,
            metadata={
                "experiment_id": config["experiment_id"],
                "model": backend.model_name,
                "backend": backend.backend_name,
                "prompt_version": method["prompt_version"],
                "confidence": response.confidence,
                "latency_ms": latency_ms,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "top_k": top_k,
                "candidate_top_k": candidate_top_k,
                "min_rerank_score": min_rerank_score,
                "reranker": retrieval_config.get("reranker", "keyword_reranker"),
                "num_candidates": len(candidate_evidence),
                "num_reranked": len(reranked_evidence),
            },
        )
        append_jsonl_record(prediction_path, record)
        records_by_sample_id[sample.sample_id] = record

    records = ordered_records_for_sample_ids(records_by_sample_id, sample_ids)
    write_jsonl(prediction_path, records)
    metrics = summarize_answer_metrics(records)
    metrics.update(
        {
            "experiment_id": config["experiment_id"],
            "backend": backend.backend_name,
            "model": backend.model_name,
            "split": dataset_config["split"],
            "scale": dataset_config["scale"],
            "retriever": retrieval_config.get("retriever", "keyword"),
            "reranker": retrieval_config.get("reranker", "keyword_reranker"),
            "top_k": top_k,
            "candidate_top_k": candidate_top_k,
            "min_rerank_score": min_rerank_score,
            "corpus_path": str(corpus_path.relative_to(resolve_project_path("."))),
            "is_valid_main_result": backend.backend_name != "mock",
            "prediction_file": str(prediction_path.relative_to(resolve_project_path("."))),
        }
    )
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return metrics


def parse_args() -> argparse.Namespace:
    """解析 RAG baseline 运行参数。"""
    parser = argparse.ArgumentParser(description="运行 Medical RAG baseline。")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/exp03_rag.yaml"),
    )
    parser.add_argument("--backend", choices=["mock", "qwen2_5_vl"], default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run(resolve_project_path(args.config), backend_name=args.backend)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
