"""运行 Phase 3 Medical RAG experiments。

支持两条不同 RAG agent 路径：

1. Knowledge RAG：先检索知识，再让 VLM reasoning。
2. Evidence RAG：先让 VLM 产生 claim，再检索证据做 verification。
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

from medreason_agent.agents.rag_agents import (
    RAGAgentResult,
    RetrievalPipeline,
    RetrievalSettings,
    create_rag_agent,
)
from medreason_agent.data.vqa_rad import VQARADSample, load_vqa_rad_split
from medreason_agent.evaluation.answer_metrics import exact_match, summarize_answer_metrics
from medreason_agent.evaluation.claim_status import summarize_claim_statuses
from medreason_agent.evaluation.error_attribution import (
    attribute_error,
    summarize_error_attribution,
)
from medreason_agent.evaluation.evidence_metrics import (
    score_evidence_quality,
    summarize_evidence_quality,
)
from medreason_agent.experiments.resume import (
    append_jsonl_record,
    load_records_by_sample_id,
    ordered_records_for_sample_ids,
)
from medreason_agent.models.vlm import create_vlm_backend
from medreason_agent.paths import resolve_project_path
from medreason_agent.retrieval.faiss_retriever import FAISSRetriever
from medreason_agent.retrieval.keyword import KeywordRetriever, load_chunks
from medreason_agent.retrieval.rerank import KeywordReranker


def load_config(path: Path) -> dict[str, Any]:
    """读取定义 RAG 实验条件的 YAML 配置。"""
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def build_prediction_record(
    sample: VQARADSample,
    agent_result: RAGAgentResult,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """构造一条项目标准 RAG prediction record。"""
    correct = exact_match(
        agent_result.prediction,
        sample.answer,
        question=sample.question,
    )
    # Evidence Quality Score 衡量检索证据和当前样本的相关性。
    # 它和 answer accuracy 分开记录，方便后续判断问题出在检索还是生成。
    evidence_quality = score_evidence_quality(
        question=sample.question,
        ground_truth=sample.answer,
        evidence_records=agent_result.retrieved_evidence,
    )
    record = {
        "experiment_id": metadata["experiment_id"],
        "model": metadata["model"],
        "backend": metadata["backend"],
        "prompt_version": metadata["prompt_version"],
        "prompt_contract": metadata["prompt_contract"],
        "rag_mode": metadata["rag_mode"],
        "dataset": sample.dataset,
        "split": sample.split,
        "sample_id": sample.sample_id,
        "image_id": sample.image_id,
        "image_path": sample.image_path,
        "question": sample.question,
        "prediction": agent_result.prediction,
        "ground_truth": sample.answer,
        "answer_type": sample.answer_type,
        "question_type": sample.question_type,
        "image_organ": sample.image_organ,
        "reasoning_output": agent_result.reasoning_output,
        "raw_output": agent_result.raw_output,
        "retrieved_evidence": agent_result.retrieved_evidence,
        "knowledge_query": agent_result.knowledge_query,
        "evidence_query": agent_result.evidence_query,
        "knowledge_evidence": agent_result.knowledge_evidence or [],
        "verification_evidence": agent_result.verification_evidence or [],
        "generated_claims": agent_result.generated_claims or [],
        "claim_statuses": agent_result.claim_statuses or [],
        "initial_prediction": agent_result.initial_prediction,
        "initial_reasoning_output": agent_result.initial_reasoning_output,
        "verified_claim": agent_result.verified_claim,
        "claim_verification_status": agent_result.claim_verification_status,
        "evidence_quality_score": evidence_quality["evidence_quality_score"],
        "evidence_quality": evidence_quality,
        "tool_calls": agent_result.tool_calls,
        "agent_route": agent_result.agent_route,
        "critic_decision": agent_result.critic_decision,
        "confidence": agent_result.confidence,
        "latency_ms": metadata["latency_ms"],
        "input_tokens": agent_result.input_tokens,
        "output_tokens": agent_result.output_tokens,
        "correct": correct,
        "error_type": "",
    }
    error_attribution = attribute_error(record)
    record["error_attribution"] = error_attribution
    record["error_type"] = error_attribution["primary_error_type"]
    return record


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """把 RAG prediction records 写成 JSONL。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def summarize_verification_status(records: list[dict[str, Any]]) -> dict[str, Any]:
    """统计 verifier 的三类 claim verification 决策。"""
    counts = Counter(
        str(record.get("claim_verification_status", "")).strip() or "NOT_APPLICABLE"
        for record in records
    )
    return {
        "claim_verification_status_counts": dict(sorted(counts.items())),
    }


def is_current_rag_record(
    record: dict[str, Any],
    method: dict[str, Any],
    expected_route: list[str],
) -> bool:
    """判断旧 prediction record 是否符合当前 RAG agent 协议。"""
    if (
        record.get("prompt_version") != method["prompt_version"]
        or record.get("prompt_contract") != method["prompt_contract"]
        or record.get("rag_mode") != method["rag_mode"]
        or record.get("agent_route") != expected_route
    ):
        return False

    required_fields = {
        "knowledge_query",
        "evidence_query",
        "generated_claims",
        "claim_statuses",
        "claim_verification_status",
        "evidence_quality_score",
        "error_attribution",
    }
    if any(field not in record for field in required_fields):
        return False

    if method["rag_mode"] in {"claim_verification", "knowledge_then_claim_verification"}:
        return record.get("claim_verification_status") in {
            "SUPPORTED",
            "UNSUPPORTED",
            "CONTRADICTED",
        }

    return True


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

    retriever_name = str(retrieval_config.get("retriever", "keyword"))
    if retriever_name == "keyword":
        retriever = KeywordRetriever(load_chunks(corpus_path))
    elif retriever_name == "faiss":
        retriever = FAISSRetriever(
            index_path=resolve_project_path(retrieval_config["index_path"]),
            metadata_path=resolve_project_path(retrieval_config["metadata_path"]),
            embedding_model=str(
                retrieval_config.get("embedding_model", "BAAI/bge-small-en-v1.5")
            ),
            device=retrieval_config.get("embedding_device"),
        )
    else:
        raise ValueError(f"Unsupported retriever: {retriever_name}")
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
    retrieval_settings = RetrievalSettings(
        candidate_top_k=candidate_top_k,
        top_k=top_k,
        min_rerank_score=min_rerank_score,
        max_chars_per_evidence=max_chars_per_evidence,
        retriever_name=str(retrieval_config.get("retriever", "keyword_retriever")),
        reranker_name=str(retrieval_config.get("reranker", "keyword_reranker")),
    )
    retrieval_pipeline = RetrievalPipeline(
        retriever=retriever,
        reranker=reranker,
        settings=retrieval_settings,
    )
    rag_agent = create_rag_agent(
        rag_mode=method["rag_mode"],
        backend=backend,
        retrieval_pipeline=retrieval_pipeline,
    )

    result_dir = resolve_project_path(output_config["result_dir"])
    prediction_path = result_dir / output_config["prediction_file"]
    metrics_path = result_dir / output_config["metrics_file"]

    # 断点续跑：RAG 每个样本都包含检索证据和 VLM 回答，跑得更慢，所以必须边跑边落盘。
    sample_ids = [sample.sample_id for sample in samples]
    existing_records_by_sample_id = load_records_by_sample_id(prediction_path)
    expected_route = rag_agent.agent_route
    records_by_sample_id = {
        sample_id: record
        for sample_id, record in existing_records_by_sample_id.items()
        if is_current_rag_record(record, method=method, expected_route=expected_route)
    }
    missing_samples = [sample for sample in samples if sample.sample_id not in records_by_sample_id]
    print(
        f"RESUME existing={len(samples) - len(missing_samples)} "
        f"missing={len(missing_samples)} total={len(samples)}"
    )

    for sample in tqdm(missing_samples, desc=config["experiment_id"], ascii=True):
        # RAG latency 包含检索和一次或多次 VLM 生成，因为用户最终感受到的是端到端耗时。
        start = time.perf_counter()
        agent_result = rag_agent.run(
            sample=sample,
            max_new_tokens=int(generation_config["max_new_tokens"]),
        )
        latency_ms = round((time.perf_counter() - start) * 1000, 3)

        record = build_prediction_record(
            sample=sample,
            agent_result=agent_result,
            metadata={
                "experiment_id": config["experiment_id"],
                "model": backend.model_name,
                "backend": backend.backend_name,
                "prompt_version": method["prompt_version"],
                "prompt_contract": method["prompt_contract"],
                "rag_mode": method["rag_mode"],
                "latency_ms": latency_ms,
            },
        )
        append_jsonl_record(prediction_path, record)
        records_by_sample_id[sample.sample_id] = record

    records = ordered_records_for_sample_ids(records_by_sample_id, sample_ids)
    write_jsonl(prediction_path, records)
    metrics = summarize_answer_metrics(records)
    metrics.update(summarize_evidence_quality(records))
    metrics.update(summarize_claim_statuses(records))
    metrics.update(summarize_verification_status(records))
    metrics.update(summarize_error_attribution(records))
    metrics.update(
        {
            "experiment_id": config["experiment_id"],
            "backend": backend.backend_name,
            "model": backend.model_name,
            "split": dataset_config["split"],
            "scale": dataset_config["scale"],
            "prompt_version": method["prompt_version"],
            "prompt_contract": method["prompt_contract"],
            "rag_mode": method["rag_mode"],
            "retriever": retriever_name,
            "reranker": retrieval_config.get("reranker", "keyword_reranker"),
            "embedding_model": retrieval_config.get("embedding_model"),
            "index_path": retrieval_config.get("index_path"),
            "metadata_path": retrieval_config.get("metadata_path"),
            "top_k": top_k,
            "candidate_top_k": candidate_top_k,
            "min_rerank_score": min_rerank_score,
            "corpus_name": retrieval_config.get("corpus_name", "seed_medical_vqa"),
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
