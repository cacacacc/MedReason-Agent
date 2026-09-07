"""RAG 证据质量评估指标。

Evidence Quality Score 用来评估“检索到的证据本身是否相关”。它不直接判断最终答案
是否正确，而是回答另一个问题：RAG 交给模型的 evidence 是否覆盖了问题关键词和答案
关键词。
"""

from __future__ import annotations

from statistics import mean
from typing import Any

from medreason_agent.evaluation.answer_metrics import normalize_answer
from medreason_agent.retrieval.keyword import tokenize

# 这些词在医学问题中经常出现，但对检索相关性帮助很小。去掉它们之后，
# Evidence Quality Score 更关注 organ、finding、modality 等有信息量的词。
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "be",
    "can",
    "do",
    "does",
    "has",
    "have",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "there",
    "this",
    "to",
    "what",
    "which",
    "with",
}

_CLOSED_ANSWER_TOKENS = {"yes", "no", "true", "false"}


def content_tokens(text: str) -> set[str]:
    """提取用于 evidence 评估的内容词。

    这里先使用轻量规则，不引入额外 embedding 模型。后续如果接入 BGE/Qdrant，
    仍然可以保留这个函数作为可解释的辅助指标。
    """
    return {token for token in tokenize(text) if token not in _STOPWORDS}


def _combined_evidence_text(evidence_records: list[dict[str, Any]]) -> str:
    """把多条 evidence 的标题和正文合并成一段待评估文本。"""
    parts: list[str] = []
    for evidence in evidence_records:
        parts.append(str(evidence.get("title", "")))
        parts.append(str(evidence.get("text", "")))
    return " ".join(parts)


def token_coverage(reference_tokens: set[str], candidate_tokens: set[str]) -> float | None:
    """计算 reference tokens 被 candidate tokens 覆盖的比例。

    如果 reference 为空，返回 None，表示这个分项不适用，而不是给 0 分。
    例如 yes/no 答案本身通常不适合作为 evidence 关键词。
    """
    if not reference_tokens:
        return None
    return len(reference_tokens & candidate_tokens) / len(reference_tokens)


def score_evidence_quality(
    question: str,
    ground_truth: str,
    evidence_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """计算单条样本的 Evidence Quality Score。

    分数范围是 0 到 1：

    - question_coverage：evidence 是否覆盖问题中的内容词。
    - answer_coverage：evidence 是否覆盖开放式标准答案中的内容词。
    - evidence_quality_score：二者的加权合成。

    对 yes/no 这类 closed answer，`answer_coverage` 不参与总分，因为 evidence 中是否出现
    yes/no 并不能说明证据质量高。
    """
    if not evidence_records:
        return {
            "evidence_quality_score": 0.0,
            "question_coverage": 0.0,
            "answer_coverage": None,
            "num_evidence": 0,
            "has_evidence": False,
        }

    evidence_tokens = content_tokens(_combined_evidence_text(evidence_records))
    question_coverage = token_coverage(content_tokens(question), evidence_tokens) or 0.0

    normalized_answer = normalize_answer(ground_truth)
    answer_tokens = content_tokens(normalized_answer) - _CLOSED_ANSWER_TOKENS
    answer_coverage = token_coverage(answer_tokens, evidence_tokens)

    if answer_coverage is None:
        evidence_quality_score = question_coverage
    else:
        # 问题覆盖衡量“检索是否围绕问题”，答案覆盖衡量“证据是否包含标准答案线索”。
        # 这里让问题覆盖占更高权重，避免 closed/open 混合任务中过度依赖答案词面。
        evidence_quality_score = 0.6 * question_coverage + 0.4 * answer_coverage

    return {
        "evidence_quality_score": round(evidence_quality_score, 6),
        "question_coverage": round(question_coverage, 6),
        "answer_coverage": None if answer_coverage is None else round(answer_coverage, 6),
        "num_evidence": len(evidence_records),
        "has_evidence": True,
    }


def summarize_evidence_quality(records: list[dict[str, Any]]) -> dict[str, float | int]:
    """汇总 RAG prediction records 的 evidence quality 指标。"""
    if not records:
        return {
            "mean_evidence_quality_score": 0.0,
            "mean_question_coverage": 0.0,
            "mean_answer_coverage": 0.0,
            "evidence_empty_rate": 0.0,
        }

    qualities = [
        record.get("evidence_quality")
        or score_evidence_quality(
            question=str(record.get("question", "")),
            ground_truth=str(record.get("ground_truth", "")),
            evidence_records=list(record.get("retrieved_evidence", [])),
        )
        for record in records
    ]
    answer_coverages = [
        item["answer_coverage"] for item in qualities if item.get("answer_coverage") is not None
    ]

    return {
        "mean_evidence_quality_score": mean(
            float(item["evidence_quality_score"]) for item in qualities
        ),
        "mean_question_coverage": mean(float(item["question_coverage"]) for item in qualities),
        "mean_answer_coverage": mean(float(item) for item in answer_coverages)
        if answer_coverages
        else 0.0,
        "evidence_empty_rate": sum(not item["has_evidence"] for item in qualities) / len(qualities),
    }
