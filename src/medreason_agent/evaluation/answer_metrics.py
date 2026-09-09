"""Medical VQA 的答案归一化和基础指标。

Phase 1 和 Phase 2 都会用这里的函数评估最终答案。Direct baseline 直接评估
模型输出；CoT baseline 会先抽取 `Final Answer`，再交给这些指标函数。
"""

from __future__ import annotations

import math
import re
from collections import Counter

from medreason_agent.answer_normalization import canonical_short_answer

_ARTICLES = {"a", "an", "the"}


def normalize_answer(answer: str) -> str:
    """把答案文本归一化，方便做 exact match 风格评估。

    归一化会统一大小写、去掉标点、移除英文冠词。这样 `The lung.` 和 `lung`
    会被视为同一个答案，避免因为表面格式差异误判。
    """
    text = _normalize_synonyms(canonical_short_answer(answer).lower().strip())
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [token for token in text.split() if token not in _ARTICLES]
    return " ".join(tokens)


def exact_match(prediction: str, ground_truth: str, question: str = "") -> bool:
    """判断预测答案和标准答案在归一化后是否完全一致。

    Exact match 是最严格的主指标之一，特别适合 yes/no 这类 closed questions。
    """
    normalized_prediction = normalize_answer(
        canonical_short_answer(prediction, question=question),
    )
    return normalized_prediction == normalize_answer(ground_truth)


def token_f1(prediction: str, ground_truth: str) -> float:
    """计算归一化后的 token-level F1。

    Token F1 比 exact match 更宽松：如果开放式答案只答对了一部分关键词，
    它能给出部分分数。
    """
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()

    if not prediction_tokens and not ground_truth_tokens:
        return 1.0
    if not prediction_tokens or not ground_truth_tokens:
        return 0.0

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0

    # Precision 表示预测 token 中有多少是正确的；recall 表示标准答案 token
    # 中有多少被预测覆盖。F1 是二者的调和平均。
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(ground_truth_tokens)
    return 2 * precision * recall / (precision + recall)


def bleu_1(prediction: str, ground_truth: str) -> float:
    """计算简化版 BLEU-1，并加入 brevity penalty。

    BLEU-1 只看 unigram 重叠，适合作为开放式短答案的辅助指标。
    它不是医学正确性的最终判断，只帮助观察词面相似度。
    """
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()

    if not prediction_tokens and not ground_truth_tokens:
        return 1.0
    if not prediction_tokens or not ground_truth_tokens:
        return 0.0

    overlap = sum((Counter(prediction_tokens) & Counter(ground_truth_tokens)).values())
    precision = overlap / len(prediction_tokens)
    brevity_penalty = 1.0
    if len(prediction_tokens) < len(ground_truth_tokens):
        # 如果预测答案明显短于标准答案，BLEU 会用 brevity penalty 轻微惩罚。
        brevity_penalty = math.exp(1 - len(ground_truth_tokens) / len(prediction_tokens))
    return brevity_penalty * precision


def summarize_answer_metrics(records: list[dict]) -> dict[str, float | int]:
    """汇总一组 prediction records 的答案指标。

    输入 records 必须包含 `prediction` 和 `ground_truth` 字段。这个函数只关心
    最终答案，不评价 reasoning 是否正确。
    """
    total = len(records)
    if total == 0:
        return {
            "num_samples": 0,
            "accuracy": 0.0,
            "mean_token_f1": 0.0,
            "mean_bleu_1": 0.0,
        }

    exact_matches = [
        exact_match(
            canonical_short_answer(
                str(record["prediction"]),
                question=str(record.get("question", "")),
            ),
            str(record["ground_truth"]),
        )
        for record in records
    ]
    f1_scores = [
        token_f1(
            canonical_short_answer(
                str(record["prediction"]),
                question=str(record.get("question", "")),
            ),
            str(record["ground_truth"]),
        )
        for record in records
    ]
    bleu_scores = [
        bleu_1(
            canonical_short_answer(
                str(record["prediction"]),
                question=str(record.get("question", "")),
            ),
            str(record["ground_truth"]),
        )
        for record in records
    ]

    return {
        "num_samples": total,
        "accuracy": sum(exact_matches) / total,
        "mean_token_f1": sum(f1_scores) / total,
        "mean_bleu_1": sum(bleu_scores) / total,
    }


def _normalize_synonyms(text: str) -> str:
    """归一化常见医学短答案同义写法。"""
    replacements = {
        "cerebrospinal fluid": "csf",
        "computed tomography": "ct",
        "magnetic resonance imaging": "mri",
        "x ray": "xray",
        "x-ray": "xray",
        "chest xray": "cxr",
        "chest x ray": "cxr",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text
