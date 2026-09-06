"""Answer normalization and metrics for Medical VQA."""

from __future__ import annotations

import math
import re
from collections import Counter

_ARTICLES = {"a", "an", "the"}


def normalize_answer(answer: str) -> str:
    """Normalize answer text for exact-match style evaluation."""
    text = answer.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [token for token in text.split() if token not in _ARTICLES]
    return " ".join(tokens)


def exact_match(prediction: str, ground_truth: str) -> bool:
    """Return whether two answers match after normalization."""
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def token_f1(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 after normalization."""
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

    precision = overlap / len(prediction_tokens)
    recall = overlap / len(ground_truth_tokens)
    return 2 * precision * recall / (precision + recall)


def bleu_1(prediction: str, ground_truth: str) -> float:
    """Compute a simple BLEU-1 score with brevity penalty."""
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
        brevity_penalty = math.exp(1 - len(ground_truth_tokens) / len(prediction_tokens))
    return brevity_penalty * precision


def summarize_answer_metrics(records: list[dict]) -> dict[str, float | int]:
    """Summarize prediction records containing prediction and ground_truth."""
    total = len(records)
    if total == 0:
        return {
            "num_samples": 0,
            "accuracy": 0.0,
            "mean_token_f1": 0.0,
            "mean_bleu_1": 0.0,
        }

    exact_matches = [
        exact_match(str(record["prediction"]), str(record["ground_truth"])) for record in records
    ]
    f1_scores = [
        token_f1(str(record["prediction"]), str(record["ground_truth"])) for record in records
    ]
    bleu_scores = [
        bleu_1(str(record["prediction"]), str(record["ground_truth"])) for record in records
    ]

    return {
        "num_samples": total,
        "accuracy": sum(exact_matches) / total,
        "mean_token_f1": sum(f1_scores) / total,
        "mean_bleu_1": sum(bleu_scores) / total,
    }
