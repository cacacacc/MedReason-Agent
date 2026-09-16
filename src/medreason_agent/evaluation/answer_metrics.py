"""Medical VQA answer normalization and basic metrics."""

from __future__ import annotations

import math
import re
from collections import Counter

from medreason_agent.answer_normalization import canonical_short_answer

_ARTICLES = {"a", "an", "the"}


def normalize_answer(answer: str) -> str:
    """Normalize answer text for exact-match style Medical VQA evaluation."""
    text = _normalize_synonyms(canonical_short_answer(answer).lower().strip())
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [token for token in text.split() if token not in _ARTICLES]
    return " ".join(tokens)


def exact_match(prediction: str, ground_truth: str, question: str = "") -> bool:
    """Return whether prediction and ground truth match after answer normalization."""
    normalized_prediction = normalize_answer(
        canonical_short_answer(prediction, question=question),
    )
    return normalized_prediction == normalize_answer(ground_truth)


def token_f1(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 after answer normalization."""
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
    """Compute a simplified BLEU-1 score with brevity penalty."""
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
    """Summarize final-answer metrics for prediction records."""
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
            question=str(record.get("question", "")),
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
    """Normalize common Medical VQA short-answer synonyms."""
    text = _normalize_laterality(text)
    replacements = {
        "anteroposterior": "ap",
        "air space": "airspace",
        "bilateral": "both",
        "cerebrospinal fluid": "csf",
        "chest radiograph": "cxr",
        "chest x ray": "cxr",
        "chest xray": "cxr",
        "computed tomography": "ct",
        "computed tomographic": "ct",
        "computerized tomography": "ct",
        "computerized tomographic": "ct",
        "coronal plane": "coronal",
        "frontal chest radiograph": "cxr",
        "frontal radiograph": "xray",
        "gall stones": "gallstones",
        "hepatic": "liver",
        "large intestine": "colon",
        "magnetic resonance": "mri",
        "magnetic resonance image": "mri",
        "magnetic resonance imaging": "mri",
        "no abnormalities": "normal",
        "no abnormality": "normal",
        "no acute abnormality": "normal",
        "not abnormal": "normal",
        "not enlarged": "normal",
        "normal size": "normal",
        "posterior anterior": "pa",
        "posteroanterior": "pa",
        "radiograph": "xray",
        "radiography": "xray",
        "renal": "kidney",
        "sagittal plane": "sagittal",
        "small intestine": "small bowel",
        "sonogram": "ultrasound",
        "sonography": "ultrasound",
        "ultrasonography": "ultrasound",
        "us": "ultrasound",
        "within normal limits": "normal",
        "x-ray": "xray",
        "x ray": "xray",
    }
    for source, target in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
        text = re.sub(rf"\b{re.escape(source)}\b", target, text)
    return text


def _normalize_laterality(text: str) -> str:
    """Normalize common left/right side variants."""
    text = re.sub(r"\bleft[-\s]+sided\b", "left", text)
    text = re.sub(r"\bright[-\s]+sided\b", "right", text)
    text = re.sub(r"\bleft\s+side\b", "left", text)
    text = re.sub(r"\bright\s+side\b", "right", text)
    return text
