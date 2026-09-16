"""Phase 6 的 oracle 蒸馏学习型路由器。

这个模块只负责轻量 router 的特征抽取、模型读写和预测。它不直接读取
ground truth；训练阶段才会使用 oracle analysis 生成的标签。
"""

from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from medreason_agent.evaluation.answer_metrics import normalize_answer

DIRECT_ROUTE = "direct"
COT_ROUTE = "structured_cot"
FULL_ROUTE = "full_multi_agent"
NONE_ROUTE = "none"

DEPLOYABLE_ROUTES = {DIRECT_ROUTE, COT_ROUTE, FULL_ROUTE}

_UNCERTAINTY_MARKERS = (
    "uncertain",
    "cannot determine",
    "cannot be determined",
    "not enough information",
    "insufficient information",
    "unable to determine",
)
_QUESTION_MARKERS = (
    "which side",
    "left or right",
    "right or left",
    "larger",
    "smaller",
    "more than",
    "less than",
    "how many",
    "measure",
    "measurement",
    "diagnosis",
    "diagnostic",
    "disease",
    "cause",
    "etiology",
    "differential",
    "abnormality",
    "abnormal",
    "modality",
    "plane",
)


@dataclass(frozen=True)
class LearnedRouterPrediction:
    """学习型 router 的预测结果和可审计信号。"""

    route: str
    confidence: float
    probabilities: dict[str, float]
    features: dict[str, Any]


def oracle_route_to_training_label(route: str) -> str:
    """把 oracle route 映射成真实部署时可执行的 route 标签。

    `none` 表示 Direct / CoT / Full 三条路线都答错。部署时这种样本没有
    可救回的升级路线，因此映射为最低成本的 Direct。
    """
    normalized = route.strip()
    if normalized == NONE_ROUTE:
        return DIRECT_ROUTE
    if normalized not in DEPLOYABLE_ROUTES:
        return DIRECT_ROUTE
    return normalized


def extract_router_features(record: dict[str, Any]) -> dict[str, Any]:
    """抽取路由特征。

    特征只能来自问题元数据，以及 Direct 首次回答后的可见输出；不能使用
    ground truth 或三条候选路线的正确性，避免训练以外的 oracle 泄漏。
    """
    question = str(record.get("question", ""))
    answer_type = str(record.get("answer_type", "")).upper()
    question_type = str(record.get("question_type", "")).upper()
    image_organ = str(record.get("image_organ", "")).upper()
    direct_prediction = str(
        record.get("direct_prediction", record.get("prediction", "")),
    )
    raw_output = str(record.get("direct_raw_output", record.get("raw_output", "")))
    normalized_prediction = normalize_answer(direct_prediction)
    question_lower = question.lower()
    raw_lower = raw_output.lower()
    question_type_parts = [
        part.strip()
        for part in question_type.split(",")
        if part.strip()
    ]
    prediction_tokens = normalized_prediction.split()

    features: dict[str, Any] = {
        "answer_type": answer_type,
        "question_type": question_type,
        "image_organ": image_organ,
        "is_closed": answer_type == "CLOSED",
        "is_open": answer_type == "OPEN",
        "question_len": len(question.split()),
        "direct_prediction": normalized_prediction,
        "direct_prediction_len": len(prediction_tokens),
        "direct_prediction_empty": not bool(normalized_prediction),
        "direct_prediction_yes": normalized_prediction == "yes",
        "direct_prediction_no": normalized_prediction == "no",
        "direct_prediction_numeric": bool(re.fullmatch(r"\d+(\.\d+)?", normalized_prediction)),
        "direct_raw_len": len(raw_output.split()),
        "direct_uncertainty": any(marker in raw_lower for marker in _UNCERTAINTY_MARKERS),
    }
    for part in question_type_parts:
        features[f"question_type_has_{part}"] = True
    for marker in _QUESTION_MARKERS:
        features[f"question_has_{_feature_name(marker)}"] = marker in question_lower
    for token in prediction_tokens[:4]:
        features[f"direct_pred_token_{_feature_name(token)}"] = True
    return features


def save_router_model(path: Path, model: Any) -> None:
    """保存 sklearn pipeline。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(model, file)


def load_router_model(path: Path) -> Any:
    """加载 sklearn pipeline。"""
    with path.open("rb") as file:
        return pickle.load(file)


def predict_route(model: Any, features: dict[str, Any]) -> LearnedRouterPrediction:
    """用训练好的 sklearn pipeline 预测 route。"""
    route = str(model.predict([features])[0])
    probabilities: dict[str, float] = {}
    confidence = 1.0
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba([features])[0]
        classes = [str(item) for item in model.classes_]
        probabilities = {
            route_name: float(score)
            for route_name, score in zip(classes, proba, strict=True)
        }
        confidence = float(max(probabilities.values())) if probabilities else 1.0
    return LearnedRouterPrediction(
        route=route if route in DEPLOYABLE_ROUTES else DIRECT_ROUTE,
        confidence=confidence,
        probabilities=probabilities,
        features=features,
    )


def _feature_name(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
