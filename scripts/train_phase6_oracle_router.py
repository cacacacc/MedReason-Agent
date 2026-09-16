"""从 oracle routing analysis 训练轻量 Phase 6 router。

这个脚本是离线蒸馏工具：读取 `analyze_oracle_routing.py` 生成的 CSV，
把 oracle route 转成监督标签，并训练一个 DictVectorizer + LogisticRegression
router。它不会调用 VLM，也不会生成新的模型答案。
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from medreason_agent.agents.learned_router import (
    extract_router_features,
    oracle_route_to_training_label,
    save_router_model,
)
from medreason_agent.paths import resolve_project_path


def load_oracle_rows(path: Path) -> list[dict[str, Any]]:
    """读取 `scripts/analyze_oracle_routing.py` 生成的 oracle CSV。"""
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def build_dataset(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """构造 sklearn 特征和可部署 route 标签。"""
    features = [extract_router_features(row) for row in rows]
    labels = [
        oracle_route_to_training_label(str(row.get("oracle_route", "")))
        for row in rows
    ]
    return features, labels


def train_router(
    features: list[dict[str, Any]],
    labels: list[str],
    seed: int,
    validation_ratio: float,
    class_weight: str | None,
) -> tuple[Pipeline, dict[str, Any]]:
    """训练 DictVectorizer + LogisticRegression router。"""
    stratify = labels if min(Counter(labels).values()) >= 2 else None
    train_x, valid_x, train_y, valid_y = train_test_split(
        features,
        labels,
        test_size=validation_ratio,
        random_state=seed,
        stratify=stratify,
    )
    model = Pipeline(
        steps=[
            ("vectorizer", DictVectorizer(sparse=False)),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight=class_weight,
                    random_state=seed,
                    solver="liblinear",
                ),
            ),
        ],
    )
    model.fit(train_x, train_y)
    train_pred = model.predict(train_x)
    valid_pred = model.predict(valid_x)
    labels_sorted = sorted(set(labels))
    metrics = {
        "num_samples": len(labels),
        "num_train": len(train_y),
        "num_validation": len(valid_y),
        "label_distribution": dict(sorted(Counter(labels).items())),
        "train_accuracy": accuracy_score(train_y, train_pred),
        "validation_accuracy": accuracy_score(valid_y, valid_pred),
        "validation_classification_report": classification_report(
            valid_y,
            valid_pred,
            labels=labels_sorted,
            zero_division=0,
            output_dict=True,
        ),
        "validation_confusion_matrix": {
            "labels": labels_sorted,
            "matrix": confusion_matrix(valid_y, valid_pred, labels=labels_sorted).tolist(),
        },
    }
    return model, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 oracle 蒸馏 Phase6 router。")
    parser.add_argument("--oracle-analysis", type=Path, required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--output-metrics", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument(
        "--class-weight",
        choices=["balanced", "none"],
        default="balanced",
        help="当 CoT/Full 标签很少时使用 balanced。",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = load_oracle_rows(resolve_project_path(args.oracle_analysis))
    features, labels = build_dataset(rows)
    class_weight = None if args.class_weight == "none" else args.class_weight
    model, metrics = train_router(
        features=features,
        labels=labels,
        seed=args.seed,
        validation_ratio=args.validation_ratio,
        class_weight=class_weight,
    )
    output_model = resolve_project_path(args.output_model)
    output_metrics = resolve_project_path(args.output_metrics)
    save_router_model(output_model, model)
    output_metrics.parent.mkdir(parents=True, exist_ok=True)
    with output_metrics.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
