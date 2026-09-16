"""构建混合 Medical VQA 短答案 SFT 数据集。

输入可以来自已经处理好的 VQA-RAD，也可以来自本地 PathVQA / SLAKE 的 JSON 或
JSONL 文件。脚本会把不同字段名统一成同一种训练格式，并默认过滤掉过长答案，
避免 LoRA 学成“解释型输出”而降低 VQA-RAD short-answer accuracy。
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from medreason_agent.evaluation.answer_metrics import normalize_answer
from medreason_agent.paths import project_root, resolve_project_path

QUESTION_KEYS = ("question", "Question", "q", "query")
ANSWER_KEYS = ("answer", "Answer", "answers", "label", "gt_answer")
IMAGE_KEYS = ("image_path", "image", "img_path", "image_name", "img_name", "file_name")
ID_KEYS = ("sample_id", "qid", "id", "question_id")


def _read_records(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL 或 JSON list 格式的原始样本。"""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        records = json.loads(text)
        if not isinstance(records, list):
            raise ValueError(f"JSON file must contain a list: {path}")
        return [record for record in records if isinstance(record, dict)]

    records = []
    for line in text.splitlines():
        if line.strip():
            record = json.loads(line)
            if isinstance(record, dict):
                records.append(record)
    return records


def _pick(record: dict[str, Any], keys: tuple[str, ...]) -> Any:
    """按多个候选字段名取值，兼容不同公开数据集的命名。"""
    for key in keys:
        if key not in record:
            continue
        value = record[key]
        if value is not None and value != "":
            return value
    return None


def _as_answer(value: Any) -> str:
    """把答案字段压成一个短字符串。"""
    if isinstance(value, list):
        if not value:
            return ""
        first = value[0]
        if isinstance(first, dict):
            first = _pick(first, ANSWER_KEYS) or _pick(first, ("text", "value"))
        return str(first or "").strip()
    if isinstance(value, dict):
        value = _pick(value, ANSWER_KEYS) or _pick(value, ("text", "value"))
    return str(value or "").strip()


def _infer_answer_type(answer: str, record: dict[str, Any]) -> str:
    """优先使用原字段，否则根据 yes/no 推断 CLOSED/OPEN。"""
    raw = record.get("answer_type") or record.get("answer_type_name")
    if raw:
        return str(raw).upper()
    return "CLOSED" if normalize_answer(answer) in {"yes", "no"} else "OPEN"


def _infer_question_type(question: str, record: dict[str, Any]) -> str:
    """给没有 question_type 的外部样本补一个粗粒度问题类型。"""
    raw = record.get("question_type") or record.get("question_type_name") or record.get("type")
    if raw:
        return str(raw).upper()

    text = question.lower()
    if "modality" in text or "type of image" in text:
        return "MODALITY"
    if "plane" in text or "view" in text or "slice" in text:
        return "PLANE"
    if text.startswith(("is ", "are ", "does ", "do ", "can ", "has ", "have ")):
        return "PRES"
    if "where" in text or "side" in text or "left" in text or "right" in text:
        return "POS"
    if "how many" in text or "number" in text or "count" in text:
        return "COUNT"
    if "size" in text or "large" in text or "small" in text:
        return "SIZE"
    if "color" in text or "colour" in text:
        return "COLOR"
    return "OTHER"


def _normalize_image_path(raw_path: str, image_root: Path | None) -> str:
    """把图片路径规范化为项目相对路径或绝对路径。"""
    image_path = Path(raw_path)
    if image_root is not None and not image_path.is_absolute():
        image_path = image_root / image_path
    if not image_path.is_absolute():
        image_path = resolve_project_path(image_path)

    try:
        return str(image_path.resolve().relative_to(project_root().resolve())).replace("\\", "/")
    except ValueError:
        return str(image_path.resolve())


def _is_english_record(record: dict[str, Any], question: str) -> bool:
    """过滤 SLAKE 这类双语数据中的非英文样本。"""
    language = str(record.get("lang") or record.get("language") or "").lower()
    if language and language not in {"en", "eng", "english"}:
        return False
    return any("a" <= char.lower() <= "z" for char in question)


def _standardize_records(
    *,
    dataset_name: str,
    source_path: Path,
    image_root: Path | None,
    split: str,
    max_samples: int | None,
    max_answer_tokens: int,
    require_images: bool,
) -> tuple[list[dict[str, Any]], Counter]:
    """把一个来源的数据转换为统一 JSONL record。"""
    counters: Counter = Counter()
    records = _read_records(source_path)
    samples: list[dict[str, Any]] = []

    for index, record in enumerate(records):
        question = str(_pick(record, QUESTION_KEYS) or "").strip()
        answer = _as_answer(_pick(record, ANSWER_KEYS))
        raw_image = str(_pick(record, IMAGE_KEYS) or "").strip()
        if not question or not answer or not raw_image:
            counters["missing_required_field"] += 1
            continue
        if not _is_english_record(record, question):
            counters["non_english"] += 1
            continue

        normalized_answer = normalize_answer(answer)
        if not normalized_answer:
            counters["empty_normalized_answer"] += 1
            continue
        if len(normalized_answer.split()) > max_answer_tokens:
            counters["long_answer"] += 1
            continue

        image_path = _normalize_image_path(raw_image, image_root)
        if require_images and not Path(image_path).is_absolute():
            if not resolve_project_path(image_path).exists():
                counters["missing_image"] += 1
                continue
        elif require_images and not Path(image_path).exists():
            counters["missing_image"] += 1
            continue

        sample_id = str(_pick(record, ID_KEYS) or f"{dataset_name}_{index}")
        samples.append(
            {
                "dataset": dataset_name,
                "sample_id": f"{dataset_name}:{sample_id}",
                "image_id": Path(image_path).name,
                "image_path": image_path,
                "question": question,
                "answer": normalized_answer,
                "answer_type": _infer_answer_type(answer, record),
                "question_type": _infer_question_type(question, record),
                "image_organ": str(record.get("image_organ") or record.get("organ") or "UNKNOWN"),
                "split": split,
            }
        )
        if max_samples is not None and len(samples) >= max_samples:
            break

    counters["kept"] = len(samples)
    counters["seen"] = len(records)
    return samples, counters


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """写出 UTF-8 JSONL 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def _add_optional_source(
    *,
    train_records: list[dict[str, Any]],
    summary: dict[str, Any],
    dataset_name: str,
    source_path: str | None,
    image_root: str | None,
    max_samples: int | None,
    max_answer_tokens: int,
    require_images: bool,
) -> None:
    """如果外部数据文件存在，就加入训练集；不存在则记录为 skipped。"""
    if not source_path:
        summary[dataset_name] = {"status": "skipped", "reason": "not_configured"}
        return

    resolved_source = resolve_project_path(source_path)
    if not resolved_source.exists():
        summary[dataset_name] = {
            "status": "skipped",
            "reason": f"missing_file: {resolved_source}",
        }
        return

    resolved_image_root = resolve_project_path(image_root) if image_root else None
    records, counters = _standardize_records(
        dataset_name=dataset_name,
        source_path=resolved_source,
        image_root=resolved_image_root,
        split="train",
        max_samples=max_samples,
        max_answer_tokens=max_answer_tokens,
        require_images=require_images,
    )
    train_records.extend(records)
    summary[dataset_name] = {"status": "loaded", **dict(counters)}


def parse_args() -> argparse.Namespace:
    """解析构建混合 SFT 数据集所需参数。"""
    parser = argparse.ArgumentParser(description="构建混合 Medical VQA 短答案 SFT 数据集")
    parser.add_argument("--output-dir", type=Path, default=Path("Data/Processed/mixed_vqa_sft"))
    parser.add_argument("--vqa-rad-train", type=str, default="Data/Processed/vqa_rad/train.jsonl")
    parser.add_argument(
        "--vqa-rad-validation",
        type=str,
        default="Data/Processed/vqa_rad/validation.jsonl",
    )
    parser.add_argument("--pathvqa-json", type=str, default=None)
    parser.add_argument("--pathvqa-image-root", type=str, default=None)
    parser.add_argument("--slake-json", type=str, default=None)
    parser.add_argument("--slake-image-root", type=str, default=None)
    parser.add_argument("--max-pathvqa", type=int, default=5000)
    parser.add_argument("--max-slake", type=int, default=3000)
    parser.add_argument("--max-answer-tokens", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-missing-images",
        action="store_true",
        help="调试字段映射时可开启；正式训练建议不要开启。",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)

    output_dir = resolve_project_path(args.output_dir)
    summary: dict[str, Any] = {"output_dir": str(output_dir)}

    train_records, train_counter = _standardize_records(
        dataset_name="VQA-RAD",
        source_path=resolve_project_path(args.vqa_rad_train),
        image_root=None,
        split="train",
        max_samples=None,
        max_answer_tokens=args.max_answer_tokens,
        require_images=not args.allow_missing_images,
    )
    validation_records, validation_counter = _standardize_records(
        dataset_name="VQA-RAD",
        source_path=resolve_project_path(args.vqa_rad_validation),
        image_root=None,
        split="validation",
        max_samples=None,
        max_answer_tokens=args.max_answer_tokens,
        require_images=not args.allow_missing_images,
    )
    summary["VQA-RAD_train"] = dict(train_counter)
    summary["VQA-RAD_validation"] = dict(validation_counter)

    _add_optional_source(
        train_records=train_records,
        summary=summary,
        dataset_name="PathVQA",
        source_path=args.pathvqa_json,
        image_root=args.pathvqa_image_root,
        max_samples=args.max_pathvqa,
        max_answer_tokens=args.max_answer_tokens,
        require_images=not args.allow_missing_images,
    )
    _add_optional_source(
        train_records=train_records,
        summary=summary,
        dataset_name="SLAKE",
        source_path=args.slake_json,
        image_root=args.slake_image_root,
        max_samples=args.max_slake,
        max_answer_tokens=args.max_answer_tokens,
        require_images=not args.allow_missing_images,
    )

    random.shuffle(train_records)
    _write_jsonl(output_dir / "train.jsonl", train_records)
    _write_jsonl(output_dir / "validation.jsonl", validation_records)

    summary["num_train_samples"] = len(train_records)
    summary["num_validation_samples"] = len(validation_records)
    summary["train_file"] = str(output_dir / "train.jsonl")
    summary["validation_file"] = str(output_dir / "validation.jsonl")
    (output_dir / "build_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
