"""Download and normalize the VQA-RAD dataset.

The script keeps raw files under Data/Raw/vqa_rad and writes normalized JSONL
files under Data/Processed/vqa_rad. It does not require third-party packages.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

DATASET_URL = "https://osf.io/download/6qdas/"
IMAGES_URL = "https://files.osf.io/v1/resources/89kps/providers/osfstorage/5b21453986d8510011c277bc/?zip="

RAW_DIR = ROOT / "Data" / "Raw" / "vqa_rad"
PROCESSED_DIR = ROOT / "Data" / "Processed" / "vqa_rad"

RAW_JSON = RAW_DIR / "VQA_RAD Dataset Public.json"
RAW_IMAGE_ZIP = RAW_DIR / "VQA_RAD Image Folder.zip"
RAW_IMAGE_DIR = RAW_DIR / "VQA_RAD Image Folder"


def download_file(url: str, destination: Path, overwrite: bool = False) -> None:
    """Download a file unless it already exists."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and not overwrite:
        print(f"SKIP existing file: {destination}")
        return

    print(f"DOWNLOAD {url}")
    print(f"TO       {destination}")
    with urllib.request.urlopen(url, timeout=120) as response:
        with destination.open("wb") as file:
            shutil.copyfileobj(response, file)


def extract_images(overwrite: bool = False) -> None:
    """Extract image zip into the raw VQA-RAD folder."""
    if RAW_IMAGE_DIR.exists() and not overwrite:
        print(f"SKIP existing image folder: {RAW_IMAGE_DIR}")
        return

    RAW_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"EXTRACT {RAW_IMAGE_ZIP}")
    with zipfile.ZipFile(RAW_IMAGE_ZIP) as archive:
        archive.extractall(RAW_IMAGE_DIR)


def load_raw_records() -> list[dict[str, Any]]:
    """Load VQA-RAD raw JSON records."""
    with RAW_JSON.open("r", encoding="utf-8-sig") as file:
        records = json.load(file)

    if not isinstance(records, list):
        raise ValueError("VQA-RAD JSON must contain a list of records.")

    return records


def find_image_path(image_name: str) -> str:
    """Return a project-relative image path when the file can be found."""
    candidates = list(RAW_DIR.rglob(image_name))
    if not candidates:
        return ""

    return candidates[0].relative_to(ROOT).as_posix()


def stable_sample_id(record: dict[str, Any], index: int) -> str:
    """Create a stable sample id even when qid is missing."""
    qid = record.get("qid") or record.get("question_id")
    if qid is not None:
        return str(qid)

    payload = json.dumps(record, sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    return f"vqa_rad_{index:05d}_{digest}"


def normalize_record(record: dict[str, Any], index: int) -> dict[str, Any]:
    """Normalize one raw VQA-RAD record into the project schema."""
    image_name = str(record.get("image_name") or record.get("image") or "")
    phrase_type = str(record.get("phrase_type") or "").strip()
    official_split = "test" if phrase_type.lower().startswith("test") else "train"

    return {
        "dataset": "VQA-RAD",
        "sample_id": stable_sample_id(record, index),
        "image_id": image_name,
        "image_path": find_image_path(image_name) if image_name else "",
        "question": str(record.get("question") or "").strip(),
        "answer": str(record.get("answer") or "").strip(),
        "answer_type": str(record.get("answer_type") or "").strip().upper(),
        "question_type": str(record.get("question_type") or "").strip().upper(),
        "image_organ": str(record.get("image_organ") or "").strip().upper(),
        "phrase_type": phrase_type,
        "official_split": official_split,
        "split": official_split,
        "raw": record,
    }


def split_records(
    records: list[dict[str, Any]],
    validation_ratio: float,
    seed: int,
) -> list[dict[str, Any]]:
    """Create train / validation / test while preserving official test."""
    train_pool = [record for record in records if record["official_split"] != "test"]
    test_records = [record for record in records if record["official_split"] == "test"]

    rng = random.Random(seed)
    shuffled_train = train_pool[:]
    rng.shuffle(shuffled_train)

    validation_count = round(len(shuffled_train) * validation_ratio)
    validation_ids = {record["sample_id"] for record in shuffled_train[:validation_count]}

    for record in records:
        if record["official_split"] == "test":
            record["split"] = "test"
        elif record["sample_id"] in validation_ids:
            record["split"] = "validation"
        else:
            record["split"] = "train"

    print(
        "SPLIT "
        f"train={sum(record['split'] == 'train' for record in records)} "
        f"validation={sum(record['split'] == 'validation' for record in records)} "
        f"test={len(test_records)}"
    )
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write JSONL records."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    """Write a lightweight CSV view without the raw nested record."""
    fieldnames = [
        "dataset",
        "sample_id",
        "image_id",
        "image_path",
        "question",
        "answer",
        "answer_type",
        "question_type",
        "image_organ",
        "phrase_type",
        "official_split",
        "split",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record[field] for field in fieldnames})


def build_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a compact dataset statistics report."""
    split_counts = Counter(record["split"] for record in records)
    official_split_counts = Counter(record["official_split"] for record in records)
    answer_type_counts = Counter(record["answer_type"] for record in records)
    question_type_counts = Counter(record["question_type"] for record in records)
    image_count = len({record["image_id"] for record in records if record["image_id"]})
    missing_images = [
        record["sample_id"]
        for record in records
        if record["image_id"] and not record["image_path"]
    ]

    return {
        "dataset": "VQA-RAD",
        "source_json_url": DATASET_URL,
        "source_images_url": IMAGES_URL,
        "num_records": len(records),
        "num_images_referenced": image_count,
        "split_counts": dict(split_counts),
        "official_split_counts": dict(official_split_counts),
        "answer_type_counts": dict(answer_type_counts),
        "question_type_counts": dict(question_type_counts),
        "missing_image_count": len(missing_images),
        "missing_image_sample_ids_preview": missing_images[:20],
    }


def write_outputs(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Write normalized dataset files and return stats."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    write_jsonl(PROCESSED_DIR / "all.jsonl", records)
    write_csv(PROCESSED_DIR / "all.csv", records)
    for split in ["train", "validation", "test"]:
        split_records_for_file = [record for record in records if record["split"] == split]
        write_jsonl(
            PROCESSED_DIR / f"{split}.jsonl",
            split_records_for_file,
        )
        write_csv(PROCESSED_DIR / f"{split}.csv", split_records_for_file)

    stats = build_stats(records)
    with (PROCESSED_DIR / "stats.json").open("w", encoding="utf-8") as file:
        json.dump(stats, file, indent=2, ensure_ascii=False)

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import VQA-RAD into project data folders.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download and re-extract raw files.",
    )
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.validation_ratio < 1.0:
        raise ValueError("--validation-ratio must be in [0.0, 1.0).")

    download_file(DATASET_URL, RAW_JSON, overwrite=args.overwrite)
    download_file(IMAGES_URL, RAW_IMAGE_ZIP, overwrite=args.overwrite)
    extract_images(overwrite=args.overwrite)

    raw_records = load_raw_records()
    normalized = [normalize_record(record, index) for index, record in enumerate(raw_records)]
    split_records(normalized, validation_ratio=args.validation_ratio, seed=args.seed)
    stats = write_outputs(normalized)

    print(json.dumps(stats, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
