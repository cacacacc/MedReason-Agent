import json

import yaml

from medreason_agent.data.mixed_vqa import load_mixed_vqa_jsonl
from scripts.build_mixed_vqa_sft_dataset import _standardize_records
from scripts.train_qwen_vl_lora import JSONLLoRADataset, build_lora_dataset


def _write_jsonl(path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def test_load_mixed_vqa_jsonl_reads_standardized_records(tmp_path) -> None:
    path = tmp_path / "train.jsonl"
    _write_jsonl(
        path,
        [
            {
                "dataset": "PathVQA",
                "sample_id": "PathVQA:1",
                "image_id": "1.jpg",
                "image_path": str(tmp_path / "1.jpg"),
                "question": "Is there a mass?",
                "answer": "yes",
                "answer_type": "CLOSED",
                "question_type": "PRES",
                "image_organ": "UNKNOWN",
                "split": "train",
            }
        ],
    )

    samples = load_mixed_vqa_jsonl(path)

    assert len(samples) == 1
    assert samples[0].dataset == "PathVQA"
    assert samples[0].absolute_image_path == tmp_path / "1.jpg"


def test_standardize_records_filters_non_english_and_long_answers(tmp_path) -> None:
    source = tmp_path / "pathvqa.jsonl"
    image_root = tmp_path / "images"
    image_root.mkdir()
    (image_root / "a.jpg").write_bytes(b"fake")
    (image_root / "b.jpg").write_bytes(b"fake")
    (image_root / "c.jpg").write_bytes(b"fake")
    _write_jsonl(
        source,
        [
            {"id": 1, "image": "a.jpg", "question": "Is this ct?", "answer": "yes"},
            {"id": 2, "image": "b.jpg", "question": "这是什么?", "answer": "肺部"},
            {
                "id": 3,
                "image": "c.jpg",
                "question": "What is shown?",
                "answer": "a very long explanatory answer with many words",
            },
        ],
    )

    records, counters = _standardize_records(
        dataset_name="PathVQA",
        source_path=source,
        image_root=image_root,
        split="train",
        max_samples=None,
        max_answer_tokens=5,
        require_images=True,
    )

    assert len(records) == 1
    assert records[0]["answer"] == "yes"
    assert records[0]["answer_type"] == "CLOSED"
    assert counters["non_english"] == 1
    assert counters["long_answer"] == 1


def test_standardize_records_supports_pathvqa_response_and_images(tmp_path) -> None:
    source = tmp_path / "pathvqa.jsonl"
    image_root = tmp_path / "images"
    image_dir = image_root / "image_train"
    image_dir.mkdir(parents=True)
    (image_dir / "000000.jpg").write_bytes(b"fake")
    _write_jsonl(
        source,
        [
            {
                "question": (
                    "<image>Given this image, please answer the question: "
                    "where are cells located?"
                ),
                "response": "in the canals of hering",
                "images": ["/path_vqa/image_train/000000.jpg"],
            }
        ],
    )

    records, counters = _standardize_records(
        dataset_name="PathVQA",
        source_path=source,
        image_root=image_root,
        split="train",
        max_samples=None,
        max_answer_tokens=5,
        require_images=True,
    )

    assert counters["kept"] == 1
    assert records[0]["question"] == "where are cells located?"
    assert records[0]["answer"] == "in canals of hering"
    assert records[0]["image_path"].endswith("images/image_train/000000.jpg")


def test_lora_dataset_can_use_standardized_jsonl(tmp_path) -> None:
    path = tmp_path / "train.jsonl"
    image_path = tmp_path / "1.jpg"
    image_path.write_bytes(b"fake")
    _write_jsonl(
        path,
        [
            {
                "dataset": "SLAKE",
                "sample_id": "SLAKE:1",
                "image_id": "1.jpg",
                "image_path": str(image_path),
                "question": "What modality is shown?",
                "answer": "ct",
                "answer_type": "OPEN",
                "question_type": "MODALITY",
                "image_organ": "CHEST",
                "split": "train",
            }
        ],
    )

    dataset = JSONLLoRADataset(path=path, prompt_template="question_type_aware_short_v2")
    selected = build_lora_dataset(
        {"train_file": path, "max_train_samples": 1},
        split_key="train_split",
        file_key="train_file",
        max_samples_key="max_train_samples",
        prompt_template="question_type_aware_short_v2",
        image_min_pixels=None,
        image_max_pixels=None,
    )

    assert len(dataset) == 1
    assert len(selected) == 1
    assert "Answer type: OPEN" in dataset[0].user_messages()[0]["content"][1]["text"]


def test_phase8_mixed_training_config_parses() -> None:
    with open(
        "configs/training/phase8_qwen_vl_lora_mixed_vqa_48gb.yaml",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    assert config["phase"] == 8
    assert config["dataset"]["train_file"].endswith("mixed_vqa_sft/train.jsonl")
    assert config["training"]["num_train_epochs"] == 1
    assert config["training"]["learning_rate"] == 0.00003
