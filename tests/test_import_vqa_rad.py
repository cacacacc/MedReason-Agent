from scripts.import_vqa_rad import normalize_record, split_records


def test_normalize_record_uses_phrase_type_for_official_split() -> None:
    record = {
        "qid": 123,
        "image_name": "sample.png",
        "question": "What organ is shown?",
        "answer": "lung",
        "answer_type": "OPEN",
        "question_type": "Modality",
        "image_organ": "CHEST",
        "phrase_type": "test_freeform",
    }

    normalized = normalize_record(record, index=0)

    assert normalized["sample_id"] == "123"
    assert normalized["official_split"] == "test"
    assert normalized["split"] == "test"
    assert normalized["question"] == "What organ is shown?"
    assert normalized["answer"] == "lung"


def test_split_records_preserves_official_test() -> None:
    records = [
        {"sample_id": "1", "official_split": "train", "split": "train"},
        {"sample_id": "2", "official_split": "train", "split": "train"},
        {"sample_id": "3", "official_split": "test", "split": "test"},
    ]

    split = split_records(records, validation_ratio=0.5, seed=42)

    test_records = [record for record in split if record["split"] == "test"]
    validation_records = [record for record in split if record["split"] == "validation"]

    assert [record["sample_id"] for record in test_records] == ["3"]
    assert len(validation_records) == 1
