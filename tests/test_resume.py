import json

from medreason_agent.experiments.resume import (
    append_jsonl_record,
    load_records_by_sample_id,
    ordered_records_for_sample_ids,
)


def test_append_and_load_records_by_sample_id(tmp_path) -> None:
    prediction_path = tmp_path / "predictions.jsonl"

    append_jsonl_record(prediction_path, {"sample_id": "a", "prediction": "old"})
    append_jsonl_record(prediction_path, {"sample_id": "b", "prediction": "yes"})
    append_jsonl_record(prediction_path, {"sample_id": "a", "prediction": "new"})

    records = load_records_by_sample_id(prediction_path)

    assert records["a"]["prediction"] == "new"
    assert records["b"]["prediction"] == "yes"


def test_ordered_records_filters_and_preserves_split_order() -> None:
    records = {
        "b": {"sample_id": "b"},
        "a": {"sample_id": "a"},
        "stale": {"sample_id": "stale"},
    }

    ordered = ordered_records_for_sample_ids(records, ["a", "missing", "b"])

    assert [record["sample_id"] for record in ordered] == ["a", "b"]


def test_load_records_reports_invalid_json_line(tmp_path) -> None:
    prediction_path = tmp_path / "predictions.jsonl"
    prediction_path.write_text(
        json.dumps({"sample_id": "a"}, ensure_ascii=False) + "\n{bad json}\n",
        encoding="utf-8",
    )

    try:
        load_records_by_sample_id(prediction_path)
    except ValueError as exc:
        assert "Invalid JSONL" in str(exc)
    else:
        raise AssertionError("invalid JSONL should raise ValueError")

