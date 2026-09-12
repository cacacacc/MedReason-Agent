import json

from scripts.analyze_oracle_routing import (
    COT_ROUTE,
    DIRECT_ROUTE,
    FULL_ROUTE,
    build_analysis_rows,
    summarize_rows,
    write_outputs,
)


def _record(sample_id: str, prediction: str, route: str = "") -> dict:
    return {
        "sample_id": sample_id,
        "question": "Is there opacity?",
        "ground_truth": "yes",
        "prediction": prediction,
        "answer_type": "CLOSED",
        "question_type": "PRES",
        "image_organ": "CHEST",
        "phase6_route": route,
    }


def test_oracle_routing_prefers_lowest_cost_correct_route() -> None:
    rows = build_analysis_rows(
        direct_records={"1": _record("1", "yes")},
        cot_records={"1": _record("1", "yes")},
        full_records={"1": _record("1", "yes")},
        routing_records={"1": _record("1", "yes", route=FULL_ROUTE)},
    )
    metrics = summarize_rows(rows)

    assert rows[0]["oracle_route"] == DIRECT_ROUTE
    assert rows[0]["depth_error"] == "over_reasoning"
    assert metrics["oracle_accuracy"] == 1.0
    assert metrics["over_reasoning_rate"] == 1.0


def test_oracle_routing_marks_under_reasoning_when_full_is_only_correct() -> None:
    rows = build_analysis_rows(
        direct_records={"1": _record("1", "no")},
        cot_records={"1": _record("1", "no")},
        full_records={"1": _record("1", "yes")},
        routing_records={"1": _record("1", "no", route=DIRECT_ROUTE)},
    )
    metrics = summarize_rows(rows)

    assert rows[0]["oracle_route"] == FULL_ROUTE
    assert rows[0]["depth_error"] == "under_reasoning"
    assert metrics["under_reasoning_rate"] == 1.0


def test_oracle_routing_writes_outputs(tmp_path) -> None:
    rows = build_analysis_rows(
        direct_records={"1": _record("1", "no")},
        cot_records={"1": _record("1", "yes")},
        full_records={"1": _record("1", "yes")},
        routing_records={"1": _record("1", "yes", route=COT_ROUTE)},
    )
    metrics = summarize_rows(rows)

    write_outputs(rows, metrics, tmp_path)

    assert (tmp_path / "oracle_metrics.json").exists()
    assert (tmp_path / "oracle_analysis.csv").exists()
    assert (tmp_path / "oracle_summary.md").exists()
    saved = json.loads((tmp_path / "oracle_metrics.json").read_text(encoding="utf-8"))
    assert saved["routing_to_oracle_match_rate"] == 1.0
