import json

from scripts.backfill_error_attribution import backfill_predictions


def test_backfill_predictions_adds_error_attribution(tmp_path) -> None:
    predictions = tmp_path / "predictions.jsonl"
    metrics = tmp_path / "metrics.json"
    records = [
        {
            "sample_id": "s1",
            "question": "Is there opacity?",
            "prediction": "yes",
            "ground_truth": "yes",
            "retrieved_evidence": [],
        },
        {
            "sample_id": "s2",
            "question": "Is there opacity?",
            "prediction": "yes",
            "ground_truth": "no",
            "agent_mode": "supervisor_multi_agent",
            "agent_route": ["retrieval_agent", "reasoning_agent"],
            "retrieved_evidence": [],
        },
    ]
    with predictions.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")

    summary = backfill_predictions(predictions, metrics)

    updated = [
        json.loads(line)
        for line in predictions.read_text(encoding="utf-8").splitlines()
    ]
    assert updated[0]["correct"] is True
    assert updated[0]["error_attribution"]["primary_error_type"] == ""
    assert updated[1]["correct"] is False
    assert updated[1]["error_type"] == "Retrieval Error"
    assert updated[1]["answer_gate"]["decision"] == "DISABLED"
    assert summary["num_incorrect"] == 1
    assert metrics.exists()
