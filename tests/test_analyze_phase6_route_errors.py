import csv

from scripts.analyze_phase6_route_errors import load_records, summarize


def test_analyze_phase6_route_errors_groups_reasons_and_markers(tmp_path) -> None:
    csv_path = tmp_path / "oracle_analysis.csv"
    rows = [
        {
            "sample_id": "1",
            "question": "Which side has more calcification?",
            "ground_truth": "left",
            "selected_route": "structured_cot",
            "oracle_route": "direct",
            "phase6_routing_reason": "v6 precision visual reasoning trigger",
            "phase6_routing_signals": (
                '{"matched_markers": [], '
                '"strong_medical_markers": [], '
                '"weak_visual_markers": ["which side"]}'
            ),
            "routing_prediction": "right",
            "direct_prediction": "left",
            "cot_prediction": "right",
            "answer_type": "OPEN",
            "question_type": "ATTRIB",
        }
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    records = load_records(csv_path)
    result = summarize(records, selected="structured_cot", oracle="direct")

    assert result["num_samples"] == 1
    assert result["reason_counts"]["v6 precision visual reasoning trigger"] == 1
    assert result["marker_counts"]["which side"] == 1
