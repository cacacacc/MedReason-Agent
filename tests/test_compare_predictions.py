from medreason_agent.analysis.compare_predictions import compare_records, outcome_label


def test_outcome_label() -> None:
    assert outcome_label(True, True) == "both_correct"
    assert outcome_label(False, True) == "cot_helped"
    assert outcome_label(True, False) == "cot_hurt"
    assert outcome_label(False, False) == "both_wrong"


def test_compare_records_counts_outcomes() -> None:
    direct = [
        {
            "sample_id": "1",
            "question": "Q1",
            "ground_truth": "yes",
            "prediction": "no",
            "correct": False,
            "answer_type": "CLOSED",
            "question_type": "PRES",
        },
        {
            "sample_id": "2",
            "question": "Q2",
            "ground_truth": "lung",
            "prediction": "lung",
            "correct": True,
            "answer_type": "OPEN",
            "question_type": "ORGAN",
        },
    ]
    cot = [
        {"sample_id": "1", "prediction": "yes", "correct": True},
        {"sample_id": "2", "prediction": "heart", "correct": False},
    ]

    comparison = compare_records(direct, cot)

    assert comparison["num_shared_records"] == 2
    assert comparison["outcome_counts"] == {"cot_helped": 1, "cot_hurt": 1}
