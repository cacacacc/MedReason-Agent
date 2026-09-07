from medreason_agent.evaluation.evidence_metrics import (
    content_tokens,
    score_evidence_quality,
    summarize_evidence_quality,
)


def test_content_tokens_removes_low_information_words() -> None:
    assert content_tokens("Is there an aortic aneurysm?") == {"aortic", "aneurysm"}


def test_score_evidence_quality_returns_zero_for_empty_evidence() -> None:
    quality = score_evidence_quality(
        question="Is there an aortic aneurysm?",
        ground_truth="yes",
        evidence_records=[],
    )

    assert quality["evidence_quality_score"] == 0.0
    assert quality["question_coverage"] == 0.0
    assert quality["answer_coverage"] is None
    assert quality["has_evidence"] is False


def test_score_evidence_quality_uses_question_coverage_for_closed_answer() -> None:
    quality = score_evidence_quality(
        question="Is there an aortic aneurysm?",
        ground_truth="yes",
        evidence_records=[
            {
                "title": "Aortic aneurysm imaging findings",
                "text": "Aortic aneurysm may show widened aortic contour.",
            }
        ],
    )

    assert quality["evidence_quality_score"] == 1.0
    assert quality["question_coverage"] == 1.0
    assert quality["answer_coverage"] is None


def test_score_evidence_quality_combines_question_and_open_answer_coverage() -> None:
    quality = score_evidence_quality(
        question="What organ is shown?",
        ground_truth="lung",
        evidence_records=[
            {
                "title": "Chest radiograph",
                "text": "The lung is visible on this chest radiograph.",
            }
        ],
    )

    assert quality["answer_coverage"] == 1.0
    assert 0.0 < quality["evidence_quality_score"] <= 1.0


def test_summarize_evidence_quality() -> None:
    records = [
        {
            "question": "Is there an aortic aneurysm?",
            "ground_truth": "yes",
            "retrieved_evidence": [
                {
                    "title": "Aortic aneurysm imaging findings",
                    "text": "Aortic aneurysm may show widened aortic contour.",
                }
            ],
        },
        {
            "question": "What organ is shown?",
            "ground_truth": "lung",
            "retrieved_evidence": [],
        },
    ]

    summary = summarize_evidence_quality(records)

    assert summary["mean_evidence_quality_score"] == 0.5
    assert summary["evidence_empty_rate"] == 0.5
