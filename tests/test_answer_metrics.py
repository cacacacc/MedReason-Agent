from medreason_agent.evaluation.answer_metrics import (
    bleu_1,
    exact_match,
    normalize_answer,
    token_f1,
)


def test_exact_match_canonicalizes_yes_no_sentence() -> None:
    assert exact_match(
        "No, the trachea is not midline.",
        "No",
        question="Is the trachea midline?",
    )


def test_exact_match_canonicalizes_yes_no_negative_phrase() -> None:
    assert exact_match(
        "There is no pleural effusion.",
        "No",
        question="Is there pleural effusion?",
    )


def test_exact_match_canonicalizes_yes_no_presence_phrase() -> None:
    assert exact_match(
        "The mass is present.",
        "Yes",
        question="Is there a mass?",
    )


def test_exact_match_canonicalizes_absent_present_tokens() -> None:
    assert exact_match("absent", "No", question="Is pneumothorax present?")
    assert exact_match("present", "Yes", question="Is pneumothorax present?")


def test_normalize_answer_lowercases_and_removes_articles() -> None:
    assert normalize_answer("The Right Lung.") == "right lung"


def test_exact_match_uses_normalized_text() -> None:
    assert exact_match("A lung", "lung")


def test_exact_match_normalizes_medical_modality_synonyms() -> None:
    assert exact_match("computed tomography", "CT")
    assert exact_match("chest radiograph", "cxr")
    assert exact_match("x-ray", "xray")
    assert exact_match("ultrasonography", "ultrasound")


def test_exact_match_normalizes_laterality_variants() -> None:
    assert exact_match("left-sided", "left")
    assert exact_match("right side", "right")


def test_normalize_answer_does_not_replace_us_inside_words() -> None:
    assert normalize_answer("sinus") == "sinus"


def test_exact_match_normalizes_common_medical_phrases() -> None:
    assert exact_match("within normal limits", "normal")
    assert exact_match("gall stones", "gallstones")
    assert exact_match("renal", "kidney")


def test_token_f1_handles_partial_overlap() -> None:
    assert token_f1("right lung", "left lung") == 0.5


def test_bleu_1_scores_unigram_overlap() -> None:
    assert bleu_1("right lung", "right lung") == 1.0
