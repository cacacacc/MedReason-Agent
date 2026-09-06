from medreason_agent.evaluation.answer_metrics import (
    bleu_1,
    exact_match,
    normalize_answer,
    token_f1,
)


def test_normalize_answer_lowercases_and_removes_articles() -> None:
    assert normalize_answer("The Right Lung.") == "right lung"


def test_exact_match_uses_normalized_text() -> None:
    assert exact_match("A lung", "lung")


def test_token_f1_handles_partial_overlap() -> None:
    assert token_f1("right lung", "left lung") == 0.5


def test_bleu_1_scores_unigram_overlap() -> None:
    assert bleu_1("right lung", "right lung") == 1.0
