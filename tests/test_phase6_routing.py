from medreason_agent.agents.phase6_routing import (
    HIGH_ROUTE,
    LOW_ROUTE,
    MEDIUM_ROUTE,
    decide_rule_based_route,
    fallback_route,
    should_fallback,
)
from medreason_agent.data.vqa_rad import VQARADSample


def _sample(
    question: str = "Is there right lung opacity?",
    answer_type: str = "CLOSED",
    question_type: str = "PRES",
) -> VQARADSample:
    return VQARADSample(
        dataset="VQA-RAD",
        sample_id="s1",
        image_id="image.jpg",
        image_path="Data/Raw/vqa_rad/images/image.jpg",
        question=question,
        answer="yes",
        answer_type=answer_type,
        question_type=question_type,
        image_organ="CHEST",
        split="test",
    )


def test_rule_based_router_sends_simple_closed_question_to_direct() -> None:
    decision = decide_rule_based_route(_sample())

    assert decision.route == LOW_ROUTE
    assert decision.complexity == "LOW"
    assert decision.evidence_support_signal == "NOT_REQUIRED"


def test_rule_based_router_sends_visual_reasoning_question_to_cot() -> None:
    decision = decide_rule_based_route(
        _sample(question="Where is the opacity located?", question_type="POS")
    )

    assert decision.route == MEDIUM_ROUTE
    assert decision.complexity == "MEDIUM"


def test_rule_based_router_sends_open_or_diagnostic_question_to_multi_agent() -> None:
    open_decision = decide_rule_based_route(
        _sample(
            question="What abnormality is shown?",
            answer_type="OPEN",
            question_type="ABN",
        )
    )
    diagnostic_decision = decide_rule_based_route(
        _sample(question="Does this finding suggest pneumonia?", question_type="PRES")
    )

    assert open_decision.route == HIGH_ROUTE
    assert diagnostic_decision.route == HIGH_ROUTE
    assert "suggest" in diagnostic_decision.signals["matched_markers"]


def test_phase6_fallback_steps_up_at_most_one_level() -> None:
    assert fallback_route(LOW_ROUTE) == MEDIUM_ROUTE
    assert fallback_route(MEDIUM_ROUTE) == HIGH_ROUTE
    assert fallback_route(HIGH_ROUTE) is None


def test_phase6_fallback_only_on_clear_uncertainty() -> None:
    should_step_up, reason = should_fallback("uncertain", "Final Answer: uncertain", LOW_ROUTE)
    should_keep, keep_reason = should_fallback("yes", "Final Answer: yes", LOW_ROUTE)

    assert should_step_up is True
    assert reason == "explicit_uncertainty_prediction"
    assert should_keep is False
    assert keep_reason == "no_fallback_signal"
