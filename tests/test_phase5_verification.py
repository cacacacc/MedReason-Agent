from medreason_agent.agents.phase5_verification import Phase5VerificationAgent
from medreason_agent.data.vqa_rad import VQARADSample
from medreason_agent.evaluation.phase5_metrics import summarize_phase5_metrics
from medreason_agent.models.vlm import MockVLMBackend, VLMRequest, VLMResponse
from medreason_agent.prompts.phase5_verification import (
    build_self_reflection_prompt,
    build_separate_verifier_prompt,
    parse_process_feedback,
)


def _sample(answer: str = "yes") -> VQARADSample:
    return VQARADSample(
        dataset="VQA-RAD",
        sample_id="s1",
        image_id="image.jpg",
        image_path="Data/Raw/vqa_rad/images/image.jpg",
        question="Is there right lung opacity?",
        answer=answer,
        answer_type="CLOSED",
        question_type="PRES",
        image_organ="CHEST",
        split="test",
    )


class RevisingVerifierBackend(MockVLMBackend):
    def generate(self, request: VLMRequest) -> VLMResponse:
        if "Act as an independent process-level medical verifier" in request.prompt:
            raw_output = (
                '{"verdict": "REVISE", '
                '"step_results": [{"step_id": "reasoning_1", '
                '"status": "UNSUPPORTED", "error_type": "UNSUPPORTED_CLAIM", '
                '"confidence": 0.87}], '
                '"grounding_score": 0.52, '
                '"logical_consistency": 0.44, '
                '"confidence_alignment": 0.40, '
                '"error_types": ["UNSUPPORTED_CLAIM"], '
                '"recommended_action": "REVISE", '
                '"revision_instruction": "Change the answer to no."}'
            )
            return VLMResponse(
                answer=raw_output,
                raw_output=raw_output,
                input_tokens=len(request.prompt.split()),
                output_tokens=len(raw_output.split()),
            )
        if "Revise the candidate answer using the independent verifier feedback" in request.prompt:
            raw_output = (
                "Reasoning: corrected mock reasoning.\n"
                "Claims:\n- no\n"
                'Claim Statuses:\n{"claim": "no", "status": "HYPOTHESIS"}\n'
                "Unsupported Assumptions: None.\n"
                "Final Answer: no"
            )
            return VLMResponse(
                answer=raw_output,
                raw_output=raw_output,
                input_tokens=len(request.prompt.split()),
                output_tokens=len(raw_output.split()),
            )
        return super().generate(request)


def test_phase5_prompt_builders_use_required_trace_inputs() -> None:
    reflection_prompt = build_self_reflection_prompt(
        question="Is there opacity?",
        vision_findings="Observation: opacity.",
        retrieved_evidence="Evidence: opacity.",
        reasoning_trace="Reasoning: yes.",
        candidate_answer="yes",
    )
    verifier_prompt = build_separate_verifier_prompt(
        question="Is there opacity?",
        vision_findings="Observation: opacity.",
        retrieved_evidence="Evidence: opacity.",
        reasoning_trace="Reasoning: yes.",
        candidate_answer="yes",
    )

    assert "Original Reasoning Steps" in reflection_prompt
    assert "Candidate Answer" in reflection_prompt
    assert "Candidate Confidence" in reflection_prompt
    assert "independent process-level medical verifier" in verifier_prompt
    assert "must not generate a new answer" in verifier_prompt
    assert "Observation Grounding" in verifier_prompt
    assert "Evidence Grounding" in verifier_prompt
    assert "Unsupported Claim" in verifier_prompt
    assert "Confidence Alignment" in verifier_prompt


def test_parse_process_feedback_json() -> None:
    feedback = parse_process_feedback(
        '{"verdict": "REVISE", '
        '"step_results": [{"step_id": "reason_2", "status": "UNSUPPORTED", '
        '"error_type": "LOGICAL_LEAP", "confidence": 0.87}], '
        '"grounding_score": 0.82, "logical_consistency": 0.75, '
        '"confidence_alignment": 0.61, "error_types": ["LOGICAL_LEAP"], '
        '"recommended_action": "REVISE", "revision_instruction": "shorten"}',
        keep_decision="PASS",
    )

    assert feedback["decision"] == "REVISE"
    assert feedback["verdict"] == "REVISE"
    assert feedback["recommended_action"] == "REVISE"
    assert feedback["step_results"][0]["error_type"] == "LOGICAL_ERROR"
    assert feedback["grounding_score"] == 0.82
    assert feedback["error_types"] == ["LOGICAL_ERROR"]
    assert feedback["revision_instruction"] == "shorten"


def test_self_reflection_keeps_candidate_when_decision_keep() -> None:
    result = Phase5VerificationAgent(
        backend=MockVLMBackend(),
        verification_mode="self_reflection",
    ).run(_sample(), max_new_tokens=64)

    assert result.prediction == result.candidate.prediction
    assert "verifier_agent" not in result.candidate.agent_route
    assert result.process_feedback["decision"] == "KEEP"
    assert result.revision_applied is False
    assert result.selective_verification_applied is True


def test_separate_verifier_revision_is_generated_by_reasoner() -> None:
    result = Phase5VerificationAgent(
        backend=RevisingVerifierBackend(),
        verification_mode="separate_verifier",
    ).run(_sample(answer="no"), max_new_tokens=64)

    assert result.candidate.prediction == "yes"
    assert result.process_feedback["decision"] == "REVISE"
    assert result.process_feedback["verdict"] == "REVISE"
    assert result.process_feedback["error_types"] == ["UNSUPPORTED_CLAIM"]
    assert result.revision_applied is True
    assert result.prediction == "no"
    assert "verifier_guided_revision" in [
        call["stage"] for call in result.tool_calls
    ]


def test_selective_risk_rule_can_skip_low_risk_candidate() -> None:
    result = Phase5VerificationAgent(
        backend=MockVLMBackend(),
        verification_mode="separate_verifier",
        selective_verification="risk_rule",
    ).run(_sample(), max_new_tokens=64)

    assert result.selective_verification_applied is False
    assert result.process_feedback["decision"] == "SKIP"
    assert result.revision_applied is False


def test_phase5_metrics_count_corrections_and_regressions() -> None:
    metrics = summarize_phase5_metrics(
        [
            {
                "candidate_correct": False,
                "correct": True,
                "selective_verification_applied": True,
                "revision_applied": True,
                "candidate_hallucination": True,
                "final_hallucination": False,
                "process_verdict": "REVISE",
                "process_recommended_action": "REVISE",
                "process_error_types": ["UNSUPPORTED_CLAIM"],
                "process_grounding_score": 0.5,
                "process_logical_consistency": 0.4,
                "process_confidence_alignment": 0.3,
            },
            {
                "candidate_correct": True,
                "correct": False,
                "selective_verification_applied": True,
                "revision_applied": True,
                "candidate_hallucination": False,
                "final_hallucination": True,
                "process_verdict": "REVISE",
                "process_recommended_action": "REVISE",
                "process_error_types": ["LOGICAL_ERROR"],
                "process_grounding_score": 0.7,
                "process_logical_consistency": 0.6,
                "process_confidence_alignment": 0.5,
            },
        ]
    )

    assert metrics["candidate_accuracy"] == 0.5
    assert metrics["final_accuracy"] == 0.5
    assert metrics["error_correction_rate"] == 1.0
    assert metrics["num_error_regressions"] == 1
    assert metrics["selective_verification_rate"] == 1.0
    assert metrics["correct_answer_preservation_rate"] == 0.0
    assert metrics["net_correction"] == 0
    assert metrics["paired_transition_counts"]["wrong_to_correct"] == 1
    assert metrics["process_error_type_counts"]["UNSUPPORTED_CLAIM"] == 1
    assert metrics["process_error_recovery"]["UNSUPPORTED_CLAIM"]["corrected"] == 1
    assert metrics["mean_process_grounding_score"] == 0.6
    assert metrics["verifier_detection"] == {
        "true_positive": 1,
        "false_positive": 1,
        "false_negative": 0,
        "true_negative": 0,
        "precision": 0.5,
        "recall": 1.0,
        "f1": 2 / 3,
    }
    assert metrics["verifier_precision"] == 0.5
    assert metrics["verifier_recall"] == 1.0
    assert metrics["verifier_f1"] == 2 / 3
