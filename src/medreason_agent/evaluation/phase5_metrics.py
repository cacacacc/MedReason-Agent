"""Phase 5 Verification / Self-Reflection 指标。

Phase 5 的关键不是只看最终 accuracy，而是看 candidate 错误经过 reflection /
verifier 后有多少被修正、原本正确答案有多少被误改，以及 Verifier 检测到了哪些
process-level error type。
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from medreason_agent.prompts.phase5_verification import PROCESS_ERROR_TYPES


def summarize_phase5_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总 Phase 5 的 process-level verification 指标。"""
    total = len(records)
    if total == 0:
        return _empty_metrics()

    candidate_correct = [bool(record.get("candidate_correct")) for record in records]
    final_correct = [bool(record.get("correct")) for record in records]
    initial_errors = [
        record for record in records if not bool(record.get("candidate_correct"))
    ]
    initial_correct = [
        record for record in records if bool(record.get("candidate_correct"))
    ]
    corrections = [
        record
        for record in records
        if not bool(record.get("candidate_correct")) and bool(record.get("correct"))
    ]
    regressions = [
        record
        for record in records
        if bool(record.get("candidate_correct")) and not bool(record.get("correct"))
    ]
    preserved = [
        record
        for record in records
        if bool(record.get("candidate_correct")) and bool(record.get("correct"))
    ]
    wrong_to_wrong = [
        record
        for record in records
        if not bool(record.get("candidate_correct")) and not bool(record.get("correct"))
    ]
    verified = [
        record for record in records if bool(record.get("selective_verification_applied"))
    ]
    revised = [record for record in records if bool(record.get("revision_applied"))]
    candidate_hallucinations = [
        bool(record.get("candidate_hallucination")) for record in records
    ]
    final_hallucinations = [bool(record.get("final_hallucination")) for record in records]

    candidate_hallucination_rate = sum(candidate_hallucinations) / total
    final_hallucination_rate = sum(final_hallucinations) / total
    verifier_detection = _verifier_detection_metrics(records)
    return {
        "candidate_accuracy": sum(candidate_correct) / total,
        "final_accuracy": sum(final_correct) / total,
        "num_initial_errors": len(initial_errors),
        "num_initial_correct": len(initial_correct),
        "num_error_corrections": len(corrections),
        "error_correction_rate": (
            len(corrections) / len(initial_errors) if initial_errors else 0.0
        ),
        "num_error_regressions": len(regressions),
        "error_regression_rate": (
            len(regressions) / len(initial_correct) if initial_correct else 0.0
        ),
        "correct_answer_preservation_rate": (
            len(preserved) / len(initial_correct) if initial_correct else 0.0
        ),
        "net_correction": len(corrections) - len(regressions),
        "paired_transition_counts": {
            "wrong_to_correct": len(corrections),
            "correct_to_wrong": len(regressions),
            "wrong_to_wrong": len(wrong_to_wrong),
            "correct_to_correct": len(preserved),
        },
        "selective_verification_rate": len(verified) / total,
        "revision_rate": len(revised) / total,
        "candidate_hallucination_rate": candidate_hallucination_rate,
        "final_hallucination_rate": final_hallucination_rate,
        "hallucination_reduction": _relative_reduction(
            candidate_hallucination_rate,
            final_hallucination_rate,
        ),
        "candidate_visual_hallucination_rate": _subtype_rate(
            records,
            "PERCEPTION_ERROR",
            before_revision=True,
        ),
        "candidate_knowledge_hallucination_rate": _subtype_rate(
            records,
            "RETRIEVAL_ERROR",
            before_revision=True,
        ),
        "candidate_unsupported_inference_rate": _unsupported_inference_rate(records),
        "final_unsupported_claim_rate": sum(final_hallucinations) / total,
        "process_error_type_counts": _process_error_type_counts(records),
        "process_error_recovery": _process_error_recovery(records),
        "verifier_detection": verifier_detection,
        "verifier_precision": verifier_detection["precision"],
        "verifier_recall": verifier_detection["recall"],
        "verifier_f1": verifier_detection["f1"],
        "process_verdict_counts": _field_counts(records, "process_verdict"),
        "process_recommended_action_counts": _field_counts(
            records,
            "process_recommended_action",
        ),
        "mean_process_grounding_score": _mean_optional(
            record.get("process_grounding_score") for record in records
        ),
        "mean_process_logical_consistency": _mean_optional(
            record.get("process_logical_consistency") for record in records
        ),
        "mean_process_confidence_alignment": _mean_optional(
            record.get("process_confidence_alignment") for record in records
        ),
        "mean_total_tokens": _sum_optional_means(
            _mean_optional(record.get("input_tokens") for record in records),
            _mean_optional(record.get("output_tokens") for record in records),
        ),
        "mean_latency_ms": _mean_optional(record.get("latency_ms") for record in records),
        "mean_agent_calls": _mean_optional(
            len(record.get("tool_calls", [])) for record in records
        ),
        "verifier_call_count": _verifier_call_count(records),
        "process_verifier_call_count": _stage_call_count(
            records,
            "separate_process_verifier",
        ),
        "outcome_verifier_call_count": _stage_call_count(records, "outcome_verifier"),
    }


def _empty_metrics() -> dict[str, Any]:
    """空结果时返回完整字段，保证 metrics schema 稳定。"""
    return {
        "candidate_accuracy": 0.0,
        "final_accuracy": 0.0,
        "num_initial_errors": 0,
        "num_initial_correct": 0,
        "num_error_corrections": 0,
        "error_correction_rate": 0.0,
        "num_error_regressions": 0,
        "error_regression_rate": 0.0,
        "correct_answer_preservation_rate": 0.0,
        "net_correction": 0,
        "paired_transition_counts": {
            "wrong_to_correct": 0,
            "correct_to_wrong": 0,
            "wrong_to_wrong": 0,
            "correct_to_correct": 0,
        },
        "selective_verification_rate": 0.0,
        "revision_rate": 0.0,
        "candidate_hallucination_rate": 0.0,
        "final_hallucination_rate": 0.0,
        "hallucination_reduction": 0.0,
        "candidate_visual_hallucination_rate": 0.0,
        "candidate_knowledge_hallucination_rate": 0.0,
        "candidate_unsupported_inference_rate": 0.0,
        "final_unsupported_claim_rate": 0.0,
        "process_error_type_counts": {error_type: 0 for error_type in PROCESS_ERROR_TYPES},
        "process_error_recovery": {
            error_type: _empty_recovery_counts() for error_type in PROCESS_ERROR_TYPES
        },
        "verifier_detection": {
            "true_positive": 0,
            "false_positive": 0,
            "false_negative": 0,
            "true_negative": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        },
        "verifier_precision": 0.0,
        "verifier_recall": 0.0,
        "verifier_f1": 0.0,
        "process_verdict_counts": {},
        "process_recommended_action_counts": {},
        "mean_process_grounding_score": None,
        "mean_process_logical_consistency": None,
        "mean_process_confidence_alignment": None,
        "mean_total_tokens": None,
        "mean_latency_ms": None,
        "mean_agent_calls": None,
        "verifier_call_count": 0,
        "process_verifier_call_count": 0,
        "outcome_verifier_call_count": 0,
    }


def _process_error_type_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    """统计 Verifier 检测到的 process-level error types。"""
    counts: Counter[str] = Counter()
    for record in records:
        for error_type in record.get("process_error_types", []):
            counts[str(error_type)] += 1
    return {error_type: counts.get(error_type, 0) for error_type in PROCESS_ERROR_TYPES}


def _process_error_recovery(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """按 Verifier error type 统计 textual revision 能否修复。"""
    recovery = {error_type: _empty_recovery_counts() for error_type in PROCESS_ERROR_TYPES}
    for record in records:
        error_types = [
            str(error_type)
            for error_type in record.get("process_error_types", [])
            if str(error_type) in PROCESS_ERROR_TYPES
        ]
        for error_type in error_types:
            recovery[error_type]["detected"] += 1
            if not record.get("candidate_correct") and record.get("correct"):
                recovery[error_type]["corrected"] += 1
            elif not record.get("candidate_correct") and not record.get("correct"):
                recovery[error_type]["uncorrected"] += 1
            elif record.get("candidate_correct") and not record.get("correct"):
                recovery[error_type]["regressed"] += 1

    for counts in recovery.values():
        detected = counts["detected"]
        counts["recovery_rate"] = counts["corrected"] / detected if detected else 0.0
    return recovery


def _verifier_detection_metrics(records: list[dict[str, Any]]) -> dict[str, int | float]:
    """用 candidate correctness 弱监督评估 verifier 是否定位到错误。

    Phase 7 的核心是 process-level error detection。没有人工 step label 时，先把
    candidate_correct=False 视为样本级 process error proxy；Verifier/Reflection 只有输出
    REVISE 或明确 error_types 时才记为 detected。单独的 step status 不计入，因为有些模型会
    在总 verdict=PASS 时仍输出 UNCERTAIN/UNSUPPORTED step，容易造成全量误检。
    """
    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0
    for record in records:
        has_error = not bool(record.get("candidate_correct"))
        detected = _process_error_detected(record)
        if detected and has_error:
            true_positive += 1
        elif detected and not has_error:
            false_positive += 1
        elif not detected and has_error:
            false_negative += 1
        else:
            true_negative += 1

    precision = _safe_divide(true_positive, true_positive + false_positive)
    recall = _safe_divide(true_positive, true_positive + false_negative)
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "precision": precision,
        "recall": recall,
        "f1": _safe_divide(2 * precision * recall, precision + recall),
    }


def _process_error_detected(record: dict[str, Any]) -> bool:
    """判断 verifier/reflection 是否明确检测到过程错误。"""
    decision = str(record.get("process_recommended_action") or "").upper()
    verdict = str(record.get("process_verdict") or "").upper()
    critic_decision = str(record.get("critic_decision") or "").upper()
    if "REVISE" in {decision, verdict, critic_decision}:
        return True
    return bool(record.get("process_error_types"))


def _empty_recovery_counts() -> dict[str, int | float]:
    """构造单个 error type 的 recovery 计数字段。"""
    return {
        "detected": 0,
        "corrected": 0,
        "uncorrected": 0,
        "regressed": 0,
        "recovery_rate": 0.0,
    }


def _field_counts(records: list[dict[str, Any]], field: str) -> dict[str, int]:
    """统计字符串字段分布，空值不计入。"""
    counts = Counter(str(record.get(field, "")).strip() for record in records)
    counts.pop("", None)
    return dict(sorted(counts.items()))


def _verifier_call_count(records: list[dict[str, Any]]) -> int:
    """统计所有 Phase 5 verifier 调用次数。"""
    return _stage_call_count(records, "separate_process_verifier") + _stage_call_count(
        records,
        "outcome_verifier",
    )


def _stage_call_count(records: list[dict[str, Any]], stage: str) -> int:
    """统计某个 VLM stage 的调用次数。"""
    return sum(
        1
        for record in records
        for call in record.get("tool_calls", [])
        if isinstance(call, dict) and call.get("stage") == stage
    )


def _mean_optional(values: Any) -> float | None:
    """计算可选数值均值。"""
    numeric_values = []
    for value in values:
        if value is None:
            continue
        try:
            numeric_values.append(float(value))
        except (TypeError, ValueError):
            continue
    if not numeric_values:
        return None
    return sum(numeric_values) / len(numeric_values)


def _sum_optional_means(*values: float | None) -> float | None:
    """汇总可选均值；任一部分缺失时保持 None。"""
    if any(value is None for value in values):
        return None
    return sum(float(value) for value in values)


def _relative_reduction(before: float, after: float) -> float:
    """计算相对下降比例；before 为 0 时返回 0。"""
    if before <= 0:
        return 0.0
    return (before - after) / before


def _safe_divide(numerator: float, denominator: float) -> float:
    """安全除法，分母为 0 时返回 0。"""
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _subtype_rate(
    records: list[dict[str, Any]],
    process_error_type: str,
    before_revision: bool,
) -> float:
    """用 process error type 启发式估计 hallucination 子类型占比。"""
    if not records:
        return 0.0
    flag_name = "candidate_hallucination" if before_revision else "final_hallucination"
    count = sum(
        1
        for record in records
        if bool(record.get(flag_name))
        and process_error_type in record.get("process_error_types", [])
    )
    return count / len(records)


def _unsupported_inference_rate(records: list[dict[str, Any]]) -> float:
    """估计 unsupported inference 占比。"""
    if not records:
        return 0.0
    unsupported_types = {
        "LOGICAL_ERROR",
        "UNSUPPORTED_CLAIM",
        "CONTRADICTION",
        "OVERCONFIDENCE",
    }
    count = sum(
        1
        for record in records
        if bool(record.get("candidate_hallucination"))
        and unsupported_types.intersection(set(record.get("process_error_types", [])))
    )
    return count / len(records)
