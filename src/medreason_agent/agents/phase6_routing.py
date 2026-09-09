"""Phase 6 rule-based adaptive routing.

Phase 6 的目标不是再增加一个固定 agent 链路，而是让每个样本先被路由到合适的
reasoning depth：简单题直接回答，中等题使用结构化 CoT，高风险题才进入完整
Multi-Agent。这里的规则是可解释的第一版 supervisor，后续 LLM-based supervisor
应该复用同一套 `RoutingDecision` schema，保证实验可比。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from medreason_agent.data.vqa_rad import VQARADSample

LOW_ROUTE = "direct"
MEDIUM_ROUTE = "structured_cot"
HIGH_ROUTE = "full_multi_agent"


@dataclass(frozen=True)
class RoutingDecision:
    """单条样本的路由决策和可审计信号。"""

    route: str
    complexity: str
    confidence_signal: str
    uncertainty_signal: str
    evidence_support_signal: str
    agent_agreement_signal: str
    reason: str
    signals: dict[str, object]

    def to_record(self) -> dict[str, object]:
        """转换成可写入 JSONL 的普通 dict。"""
        return asdict(self)


def decide_rule_based_route(sample: VQARADSample) -> RoutingDecision:
    """根据 VQA-RAD 元数据和问题文本选择 reasoning depth。

    规则只使用推理前可得信息，避免偷看模型输出或 ground truth：
    - LOW：简单 closed visual question，优先 Direct。
    - MEDIUM：需要位置、大小、属性等轻量视觉推理，使用 Structured CoT。
    - HIGH：诊断、异常、病因、开放医学知识问题，使用完整 Multi-Agent。
    """
    question_type = sample.question_type.upper()
    answer_type = sample.answer_type.upper()
    question = sample.question.lower()
    markers = _matched_markers(question)

    if _is_high_complexity(sample, markers):
        return RoutingDecision(
            route=HIGH_ROUTE,
            complexity="HIGH",
            confidence_signal="LOW",
            uncertainty_signal="HIGH",
            evidence_support_signal="NEEDED",
            agent_agreement_signal="UNKNOWN_PRE_ROUTE",
            reason=(
                "question requires external medical knowledge or open-ended "
                "abnormality reasoning"
            ),
            signals={
                "answer_type": answer_type,
                "question_type": question_type,
                "matched_markers": markers,
            },
        )

    if _is_low_complexity(sample):
        return RoutingDecision(
            route=LOW_ROUTE,
            complexity="LOW",
            confidence_signal="HIGH",
            uncertainty_signal="LOW",
            evidence_support_signal="NOT_REQUIRED",
            agent_agreement_signal="UNKNOWN_PRE_ROUTE",
            reason="closed visual recognition question",
            signals={
                "answer_type": answer_type,
                "question_type": question_type,
                "matched_markers": markers,
            },
        )

    return RoutingDecision(
        route=MEDIUM_ROUTE,
        complexity="MEDIUM",
        confidence_signal="MEDIUM",
        uncertainty_signal="MEDIUM",
        evidence_support_signal="OPTIONAL",
        agent_agreement_signal="UNKNOWN_PRE_ROUTE",
        reason="visual reasoning question without explicit external-knowledge trigger",
        signals={
            "answer_type": answer_type,
            "question_type": question_type,
            "matched_markers": markers,
        },
    )


def fallback_route(route: str) -> str | None:
    """返回当前 route 的下一层 fallback，最多升一级。"""
    if route == LOW_ROUTE:
        return MEDIUM_ROUTE
    if route == MEDIUM_ROUTE:
        return HIGH_ROUTE
    return None


def should_fallback(prediction: str, raw_output: str, route: str) -> tuple[bool, str]:
    """判断是否需要升一级 reasoning depth。

    第一版只处理明显失败信号：空答案、unknown/uncertain、模型明确说无法判断。这样可以
    避免 fallback 把大量本来正确的短答案又送入更复杂链路。
    """
    if route == HIGH_ROUTE:
        return False, "already_highest_route"

    normalized_prediction = " ".join(prediction.lower().split())
    normalized_output = raw_output.lower()
    if not normalized_prediction:
        return True, "empty_prediction"
    if normalized_prediction in {"unknown", "uncertain", "cannot determine"}:
        return True, "explicit_uncertainty_prediction"
    uncertainty_markers = (
        "cannot determine",
        "cannot be determined",
        "not enough information",
        "insufficient information",
        "unable to determine",
    )
    if any(marker in normalized_output for marker in uncertainty_markers):
        return True, "explicit_uncertainty_output"
    return False, "no_fallback_signal"


def _is_low_complexity(sample: VQARADSample) -> bool:
    """识别适合 Direct 的简单闭合视觉题。"""
    return sample.answer_type.upper() == "CLOSED" and sample.question_type.upper() in {
        "PRES",
        "MODALITY",
        "PLANE",
        "ORGAN",
        "COUNT",
        "COLOR",
    }


def _is_high_complexity(sample: VQARADSample, markers: list[str]) -> bool:
    """识别需要外部知识或完整验证链路的高复杂度题。"""
    if sample.answer_type.upper() == "OPEN":
        return True
    if sample.question_type.upper() in {"ABN", "OTHER"}:
        return True
    return bool(markers)


def _matched_markers(question: str) -> list[str]:
    """匹配外部医学知识触发词。"""
    markers = (
        "diagnosis",
        "diagnostic",
        "disease",
        "cause",
        "represent",
        "compatible",
        "suggest",
        "consistent with",
        "abnormality",
        "abnormal",
        "pathology",
        "etiology",
        "differential",
    )
    return [marker for marker in markers if marker in question]
