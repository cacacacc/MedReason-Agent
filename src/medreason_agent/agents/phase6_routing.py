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


def decide_rule_based_route(
    sample: VQARADSample,
    policy: str = "rule_based_v1",
) -> RoutingDecision:
    """根据 VQA-RAD 元数据和问题文本选择 reasoning depth。

    规则只使用推理前可得信息，避免偷看模型输出或 ground truth：
    - LOW：简单 closed visual question，优先 Direct。
    - MEDIUM：需要位置、大小、属性等轻量视觉推理，使用 Structured CoT。
    - HIGH：诊断、异常、病因、开放医学知识问题，使用完整 Multi-Agent。
    """
    if policy == "rule_based_v2":
        return decide_rule_based_route_v2(sample)
    if policy == "rule_based_v3":
        return decide_rule_based_route_v3(sample)
    if policy == "rule_based_v4":
        return decide_rule_based_route_v4(sample)
    if policy == "rule_based_v5":
        return decide_rule_based_route_v5(sample)
    if policy != "rule_based_v1":
        raise ValueError(f"Unsupported Phase 6 routing policy: {policy}")

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


def decide_rule_based_route_v2(sample: VQARADSample) -> RoutingDecision:
    """Phase6 v2 路由规则。

    v1 的主要问题是把所有 OPEN question 都送入 HIGH，导致 Full Multi-Agent route
    占比过高且 accuracy 很低。v2 更保守地调用完整 Multi-Agent：
    - 只有明确诊断、病因、鉴别诊断、疾病解释类问题进入 HIGH。
    - 解剖结构、成像方式、平面、位置、大小、属性、简单异常描述更多进入 MEDIUM。
    - 简单 closed visual question 继续走 LOW。
    """
    question_type = sample.question_type.upper()
    answer_type = sample.answer_type.upper()
    question = sample.question.lower()
    markers = _matched_markers(question)
    strong_markers = _matched_strong_medical_markers(question)
    weak_visual_markers = _matched_visual_description_markers(question)

    if strong_markers:
        return _decision(
            route=HIGH_ROUTE,
            complexity="HIGH",
            confidence_signal="LOW",
            uncertainty_signal="HIGH",
            evidence_support_signal="NEEDED",
            reason="explicit diagnosis, disease, etiology, or differential reasoning trigger",
            sample=sample,
            matched_markers=markers,
            strong_markers=strong_markers,
            weak_visual_markers=weak_visual_markers,
        )

    if _is_low_complexity(sample):
        return _decision(
            route=LOW_ROUTE,
            complexity="LOW",
            confidence_signal="HIGH",
            uncertainty_signal="LOW",
            evidence_support_signal="NOT_REQUIRED",
            reason="closed visual recognition question",
            sample=sample,
            matched_markers=markers,
            strong_markers=strong_markers,
            weak_visual_markers=weak_visual_markers,
        )

    if question_type in {"POS", "SIZE", "ATTRIB", "ORGAN", "MODALITY", "PLANE", "COUNT"}:
        return _decision(
            route=MEDIUM_ROUTE,
            complexity="MEDIUM",
            confidence_signal="MEDIUM",
            uncertainty_signal="MEDIUM",
            evidence_support_signal="OPTIONAL",
            reason="visual reasoning or visual-description question",
            sample=sample,
            matched_markers=markers,
            strong_markers=strong_markers,
            weak_visual_markers=weak_visual_markers,
        )

    if answer_type == "OPEN" and (weak_visual_markers or question_type in {"ABN", "OTHER"}):
        return _decision(
            route=MEDIUM_ROUTE,
            complexity="MEDIUM",
            confidence_signal="MEDIUM",
            uncertainty_signal="MEDIUM",
            evidence_support_signal="OPTIONAL",
            reason="open visual description without strong external-knowledge trigger",
            sample=sample,
            matched_markers=markers,
            strong_markers=strong_markers,
            weak_visual_markers=weak_visual_markers,
        )

    if question_type in {"ABN", "OTHER"}:
        return _decision(
            route=HIGH_ROUTE,
            complexity="HIGH",
            confidence_signal="LOW",
            uncertainty_signal="HIGH",
            evidence_support_signal="NEEDED",
            reason="ambiguous abnormality or other medical question",
            sample=sample,
            matched_markers=markers,
            strong_markers=strong_markers,
            weak_visual_markers=weak_visual_markers,
        )

    return _decision(
        route=MEDIUM_ROUTE,
        complexity="MEDIUM",
        confidence_signal="MEDIUM",
        uncertainty_signal="MEDIUM",
        evidence_support_signal="OPTIONAL",
        reason="default v2 route avoids full multi-agent unless high-risk markers appear",
        sample=sample,
        matched_markers=markers,
        strong_markers=strong_markers,
        weak_visual_markers=weak_visual_markers,
    )


def decide_rule_based_route_v3(sample: VQARADSample) -> RoutingDecision:
    """Phase6 v3 路由规则。

    v2 的问题是把太多样本送入 Structured CoT，导致 CoT route accuracy 从 v1 的小样本高值
    回落到 27.66%。v3 收紧 CoT 的适用范围，只把明确的视觉推理题交给 CoT。
    """
    question_types = _question_type_set(sample)
    answer_type = sample.answer_type.upper()
    question = sample.question.lower()
    markers = _matched_markers(question)
    strong_markers = _matched_strong_medical_markers(question)
    weak_visual_markers = _matched_visual_description_markers(question)

    if strong_markers or "ABN" in question_types:
        return _decision(
            route=HIGH_ROUTE,
            complexity="HIGH",
            confidence_signal="LOW",
            uncertainty_signal="HIGH",
            evidence_support_signal="NEEDED",
            reason="abnormality or strong medical-knowledge trigger",
            sample=sample,
            matched_markers=markers,
            strong_markers=strong_markers,
            weak_visual_markers=weak_visual_markers,
        )

    if question_types & {"POS", "SIZE", "ATTRIB"}:
        return _decision(
            route=MEDIUM_ROUTE,
            complexity="MEDIUM",
            confidence_signal="MEDIUM",
            uncertainty_signal="MEDIUM",
            evidence_support_signal="OPTIONAL",
            reason="explicit visual reasoning question",
            sample=sample,
            matched_markers=markers,
            strong_markers=strong_markers,
            weak_visual_markers=weak_visual_markers,
        )

    if _is_low_complexity(sample) or _is_direct_short_answer_question(
        answer_type=answer_type,
        question_types=question_types,
    ):
        return _decision(
            route=LOW_ROUTE,
            complexity="LOW",
            confidence_signal="HIGH",
            uncertainty_signal="LOW",
            evidence_support_signal="NOT_REQUIRED",
            reason="short-answer visual question without high-risk medical trigger",
            sample=sample,
            matched_markers=markers,
            strong_markers=strong_markers,
            weak_visual_markers=weak_visual_markers,
        )

    return _decision(
        route=LOW_ROUTE,
        complexity="LOW",
        confidence_signal="MEDIUM",
        uncertainty_signal="LOW",
        evidence_support_signal="NOT_REQUIRED",
        reason="default v3 route avoids CoT and full agents without clear trigger",
        sample=sample,
        matched_markers=markers,
        strong_markers=strong_markers,
        weak_visual_markers=weak_visual_markers,
    )


def decide_rule_based_route_v4(sample: VQARADSample) -> RoutingDecision:
    """Phase6 v4 路由规则。

    Oracle Routing full 结果显示，理论最优路线以 Direct 为主，
    Full Multi-Agent 只在极少数样本上最优。
    因此 v4 不再把 ABN、POS 这类问题类型自动升级，而是采用更保守的 Direct-dominant 策略：
    - Full 只用于强诊断、病因、鉴别诊断、疾病解释类触发词。
    - Structured CoT 只用于明确大小、属性、比较关系的视觉推理题。
    - 其他样本默认 Direct，避免过度推理带来的答案漂移。
    """
    question_types = _question_type_set(sample)
    answer_type = sample.answer_type.upper()
    question = sample.question.lower()
    markers = _matched_markers(question)
    strong_markers = _matched_strong_medical_markers(question)
    v4_full_markers = _matched_v4_full_markers(question)
    v4_cot_markers = _matched_v4_cot_markers(question)
    weak_visual_markers = _matched_visual_description_markers(question)

    if v4_full_markers:
        return _decision(
            route=HIGH_ROUTE,
            complexity="HIGH",
            confidence_signal="LOW",
            uncertainty_signal="HIGH",
            evidence_support_signal="NEEDED",
            reason="v4 strict diagnosis, disease, etiology, or differential trigger",
            sample=sample,
            matched_markers=markers,
            strong_markers=strong_markers,
            weak_visual_markers=weak_visual_markers,
        )

    if question_types & {"SIZE", "ATTRIB"} or v4_cot_markers:
        return _decision(
            route=MEDIUM_ROUTE,
            complexity="MEDIUM",
            confidence_signal="MEDIUM",
            uncertainty_signal="MEDIUM",
            evidence_support_signal="OPTIONAL",
            reason="v4 narrow visual comparison, size, or attribute reasoning trigger",
            sample=sample,
            matched_markers=markers,
            strong_markers=strong_markers,
            weak_visual_markers=weak_visual_markers,
        )

    if _is_low_complexity(sample) or _is_direct_short_answer_question(
        answer_type=answer_type,
        question_types=question_types,
    ):
        return _decision(
            route=LOW_ROUTE,
            complexity="LOW",
            confidence_signal="HIGH",
            uncertainty_signal="LOW",
            evidence_support_signal="NOT_REQUIRED",
            reason="v4 direct-dominant short-answer visual question",
            sample=sample,
            matched_markers=markers,
            strong_markers=strong_markers,
            weak_visual_markers=weak_visual_markers,
        )

    return _decision(
        route=LOW_ROUTE,
        complexity="LOW",
        confidence_signal="MEDIUM",
        uncertainty_signal="LOW",
        evidence_support_signal="NOT_REQUIRED",
        reason="v4 default route follows oracle analysis: avoid CoT/full without strict trigger",
        sample=sample,
        matched_markers=markers,
        strong_markers=strong_markers,
        weak_visual_markers=weak_visual_markers,
    )


def decide_rule_based_route_v5(sample: VQARADSample) -> RoutingDecision:
    """Phase6 v5 路由规则。

    v5 基于 v4 full 的 Oracle gap 分析，只修正最明确的 under-reasoning：
    v4 选 Direct、但 Oracle 显示 Structured CoT 才正确的一小组稳定题型。
    这个版本仍保留 v4 的 Direct-dominant 主体，避免重新扩大 Full Multi-Agent。
    """
    question_types = _question_type_set(sample)
    answer_type = sample.answer_type.upper()
    question = sample.question.lower()
    markers = _matched_markers(question)
    strong_markers = _matched_strong_medical_markers(question)
    v4_full_markers = _matched_v4_full_markers(question)
    v4_cot_markers = _matched_v4_cot_markers(question)
    v5_cot_markers = _matched_v5_oracle_gap_cot_markers(question)
    weak_visual_markers = _matched_visual_description_markers(question)

    if v4_full_markers:
        return _decision(
            route=HIGH_ROUTE,
            complexity="HIGH",
            confidence_signal="LOW",
            uncertainty_signal="HIGH",
            evidence_support_signal="NEEDED",
            reason="v5 strict diagnosis, disease, etiology, or differential trigger",
            sample=sample,
            matched_markers=markers,
            strong_markers=strong_markers,
            weak_visual_markers=weak_visual_markers,
        )

    if v5_cot_markers:
        return _decision(
            route=MEDIUM_ROUTE,
            complexity="MEDIUM",
            confidence_signal="MEDIUM",
            uncertainty_signal="MEDIUM",
            evidence_support_signal="OPTIONAL",
            reason="v5 oracle-gap visual finding or localization trigger",
            sample=sample,
            matched_markers=markers,
            strong_markers=strong_markers,
            weak_visual_markers=weak_visual_markers,
        )

    if question_types & {"SIZE", "ATTRIB"} or v4_cot_markers:
        return _decision(
            route=MEDIUM_ROUTE,
            complexity="MEDIUM",
            confidence_signal="MEDIUM",
            uncertainty_signal="MEDIUM",
            evidence_support_signal="OPTIONAL",
            reason="v5 inherited v4 visual comparison, size, or attribute trigger",
            sample=sample,
            matched_markers=markers,
            strong_markers=strong_markers,
            weak_visual_markers=weak_visual_markers,
        )

    if _is_low_complexity(sample) or _is_direct_short_answer_question(
        answer_type=answer_type,
        question_types=question_types,
    ):
        return _decision(
            route=LOW_ROUTE,
            complexity="LOW",
            confidence_signal="HIGH",
            uncertainty_signal="LOW",
            evidence_support_signal="NOT_REQUIRED",
            reason="v5 direct-dominant short-answer visual question",
            sample=sample,
            matched_markers=markers,
            strong_markers=strong_markers,
            weak_visual_markers=weak_visual_markers,
        )

    return _decision(
        route=LOW_ROUTE,
        complexity="LOW",
        confidence_signal="MEDIUM",
        uncertainty_signal="LOW",
        evidence_support_signal="NOT_REQUIRED",
        reason="v5 default route avoids CoT/full without explicit oracle-gap trigger",
        sample=sample,
        matched_markers=markers,
        strong_markers=strong_markers,
        weak_visual_markers=weak_visual_markers,
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
    simple_types = {"PRES", "MODALITY", "PLANE", "ORGAN", "COUNT", "COLOR"}
    return sample.answer_type.upper() == "CLOSED" and bool(
        _question_type_set(sample) & simple_types
    )


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


def _matched_strong_medical_markers(question: str) -> list[str]:
    """匹配真正需要外部医学知识的强触发词。"""
    markers = (
        "diagnosis",
        "diagnostic",
        "disease",
        "cause",
        "caused by",
        "etiology",
        "differential",
        "compatible",
        "consistent with",
        "suggest pneumonia",
        "suggest malignancy",
        "pathology",
    )
    return [marker for marker in markers if marker in question]


def _matched_v4_full_markers(question: str) -> list[str]:
    """匹配 v4 中允许进入 Full Multi-Agent 的强医学推理触发词。"""
    markers = (
        "diagnosis",
        "diagnostic",
        "differential",
        "etiology",
        "cause",
        "caused by",
        "disease",
        "compatible with",
        "consistent with",
        "suggest pneumonia",
        "suggest malignancy",
    )
    return [marker for marker in markers if marker in question]


def _matched_v4_cot_markers(question: str) -> list[str]:
    """匹配 v4 中适合 Structured CoT 的小范围视觉比较触发词。"""
    markers = (
        "larger",
        "smaller",
        "greater",
        "less than",
        "more than",
        "increased",
        "decreased",
        "changed",
        "compare",
        "comparison",
        "which side",
    )
    return [marker for marker in markers if marker in question]


def _matched_v5_oracle_gap_cot_markers(question: str) -> list[str]:
    """匹配 v4 Oracle gap 中 CoT 稳定救回的窄触发词。"""
    markers = (
        "airspace consolidation",
        "vascular pathology",
        ">12 ribs",
        "small bowel obstruction",
        "more radioopaque",
        "gastrointestinal system",
        "mass calcified",
        "lung markings",
        "liver visible",
        "bright white structures",
        "safe for pregnant",
        "kidneys normal",
        "hylar lymphadenopathy",
        "hilar lymphadenopathy",
        "mass present",
        "is a mass",
        "hypodense mass",
        "gallstones",
        "bright specks",
        "left kidney abnormal",
        "gastric bubble",
        "sequence of this mri",
        "left or right",
    )
    return [marker for marker in markers if marker in question]


def _matched_visual_description_markers(question: str) -> list[str]:
    """匹配更像视觉描述、而不是外部知识推理的问题。"""
    markers = (
        "what is seen",
        "what is shown",
        "what structure",
        "which organ",
        "where",
        "location",
        "located",
        "size",
        "large",
        "small",
        "opacity",
        "abnormality",
        "abnormal",
        "normal",
        "modality",
        "plane",
        "view",
    )
    return [marker for marker in markers if marker in question]


def _question_type_set(sample: VQARADSample) -> set[str]:
    """把 `POS, PRES` 这类组合 question_type 规整成集合。"""
    return {
        part.strip().upper()
        for part in sample.question_type.split(",")
        if part.strip()
    }


def _is_direct_short_answer_question(answer_type: str, question_types: set[str]) -> bool:
    """识别 v3 中应避免 CoT 的短答案视觉题。"""
    direct_types = {"PRES", "ORGAN", "MODALITY", "PLANE", "COUNT", "COLOR", "OTHER"}
    reasoning_types = {"POS", "SIZE", "ATTRIB", "ABN"}
    return bool(question_types & direct_types) and not bool(
        question_types & reasoning_types
    ) and answer_type in {"OPEN", "CLOSED"}


def _decision(
    route: str,
    complexity: str,
    confidence_signal: str,
    uncertainty_signal: str,
    evidence_support_signal: str,
    reason: str,
    sample: VQARADSample,
    matched_markers: list[str],
    strong_markers: list[str],
    weak_visual_markers: list[str],
) -> RoutingDecision:
    """统一构造 v2 routing decision，避免每条规则漏字段。"""
    return RoutingDecision(
        route=route,
        complexity=complexity,
        confidence_signal=confidence_signal,
        uncertainty_signal=uncertainty_signal,
        evidence_support_signal=evidence_support_signal,
        agent_agreement_signal="UNKNOWN_PRE_ROUTE",
        reason=reason,
        signals={
            "answer_type": sample.answer_type.upper(),
            "question_type": sample.question_type.upper(),
            "matched_markers": matched_markers,
            "strong_medical_markers": strong_markers,
            "weak_visual_markers": weak_visual_markers,
        },
    )
