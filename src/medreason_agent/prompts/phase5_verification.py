"""Phase 5 Verification / Self-Reflection prompt 模板。

Phase 5 的 Critic 只发生在 candidate reasoning + answer 之后。Self-Reflection
使用同一个 Reasoning Agent 检查自己的 trace；Separate Verifier 使用相同 backbone
但独立 role prompt，只负责 detect / localize / classify / recommend，不直接生成答案。
"""

from __future__ import annotations

import json
import re
from typing import Any

PROCESS_ERROR_TYPES = (
    "PERCEPTION_ERROR",
    "RETRIEVAL_ERROR",
    "LOGICAL_ERROR",
    "UNSUPPORTED_CLAIM",
    "CONTRADICTION",
    "OVERCONFIDENCE",
)

PROCESS_STEP_STATUSES = (
    "SUPPORTED",
    "UNSUPPORTED",
    "CONTRADICTED",
    "UNCERTAIN",
)

SELF_REFLECTION_PROMPT_V1 = """Self-Reflection for the same clinical reasoning agent.

You are the original Clinical Reasoning Agent. Check your own immediately
previous reasoning trace and candidate answer. Do not re-plan. Do not call
Vision, Retrieval, Supervisor, or any external tool.

Question:
{question}

Vision Findings:
{vision_findings}

Retrieved Evidence:
{retrieved_evidence}

Original Reasoning Steps:
{reasoning_trace}

Candidate Answer:
{candidate_answer}

Candidate Confidence:
{confidence}

Check exactly these process-level issues:
1. whether visual findings are used correctly
2. whether evidence supports the inference
3. whether the reasoning has a logical leap
4. whether there is a contradiction
5. whether there is an unsupported claim
6. whether confidence is reasonable

Return only JSON:
{{
  "decision": "KEEP or REVISE",
  "issues": [
    {{
      "type": "PROCESS_ERROR_TYPE",
      "step_id": "",
      "description": ""
    }}
  ],
  "reflection": "short process-level reflection",
  "revision_instruction": ""
}}"""


SELF_REVISION_PROMPT_V1 = """Act as the same Clinical Reasoning Agent.

Revise your previous clinical reasoning answer using only your self-reflection.
Do not call Vision, Retrieval, Supervisor, or any external tool.

Question:
{question}

Vision Findings:
{vision_findings}

Retrieved Evidence:
{retrieved_evidence}

Original Reasoning Steps:
{reasoning_trace}

Candidate Answer:
{candidate_answer}

Candidate Confidence:
{confidence}

Self-Reflection:
{feedback}

Use exactly this format:

Reasoning: corrected concise reasoning.
Claims: list one to three checkable claims, one claim per line.
Claim Statuses: write JSON objects, one per line.
Unsupported Assumptions: list unsupported assumptions, or write None.
Final Answer: provide only the short benchmark answer."""


SEPARATE_VERIFIER_PROMPT_V1 = """Act as an independent process-level medical verifier.

Use the same backbone but a completely independent verifier role and context.
You must not generate a new answer. Your job is only to detect, localize,
classify, and recommend correction for process-level errors in the candidate
reasoning and answer.

Question:
{question}

Vision Findings:
{vision_findings}

Retrieved Evidence:
{retrieved_evidence}

Candidate Reasoning:
{reasoning_trace}

Candidate Answer:
{candidate_answer}

Candidate Confidence:
{confidence}

Check the reasoning process with exactly these six layers:
A. Observation Grounding
- Check whether each observation claim in reasoning is truly grounded in Vision Findings.
- Mark contradictions with Vision Findings as CONTRADICTED.

B. Evidence Grounding
- If Retrieval was used, check whether reasoning claims are supported by retrieved evidence.
- Detect evidence misuse, overclaim, and knowledge conflict.

C. Logical Consistency
- Check whether Observation -> Evidence -> Inference -> Conclusion is valid.
- Detect logical leap, invalid inference, causal overclaim, and uncertainty amplification.

D. Unsupported Claim
- Check whether the reasoning introduces medical claims that are not grounded in
  Vision Findings or Retrieved Evidence.
- Mark unsupported assumptions as UNSUPPORTED.

E. Contradiction
- Check internal contradictions in the reasoning trace and contradictions against Vision / Evidence.

F. Confidence Alignment
- Check whether confidence matches visual confidence, evidence strength, reasoning uncertainty,
  and multiple plausible hypotheses.
- Detect OVERCONFIDENCE and UNDERCONFIDENCE.

Return at least one step_result for each checked layer. Use step_id prefixes:
obs_, evidence_, reason_, unsupported_, contradiction_, confidence_.

Decision rule:
- Use verdict=REVISE when any step has status UNSUPPORTED or CONTRADICTED.
- Use verdict=REVISE when any step has a non-null error_type.
- Use verdict=PASS only when all checked layers are supported or not applicable.
- recommended_action must be identical to verdict.

Allowed error_types:
PERCEPTION_ERROR, RETRIEVAL_ERROR, LOGICAL_ERROR, UNSUPPORTED_CLAIM,
CONTRADICTION, OVERCONFIDENCE

Return only JSON with exactly this schema:
{{
  "verdict": "PASS or REVISE",
  "step_results": [
    {{
      "step_id": "obs_1 or evidence_1 or reason_1 or conclusion_1 or confidence_1",
      "status": "SUPPORTED or UNSUPPORTED or CONTRADICTED or UNCERTAIN",
      "error_type": null,
      "confidence": 0.0
    }}
  ],
  "grounding_score": 0.0,
  "logical_consistency": 0.0,
  "confidence_alignment": 0.0,
  "error_types": [],
  "recommended_action": "PASS or REVISE",
  "revision_instruction": ""
}}"""


OUTCOME_VERIFIER_PROMPT_V1 = """Act as an independent outcome-level medical verifier.

Use the same backbone but a completely independent verifier role and context.
You must not generate a new answer. You only judge whether the candidate final
answer is plausible for the question. You cannot inspect Vision Findings,
Retrieved Evidence, or Reasoning Steps.

Question:
{question}

Candidate Answer:
{candidate_answer}

Candidate Confidence:
{confidence}

Return only JSON:
{{
  "verdict": "PASS or REVISE",
  "step_results": [
    {{
      "step_id": "answer_1",
      "status": "SUPPORTED or UNSUPPORTED or CONTRADICTED or UNCERTAIN",
      "error_type": null,
      "confidence": 0.0
    }}
  ],
  "grounding_score": null,
  "logical_consistency": null,
  "confidence_alignment": 0.0,
  "error_types": [],
  "recommended_action": "PASS or REVISE",
  "revision_instruction": ""
}}"""


VERIFIER_REVISION_PROMPT_V1 = """Act as the Clinical Reasoning Agent.

Revise the candidate answer using the independent verifier feedback. The
Verifier does not generate the answer; you generate exactly one revised final
answer. Do not call Vision, Retrieval, Supervisor, or any external tool.

Question:
{question}

Vision Findings:
{vision_findings}

Retrieved Evidence:
{retrieved_evidence}

Candidate Reasoning:
{reasoning_trace}

Candidate Answer:
{candidate_answer}

Candidate Confidence:
{confidence}

Verifier Feedback:
{feedback}

Use exactly this format:

Reasoning: corrected concise reasoning.
Claims: list one to three checkable claims, one claim per line.
Claim Statuses: write JSON objects, one per line.
Unsupported Assumptions: list unsupported assumptions, or write None.
Final Answer: provide only the short benchmark answer."""


_DECISION_PATTERN = re.compile(
    r'"?(?:decision|recommended_action|verdict)"?\s*:\s*"?(?P<decision>KEEP|PASS|REVISE)"?',
    flags=re.IGNORECASE,
)
_REVISION_INSTRUCTION_PATTERN = re.compile(
    r'"?revision_instruction"?\s*:\s*"(?P<instruction>.*?)"',
    flags=re.IGNORECASE | re.DOTALL,
)


def build_self_reflection_prompt(
    question: str,
    vision_findings: str,
    retrieved_evidence: str,
    reasoning_trace: str,
    candidate_answer: str,
    confidence: float | None = None,
) -> str:
    """构造严格 Self-Reflection prompt。"""
    return SELF_REFLECTION_PROMPT_V1.format(
        question=question,
        vision_findings=vision_findings or "None",
        retrieved_evidence=retrieved_evidence or "None",
        reasoning_trace=reasoning_trace or "None",
        candidate_answer=candidate_answer,
        confidence=_format_confidence(confidence),
    )


def build_self_revision_prompt(
    question: str,
    vision_findings: str,
    retrieved_evidence: str,
    reasoning_trace: str,
    candidate_answer: str,
    feedback: str,
    confidence: float | None = None,
) -> str:
    """构造同一个 Reasoning Agent 的 revision prompt。"""
    return SELF_REVISION_PROMPT_V1.format(
        question=question,
        vision_findings=vision_findings or "None",
        retrieved_evidence=retrieved_evidence or "None",
        reasoning_trace=reasoning_trace or "None",
        candidate_answer=candidate_answer,
        confidence=_format_confidence(confidence),
        feedback=feedback or "None",
    )


def build_separate_verifier_prompt(
    question: str,
    vision_findings: str,
    retrieved_evidence: str,
    reasoning_trace: str,
    candidate_answer: str,
    confidence: float | None = None,
) -> str:
    """构造独立 Verifier 的五层 process-level verification prompt。"""
    return SEPARATE_VERIFIER_PROMPT_V1.format(
        question=question,
        vision_findings=vision_findings or "None",
        retrieved_evidence=retrieved_evidence or "None",
        reasoning_trace=reasoning_trace or "None",
        candidate_answer=candidate_answer,
        confidence=_format_confidence(confidence),
    )


def build_outcome_verifier_prompt(
    question: str,
    candidate_answer: str,
    confidence: float | None = None,
) -> str:
    """构造只看最终答案的 outcome-level verifier prompt。"""
    return OUTCOME_VERIFIER_PROMPT_V1.format(
        question=question,
        candidate_answer=candidate_answer,
        confidence=_format_confidence(confidence),
    )


def build_verifier_revision_prompt(
    question: str,
    vision_findings: str,
    retrieved_evidence: str,
    reasoning_trace: str,
    candidate_answer: str,
    feedback: str,
    confidence: float | None = None,
) -> str:
    """构造 Reasoning Agent 根据外部 Verifier feedback 的 revision prompt。"""
    return VERIFIER_REVISION_PROMPT_V1.format(
        question=question,
        vision_findings=vision_findings or "None",
        retrieved_evidence=retrieved_evidence or "None",
        reasoning_trace=reasoning_trace or "None",
        candidate_answer=candidate_answer,
        confidence=_format_confidence(confidence),
        feedback=feedback or "None",
    )


def parse_process_feedback(output: str, keep_decision: str = "KEEP") -> dict[str, Any]:
    """解析 reflection / verifier JSON；失败时用保守 KEEP/PASS。

    返回值保留 `decision` 兼容旧 runner，同时把 Verifier 的 `verdict`、
    `recommended_action`、五层 step results 和分数标准化，方便 Phase 6 直接复用。
    """
    stripped = output.strip()
    parsed = _parse_json_object(stripped)
    if parsed:
        return _normalize_feedback(parsed, keep_decision)

    decision_match = _DECISION_PATTERN.search(stripped)
    instruction_match = _REVISION_INSTRUCTION_PATTERN.search(stripped)
    decision = decision_match.group("decision").upper() if decision_match else keep_decision
    return _normalize_feedback(
        {
            "decision": decision,
            "revision_instruction": (
                instruction_match.group("instruction").strip()
                if instruction_match
                else ""
            ),
        },
        keep_decision,
    )


def _normalize_feedback(parsed: dict[str, Any], keep_decision: str) -> dict[str, Any]:
    """把 Self-Reflection 和 Separate Verifier schema 规整到同一套字段。"""
    verdict = _normalize_verdict(str(parsed.get("verdict", "")))
    recommended_action = _normalize_action(str(parsed.get("recommended_action", "")))
    raw_decision = str(parsed.get("decision", "")).upper()
    if keep_decision == "PASS" and recommended_action:
        decision = recommended_action
    elif keep_decision == "PASS" and verdict:
        decision = verdict
    elif recommended_action:
        decision = recommended_action
    elif raw_decision:
        decision = _normalize_decision(raw_decision, keep_decision)
    elif verdict:
        decision = verdict
    else:
        decision = keep_decision

    issues = parsed.get("issues", [])
    step_results = _normalize_step_results(parsed.get("step_results", []))
    error_types = _normalize_error_types(parsed.get("error_types", []))
    if not error_types:
        error_types = _error_types_from_items([*issues, *step_results])

    return {
        "decision": _normalize_decision(decision, keep_decision),
        "verdict": verdict or ("PASS" if keep_decision == "PASS" else ""),
        "recommended_action": recommended_action or "",
        "issues": issues if isinstance(issues, list) else [],
        "step_results": step_results,
        "grounding_score": _optional_float(parsed.get("grounding_score")),
        "logical_consistency": _optional_float(parsed.get("logical_consistency")),
        "confidence_alignment": _optional_float(parsed.get("confidence_alignment")),
        "error_types": error_types,
        "revision_instruction": str(parsed.get("revision_instruction", "")),
        "raw": parsed,
    }


def _parse_json_object(text: str) -> dict[str, Any] | None:
    """从模型输出里解析第一个 JSON object。"""
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _normalize_decision(decision: str, keep_decision: str) -> str:
    """把 process-level decision 规整到 KEEP/PASS/REVISE。"""
    normalized = decision.strip().upper()
    if normalized in {"KEEP", "PASS", "REVISE"}:
        return normalized
    return keep_decision


def _normalize_verdict(verdict: str) -> str:
    """规整独立 Verifier 的 verdict。"""
    normalized = verdict.strip().upper()
    return normalized if normalized in {"PASS", "REVISE"} else ""


def _normalize_action(action: str) -> str:
    """规整独立 Verifier 的 recommended_action。"""
    normalized = action.strip().upper()
    return normalized if normalized in {"PASS", "REVISE"} else ""


def _normalize_step_results(items: Any) -> list[dict[str, Any]]:
    """标准化 Verifier 五层检查结果。"""
    if not isinstance(items, list):
        return []

    results = []
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        results.append(
            {
                "step_id": str(item.get("step_id") or f"step_{index}"),
                "status": _normalize_step_status(str(item.get("status", ""))),
                "error_type": _normalize_error_type(item.get("error_type")),
                "confidence": _optional_float(item.get("confidence")),
            }
        )
    return results


def _normalize_step_status(status: str) -> str:
    """规整 step-level grounding 状态。"""
    normalized = status.strip().upper()
    if normalized in PROCESS_STEP_STATUSES:
        return normalized
    return "UNCERTAIN"


def _normalize_error_types(items: Any) -> list[str]:
    """规整 Verifier 输出的 error_types 列表。"""
    if not isinstance(items, list):
        return []
    normalized = [_normalize_error_type(item) for item in items]
    return [item for item in dict.fromkeys(normalized) if item]


def _normalize_error_type(item: Any) -> str | None:
    """把常见错误类型别名归并到 Phase 5 枚举。"""
    if item is None:
        return None
    normalized = str(item).strip().upper().replace(" ", "_").replace("-", "_")
    aliases = {
        "PERCEPTION": "PERCEPTION_ERROR",
        "VISUAL_HALLUCINATION": "PERCEPTION_ERROR",
        "OBSERVATION_MISMATCH": "PERCEPTION_ERROR",
        "RETRIEVAL": "RETRIEVAL_ERROR",
        "EVIDENCE_UNSUPPORTED": "UNSUPPORTED_CLAIM",
        "EVIDENCE_MISUSE": "RETRIEVAL_ERROR",
        "OVERCLAIM": "UNSUPPORTED_CLAIM",
        "LOGICAL": "LOGICAL_ERROR",
        "LOGICAL_LEAP": "LOGICAL_ERROR",
        "INVALID_INFERENCE": "LOGICAL_ERROR",
        "CAUSAL_OVERCLAIM": "LOGICAL_ERROR",
        "UNCERTAINTY_AMPLIFICATION": "OVERCONFIDENCE",
        "UNSUPPORTED": "UNSUPPORTED_CLAIM",
        "CONTRADICTED": "CONTRADICTION",
    }
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in PROCESS_ERROR_TYPES else None


def _error_types_from_items(items: list[Any]) -> list[str]:
    """从 issues / step_results 里抽取错误类型。"""
    error_types = []
    for item in items:
        if not isinstance(item, dict):
            continue
        error_type = _normalize_error_type(item.get("type") or item.get("error_type"))
        if error_type:
            error_types.append(error_type)
    return [item for item in dict.fromkeys(error_types) if item]


def _optional_float(value: Any) -> float | None:
    """解析 0-1 分数；失败时返回 None。"""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_confidence(confidence: float | None) -> str:
    """格式化 candidate confidence。"""
    if confidence is None:
        return "Unknown"
    return f"{float(confidence):.4f}"
