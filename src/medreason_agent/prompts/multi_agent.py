"""Phase 4 Supervisor / Multi-Agent prompt 模板。

所有 agent 仍然调用同一个 frozen VLM，但每个 agent 使用不同 role prompt。
这样实验变量是 agent collaboration，而不是 backbone 或训练。
"""

from __future__ import annotations

import re

VISION_AGENT_V1 = """Act as a medical image specialist.

Question: {question}

Use exactly this format:

Observation: describe only visible image findings relevant to the question.
Claim Statuses: write JSON objects, one per line. Use OBSERVED only for visible image findings.
Uncertainty: state visual uncertainty if the image is insufficient.

Do not provide a final diagnosis. Do not provide clinical advice."""


REASONING_AGENT_V1 = """Act as a clinical reasoning expert.

Question: {question}

Shared State:
{shared_state_context}

Vision Observation:
{vision_output}

Retrieved Evidence:
{evidence_block}

Use exactly this format:

Reasoning: connect visual observations, question, and evidence.
Claims: list one to three checkable claims, one claim per line.
Claim Statuses: write JSON objects, one per line. Use OBSERVED for visible
findings, SUPPORTED for evidence-backed facts, HYPOTHESIS for possible
conclusions, UNSUPPORTED for unsupported claims, and CONTRADICTED for
evidence-conflicting claims. Do not treat HYPOTHESIS as fact.
Preliminary Answer: provide a short answer candidate.
Unsupported Assumptions: list unsupported assumptions, or write None.

Do not provide clinical advice."""


CRITIC_AGENT_V1 = """Check logical errors in the reasoning.

Question: {question}

Shared State:
{shared_state_context}

Vision Observation:
{vision_output}

Reasoning Output:
{reasoning_output}

Use exactly this format:

Critique: identify logical errors, unsupported assumptions, or missing evidence.
Critic Decision: ACCEPT / REVISE / REJECT.
Claim Statuses: write JSON objects, one per line for claims you checked. Use
SUPPORTED, UNSUPPORTED, or CONTRADICTED when evidence is available; otherwise
use HYPOTHESIS.
Unsupported Medical Claims: list unsupported medical claims, or write None.

Do not provide clinical advice."""


ANSWER_AGENT_V1 = """Act as the final answer agent for a benchmark Medical VQA task.

Question: {question}

Shared State:
{shared_state_context}

Vision Observation:
{vision_output}

Reasoning Output:
{reasoning_output}

Critic Output:
{critic_output}

Use exactly this format:

Conclusion: provide the concise benchmark conclusion.
Final Answer: provide only the short benchmark answer.

Do not provide clinical advice. Do not claim a real clinical diagnosis."""


SUPERVISOR_AGENT_V1 = """Act as a supervisor for a multimodal medical reasoning system.

Question: {question}

Available tools:
- Vision Agent
- Reasoning Agent
- Retrieval Agent
- Verifier Agent
- Answer Agent

Select the tools needed for a reliable benchmark answer.

Use exactly this format:

Selected Tools: comma-separated tool names.
Rationale: one concise sentence.

Do not provide clinical advice."""


VERIFIER_AGENT_V1 = """Verify the reasoning claims against retrieved evidence.

Question: {question}

Shared State:
{shared_state_context}

Claims:
{claims_block}

Retrieved Evidence:
{evidence_block}

Use exactly this format:

Verification: compare each claim against the retrieved evidence.
Verification Status: choose exactly one of SUPPORTED, UNSUPPORTED, or CONTRADICTED.
Claim Statuses: write JSON objects, one per line. Each input claim must receive
SUPPORTED, UNSUPPORTED, or CONTRADICTED. Do not mark a hypothesis as fact unless
retrieved evidence supports it.
Unsupported Medical Claims: list unsupported claims, or write None.

Do not provide clinical advice."""


_FINAL_ANSWER_PATTERN = re.compile(
    r"final\s+answer\s*:\s*(?P<answer>.+)",
    flags=re.IGNORECASE | re.DOTALL,
)
_PRELIMINARY_ANSWER_PATTERN = re.compile(
    r"preliminary\s+answer\s*:\s*(?P<answer>.+)",
    flags=re.IGNORECASE | re.DOTALL,
)
_CRITIC_DECISION_PATTERN = re.compile(
    r"critic\s+decision\s*:\s*(?P<decision>ACCEPT|REVISE|REJECT)",
    flags=re.IGNORECASE,
)
_SELECTED_TOOLS_PATTERN = re.compile(
    r"selected\s+tools\s*:\s*(?P<tools>.+)",
    flags=re.IGNORECASE,
)
_CLAIMS_PATTERN = re.compile(
    r"claims?\s*:\s*(?P<claims>.*?)(?:\n[A-Z][A-Za-z ]{1,40}:|\Z)",
    flags=re.IGNORECASE | re.DOTALL,
)
_VERIFICATION_STATUS_PATTERN = re.compile(
    r"verification\s+status\s*:\s*(?P<status>SUPPORTED|UNSUPPORTED|CONTRADICTED)",
    flags=re.IGNORECASE,
)


def build_vision_prompt(question: str) -> str:
    """构造 Vision Agent prompt。"""
    return VISION_AGENT_V1.format(question=question)


def build_reasoning_prompt(
    question: str,
    vision_output: str = "",
    evidence_block: str = "No retrieved evidence.",
    shared_state_context: str = "No shared state yet.",
) -> str:
    """构造 Reasoning Agent prompt。"""
    return REASONING_AGENT_V1.format(
        question=question,
        shared_state_context=shared_state_context,
        vision_output=vision_output,
        evidence_block=evidence_block,
    )


def build_critic_prompt(
    question: str,
    vision_output: str = "",
    reasoning_output: str = "",
    shared_state_context: str = "No shared state yet.",
) -> str:
    """构造 Critic Agent prompt。"""
    return CRITIC_AGENT_V1.format(
        question=question,
        shared_state_context=shared_state_context,
        vision_output=vision_output,
        reasoning_output=reasoning_output,
    )


def build_answer_prompt(
    question: str,
    vision_output: str = "",
    reasoning_output: str = "",
    critic_output: str = "",
    shared_state_context: str = "No shared state yet.",
) -> str:
    """构造 Answer Agent prompt。"""
    return ANSWER_AGENT_V1.format(
        question=question,
        shared_state_context=shared_state_context,
        vision_output=vision_output,
        reasoning_output=reasoning_output,
        critic_output=critic_output,
    )


def build_supervisor_prompt(question: str) -> str:
    """构造 Supervisor prompt。"""
    return SUPERVISOR_AGENT_V1.format(question=question)


def build_verifier_prompt(
    question: str,
    claims: list[str],
    evidence_block: str,
    shared_state_context: str = "No shared state yet.",
) -> str:
    """构造 Verifier Agent prompt。"""
    claims_block = "\n".join(f"- {claim}" for claim in claims) if claims else "None"
    return VERIFIER_AGENT_V1.format(
        question=question,
        shared_state_context=shared_state_context,
        claims_block=claims_block,
        evidence_block=evidence_block,
    )


def extract_final_answer(output: str) -> str:
    """从 Answer Agent 或 Reasoning Agent 输出中抽取短答案。"""
    stripped = output.strip()
    match = _FINAL_ANSWER_PATTERN.search(stripped) or _PRELIMINARY_ANSWER_PATTERN.search(stripped)
    if not match:
        return stripped
    return match.group("answer").strip().splitlines()[0].strip()


def extract_critic_decision(output: str) -> str:
    """抽取 Critic Agent 决策。"""
    match = _CRITIC_DECISION_PATTERN.search(output)
    if not match:
        return ""
    return match.group("decision").upper()


def extract_claims(output: str) -> list[str]:
    """从 Reasoning Agent 输出中抽取 claims。"""
    match = _CLAIMS_PATTERN.search(output.strip())
    if not match:
        return []

    claims: list[str] = []
    for line in match.group("claims").splitlines():
        claim = line.strip().lstrip("-*0123456789. )").strip()
        if claim:
            claims.append(claim)
    return claims


def extract_selected_tools(output: str) -> list[str]:
    """从 Supervisor 输出中抽取工具选择结果。"""
    match = _SELECTED_TOOLS_PATTERN.search(output)
    if not match:
        return []
    raw_tools = match.group("tools").splitlines()[0]
    return [tool.strip() for tool in raw_tools.split(",") if tool.strip()]


def extract_verification_status(output: str) -> str:
    """抽取 Verifier 的三分类状态。"""
    match = _VERIFICATION_STATUS_PATTERN.search(output)
    if not match:
        return "UNSUPPORTED"
    return match.group("status").upper()
