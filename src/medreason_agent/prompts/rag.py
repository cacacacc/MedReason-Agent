"""RAG prompt 模板。

Phase 3 同时支持两类 RAG：

- Knowledge RAG：推理前检索，用于知识获取。
- Evidence RAG：推理后围绕初始 claim 检索，用于证据验证。
"""

import re

KNOWLEDGE_RAG_V3 = """Answer the medical question using image and retrieved medical evidence.

Question: {question}

Retrieved Evidence:
{evidence_block}

You must follow these rules:

1. Use retrieved evidence explicitly when it is relevant.
2. Do not introduce unsupported medical claims.
3. Separate image observation from external knowledge.
4. State uncertainty if the evidence is insufficient.
5. Label every claim with one status from OBSERVED, SUPPORTED, HYPOTHESIS,
   UNSUPPORTED, CONTRADICTED.

Use exactly the following format:

Observation: describe only visual findings that are visible in the image.
Retrieved Evidence: summarize only the retrieved evidence that is relevant to the question.
Reasoning: connect the visual findings, question, and retrieved evidence.
Claim Statuses: write JSON objects, one per line. Use OBSERVED for visible image
findings, SUPPORTED for evidence-backed facts, HYPOTHESIS for possible
conclusions, UNSUPPORTED for unsupported claims, and CONTRADICTED for
evidence-conflicting claims.
Unsupported Assumptions: list unsupported assumptions, or write None.
Uncertainty: state uncertainty if the image or evidence is insufficient.
Conclusion: provide the concise conclusion.
Final Answer: provide only the short benchmark answer.

Do not provide clinical advice. Do not claim a real clinical diagnosis."""


KNOWLEDGE_CLAIM_RAG_V1 = """Answer the medical question using image and retrieved knowledge.

Question: {question}

Retrieved Knowledge:
{evidence_block}

You must follow these rules:

1. Use retrieved knowledge explicitly when it is relevant.
2. Do not introduce unsupported medical claims.
3. Separate image observation from external knowledge.
4. State uncertainty if the knowledge is insufficient.
5. Generate concise claims that can be checked by a verifier.
6. Label every claim with one status from OBSERVED, SUPPORTED, HYPOTHESIS,
   UNSUPPORTED, CONTRADICTED.

Use exactly the following format:

Observation: describe only visual findings that are visible in the image.
Retrieved Knowledge: summarize only retrieved knowledge relevant to the question.
Reasoning: connect the visual findings, question, and retrieved knowledge.
Claims: list one to three checkable claims, one claim per line.
Claim Statuses: write JSON objects, one per line. Use OBSERVED for visible image
findings, SUPPORTED for retrieved-knowledge facts, HYPOTHESIS for possible
conclusions, UNSUPPORTED for unsupported claims, and CONTRADICTED for
evidence-conflicting claims.
Unsupported Assumptions: list unsupported assumptions, or write None.
Uncertainty: state uncertainty if the image or knowledge is insufficient.
Conclusion: provide the concise conclusion.
Final Answer: provide only the short benchmark answer.

Do not provide clinical advice. Do not claim a real clinical diagnosis."""


EVIDENCE_RAG_V1 = """Verify the initial medical claim using image and retrieved evidence.

Question: {question}

Initial Claim:
{initial_claim}

Initial Reasoning:
{initial_reasoning_output}

Retrieved Evidence:
{evidence_block}

You must follow these rules:

1. Use retrieved evidence explicitly when it is relevant.
2. Do not introduce unsupported medical claims.
3. Separate image observation from external knowledge.
4. State uncertainty if the evidence is insufficient.
5. Compare the initial claim against the retrieved evidence.
6. Label the verified claim with exactly one status: SUPPORTED, UNSUPPORTED, or CONTRADICTED.

Use exactly the following format:

Observation: describe only visual findings that are visible in the image.
Initial Claim: restate the claim being verified.
Retrieved Evidence: summarize only the retrieved evidence that is relevant to the claim.
Verification: state whether evidence supports, contradicts, or is insufficient for the claim.
Verification Status: choose exactly one of SUPPORTED, UNSUPPORTED, or CONTRADICTED.
Claim Statuses: write one JSON object for the initial claim, using the same
status as Verification Status.
Unsupported Assumptions: list unsupported assumptions, or write None.
Uncertainty: state uncertainty if the image or evidence is insufficient.
Conclusion: provide the concise verified conclusion.
Final Answer: provide only the short benchmark answer.

Do not provide clinical advice. Do not claim a real clinical diagnosis."""


_FINAL_ANSWER_PATTERN = re.compile(
    r"final\s+answer\s*:\s*(?P<answer>.+)",
    flags=re.IGNORECASE | re.DOTALL,
)
_CONCLUSION_PATTERN = re.compile(
    r"conclusion\s*:\s*(?P<answer>.+)",
    flags=re.IGNORECASE | re.DOTALL,
)
_CLAIMS_PATTERN = re.compile(
    r"claims?\s*:\s*(?P<claims>.*?)(?:\n[A-Z][A-Za-z ]{1,40}:|\Z)",
    flags=re.IGNORECASE | re.DOTALL,
)
_VERIFICATION_STATUS_PATTERN = re.compile(
    r"verification\s+status\s*:\s*(?P<status>SUPPORTED|UNSUPPORTED|CONTRADICTED)",
    flags=re.IGNORECASE,
)


def format_evidence_block(evidence_records: list[dict], max_chars_per_evidence: int = 700) -> str:
    """把检索到的证据格式化成 prompt 中的证据块。

    每条证据保留标题、分数和正文。限制字符数是为了避免 RAG prompt 过长，
    影响显存、速度和模型注意力。
    """
    if not evidence_records:
        return "No retrieved evidence."

    lines: list[str] = []
    for index, evidence in enumerate(evidence_records, start=1):
        title = str(evidence.get("title", "")).strip() or "Untitled"
        score = evidence.get("score", "")
        text = str(evidence.get("text", "")).strip()
        if len(text) > max_chars_per_evidence:
            text = text[:max_chars_per_evidence].rstrip() + "..."
        lines.append(f"[{index}] title={title}; score={score}\n{text}")
    return "\n\n".join(lines)


def build_knowledge_rag_prompt(
    question: str,
    evidence_records: list[dict],
    max_chars_per_evidence: int = 700,
) -> str:
    """为 Knowledge RAG 构造 Structured RAG Prompt。

    这个 prompt 明确分开图像观察、外部证据、推理、unsupported assumptions、
    uncertainty 和最终答案，便于后续做 hallucination 与 evidence grounding 分析。
    """
    evidence_block = format_evidence_block(
        evidence_records,
        max_chars_per_evidence=max_chars_per_evidence,
    )
    return KNOWLEDGE_RAG_V3.format(question=question, evidence_block=evidence_block)


def build_rag_prompt(
    question: str,
    evidence_records: list[dict],
    max_chars_per_evidence: int = 700,
) -> str:
    """兼容旧调用：默认构造 Knowledge RAG prompt。"""
    return build_knowledge_rag_prompt(
        question=question,
        evidence_records=evidence_records,
        max_chars_per_evidence=max_chars_per_evidence,
    )


def build_knowledge_claim_generation_prompt(
    question: str,
    evidence_records: list[dict],
    max_chars_per_evidence: int = 700,
) -> str:
    """为组合 RAG 构造“先知识增强，再生成可验证 claims”的 prompt。"""
    evidence_block = format_evidence_block(
        evidence_records,
        max_chars_per_evidence=max_chars_per_evidence,
    )
    return KNOWLEDGE_CLAIM_RAG_V1.format(question=question, evidence_block=evidence_block)


def build_claim_verification_query(
    question: str,
    claim: str,
    reasoning_output: str,
    max_reasoning_chars: int = 350,
) -> str:
    """为 Evidence RAG 构造检索 query。

    和 Knowledge RAG 不同，这里 query 的中心不是原始问题，而是模型已经提出的
    claim。这样检索结果服务于 claim verification，而不是回答前知识补全。
    """
    compact_reasoning = " ".join(reasoning_output.split())[:max_reasoning_chars]
    return (
        "medical evidence supporting or contradicting this claim. "
        f"Question: {question} Claim: {claim} Reasoning: {compact_reasoning}"
    )


def build_evidence_verification_prompt(
    question: str,
    initial_claim: str,
    initial_reasoning_output: str,
    evidence_records: list[dict],
    max_chars_per_evidence: int = 700,
) -> str:
    """为 Evidence RAG 构造 claim verification prompt。"""
    evidence_block = format_evidence_block(
        evidence_records,
        max_chars_per_evidence=max_chars_per_evidence,
    )
    return EVIDENCE_RAG_V1.format(
        question=question,
        initial_claim=initial_claim,
        initial_reasoning_output=initial_reasoning_output,
        evidence_block=evidence_block,
    )


def extract_claims(output: str) -> list[str]:
    """从模型输出中抽取 verifier 需要检查的 claims。

    这里先实现确定性的轻量解析：读取 `Claims:` 段落中的每一行。如果模型没有按格式
    输出 claims，调用方可以退回使用 `Final Answer:` 作为最小 claim。
    """
    match = _CLAIMS_PATTERN.search(output.strip())
    if not match:
        return []

    claims: list[str] = []
    for line in match.group("claims").splitlines():
        claim = line.strip().lstrip("-*0123456789. )").strip()
        if claim:
            claims.append(claim)
    return claims


def extract_verification_status(output: str) -> str:
    """从 verifier 输出中抽取 SUPPORTED / UNSUPPORTED / CONTRADICTED。"""
    match = _VERIFICATION_STATUS_PATTERN.search(output)
    if not match:
        return "UNSUPPORTED"
    return match.group("status").upper()


def extract_final_answer(output: str) -> str:
    """从 RAG 模型输出中抽取用于指标计算的最终短答案。

    `Final Answer:` 是首选字段。为了兼容模型只输出 `Conclusion:` 的情况，
    这里也支持从 Conclusion 中兜底抽取。
    """
    match = _FINAL_ANSWER_PATTERN.search(output.strip())
    if not match:
        match = _CONCLUSION_PATTERN.search(output.strip())
    if not match:
        return output.strip()

    answer = match.group("answer").strip()
    return answer.splitlines()[0].strip()
