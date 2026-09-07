"""RAG baseline prompt 模板。

Phase 3 在 Phase 2 CoT 的基础上加入外部医学证据。模型仍然是同一个 frozen VLM，
实验变量变成：回答前是否提供 retrieved evidence。
"""

import re


RAG_V1 = """Answer the medical question based on the image and the retrieved medical evidence.

Question: {question}

Retrieved Evidence:
{evidence_block}

Use the following format:

Observation: describe only visual findings that are visible in the image.
Evidence Use: explain which retrieved evidence is relevant or not relevant.
Reasoning: connect the visual findings, question, and evidence.
Final Answer: provide only the short final answer.

Do not provide clinical advice. If the evidence is not relevant, rely on the image and say so briefly."""


_FINAL_ANSWER_PATTERN = re.compile(
    r"final\s+answer\s*:\s*(?P<answer>.+)",
    flags=re.IGNORECASE | re.DOTALL,
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


def build_rag_prompt(
    question: str,
    evidence_records: list[dict],
    max_chars_per_evidence: int = 700,
) -> str:
    """为一条样本构造 RAG prompt。"""
    evidence_block = format_evidence_block(
        evidence_records,
        max_chars_per_evidence=max_chars_per_evidence,
    )
    return RAG_V1.format(question=question, evidence_block=evidence_block)


def extract_final_answer(output: str) -> str:
    """从 RAG 模型输出中抽取用于指标计算的最终短答案。"""
    match = _FINAL_ANSWER_PATTERN.search(output.strip())
    if not match:
        return output.strip()

    answer = match.group("answer").strip()
    return answer.splitlines()[0].strip()
