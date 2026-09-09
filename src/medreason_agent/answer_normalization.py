"""医学 VQA 短答案规整工具。

VQA-RAD 的主指标依赖短答案。后续阶段的 CoT / RAG / Agent 往往会输出解释型文本，
如果直接拿长句做 exact match，会把“语义上是 yes/no”的答案错判。因此这里集中处理
benchmark 预测答案的短化和归一化。
"""

from __future__ import annotations

import re

_YES_NO_QUESTION_PREFIXES = (
    "is ",
    "are ",
    "was ",
    "were ",
    "does ",
    "do ",
    "did ",
    "can ",
    "could ",
    "has ",
    "have ",
    "had ",
    "should ",
)
_YES_PATTERN = re.compile(r"^\s*(yes|y)\b", flags=re.IGNORECASE)
_NO_PATTERN = re.compile(r"^\s*(no|n)\b", flags=re.IGNORECASE)


def is_yes_no_question(question: str) -> bool:
    """判断问题是否是 closed yes/no 问题。"""
    normalized = " ".join(question.lower().strip().split())
    return normalized.startswith(_YES_NO_QUESTION_PREFIXES)


def canonical_short_answer(answer: str, question: str = "") -> str:
    """把模型最终答案规整成适合 VQA-RAD 评估的短答案。

    对 yes/no 问题，只要最终答案以 yes 或 no 开头，就压缩为单词 `yes` / `no`。
    对开放问题，去掉常见引导词并保留第一行，避免 explanation 泄漏进 prediction。
    """
    text = answer.strip()
    if not text:
        return text

    first_line = text.splitlines()[0].strip()
    first_line = _strip_answer_prefix(first_line)
    if is_yes_no_question(question):
        yes_match = _YES_PATTERN.match(first_line)
        no_match = _NO_PATTERN.match(first_line)
        if yes_match:
            return "yes"
        if no_match:
            return "no"
    return first_line.strip().strip(" .")


def _strip_answer_prefix(answer: str) -> str:
    """移除模型常见的答案前缀。"""
    return re.sub(
        r"^\s*(final\s+answer|answer|conclusion)\s*:\s*",
        "",
        answer,
        flags=re.IGNORECASE,
    ).strip()
