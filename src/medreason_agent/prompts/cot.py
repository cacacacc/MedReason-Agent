"""Chain-of-Thought prompt 模板。

Phase 2 保持同一个 frozen VLM 和同一个数据划分，只改变 prompt 策略。
目标是测试显式中间推理是否比 Phase 1 的直接回答 baseline 更有帮助。
"""

import re

# Vanilla CoT 只给出自然语言 step-by-step 指令，不强制 Observation / Reasoning 分段。
# 它用于回答“普通显式推理是否有帮助”。
VANILLA_COT_V1 = """Answer the medical question based on the image.

Question: {question}

Think step by step, then provide the final short answer.

Use this final line:
Final Answer: provide only the short final answer.

Do not provide clinical advice."""


# Structured CoT 固定 Observation / Reasoning / Final Answer 段落。
# evaluator 只抽取 `Final Answer:` 后面的短答案计算 accuracy。
STRUCTURED_COT_V1 = """Answer the medical question based on the image.

Question: {question}

Use the following format:

Observation: describe only visual findings that are visible in the image.
Reasoning: explain how the visual findings relate to the question.
Final Answer: provide only the short final answer.

Do not provide clinical advice."""


_FINAL_ANSWER_PATTERN = re.compile(
    r"final\s+answer\s*:\s*(?P<answer>.+)",
    flags=re.IGNORECASE | re.DOTALL,
)


def build_cot_prompt(question: str, prompt_style: str = "structured") -> str:
    """为一条 VQA-RAD 问题构造 CoT prompt。"""
    if prompt_style == "vanilla":
        return VANILLA_COT_V1.format(question=question)
    if prompt_style == "structured":
        return STRUCTURED_COT_V1.format(question=question)
    raise ValueError(f"Unsupported CoT prompt_style: {prompt_style}")


def extract_final_answer(output: str) -> str:
    """抽取用于指标计算的短答案。

    CoT 完整输出适合做推理分析，但 exact match 等答案指标应该只比较最终答案和
    ground truth。
    """
    match = _FINAL_ANSWER_PATTERN.search(output.strip())
    if not match:
        # 如果模型在早期 smoke test 中没有遵守格式，就退回使用原始输出，
        # 这样 runner 不会因为格式问题直接崩掉。
        return output.strip()

    answer = match.group("answer").strip()
    first_line = answer.splitlines()[0].strip()
    return first_line
