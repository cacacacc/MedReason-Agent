"""Chain-of-Thought prompt 模板。

Phase 2 保持同一个 frozen VLM 和同一个数据划分，只改变 prompt 策略。
目标是测试显式中间推理是否比 Phase 1 的直接回答 baseline 更有帮助。
"""

import re

# 下面的格式标记属于实验协议。`Final Answer:` 尤其重要：
# evaluator 只抽取这个短答案计算 accuracy，完整文本会保存到 `reasoning_output`。
COT_V1 = """Answer the medical question based on the image.

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


def build_cot_prompt(question: str) -> str:
    """为一条 VQA-RAD 问题构造 CoT prompt。"""
    return COT_V1.format(question=question)


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
