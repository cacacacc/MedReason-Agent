"""Direct VLM baseline 的提示词。

Phase 1 用这个 prompt 单独测试 frozen VLM 本身的能力。这里故意不要求模型写推理、
不加入检索、不加入验证，这样后续阶段才能和这个简单 baseline 做公平对比。
"""

# 直接回答 prompt 只要求最终短答案。这样 Phase 1 只测答案准确率，
# 不会提前混入 Chain-of-Thought 行为。
DIRECT_V1 = """Answer the medical question based on the image.

Question: {question}

Return only the short final answer.
For yes/no questions, answer exactly yes or no.
For open questions, answer with one short phrase, not a sentence.
Do not provide clinical advice."""


def build_direct_prompt(question: str) -> str:
    """为一条 VQA-RAD 问题构造 direct-answer prompt。"""
    return DIRECT_V1.format(question=question)
