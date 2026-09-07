"""Chain-of-thought prompt templates."""

import re

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
    """Build the chain-of-thought direct-answer prompt."""
    return COT_V1.format(question=question)


def extract_final_answer(output: str) -> str:
    """Extract the short final answer from a CoT model output."""
    match = _FINAL_ANSWER_PATTERN.search(output.strip())
    if not match:
        return output.strip()

    answer = match.group("answer").strip()
    first_line = answer.splitlines()[0].strip()
    return first_line
