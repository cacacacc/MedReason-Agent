"""Direct VLM baseline prompts."""

DIRECT_V1 = """Answer the medical question based on the image.

Question: {question}

Return only the short final answer. Do not provide clinical advice."""


def build_direct_prompt(question: str) -> str:
    """Build the direct-answer prompt."""
    return DIRECT_V1.format(question=question)
