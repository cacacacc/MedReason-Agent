"""RAG 证据重排与过滤模块。

Phase 3 的第一版检索链路现在是：

Retriever -> Reranker -> Evidence Filter -> Reasoning Agent

这里先实现一个轻量级关键词 reranker，不额外加载大模型，适合 RTX 4070 上做实验闭环。
后续如果接入 BGE reranker 或 cross-encoder，只需要保持输入输出仍然是 `RetrievedEvidence`。
"""

from __future__ import annotations

from collections import Counter

from medreason_agent.retrieval.keyword import RetrievedEvidence, tokenize

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "be",
    "can",
    "do",
    "does",
    "has",
    "have",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "there",
    "this",
    "to",
    "what",
    "which",
    "with",
}

_MEDICAL_FINDING_TOKENS = {
    "abnormal",
    "abnormality",
    "aneurysm",
    "calcification",
    "calcified",
    "consolidation",
    "edema",
    "effusion",
    "fracture",
    "hemorrhage",
    "hypodense",
    "infarct",
    "lesion",
    "mass",
    "nodule",
    "opacity",
    "pneumonia",
    "tumor",
}

_ANATOMY_TOKENS = {
    "abdomen",
    "aorta",
    "brain",
    "chest",
    "colon",
    "heart",
    "kidney",
    "liver",
    "lung",
    "pancreas",
    "spleen",
    "trachea",
    "ventricle",
}

_MODALITY_TOKENS = {
    "ct",
    "mri",
    "radiograph",
    "ultrasound",
    "xray",
}


class KeywordReranker:
    """基于问题和 evidence 文本重叠度的轻量重排器。

    Retriever 负责快速召回候选 chunk；reranker 再更细地比较 query 与每条 evidence 的标题、
    正文是否匹配。标题里的医学实体通常更关键，所以标题 token 会获得更高权重。
    """

    def rerank(
        self,
        query: str,
        evidence: list[RetrievedEvidence],
    ) -> list[RetrievedEvidence]:
        """对候选 evidence 重新打分并按质量降序排序。"""
        query_tokens = tokenize(query)
        if not query_tokens:
            return evidence

        query_counts = Counter(query_tokens)
        reranked: list[RetrievedEvidence] = []
        for item in evidence:
            title_counts = Counter(tokenize(item.title))
            text_counts = Counter(tokenize(item.text))

            overlap_score = 0.0
            for token, query_count in query_counts.items():
                # 标题通常概括医学主题，因此标题命中比正文命中更重要。
                overlap_score += query_count * title_counts[token] * 2.0
                overlap_score += query_count * text_counts[token]

            # 保留 retriever 原始分数的一小部分，避免完全丢掉第一阶段召回信号。
            final_score = overlap_score + item.score * 0.1
            reranked.append(
                RetrievedEvidence(
                    chunk_id=item.chunk_id,
                    doc_id=item.doc_id,
                    title=item.title,
                    text=item.text,
                    score=round(final_score, 6),
                    source=item.source,
                )
            )

        reranked.sort(key=lambda item: item.score, reverse=True)
        return reranked


class MedicalHeuristicReranker:
    """Medical VQA reranker with normalized lexical and concept-aware scoring.

    This remains lightweight and deterministic. It is meant to test whether a
    better rule-based reranker can improve evidence quality before introducing a
    neural cross-encoder reranker.
    """

    def rerank(
        self,
        query: str,
        evidence: list[RetrievedEvidence],
    ) -> list[RetrievedEvidence]:
        """Rerank evidence with medical token coverage and phrase bonuses."""
        query_tokens = _content_tokens(query)
        if not query_tokens:
            return evidence

        query_token_set = set(query_tokens)
        query_phrases = _content_bigrams(query_tokens)
        reranked: list[RetrievedEvidence] = []
        for item in evidence:
            title_tokens = _content_tokens(item.title)
            text_tokens = _content_tokens(item.text)
            title_set = set(title_tokens)
            text_set = set(text_tokens)
            evidence_set = title_set | text_set

            title_coverage = _coverage(query_token_set, title_set)
            text_coverage = _coverage(query_token_set, text_set)
            evidence_coverage = _coverage(query_token_set, evidence_set)
            phrase_score = _phrase_coverage(
                query_phrases,
                _content_bigrams(title_tokens) | _content_bigrams(text_tokens),
            )
            concept_score = _concept_match_score(query_token_set, evidence_set)

            final_score = (
                3.0 * title_coverage
                + 2.0 * evidence_coverage
                + 1.0 * text_coverage
                + 1.5 * phrase_score
                + 1.2 * concept_score
                + 0.05 * item.score
            )
            reranked.append(
                RetrievedEvidence(
                    chunk_id=item.chunk_id,
                    doc_id=item.doc_id,
                    title=item.title,
                    text=item.text,
                    score=round(final_score, 6),
                    source=item.source,
                )
            )

        reranked.sort(key=lambda item: item.score, reverse=True)
        return reranked


def create_reranker(name: str = "keyword_reranker") -> KeywordReranker | MedicalHeuristicReranker:
    """Create a configured reranker by name."""
    if name == "keyword_reranker":
        return KeywordReranker()
    if name == "medical_heuristic_reranker":
        return MedicalHeuristicReranker()
    raise ValueError(f"Unsupported reranker: {name}")


def filter_evidence(
    evidence: list[RetrievedEvidence],
    top_k: int,
    min_score: float = 0.0,
) -> list[RetrievedEvidence]:
    """过滤低质量 evidence，只把最终证据交给 Reasoning Agent。

    `top_k` 控制最多保留多少条证据；`min_score` 控制最低重排分数。这样可以避免把明显
    无关的 chunk 塞进 prompt，减少 hallucination 和额外推理成本。
    """
    if top_k <= 0:
        return []

    return [item for item in evidence if item.score >= min_score][:top_k]


def _content_tokens(text: str) -> list[str]:
    """Tokenize and remove generic question words."""
    return [token for token in tokenize(text) if token not in _STOPWORDS]


def _content_bigrams(tokens: list[str]) -> set[tuple[str, str]]:
    """Build adjacent token pairs for phrase-level matching."""
    return set(zip(tokens, tokens[1:], strict=False))


def _coverage(reference: set[str], candidate: set[str]) -> float:
    """Compute normalized token coverage."""
    if not reference:
        return 0.0
    return len(reference & candidate) / len(reference)


def _phrase_coverage(
    reference_phrases: set[tuple[str, str]],
    candidate_phrases: set[tuple[str, str]],
) -> float:
    """Compute normalized bigram coverage."""
    if not reference_phrases:
        return 0.0
    return len(reference_phrases & candidate_phrases) / len(reference_phrases)


def _concept_match_score(query_tokens: set[str], evidence_tokens: set[str]) -> float:
    """Reward matching medically informative token groups."""
    groups = (_MEDICAL_FINDING_TOKENS, _ANATOMY_TOKENS, _MODALITY_TOKENS)
    active_groups = [group for group in groups if query_tokens & group]
    if not active_groups:
        return 0.0
    matched = sum(1 for group in active_groups if query_tokens & group & evidence_tokens)
    return matched / len(active_groups)

