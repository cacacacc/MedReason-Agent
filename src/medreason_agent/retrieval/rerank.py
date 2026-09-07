"""RAG 证据重排与过滤模块。

Phase 3 的第一版检索链路现在是：

Retriever -> Reranker -> Evidence Filter -> Reasoning Agent

这里先实现一个轻量级关键词 reranker，不额外加载大模型，适合 RTX 4070 上做实验闭环。
后续如果接入 BGE reranker 或 cross-encoder，只需要保持输入输出仍然是 `RetrievedEvidence`。
"""

from __future__ import annotations

from collections import Counter

from medreason_agent.retrieval.keyword import RetrievedEvidence, tokenize


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

