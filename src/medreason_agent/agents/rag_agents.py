"""Phase 3 使用的 RAG agent。

这里把 RAG 拆成三条真实执行路径：

- Knowledge RAG：先根据问题检索医学知识，再让 VLM 进行临床推理。
- Evidence RAG：先让 VLM 给出初始 claim，再围绕 claim 检索证据并验证。
- Knowledge + Evidence RAG：先查知识辅助推理，再查证据验证推理 claims。

这样实验结果中的 `rag_mode` 不只是标签，而是对应不同的检索时机、query 来源和
agent route。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from medreason_agent.data.vqa_rad import VQARADSample
from medreason_agent.models.vlm import VLMBackend, VLMRequest
from medreason_agent.prompts.cot import build_cot_prompt
from medreason_agent.prompts.rag import (
    build_claim_verification_query,
    build_evidence_verification_prompt,
    build_knowledge_claim_generation_prompt,
    build_knowledge_rag_prompt,
    extract_claims,
    extract_final_answer,
    extract_verification_status,
)
from medreason_agent.retrieval.keyword import RetrievedEvidence, evidence_to_record
from medreason_agent.retrieval.rerank import filter_evidence


class Retriever(Protocol):
    """RAG agent 依赖的最小检索器接口。"""

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedEvidence]:
        """根据 query 召回候选 evidence。"""


class Reranker(Protocol):
    """RAG agent 依赖的最小重排器接口。"""

    def rerank(
        self,
        query: str,
        evidence: list[RetrievedEvidence],
    ) -> list[RetrievedEvidence]:
        """根据 query 对候选 evidence 重排。"""


@dataclass(frozen=True)
class RetrievalSettings:
    """检索链路的可配置参数。"""

    candidate_top_k: int
    top_k: int
    min_rerank_score: float
    max_chars_per_evidence: int
    retriever_name: str = "keyword_retriever"
    reranker_name: str = "keyword_reranker"


@dataclass(frozen=True)
class RetrievalTrace:
    """一次检索链路的中间结果。"""

    query: str
    candidates: list[RetrievedEvidence]
    reranked: list[RetrievedEvidence]
    filtered: list[RetrievedEvidence]
    evidence_records: list[dict]


@dataclass(frozen=True)
class RAGAgentResult:
    """RAG agent 对单个样本的输出。"""

    prediction: str
    reasoning_output: str
    raw_output: str
    retrieved_evidence: list[dict]
    evidence_query: str
    agent_route: list[str]
    tool_calls: list[dict]
    knowledge_query: str = ""
    knowledge_evidence: list[dict] | None = None
    verification_evidence: list[dict] | None = None
    generated_claims: list[str] | None = None
    claim_verification_status: str = ""
    critic_decision: str = ""
    initial_prediction: str = ""
    initial_reasoning_output: str = ""
    verified_claim: str = ""
    confidence: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


class RetrievalPipeline:
    """统一封装 Retriever -> Reranker -> Evidence Filter。

    两类 RAG agent 复用同一条检索链路，但传入的 query 不同：
    Knowledge RAG 使用原始问题，Evidence RAG 使用初始 claim。
    """

    def __init__(
        self,
        retriever: Retriever,
        reranker: Reranker,
        settings: RetrievalSettings,
    ) -> None:
        self.retriever = retriever
        self.reranker = reranker
        self.settings = settings

    def retrieve(self, query: str) -> RetrievalTrace:
        """执行一次完整检索链路。"""
        candidates = self.retriever.retrieve(query, self.settings.candidate_top_k)
        reranked = self.reranker.rerank(query, candidates)
        filtered = filter_evidence(
            reranked,
            top_k=self.settings.top_k,
            min_score=self.settings.min_rerank_score,
        )
        return RetrievalTrace(
            query=query,
            candidates=candidates,
            reranked=reranked,
            filtered=filtered,
            evidence_records=[evidence_to_record(item) for item in filtered],
        )

    def build_tool_calls(self, trace: RetrievalTrace, stage: str) -> list[dict]:
        """把检索链路写成可审计的 tool call 记录。"""
        return [
            {
                "tool": self.settings.retriever_name,
                "stage": stage,
                "query": trace.query,
                "top_k": self.settings.candidate_top_k,
                "num_candidates": len(trace.candidates),
            },
            {
                "tool": self.settings.reranker_name,
                "stage": stage,
                "query": trace.query,
                "num_candidates": len(trace.candidates),
                "num_reranked": len(trace.reranked),
            },
            {
                "tool": "evidence_filter",
                "stage": stage,
                "top_k": self.settings.top_k,
                "min_score": self.settings.min_rerank_score,
                "num_evidence": len(trace.evidence_records),
            },
        ]


class KnowledgeRAGAgent:
    """先检索知识，再让 VLM 推理回答。"""

    rag_mode = "knowledge_acquisition"
    agent_route = [
        "knowledge_agent",
        "retriever",
        "reranker",
        "evidence_filter",
        "clinical_reasoning_agent",
    ]

    def __init__(self, backend: VLMBackend, retrieval_pipeline: RetrievalPipeline) -> None:
        self.backend = backend
        self.retrieval_pipeline = retrieval_pipeline

    def run(self, sample: VQARADSample, max_new_tokens: int) -> RAGAgentResult:
        """执行 Knowledge RAG：Question -> Retrieve -> Reason -> Answer。"""
        evidence_query = sample.question
        trace = self.retrieval_pipeline.retrieve(evidence_query)
        prompt = build_knowledge_rag_prompt(
            question=sample.question,
            evidence_records=trace.evidence_records,
            max_chars_per_evidence=self.retrieval_pipeline.settings.max_chars_per_evidence,
        )
        response = self.backend.generate(
            VLMRequest(
                image_path=str(sample.absolute_image_path),
                question=sample.question,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
            )
        )
        return RAGAgentResult(
            prediction=extract_final_answer(response.raw_output),
            reasoning_output=response.raw_output,
            raw_output=response.raw_output,
            retrieved_evidence=trace.evidence_records,
            evidence_query=evidence_query,
            agent_route=list(self.agent_route),
            tool_calls=[
                *self.retrieval_pipeline.build_tool_calls(trace, stage="pre_reasoning"),
                {"tool": "vlm_generate", "stage": "clinical_reasoning"},
            ],
            confidence=response.confidence,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )


class EvidenceRAGAgent:
    """先产生初始 claim，再检索证据进行验证。"""

    rag_mode = "claim_verification"
    agent_route = [
        "reasoning_agent",
        "claim_extractor",
        "evidence_agent",
        "retriever",
        "reranker",
        "evidence_filter",
        "verifier_agent",
    ]

    def __init__(self, backend: VLMBackend, retrieval_pipeline: RetrievalPipeline) -> None:
        self.backend = backend
        self.retrieval_pipeline = retrieval_pipeline

    def run(self, sample: VQARADSample, max_new_tokens: int) -> RAGAgentResult:
        """执行 Evidence RAG：Reason -> Claim -> Retrieve -> Verify -> Answer。"""
        initial_prompt = build_cot_prompt(sample.question)
        initial_response = self.backend.generate(
            VLMRequest(
                image_path=str(sample.absolute_image_path),
                question=sample.question,
                prompt=initial_prompt,
                max_new_tokens=max_new_tokens,
            )
        )
        initial_prediction = extract_final_answer(initial_response.raw_output)
        evidence_query = build_claim_verification_query(
            question=sample.question,
            claim=initial_prediction,
            reasoning_output=initial_response.raw_output,
        )
        trace = self.retrieval_pipeline.retrieve(evidence_query)
        verification_prompt = build_evidence_verification_prompt(
            question=sample.question,
            initial_claim=initial_prediction,
            initial_reasoning_output=initial_response.raw_output,
            evidence_records=trace.evidence_records,
            max_chars_per_evidence=self.retrieval_pipeline.settings.max_chars_per_evidence,
        )
        verification_response = self.backend.generate(
            VLMRequest(
                image_path=str(sample.absolute_image_path),
                question=sample.question,
                prompt=verification_prompt,
                max_new_tokens=max_new_tokens,
            )
        )
        verification_status = extract_verification_status(verification_response.raw_output)
        return RAGAgentResult(
            prediction=extract_final_answer(verification_response.raw_output),
            reasoning_output=verification_response.raw_output,
            raw_output=verification_response.raw_output,
            retrieved_evidence=trace.evidence_records,
            evidence_query=evidence_query,
            agent_route=list(self.agent_route),
            tool_calls=[
                {"tool": "vlm_generate", "stage": "initial_reasoning"},
                {"tool": "claim_extractor", "stage": "post_reasoning", "claim": initial_prediction},
                *self.retrieval_pipeline.build_tool_calls(trace, stage="claim_verification"),
                {"tool": "vlm_generate", "stage": "verification"},
            ],
            initial_prediction=initial_prediction,
            initial_reasoning_output=initial_response.raw_output,
            verified_claim=initial_prediction,
            claim_verification_status=verification_status,
            critic_decision=verification_status,
            confidence=verification_response.confidence,
            input_tokens=_sum_optional(
                initial_response.input_tokens,
                verification_response.input_tokens,
            ),
            output_tokens=_sum_optional(
                initial_response.output_tokens,
                verification_response.output_tokens,
            ),
        )


class KnowledgeThenEvidenceRAGAgent:
    """Knowledge RAG 和 Evidence RAG 的串联协作路径。

    完整流程：
    Knowledge Agent 先检索回答前知识；Reasoning Agent 基于图像和知识生成 claims；
    Verifier 抽取 claim；Evidence Agent 围绕 claim 再检索；Verifier 比较 claim 和
    evidence，输出 SUPPORTED / UNSUPPORTED / CONTRADICTED。
    """

    rag_mode = "knowledge_then_claim_verification"
    agent_route = [
        "knowledge_agent",
        "retriever",
        "reranker",
        "evidence_filter",
        "reasoning_agent",
        "claim_extractor",
        "evidence_agent",
        "retriever",
        "reranker",
        "evidence_filter",
        "verifier_agent",
    ]

    def __init__(self, backend: VLMBackend, retrieval_pipeline: RetrievalPipeline) -> None:
        self.backend = backend
        self.retrieval_pipeline = retrieval_pipeline

    def run(self, sample: VQARADSample, max_new_tokens: int) -> RAGAgentResult:
        """执行组合路径：先知识增强推理，再证据检索验证。"""
        knowledge_query = sample.question
        knowledge_trace = self.retrieval_pipeline.retrieve(knowledge_query)
        initial_prompt = build_knowledge_claim_generation_prompt(
            question=sample.question,
            evidence_records=knowledge_trace.evidence_records,
            max_chars_per_evidence=self.retrieval_pipeline.settings.max_chars_per_evidence,
        )
        initial_response = self.backend.generate(
            VLMRequest(
                image_path=str(sample.absolute_image_path),
                question=sample.question,
                prompt=initial_prompt,
                max_new_tokens=max_new_tokens,
            )
        )
        generated_claims = extract_claims(initial_response.raw_output)
        initial_prediction = extract_final_answer(initial_response.raw_output)
        claim = generated_claims[0] if generated_claims else initial_prediction
        evidence_query = build_claim_verification_query(
            question=sample.question,
            claim=claim,
            reasoning_output=initial_response.raw_output,
        )
        verification_trace = self.retrieval_pipeline.retrieve(evidence_query)
        verification_prompt = build_evidence_verification_prompt(
            question=sample.question,
            initial_claim=claim,
            initial_reasoning_output=initial_response.raw_output,
            evidence_records=verification_trace.evidence_records,
            max_chars_per_evidence=self.retrieval_pipeline.settings.max_chars_per_evidence,
        )
        verification_response = self.backend.generate(
            VLMRequest(
                image_path=str(sample.absolute_image_path),
                question=sample.question,
                prompt=verification_prompt,
                max_new_tokens=max_new_tokens,
            )
        )
        verification_status = extract_verification_status(verification_response.raw_output)
        knowledge_evidence = _tag_evidence_stage(
            knowledge_trace.evidence_records,
            stage="knowledge_acquisition",
        )
        verification_evidence = _tag_evidence_stage(
            verification_trace.evidence_records,
            stage="claim_verification",
        )
        return RAGAgentResult(
            prediction=extract_final_answer(verification_response.raw_output),
            reasoning_output=verification_response.raw_output,
            raw_output=verification_response.raw_output,
            retrieved_evidence=[*knowledge_evidence, *verification_evidence],
            evidence_query=evidence_query,
            agent_route=list(self.agent_route),
            tool_calls=[
                *self.retrieval_pipeline.build_tool_calls(
                    knowledge_trace,
                    stage="knowledge_acquisition",
                ),
                {"tool": "vlm_generate", "stage": "knowledge_guided_reasoning"},
                {"tool": "claim_extractor", "stage": "post_reasoning", "claim": claim},
                *self.retrieval_pipeline.build_tool_calls(
                    verification_trace,
                    stage="claim_verification",
                ),
                {
                    "tool": "verifier",
                    "stage": "verification",
                    "decision": verification_status,
                },
            ],
            knowledge_query=knowledge_query,
            knowledge_evidence=knowledge_evidence,
            verification_evidence=verification_evidence,
            generated_claims=generated_claims,
            claim_verification_status=verification_status,
            critic_decision=verification_status,
            initial_prediction=initial_prediction,
            initial_reasoning_output=initial_response.raw_output,
            verified_claim=claim,
            confidence=verification_response.confidence,
            input_tokens=_sum_optional(
                initial_response.input_tokens,
                verification_response.input_tokens,
            ),
            output_tokens=_sum_optional(
                initial_response.output_tokens,
                verification_response.output_tokens,
            ),
        )


def _sum_optional(left: int | None, right: int | None) -> int | None:
    """汇总两次 VLM 调用的 token 数；任一侧缺失时保留缺失状态。"""
    if left is None or right is None:
        return None
    return left + right


def _tag_evidence_stage(evidence_records: list[dict], stage: str) -> list[dict]:
    """给 evidence 标记来源阶段，方便分析知识检索和验证检索的作用。"""
    return [dict(record, evidence_stage=stage) for record in evidence_records]


def create_rag_agent(
    rag_mode: str,
    backend: VLMBackend,
    retrieval_pipeline: RetrievalPipeline,
) -> KnowledgeRAGAgent | EvidenceRAGAgent | KnowledgeThenEvidenceRAGAgent:
    """根据 `rag_mode` 创建对应的 RAG agent。"""
    if rag_mode == KnowledgeRAGAgent.rag_mode:
        return KnowledgeRAGAgent(backend=backend, retrieval_pipeline=retrieval_pipeline)
    if rag_mode == EvidenceRAGAgent.rag_mode:
        return EvidenceRAGAgent(backend=backend, retrieval_pipeline=retrieval_pipeline)
    if rag_mode == KnowledgeThenEvidenceRAGAgent.rag_mode:
        return KnowledgeThenEvidenceRAGAgent(
            backend=backend,
            retrieval_pipeline=retrieval_pipeline,
        )
    raise ValueError(
        "Unsupported rag_mode="
        f"{rag_mode!r}. Use 'knowledge_acquisition', 'claim_verification', "
        "or 'knowledge_then_claim_verification'."
    )
