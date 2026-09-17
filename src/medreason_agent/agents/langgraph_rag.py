"""LangGraph orchestration for Phase 3 RAG agents.

The legacy RAG agents remain the baseline. These graph agents keep the same
inputs, outputs, prompts, retriever, reranker, and metrics, but make the state
transitions explicit for controlled orchestration ablations.
"""

from __future__ import annotations

from typing import Any, TypedDict

from medreason_agent.agents.rag_agents import (
    RAGAgentResult,
    RetrievalPipeline,
    _sum_optional,
    _tag_evidence_stage,
)
from medreason_agent.data.vqa_rad import VQARADSample
from medreason_agent.evaluation.claim_status import (
    build_claim_record,
    claims_to_status_records,
    parse_claim_status_lines,
)
from medreason_agent.models.vlm import VLMBackend, VLMRequest, VLMResponse
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


class _RAGGraphState(TypedDict, total=False):
    sample: VQARADSample
    max_new_tokens: int
    knowledge_query: str
    evidence_query: str
    knowledge_trace: Any
    verification_trace: Any
    initial_prompt: str
    verification_prompt: str
    initial_response: VLMResponse
    verification_response: VLMResponse
    generated_claims: list[str]
    initial_prediction: str
    verified_claim: str
    result: RAGAgentResult


class LangGraphKnowledgeRAGAgent:
    """Knowledge RAG implemented as explicit LangGraph nodes."""

    rag_mode = "knowledge_acquisition"
    agent_route = [
        "langgraph",
        "knowledge_agent",
        "retriever",
        "reranker",
        "evidence_filter",
        "clinical_reasoning_agent",
    ]

    def __init__(self, backend: VLMBackend, retrieval_pipeline: RetrievalPipeline) -> None:
        self.backend = backend
        self.retrieval_pipeline = retrieval_pipeline
        self.graph = _compile_graph(
            nodes={
                "knowledge_agent": self._knowledge_agent,
                "clinical_reasoning_agent": self._clinical_reasoning_agent,
            },
            edges=[
                ("knowledge_agent", "clinical_reasoning_agent"),
            ],
            entrypoint="knowledge_agent",
        )

    def run(self, sample: VQARADSample, max_new_tokens: int) -> RAGAgentResult:
        state = self.graph.invoke({"sample": sample, "max_new_tokens": max_new_tokens})
        return state["result"]

    def _knowledge_agent(self, state: _RAGGraphState) -> dict[str, Any]:
        sample = state["sample"]
        query = sample.question
        return {
            "knowledge_query": query,
            "evidence_query": query,
            "knowledge_trace": self.retrieval_pipeline.retrieve(query),
        }

    def _clinical_reasoning_agent(self, state: _RAGGraphState) -> dict[str, Any]:
        sample = state["sample"]
        trace = state["knowledge_trace"]
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
                max_new_tokens=state["max_new_tokens"],
            )
        )
        return {
            "result": RAGAgentResult(
                prediction=extract_final_answer(response.raw_output, question=sample.question),
                reasoning_output=response.raw_output,
                raw_output=response.raw_output,
                retrieved_evidence=trace.evidence_records,
                evidence_query=state["evidence_query"],
                agent_route=list(self.agent_route),
                tool_calls=[
                    {"tool": "langgraph_state_graph", "stage": "orchestration"},
                    *self.retrieval_pipeline.build_tool_calls(trace, stage="pre_reasoning"),
                    {"tool": "vlm_generate", "stage": "clinical_reasoning"},
                ],
                claim_statuses=parse_claim_status_lines(
                    response.raw_output,
                    source_agent="clinical_reasoning_agent",
                ),
                confidence=response.confidence,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
            )
        }


class LangGraphEvidenceRAGAgent:
    """Claim-verification RAG implemented as explicit LangGraph nodes."""

    rag_mode = "claim_verification"
    agent_route = [
        "langgraph",
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
        self.graph = _compile_graph(
            nodes={
                "reasoning_agent": self._reasoning_agent,
                "evidence_agent": self._evidence_agent,
                "verifier_agent": self._verifier_agent,
            },
            edges=[
                ("reasoning_agent", "evidence_agent"),
                ("evidence_agent", "verifier_agent"),
            ],
            entrypoint="reasoning_agent",
        )

    def run(self, sample: VQARADSample, max_new_tokens: int) -> RAGAgentResult:
        state = self.graph.invoke({"sample": sample, "max_new_tokens": max_new_tokens})
        return state["result"]

    def _reasoning_agent(self, state: _RAGGraphState) -> dict[str, Any]:
        sample = state["sample"]
        response = self.backend.generate(
            VLMRequest(
                image_path=str(sample.absolute_image_path),
                question=sample.question,
                prompt=build_cot_prompt(sample.question),
                max_new_tokens=state["max_new_tokens"],
            )
        )
        prediction = extract_final_answer(response.raw_output, question=sample.question)
        return {
            "initial_response": response,
            "initial_prediction": prediction,
        }

    def _evidence_agent(self, state: _RAGGraphState) -> dict[str, Any]:
        sample = state["sample"]
        response = state["initial_response"]
        query = build_claim_verification_query(
            question=sample.question,
            claim=state["initial_prediction"],
            reasoning_output=response.raw_output,
        )
        return {
            "evidence_query": query,
            "verification_trace": self.retrieval_pipeline.retrieve(query),
        }

    def _verifier_agent(self, state: _RAGGraphState) -> dict[str, Any]:
        sample = state["sample"]
        initial_response = state["initial_response"]
        trace = state["verification_trace"]
        verification_prompt = build_evidence_verification_prompt(
            question=sample.question,
            initial_claim=state["initial_prediction"],
            initial_reasoning_output=initial_response.raw_output,
            evidence_records=trace.evidence_records,
            max_chars_per_evidence=self.retrieval_pipeline.settings.max_chars_per_evidence,
        )
        verification_response = self.backend.generate(
            VLMRequest(
                image_path=str(sample.absolute_image_path),
                question=sample.question,
                prompt=verification_prompt,
                max_new_tokens=state["max_new_tokens"],
            )
        )
        verification_status = extract_verification_status(verification_response.raw_output)
        verifier_claim_statuses = parse_claim_status_lines(
            verification_response.raw_output,
            source_agent="verifier_agent",
        )
        if not verifier_claim_statuses:
            verifier_claim_statuses = [
                build_claim_record(
                    claim=state["initial_prediction"],
                    status=verification_status,
                    source_agent="verifier_agent",
                )
            ]
        return {
            "result": RAGAgentResult(
                prediction=extract_final_answer(
                    verification_response.raw_output,
                    question=sample.question,
                ),
                reasoning_output=verification_response.raw_output,
                raw_output=verification_response.raw_output,
                retrieved_evidence=trace.evidence_records,
                evidence_query=state["evidence_query"],
                agent_route=list(self.agent_route),
                tool_calls=[
                    {"tool": "langgraph_state_graph", "stage": "orchestration"},
                    {"tool": "vlm_generate", "stage": "initial_reasoning"},
                    {
                        "tool": "claim_extractor",
                        "stage": "post_reasoning",
                        "claim": state["initial_prediction"],
                    },
                    *self.retrieval_pipeline.build_tool_calls(
                        trace,
                        stage="claim_verification",
                    ),
                    {"tool": "vlm_generate", "stage": "verification"},
                ],
                initial_prediction=state["initial_prediction"],
                initial_reasoning_output=initial_response.raw_output,
                verified_claim=state["initial_prediction"],
                claim_statuses=[
                    build_claim_record(
                        claim=state["initial_prediction"],
                        status="HYPOTHESIS",
                        source_agent="reasoning_agent",
                    ),
                    *verifier_claim_statuses,
                ],
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
        }


class LangGraphKnowledgeThenEvidenceRAGAgent:
    """Knowledge-then-verification RAG implemented as explicit LangGraph nodes."""

    rag_mode = "knowledge_then_claim_verification"
    agent_route = [
        "langgraph",
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
        self.graph = _compile_graph(
            nodes={
                "knowledge_agent": self._knowledge_agent,
                "reasoning_agent": self._reasoning_agent,
                "evidence_agent": self._evidence_agent,
                "verifier_agent": self._verifier_agent,
            },
            edges=[
                ("knowledge_agent", "reasoning_agent"),
                ("reasoning_agent", "evidence_agent"),
                ("evidence_agent", "verifier_agent"),
            ],
            entrypoint="knowledge_agent",
        )

    def run(self, sample: VQARADSample, max_new_tokens: int) -> RAGAgentResult:
        state = self.graph.invoke({"sample": sample, "max_new_tokens": max_new_tokens})
        return state["result"]

    def _knowledge_agent(self, state: _RAGGraphState) -> dict[str, Any]:
        sample = state["sample"]
        query = sample.question
        return {
            "knowledge_query": query,
            "knowledge_trace": self.retrieval_pipeline.retrieve(query),
        }

    def _reasoning_agent(self, state: _RAGGraphState) -> dict[str, Any]:
        sample = state["sample"]
        trace = state["knowledge_trace"]
        prompt = build_knowledge_claim_generation_prompt(
            question=sample.question,
            evidence_records=trace.evidence_records,
            max_chars_per_evidence=self.retrieval_pipeline.settings.max_chars_per_evidence,
        )
        response = self.backend.generate(
            VLMRequest(
                image_path=str(sample.absolute_image_path),
                question=sample.question,
                prompt=prompt,
                max_new_tokens=state["max_new_tokens"],
            )
        )
        generated_claims = extract_claims(response.raw_output)
        initial_prediction = extract_final_answer(response.raw_output, question=sample.question)
        return {
            "initial_response": response,
            "generated_claims": generated_claims,
            "initial_prediction": initial_prediction,
            "verified_claim": generated_claims[0] if generated_claims else initial_prediction,
        }

    def _evidence_agent(self, state: _RAGGraphState) -> dict[str, Any]:
        sample = state["sample"]
        response = state["initial_response"]
        query = build_claim_verification_query(
            question=sample.question,
            claim=state["verified_claim"],
            reasoning_output=response.raw_output,
        )
        return {
            "evidence_query": query,
            "verification_trace": self.retrieval_pipeline.retrieve(query),
        }

    def _verifier_agent(self, state: _RAGGraphState) -> dict[str, Any]:
        sample = state["sample"]
        initial_response = state["initial_response"]
        verification_trace = state["verification_trace"]
        verification_prompt = build_evidence_verification_prompt(
            question=sample.question,
            initial_claim=state["verified_claim"],
            initial_reasoning_output=initial_response.raw_output,
            evidence_records=verification_trace.evidence_records,
            max_chars_per_evidence=self.retrieval_pipeline.settings.max_chars_per_evidence,
        )
        verification_response = self.backend.generate(
            VLMRequest(
                image_path=str(sample.absolute_image_path),
                question=sample.question,
                prompt=verification_prompt,
                max_new_tokens=state["max_new_tokens"],
            )
        )
        verification_status = extract_verification_status(verification_response.raw_output)
        initial_claim_statuses = parse_claim_status_lines(
            initial_response.raw_output,
            source_agent="reasoning_agent",
        )
        if not initial_claim_statuses:
            initial_claim_statuses = claims_to_status_records(
                state["generated_claims"],
                status="HYPOTHESIS",
                source_agent="reasoning_agent",
            )
        verifier_claim_statuses = parse_claim_status_lines(
            verification_response.raw_output,
            source_agent="verifier_agent",
        )
        if not verifier_claim_statuses:
            verifier_claim_statuses = [
                build_claim_record(
                    claim=state["verified_claim"],
                    status=verification_status,
                    source_agent="verifier_agent",
                )
            ]
        knowledge_evidence = _tag_evidence_stage(
            state["knowledge_trace"].evidence_records,
            stage="knowledge_acquisition",
        )
        verification_evidence = _tag_evidence_stage(
            verification_trace.evidence_records,
            stage="claim_verification",
        )
        return {
            "result": RAGAgentResult(
                prediction=extract_final_answer(
                    verification_response.raw_output,
                    question=sample.question,
                ),
                reasoning_output=verification_response.raw_output,
                raw_output=verification_response.raw_output,
                retrieved_evidence=[*knowledge_evidence, *verification_evidence],
                evidence_query=state["evidence_query"],
                agent_route=list(self.agent_route),
                tool_calls=[
                    {"tool": "langgraph_state_graph", "stage": "orchestration"},
                    *self.retrieval_pipeline.build_tool_calls(
                        state["knowledge_trace"],
                        stage="knowledge_acquisition",
                    ),
                    {"tool": "vlm_generate", "stage": "knowledge_guided_reasoning"},
                    {
                        "tool": "claim_extractor",
                        "stage": "post_reasoning",
                        "claim": state["verified_claim"],
                    },
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
                knowledge_query=state["knowledge_query"],
                knowledge_evidence=knowledge_evidence,
                verification_evidence=verification_evidence,
                generated_claims=state["generated_claims"],
                claim_statuses=[*initial_claim_statuses, *verifier_claim_statuses],
                claim_verification_status=verification_status,
                critic_decision=verification_status,
                initial_prediction=state["initial_prediction"],
                initial_reasoning_output=initial_response.raw_output,
                verified_claim=state["verified_claim"],
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
        }


def create_langgraph_rag_agent(
    rag_mode: str,
    backend: VLMBackend,
    retrieval_pipeline: RetrievalPipeline,
) -> (
    LangGraphKnowledgeRAGAgent
    | LangGraphEvidenceRAGAgent
    | LangGraphKnowledgeThenEvidenceRAGAgent
):
    """Create a LangGraph RAG agent for the requested RAG mode."""
    if rag_mode == LangGraphKnowledgeRAGAgent.rag_mode:
        return LangGraphKnowledgeRAGAgent(backend=backend, retrieval_pipeline=retrieval_pipeline)
    if rag_mode == LangGraphEvidenceRAGAgent.rag_mode:
        return LangGraphEvidenceRAGAgent(backend=backend, retrieval_pipeline=retrieval_pipeline)
    if rag_mode == LangGraphKnowledgeThenEvidenceRAGAgent.rag_mode:
        return LangGraphKnowledgeThenEvidenceRAGAgent(
            backend=backend,
            retrieval_pipeline=retrieval_pipeline,
        )
    raise ValueError(
        "Unsupported rag_mode="
        f"{rag_mode!r}. Use 'knowledge_acquisition', 'claim_verification', "
        "or 'knowledge_then_claim_verification'."
    )


def _compile_graph(nodes: dict[str, Any], edges: list[tuple[str, str]], entrypoint: str) -> Any:
    """Compile a LangGraph StateGraph only when the optional package is available."""
    try:
        from langgraph.graph import END, StateGraph
    except ImportError as exc:
        raise RuntimeError(
            "LangGraph orchestration requires langgraph. Install with: "
            "pip install -r requirements/agent.txt"
        ) from exc

    graph = StateGraph(_RAGGraphState)
    for name, node in nodes.items():
        graph.add_node(name, node)
    graph.set_entry_point(entrypoint)
    for source, target in edges:
        graph.add_edge(source, target)
    terminal_nodes = set(nodes) - {source for source, _target in edges}
    for terminal_node in terminal_nodes:
        graph.add_edge(terminal_node, END)
    return graph.compile()
