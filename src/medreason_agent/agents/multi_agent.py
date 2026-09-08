"""Phase 4 Fixed / Supervisor Multi-Agent 执行器。"""

from __future__ import annotations

from dataclasses import dataclass

from medreason_agent.agents.memory import PersistentAgentMemoryStore
from medreason_agent.agents.rag_agents import RetrievalPipeline
from medreason_agent.agents.state import SharedAgentState
from medreason_agent.data.vqa_rad import VQARADSample
from medreason_agent.evaluation.claim_status import (
    claims_to_status_records,
    parse_claim_status_lines,
)
from medreason_agent.models.vlm import VLMBackend, VLMRequest
from medreason_agent.prompts.multi_agent import (
    build_answer_prompt,
    build_critic_prompt,
    build_reasoning_prompt,
    build_supervisor_prompt,
    build_verifier_prompt,
    build_vision_prompt,
    extract_claims,
    extract_critic_decision,
    extract_final_answer,
    extract_selected_tools,
    extract_verification_status,
)
from medreason_agent.prompts.rag import build_claim_verification_query, format_evidence_block


@dataclass(frozen=True)
class MultiAgentResult:
    """单条样本的 Multi-Agent 输出。"""

    prediction: str
    reasoning_output: str
    raw_output: str
    agent_outputs: dict[str, str]
    agent_route: list[str]
    expected_agent_route: list[str]
    selected_tools: list[str]
    expected_selected_tools: list[str]
    tool_calls: list[dict]
    retrieved_evidence: list[dict]
    generated_claims: list[str]
    claim_statuses: list[dict]
    shared_state: dict
    critic_decision: str
    claim_verification_status: str = ""
    evidence_query: str = ""
    memory_records: list[dict] | None = None
    memory_write_record: dict | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    confidence: float | None = None


class FixedMultiAgent:
    """固定链路 baseline：Vision -> Reasoning -> Critic -> Answer。"""

    mode = "fixed_multi_agent"
    expected_agent_route = ["vision_agent", "reasoning_agent", "critic_agent", "answer_agent"]

    def __init__(
        self,
        backend: VLMBackend,
        memory_store: PersistentAgentMemoryStore | None = None,
    ) -> None:
        self.backend = backend
        self.memory_store = memory_store

    def run(
        self,
        sample: VQARADSample,
        max_new_tokens: int,
        experiment_id: str = "",
    ) -> MultiAgentResult:
        """执行固定 Multi-Agent baseline。"""
        shared_state = SharedAgentState(question=sample.question)
        memory_records = _retrieve_persistent_memory(self.memory_store, sample)
        shared_state.add_persistent_memories(memory_records)
        vision = self._generate(sample, build_vision_prompt(sample.question), max_new_tokens)
        vision_claim_statuses = parse_claim_status_lines(
            vision.raw_output,
            source_agent="vision_agent",
        )
        shared_state.add_agent_output(
            "vision_agent",
            vision.raw_output,
            claim_statuses=vision_claim_statuses,
        )

        reasoning = self._generate(
            sample,
            build_reasoning_prompt(
                sample.question,
                shared_state_context=shared_state.compressed_context(),
            ),
            max_new_tokens,
        )
        reasoning_claim_statuses = parse_claim_status_lines(
            reasoning.raw_output,
            source_agent="reasoning_agent",
        )
        shared_state.add_agent_output(
            "reasoning_agent",
            reasoning.raw_output,
            claim_statuses=reasoning_claim_statuses,
        )

        critic = self._generate(
            sample,
            build_critic_prompt(
                sample.question,
                shared_state_context=shared_state.compressed_context(),
            ),
            max_new_tokens,
        )
        critic_claim_statuses = parse_claim_status_lines(
            critic.raw_output,
            source_agent="critic_agent",
        )
        shared_state.add_agent_output(
            "critic_agent",
            critic.raw_output,
            claim_statuses=critic_claim_statuses,
        )

        answer = self._generate(
            sample,
            build_answer_prompt(
                sample.question,
                shared_state_context=shared_state.compressed_context(),
            ),
            max_new_tokens,
        )
        shared_state.add_agent_output("answer_agent", answer.raw_output)
        generated_claims = extract_claims(reasoning.raw_output)
        if not shared_state.claim_statuses:
            shared_state.extend_claim_statuses(
                claims_to_status_records(
                    generated_claims,
                    status="HYPOTHESIS",
                    source_agent="reasoning_agent",
                )
            )
        elif not _has_reasoning_claim_status(shared_state):
            shared_state.extend_claim_statuses(
                claims_to_status_records(
                    generated_claims,
                    status="HYPOTHESIS",
                    source_agent="reasoning_agent",
                )
            )
        claim_statuses = shared_state.claim_statuses
        shared_state_record = shared_state.to_record()
        memory_write_record = _append_persistent_memory(
            self.memory_store,
            sample=sample,
            shared_state=shared_state_record,
            prediction=extract_final_answer(answer.raw_output),
            experiment_id=experiment_id,
        )
        return MultiAgentResult(
            prediction=extract_final_answer(answer.raw_output),
            reasoning_output=reasoning.raw_output,
            raw_output=answer.raw_output,
            agent_outputs={
                "vision": vision.raw_output,
                "reasoning": reasoning.raw_output,
                "critic": critic.raw_output,
                "answer": answer.raw_output,
            },
            agent_route=list(self.expected_agent_route),
            expected_agent_route=list(self.expected_agent_route),
            selected_tools=[],
            expected_selected_tools=[],
            tool_calls=[
                {"tool": "vlm_generate", "stage": "vision_agent"},
                {"tool": "vlm_generate", "stage": "reasoning_agent"},
                {"tool": "vlm_generate", "stage": "critic_agent"},
                {"tool": "vlm_generate", "stage": "answer_agent"},
            ],
            retrieved_evidence=[],
            generated_claims=generated_claims,
            claim_statuses=claim_statuses,
            shared_state=shared_state_record,
            critic_decision=extract_critic_decision(critic.raw_output),
            memory_records=memory_records,
            memory_write_record=memory_write_record,
            input_tokens=_sum_optional_many(
                vision.input_tokens,
                reasoning.input_tokens,
                critic.input_tokens,
                answer.input_tokens,
            ),
            output_tokens=_sum_optional_many(
                vision.output_tokens,
                reasoning.output_tokens,
                critic.output_tokens,
                answer.output_tokens,
            ),
            confidence=answer.confidence,
        )

    def _generate(self, sample: VQARADSample, prompt: str, max_new_tokens: int):
        """调用同一个 frozen VLM backend。"""
        return self.backend.generate(
            VLMRequest(
                image_path=str(sample.absolute_image_path),
                question=sample.question,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
            )
        )


class SupervisorMultiAgent:
    """Supervisor-guided method：先选工具，再执行多 agent 协作。"""

    mode = "supervisor_multi_agent"
    expected_agent_route = [
        "supervisor_agent",
        "vision_agent",
        "retrieval_agent",
        "reasoning_agent",
        "verifier_agent",
        "answer_agent",
    ]
    expected_selected_tools = [
        "Vision Agent",
        "Retrieval Agent",
        "Reasoning Agent",
        "Verifier Agent",
        "Answer Agent",
    ]

    def __init__(
        self,
        backend: VLMBackend,
        retrieval_pipeline: RetrievalPipeline | None = None,
        memory_store: PersistentAgentMemoryStore | None = None,
    ) -> None:
        self.backend = backend
        self.retrieval_pipeline = retrieval_pipeline
        self.memory_store = memory_store

    def run(
        self,
        sample: VQARADSample,
        max_new_tokens: int,
        experiment_id: str = "",
    ) -> MultiAgentResult:
        """执行 Supervisor Multi-Agent method。"""
        shared_state = SharedAgentState(question=sample.question)
        memory_records = _retrieve_persistent_memory(self.memory_store, sample)
        shared_state.add_persistent_memories(memory_records)
        supervisor = self._generate(
            sample,
            build_supervisor_prompt(sample.question),
            max_new_tokens,
        )
        selected_tools = extract_selected_tools(supervisor.raw_output)
        shared_state.set_selected_tools(selected_tools)
        shared_state.add_agent_output("supervisor_agent", supervisor.raw_output)
        agent_route = ["supervisor_agent"]

        vision = self._generate(sample, build_vision_prompt(sample.question), max_new_tokens)
        vision_claim_statuses = parse_claim_status_lines(
            vision.raw_output,
            source_agent="vision_agent",
        )
        shared_state.add_agent_output(
            "vision_agent",
            vision.raw_output,
            claim_statuses=vision_claim_statuses,
        )
        agent_route.append("vision_agent")

        evidence_query = sample.question
        retrieved_evidence: list[dict] = []
        retrieval_tool_calls: list[dict] = []
        if self.retrieval_pipeline is not None:
            trace = self.retrieval_pipeline.retrieve(evidence_query)
            retrieved_evidence = trace.evidence_records
            shared_state.add_retrieved_evidence(
                retrieved_evidence,
                stage="retrieval_agent",
            )
            retrieval_tool_calls = self.retrieval_pipeline.build_tool_calls(
                trace,
                stage="retrieval_agent",
            )
            agent_route.append("retrieval_agent")

        max_chars = (
            self.retrieval_pipeline.settings.max_chars_per_evidence
            if self.retrieval_pipeline is not None
            else 700
        )
        evidence_block = format_evidence_block(retrieved_evidence, max_chars_per_evidence=max_chars)
        reasoning = self._generate(
            sample,
            build_reasoning_prompt(
                sample.question,
                evidence_block=evidence_block,
                shared_state_context=shared_state.compressed_context(),
            ),
            max_new_tokens,
        )
        reasoning_claim_statuses = parse_claim_status_lines(
            reasoning.raw_output,
            source_agent="reasoning_agent",
        )
        shared_state.add_agent_output(
            "reasoning_agent",
            reasoning.raw_output,
            claim_statuses=reasoning_claim_statuses,
        )
        agent_route.append("reasoning_agent")

        claims = extract_claims(reasoning.raw_output)
        if not _has_reasoning_claim_status(shared_state):
            shared_state.extend_claim_statuses(
                claims_to_status_records(
                    claims,
                    status="HYPOTHESIS",
                    source_agent="reasoning_agent",
                )
            )
        verification_status = ""
        verifier_output = ""
        if self.retrieval_pipeline is not None:
            claim = claims[0] if claims else extract_final_answer(reasoning.raw_output)
            evidence_query = build_claim_verification_query(
                question=sample.question,
                claim=claim,
                reasoning_output=reasoning.raw_output,
            )
            verifier_trace = self.retrieval_pipeline.retrieve(evidence_query)
            verifier_evidence = verifier_trace.evidence_records
            retrieved_evidence = [*retrieved_evidence, *verifier_evidence]
            shared_state.add_retrieved_evidence(
                verifier_evidence,
                stage="verifier_retrieval",
            )
            retrieval_tool_calls.extend(
                self.retrieval_pipeline.build_tool_calls(
                    verifier_trace,
                    stage="verifier_retrieval",
                )
            )
            verifier = self._generate(
                sample,
                build_verifier_prompt(
                    sample.question,
                    claims=claims,
                    evidence_block=format_evidence_block(
                        verifier_evidence,
                        max_chars_per_evidence=max_chars,
                    ),
                    shared_state_context=shared_state.compressed_context(),
                ),
                max_new_tokens,
            )
            verifier_output = verifier.raw_output
            verification_status = extract_verification_status(verifier.raw_output)
            verifier_claim_statuses = parse_claim_status_lines(
                verifier.raw_output,
                source_agent="verifier_agent",
            )
            if not verifier_claim_statuses:
                verifier_claim_statuses = claims_to_status_records(
                    claims,
                    status=verification_status,
                    source_agent="verifier_agent",
                )
            shared_state.add_agent_output(
                "verifier_agent",
                verifier.raw_output,
                claim_statuses=verifier_claim_statuses,
            )
            agent_route.append("verifier_agent")
        else:
            verifier = None

        answer = self._generate(
            sample,
            build_answer_prompt(
                sample.question,
                shared_state_context=shared_state.compressed_context(),
            ),
            max_new_tokens,
        )
        shared_state.add_agent_output("answer_agent", answer.raw_output)
        agent_route.append("answer_agent")
        shared_state_record = shared_state.to_record()
        memory_write_record = _append_persistent_memory(
            self.memory_store,
            sample=sample,
            shared_state=shared_state_record,
            prediction=extract_final_answer(answer.raw_output),
            experiment_id=experiment_id,
        )
        return MultiAgentResult(
            prediction=extract_final_answer(answer.raw_output),
            reasoning_output=reasoning.raw_output,
            raw_output=answer.raw_output,
            agent_outputs={
                "supervisor": supervisor.raw_output,
                "vision": vision.raw_output,
                "reasoning": reasoning.raw_output,
                "verifier": verifier_output,
                "answer": answer.raw_output,
            },
            agent_route=agent_route,
            expected_agent_route=list(self.expected_agent_route),
            selected_tools=selected_tools,
            expected_selected_tools=list(self.expected_selected_tools),
            tool_calls=[
                {"tool": "vlm_generate", "stage": "supervisor_agent"},
                {"tool": "vlm_generate", "stage": "vision_agent"},
                *retrieval_tool_calls,
                {"tool": "vlm_generate", "stage": "reasoning_agent"},
                {"tool": "vlm_generate", "stage": "verifier_agent"},
                {"tool": "vlm_generate", "stage": "answer_agent"},
            ],
            retrieved_evidence=retrieved_evidence,
            generated_claims=claims,
            claim_statuses=shared_state.claim_statuses,
            shared_state=shared_state_record,
            critic_decision=verification_status,
            claim_verification_status=verification_status,
            evidence_query=evidence_query,
            memory_records=memory_records,
            memory_write_record=memory_write_record,
            input_tokens=_sum_optional_many(
                supervisor.input_tokens,
                vision.input_tokens,
                reasoning.input_tokens,
                verifier.input_tokens if verifier is not None else None,
                answer.input_tokens,
            ),
            output_tokens=_sum_optional_many(
                supervisor.output_tokens,
                vision.output_tokens,
                reasoning.output_tokens,
                verifier.output_tokens if verifier is not None else None,
                answer.output_tokens,
            ),
            confidence=answer.confidence,
        )

    def _generate(self, sample: VQARADSample, prompt: str, max_new_tokens: int):
        """调用同一个 frozen VLM backend。"""
        return self.backend.generate(
            VLMRequest(
                image_path=str(sample.absolute_image_path),
                question=sample.question,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
            )
        )


def create_multi_agent(
    mode: str,
    backend: VLMBackend,
    retrieval_pipeline: RetrievalPipeline | None = None,
    memory_store: PersistentAgentMemoryStore | None = None,
) -> FixedMultiAgent | SupervisorMultiAgent:
    """根据配置创建 Phase 4 agent。"""
    if mode == FixedMultiAgent.mode:
        return FixedMultiAgent(backend=backend, memory_store=memory_store)
    if mode == SupervisorMultiAgent.mode:
        return SupervisorMultiAgent(
            backend=backend,
            retrieval_pipeline=retrieval_pipeline,
            memory_store=memory_store,
        )
    raise ValueError(f"Unsupported multi-agent mode: {mode}")


def _sum_optional_many(*values: int | None) -> int | None:
    """汇总多次 VLM 调用 token 数；任一侧缺失时保留缺失状态。"""
    if any(value is None for value in values):
        return None
    return sum(int(value) for value in values)


def _has_reasoning_claim_status(shared_state: SharedAgentState) -> bool:
    """判断 Reasoning Agent 是否已经写入 claim status。"""
    return any(
        item.get("source_agent") == "reasoning_agent"
        for item in shared_state.claim_statuses
    )


def _retrieve_persistent_memory(
    memory_store: PersistentAgentMemoryStore | None,
    sample: VQARADSample,
) -> list[dict]:
    """读取与当前样本相似的跨运行 memory。"""
    if memory_store is None:
        return []
    return memory_store.retrieve(sample)


def _append_persistent_memory(
    memory_store: PersistentAgentMemoryStore | None,
    sample: VQARADSample,
    shared_state: dict,
    prediction: str,
    experiment_id: str,
) -> dict | None:
    """把当前样本经验写入跨运行 memory。"""
    if memory_store is None:
        return None
    return memory_store.append_from_state(
        sample=sample,
        shared_state=shared_state,
        prediction=prediction,
        experiment_id=experiment_id,
    )
