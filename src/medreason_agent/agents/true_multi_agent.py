"""Contract-based multi-agent runtime for Phase 4 experiments.

This module keeps the same prompts, model backend, retrieval pipeline, metrics,
and output schema as ``SupervisorMultiAgent``. The difference is architectural:
each role is implemented as a separate agent object with a narrow input/output
contract, and a runtime commits each step to shared state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from medreason_agent.agents.memory import PersistentAgentMemoryStore
from medreason_agent.agents.multi_agent import (
    MultiAgentResult,
    _apply_answer_gate,
    _build_answer_gate,
    _fixed_supervisor_route,
    _format_answer_gate,
    _heuristic_selected_tools,
    _normalize_selected_tools,
    _planned_route_from_tools,
    _run_vision_stage,
)
from medreason_agent.agents.rag_agents import RetrievalPipeline
from medreason_agent.agents.state import SharedAgentState
from medreason_agent.data.vqa_rad import VQARADSample
from medreason_agent.evaluation.claim_status import claims_to_status_records, parse_claim_status_lines
from medreason_agent.models.vlm import VLMBackend, VLMRequest, VLMResponse
from medreason_agent.prompts.multi_agent import (
    build_answer_prompt,
    build_reasoning_prompt,
    build_supervisor_prompt,
    build_verifier_prompt,
    extract_claims,
    extract_final_answer,
    extract_selected_tools,
    extract_verification_status,
)
from medreason_agent.prompts.rag import build_claim_verification_query, format_evidence_block


@dataclass(frozen=True)
class AgentStepResult:
    """Structured result produced by one independently executable agent."""

    agent_name: str
    raw_output: str = ""
    parsed_output: dict[str, Any] = field(default_factory=dict)
    claim_statuses: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    retrieved_evidence: list[dict[str, Any]] = field(default_factory=list)
    input_tokens: int | None = None
    output_tokens: int | None = None
    confidence: float | None = None


@dataclass
class AgentRuntimeContext:
    """Mutable per-sample context shared by runtime and agent objects."""

    sample: VQARADSample
    shared_state: SharedAgentState
    max_new_tokens: int
    retrieval_pipeline: RetrievalPipeline | None = None
    vision_output: str = ""
    vision_observations: list[str] = field(default_factory=list)
    vision_consensus: str = ""
    reasoning_output: str = ""
    verifier_output: str = ""
    evidence_query: str = ""
    retrieved_evidence: list[dict[str, Any]] = field(default_factory=list)
    generated_claims: list[str] = field(default_factory=list)
    verification_status: str = ""
    answer_gate: dict[str, Any] = field(default_factory=dict)


class RuntimeAgent(Protocol):
    """Minimal interface for independently executable agents."""

    name: str

    def run(self, context: AgentRuntimeContext) -> AgentStepResult:
        """Run one agent step against the current runtime context."""


class BaseVLMAgent:
    """Base class for VLM-backed role agents."""

    name = "base_agent"

    def __init__(self, backend: VLMBackend) -> None:
        self.backend = backend

    def _generate(self, context: AgentRuntimeContext, prompt: str) -> VLMResponse:
        return self.backend.generate(
            VLMRequest(
                image_path=str(context.sample.absolute_image_path),
                question=context.sample.question,
                prompt=prompt,
                max_new_tokens=context.max_new_tokens,
            )
        )


class SupervisorAgent(BaseVLMAgent):
    """Agent that selects which specialist agents should run."""

    name = "supervisor_agent"

    def run(self, context: AgentRuntimeContext) -> AgentStepResult:
        response = self._generate(context, build_supervisor_prompt(context.sample.question))
        selected_tools = extract_selected_tools(response.raw_output)
        return AgentStepResult(
            agent_name=self.name,
            raw_output=response.raw_output,
            parsed_output={"selected_tools": selected_tools},
            tool_calls=[{"tool": "vlm_generate", "stage": self.name}],
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            confidence=response.confidence,
        )


class VisionAgent(RuntimeAgent):
    """Agent with exclusive responsibility for visual observations."""

    name = "vision_agent"

    def __init__(
        self,
        backend: VLMBackend,
        consistency_enabled: bool = False,
        observation_count: int = 3,
    ) -> None:
        self.backend = backend
        self.consistency_enabled = consistency_enabled
        self.observation_count = observation_count

    def run(self, context: AgentRuntimeContext) -> AgentStepResult:
        stage = _run_vision_stage(
            backend=self.backend,
            sample=context.sample,
            max_new_tokens=context.max_new_tokens,
            enabled=self.consistency_enabled,
            observation_count=self.observation_count,
        )
        return AgentStepResult(
            agent_name=self.name,
            raw_output=stage.output,
            parsed_output={
                "vision_observations": stage.observations,
                "vision_consensus": stage.consensus,
            },
            claim_statuses=stage.claim_statuses,
            tool_calls=stage.tool_calls,
            input_tokens=stage.input_tokens,
            output_tokens=stage.output_tokens,
            confidence=stage.confidence,
        )


class RetrievalAgent(RuntimeAgent):
    """Agent with exclusive access to the retrieval pipeline."""

    name = "retrieval_agent"

    def run(self, context: AgentRuntimeContext) -> AgentStepResult:
        if context.retrieval_pipeline is None:
            return AgentStepResult(agent_name=self.name)
        query = context.sample.question
        trace = context.retrieval_pipeline.retrieve(query)
        return AgentStepResult(
            agent_name=self.name,
            parsed_output={"evidence_query": query},
            retrieved_evidence=trace.evidence_records,
            tool_calls=context.retrieval_pipeline.build_tool_calls(
                trace,
                stage=self.name,
            ),
        )


class ReasoningAgent(BaseVLMAgent):
    """Agent that turns observations and evidence into checkable hypotheses."""

    name = "reasoning_agent"

    def run(self, context: AgentRuntimeContext) -> AgentStepResult:
        max_chars = _max_evidence_chars(context)
        evidence_block = format_evidence_block(
            context.retrieved_evidence,
            max_chars_per_evidence=max_chars,
        )
        response = self._generate(
            context,
            build_reasoning_prompt(
                context.sample.question,
                vision_output=context.vision_output,
                evidence_block=evidence_block,
                shared_state_context=context.shared_state.compressed_context(),
            ),
        )
        claims = extract_claims(response.raw_output)
        claim_statuses = parse_claim_status_lines(
            response.raw_output,
            source_agent=self.name,
        )
        if not claim_statuses:
            claim_statuses = claims_to_status_records(
                claims,
                status="HYPOTHESIS",
                source_agent=self.name,
            )
        return AgentStepResult(
            agent_name=self.name,
            raw_output=response.raw_output,
            parsed_output={"claims": claims},
            claim_statuses=claim_statuses,
            tool_calls=[{"tool": "vlm_generate", "stage": self.name}],
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            confidence=response.confidence,
        )


class VerifierAgent(BaseVLMAgent):
    """Agent that verifies reasoning claims against fresh retrieved evidence."""

    name = "verifier_agent"

    def run(self, context: AgentRuntimeContext) -> AgentStepResult:
        if context.retrieval_pipeline is None:
            return AgentStepResult(agent_name=self.name)
        claim = (
            context.generated_claims[0]
            if context.generated_claims
            else extract_final_answer(context.reasoning_output, question=context.sample.question)
        )
        evidence_query = build_claim_verification_query(
            question=context.sample.question,
            claim=claim,
            reasoning_output=context.reasoning_output,
        )
        trace = context.retrieval_pipeline.retrieve(evidence_query)
        max_chars = _max_evidence_chars(context)
        response = self._generate(
            context,
            build_verifier_prompt(
                context.sample.question,
                claims=context.generated_claims,
                evidence_block=format_evidence_block(
                    trace.evidence_records,
                    max_chars_per_evidence=max_chars,
                ),
                shared_state_context=context.shared_state.compressed_context(),
            ),
        )
        verification_status = extract_verification_status(response.raw_output)
        claim_statuses = parse_claim_status_lines(
            response.raw_output,
            source_agent=self.name,
        )
        if not claim_statuses:
            claim_statuses = claims_to_status_records(
                context.generated_claims,
                status=verification_status,
                source_agent=self.name,
            )
        return AgentStepResult(
            agent_name=self.name,
            raw_output=response.raw_output,
            parsed_output={
                "evidence_query": evidence_query,
                "verification_status": verification_status,
            },
            claim_statuses=claim_statuses,
            retrieved_evidence=trace.evidence_records,
            tool_calls=[
                *context.retrieval_pipeline.build_tool_calls(
                    trace,
                    stage="verifier_retrieval",
                ),
                {"tool": "vlm_generate", "stage": self.name},
            ],
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            confidence=response.confidence,
        )


class AnswerAgent(BaseVLMAgent):
    """Agent that produces the final benchmark answer from committed state."""

    name = "answer_agent"

    def run(self, context: AgentRuntimeContext) -> AgentStepResult:
        response = self._generate(
            context,
            build_answer_prompt(
                context.sample.question,
                vision_output=context.vision_output,
                reasoning_output=context.reasoning_output,
                critic_output=context.verifier_output,
                shared_state_context=context.shared_state.compressed_context(),
            ),
        )
        prediction = extract_final_answer(response.raw_output, question=context.sample.question)
        return AgentStepResult(
            agent_name=self.name,
            raw_output=response.raw_output,
            parsed_output={"prediction": prediction},
            tool_calls=[{"tool": "vlm_generate", "stage": self.name}],
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            confidence=response.confidence,
        )


class SupervisorPolicy:
    """Planner that converts supervisor output into a concrete agent schedule."""

    def __init__(
        self,
        expected_tools: list[str],
        dynamic_routing: bool = False,
        question_routing: str = "none",
        internal_verifier: bool = True,
    ) -> None:
        self.expected_tools = list(expected_tools)
        self.dynamic_routing = dynamic_routing
        self.question_routing = question_routing
        self.internal_verifier = internal_verifier

    def build_plan(
        self,
        sample: VQARADSample,
        supervisor_result: AgentStepResult,
        can_retrieve: bool,
    ) -> tuple[list[str], list[str], list[str]]:
        selected_tools = _normalize_selected_tools(
            list(supervisor_result.parsed_output.get("selected_tools", [])),
            fallback=self.expected_tools,
        )
        if self.question_routing == "heuristic":
            selected_tools = _heuristic_selected_tools(sample, selected_tools)
        elif self.question_routing != "none":
            raise ValueError(f"Unsupported question_routing: {self.question_routing}")

        if self.dynamic_routing:
            route = _planned_route_from_tools(
                selected_tools=selected_tools,
                can_retrieve=can_retrieve,
            )
            expected_tools = list(selected_tools)
        else:
            route = _fixed_supervisor_route(
                can_retrieve=can_retrieve,
                internal_verifier=self.internal_verifier,
            )
            expected_tools = list(self.expected_tools)

        if not self.internal_verifier and "verifier_agent" in route:
            route = [agent_name for agent_name in route if agent_name != "verifier_agent"]
        if not can_retrieve:
            route = [
                agent_name
                for agent_name in route
                if agent_name not in {"retrieval_agent", "verifier_agent"}
            ]
            if "answer_agent" not in route:
                route.append("answer_agent")
        return route, selected_tools, expected_tools


class AgentRuntimeMultiAgent:
    """True multi-agent runtime with separate agent objects and contracts."""

    mode = "agent_runtime_multi_agent"
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
        dynamic_routing: bool = False,
        deterministic_answer_gate: bool = False,
        internal_verifier: bool = True,
        question_routing: str = "none",
        vision_consistency_enabled: bool = False,
        vision_observation_count: int = 3,
    ) -> None:
        self.retrieval_pipeline = retrieval_pipeline
        self.memory_store = memory_store
        self.deterministic_answer_gate = deterministic_answer_gate
        self.policy = SupervisorPolicy(
            expected_tools=self.expected_selected_tools,
            dynamic_routing=dynamic_routing,
            question_routing=question_routing,
            internal_verifier=internal_verifier,
        )
        self.agents: dict[str, RuntimeAgent] = {
            "supervisor_agent": SupervisorAgent(backend),
            "vision_agent": VisionAgent(
                backend,
                consistency_enabled=vision_consistency_enabled,
                observation_count=vision_observation_count,
            ),
            "retrieval_agent": RetrievalAgent(),
            "reasoning_agent": ReasoningAgent(backend),
            "verifier_agent": VerifierAgent(backend),
            "answer_agent": AnswerAgent(backend),
        }

    def run(
        self,
        sample: VQARADSample,
        max_new_tokens: int,
        experiment_id: str = "",
    ) -> MultiAgentResult:
        memory_records = _retrieve_memory(self.memory_store, sample)
        shared_state = SharedAgentState(question=sample.question)
        shared_state.add_persistent_memories(memory_records)
        context = AgentRuntimeContext(
            sample=sample,
            shared_state=shared_state,
            max_new_tokens=max_new_tokens,
            retrieval_pipeline=self.retrieval_pipeline,
        )
        step_results: list[AgentStepResult] = []

        supervisor_result = self.agents["supervisor_agent"].run(context)
        self._commit_step(context, supervisor_result)
        step_results.append(supervisor_result)

        planned_route, selected_tools, expected_tools = self.policy.build_plan(
            sample=sample,
            supervisor_result=supervisor_result,
            can_retrieve=self.retrieval_pipeline is not None,
        )
        shared_state.set_selected_tools(selected_tools)

        actual_route = ["supervisor_agent"]
        for agent_name in planned_route:
            if agent_name == "supervisor_agent":
                continue
            step = self.agents[agent_name].run(context)
            self._commit_step(context, step)
            step_results.append(step)
            actual_route.append(agent_name)

        if not context.answer_gate:
            context.answer_gate = _build_answer_gate(
                shared_state.claim_statuses,
                verification_status=context.verification_status,
                enabled=self.deterministic_answer_gate,
            )
            shared_state.add_agent_output("answer_gate", _format_answer_gate(context.answer_gate))

        answer_step = _last_step(step_results, "answer_agent")
        raw_prediction = str(answer_step.parsed_output.get("prediction", ""))
        gated_prediction = _apply_answer_gate(raw_prediction, context.answer_gate)
        shared_state_record = shared_state.to_record()
        memory_write_record = _append_memory(
            self.memory_store,
            sample=sample,
            shared_state=shared_state_record,
            prediction=gated_prediction,
            experiment_id=experiment_id,
        )
        return MultiAgentResult(
            prediction=gated_prediction,
            reasoning_output=context.reasoning_output,
            raw_output=answer_step.raw_output,
            agent_outputs={
                "supervisor": supervisor_result.raw_output,
                "vision": context.vision_output,
                "vision_observations": "\n\n".join(context.vision_observations),
                "vision_consensus": context.vision_consensus,
                "reasoning": context.reasoning_output,
                "verifier": context.verifier_output,
                "answer_gate": _format_answer_gate(context.answer_gate),
                "answer": answer_step.raw_output,
            },
            vision_observations=context.vision_observations,
            vision_consensus=context.vision_consensus,
            agent_route=actual_route,
            expected_agent_route=planned_route,
            selected_tools=selected_tools,
            expected_selected_tools=expected_tools,
            tool_calls=[
                call
                for step in step_results
                for call in step.tool_calls
            ],
            retrieved_evidence=context.retrieved_evidence,
            generated_claims=context.generated_claims,
            claim_statuses=shared_state.claim_statuses,
            shared_state=shared_state_record,
            critic_decision=context.verification_status,
            answer_gate=context.answer_gate,
            claim_verification_status=context.verification_status,
            evidence_query=context.evidence_query,
            memory_records=memory_records,
            memory_write_record=memory_write_record,
            input_tokens=_sum_step_tokens(step_results, "input_tokens"),
            output_tokens=_sum_step_tokens(step_results, "output_tokens"),
            confidence=answer_step.confidence,
        )

    def _commit_step(
        self,
        context: AgentRuntimeContext,
        step: AgentStepResult,
    ) -> None:
        if step.agent_name == "supervisor_agent":
            context.shared_state.add_agent_output(step.agent_name, step.raw_output)
            return

        if step.agent_name == "vision_agent":
            context.vision_output = step.raw_output
            context.vision_observations = list(
                step.parsed_output.get("vision_observations", [])
            )
            context.vision_consensus = str(step.parsed_output.get("vision_consensus", ""))
            context.shared_state.add_agent_output(
                step.agent_name,
                step.raw_output,
                claim_statuses=step.claim_statuses,
            )
            return

        if step.agent_name == "retrieval_agent":
            context.evidence_query = str(step.parsed_output.get("evidence_query", ""))
            context.retrieved_evidence = [
                *context.retrieved_evidence,
                *step.retrieved_evidence,
            ]
            context.shared_state.add_retrieved_evidence(
                step.retrieved_evidence,
                stage=step.agent_name,
            )
            return

        if step.agent_name == "reasoning_agent":
            context.reasoning_output = step.raw_output
            context.generated_claims = list(step.parsed_output.get("claims", []))
            context.shared_state.add_agent_output(
                step.agent_name,
                step.raw_output,
                claim_statuses=step.claim_statuses,
            )
            return

        if step.agent_name == "verifier_agent":
            context.verifier_output = step.raw_output
            context.verification_status = str(
                step.parsed_output.get("verification_status", "")
            )
            context.evidence_query = str(step.parsed_output.get("evidence_query", ""))
            context.retrieved_evidence = [
                *context.retrieved_evidence,
                *step.retrieved_evidence,
            ]
            context.shared_state.add_retrieved_evidence(
                step.retrieved_evidence,
                stage="verifier_retrieval",
            )
            context.shared_state.add_agent_output(
                step.agent_name,
                step.raw_output,
                claim_statuses=step.claim_statuses,
            )
            return

        if step.agent_name == "answer_agent":
            context.answer_gate = _build_answer_gate(
                context.shared_state.claim_statuses,
                verification_status=context.verification_status,
                enabled=self.deterministic_answer_gate,
            )
            context.shared_state.add_agent_output(
                "answer_gate",
                _format_answer_gate(context.answer_gate),
            )
            context.shared_state.add_agent_output(step.agent_name, step.raw_output)

def _max_evidence_chars(context: AgentRuntimeContext) -> int:
    if context.retrieval_pipeline is None:
        return 700
    return context.retrieval_pipeline.settings.max_chars_per_evidence


def _sum_step_tokens(steps: list[AgentStepResult], attribute: str) -> int | None:
    values = [getattr(step, attribute) for step in steps]
    if any(value is None for value in values):
        return None
    return sum(int(value) for value in values)


def _last_step(steps: list[AgentStepResult], agent_name: str) -> AgentStepResult:
    for step in reversed(steps):
        if step.agent_name == agent_name:
            return step
    return AgentStepResult(agent_name=agent_name)


def _retrieve_memory(
    memory_store: PersistentAgentMemoryStore | None,
    sample: VQARADSample,
) -> list[dict]:
    if memory_store is None:
        return []
    return memory_store.retrieve(sample)


def _append_memory(
    memory_store: PersistentAgentMemoryStore | None,
    sample: VQARADSample,
    shared_state: dict,
    prediction: str,
    experiment_id: str,
) -> dict | None:
    if memory_store is None:
        return None
    return memory_store.append_from_state(
        sample=sample,
        shared_state=shared_state,
        prediction=prediction,
        experiment_id=experiment_id,
    )
