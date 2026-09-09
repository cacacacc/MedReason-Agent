"""Phase 4 Fixed / Supervisor Multi-Agent 执行器。"""

from __future__ import annotations

from dataclasses import dataclass

from medreason_agent.agents.memory import PersistentAgentMemoryStore
from medreason_agent.agents.rag_agents import RetrievalPipeline
from medreason_agent.agents.state import SharedAgentState
from medreason_agent.data.vqa_rad import VQARADSample
from medreason_agent.evaluation.claim_status import (
    claims_to_status_records,
    normalize_claim_status,
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
    answer_gate: dict
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

        answer_gate = _build_answer_gate(shared_state.claim_statuses, enabled=False)
        shared_state.add_agent_output("answer_gate", _format_answer_gate(answer_gate))

        answer = self._generate(
            sample,
            build_answer_prompt(
                sample.question,
                shared_state_context=shared_state.compressed_context(),
            ),
            max_new_tokens,
        )
        shared_state.add_agent_output("answer_agent", answer.raw_output)
        gated_prediction = _apply_answer_gate(
            extract_final_answer(answer.raw_output, question=sample.question),
            answer_gate,
        )
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
            prediction=gated_prediction,
            experiment_id=experiment_id,
        )
        return MultiAgentResult(
            prediction=gated_prediction,
            reasoning_output=reasoning.raw_output,
            raw_output=answer.raw_output,
            agent_outputs={
                "vision": vision.raw_output,
                "reasoning": reasoning.raw_output,
                "critic": critic.raw_output,
                "answer_gate": _format_answer_gate(answer_gate),
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
            answer_gate=answer_gate,
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
        dynamic_routing: bool = False,
        deterministic_answer_gate: bool = False,
        internal_verifier: bool = True,
        question_routing: str = "none",
    ) -> None:
        self.backend = backend
        self.retrieval_pipeline = retrieval_pipeline
        self.memory_store = memory_store
        self.dynamic_routing = dynamic_routing
        self.deterministic_answer_gate = deterministic_answer_gate
        self.internal_verifier = internal_verifier
        self.question_routing = question_routing

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
        selected_tools = _normalize_selected_tools(
            extract_selected_tools(supervisor.raw_output),
            fallback=self.expected_selected_tools,
        )
        if self.question_routing == "heuristic":
            selected_tools = _heuristic_selected_tools(sample, selected_tools)
        elif self.question_routing != "none":
            raise ValueError(f"Unsupported question_routing: {self.question_routing}")
        shared_state.set_selected_tools(selected_tools)
        shared_state.add_agent_output("supervisor_agent", supervisor.raw_output)
        agent_route = ["supervisor_agent"]
        tool_calls = [{"tool": "vlm_generate", "stage": "supervisor_agent"}]
        can_retrieve = self.retrieval_pipeline is not None
        if self.dynamic_routing:
            should_run_vision = _tool_selected(selected_tools, "Vision Agent")
            should_run_retrieval = (
                _tool_selected(selected_tools, "Retrieval Agent") and can_retrieve
            )
            should_run_verifier = (
                _tool_selected(selected_tools, "Verifier Agent")
                and can_retrieve
                and self.internal_verifier
            )
            should_run_reasoning = (
                _tool_selected(selected_tools, "Reasoning Agent") or should_run_verifier
            )
            planned_agent_route = _planned_route_from_tools(
                selected_tools=selected_tools,
                can_retrieve=can_retrieve,
            )
            expected_selected_tools = list(selected_tools)
        else:
            should_run_vision = True
            should_run_retrieval = can_retrieve
            should_run_reasoning = True
            should_run_verifier = can_retrieve and self.internal_verifier
            planned_agent_route = _fixed_supervisor_route(
                can_retrieve=can_retrieve,
                internal_verifier=self.internal_verifier,
            )
            expected_selected_tools = list(self.expected_selected_tools)

        vision = None
        vision_output = ""
        if should_run_vision:
            vision = self._generate(sample, build_vision_prompt(sample.question), max_new_tokens)
            vision_output = vision.raw_output
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
            tool_calls.append({"tool": "vlm_generate", "stage": "vision_agent"})

        evidence_query = sample.question
        retrieved_evidence: list[dict] = []
        if should_run_retrieval:
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
            tool_calls.extend(retrieval_tool_calls)
            agent_route.append("retrieval_agent")

        max_chars = (
            self.retrieval_pipeline.settings.max_chars_per_evidence
            if self.retrieval_pipeline is not None
            else 700
        )
        reasoning = None
        reasoning_output = ""
        evidence_block = format_evidence_block(retrieved_evidence, max_chars_per_evidence=max_chars)
        if should_run_reasoning:
            reasoning = self._generate(
                sample,
                build_reasoning_prompt(
                    sample.question,
                    vision_output=vision_output,
                    evidence_block=evidence_block,
                    shared_state_context=shared_state.compressed_context(),
                ),
                max_new_tokens,
            )
            reasoning_output = reasoning.raw_output
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
            tool_calls.append({"tool": "vlm_generate", "stage": "reasoning_agent"})

        claims = extract_claims(reasoning_output)
        if should_run_reasoning and not _has_reasoning_claim_status(shared_state):
            shared_state.extend_claim_statuses(
                claims_to_status_records(
                    claims,
                    status="HYPOTHESIS",
                    source_agent="reasoning_agent",
                )
            )
        verification_status = ""
        verifier_output = ""
        verifier = None
        if should_run_verifier:
            claim = (
                claims[0]
                if claims
                else extract_final_answer(reasoning_output, question=sample.question)
            )
            evidence_query = build_claim_verification_query(
                question=sample.question,
                claim=claim,
                reasoning_output=reasoning_output,
            )
            verifier_trace = self.retrieval_pipeline.retrieve(evidence_query)
            verifier_evidence = verifier_trace.evidence_records
            retrieved_evidence = [*retrieved_evidence, *verifier_evidence]
            shared_state.add_retrieved_evidence(
                verifier_evidence,
                stage="verifier_retrieval",
            )
            verifier_retrieval_tool_calls = self.retrieval_pipeline.build_tool_calls(
                verifier_trace,
                stage="verifier_retrieval",
            )
            tool_calls.extend(verifier_retrieval_tool_calls)
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
            tool_calls.append({"tool": "vlm_generate", "stage": "verifier_agent"})

        answer_gate = _build_answer_gate(
            shared_state.claim_statuses,
            verification_status=verification_status,
            enabled=self.deterministic_answer_gate,
        )
        shared_state.add_agent_output("answer_gate", _format_answer_gate(answer_gate))
        answer = self._generate(
            sample,
            build_answer_prompt(
                sample.question,
                vision_output=vision_output,
                reasoning_output=reasoning_output,
                critic_output=verifier_output,
                shared_state_context=shared_state.compressed_context(),
            ),
            max_new_tokens,
        )
        shared_state.add_agent_output("answer_agent", answer.raw_output)
        agent_route.append("answer_agent")
        tool_calls.append({"tool": "vlm_generate", "stage": "answer_agent"})
        gated_prediction = _apply_answer_gate(
            extract_final_answer(answer.raw_output, question=sample.question),
            answer_gate,
        )
        shared_state_record = shared_state.to_record()
        memory_write_record = _append_persistent_memory(
            self.memory_store,
            sample=sample,
            shared_state=shared_state_record,
            prediction=gated_prediction,
            experiment_id=experiment_id,
        )
        return MultiAgentResult(
            prediction=gated_prediction,
            reasoning_output=reasoning_output,
            raw_output=answer.raw_output,
            agent_outputs={
                "supervisor": supervisor.raw_output,
                "vision": vision_output,
                "reasoning": reasoning_output,
                "verifier": verifier_output,
                "answer_gate": _format_answer_gate(answer_gate),
                "answer": answer.raw_output,
            },
            agent_route=agent_route,
            expected_agent_route=planned_agent_route,
            selected_tools=selected_tools,
            expected_selected_tools=expected_selected_tools,
            tool_calls=tool_calls,
            retrieved_evidence=retrieved_evidence,
            generated_claims=claims,
            claim_statuses=shared_state.claim_statuses,
            shared_state=shared_state_record,
            critic_decision=verification_status,
            answer_gate=answer_gate,
            claim_verification_status=verification_status,
            evidence_query=evidence_query,
            memory_records=memory_records,
            memory_write_record=memory_write_record,
            input_tokens=_sum_response_tokens(
                [supervisor, vision, reasoning, verifier, answer],
                "input_tokens",
            ),
            output_tokens=_sum_response_tokens(
                [supervisor, vision, reasoning, verifier, answer],
                "output_tokens",
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
    dynamic_routing: bool = False,
    deterministic_answer_gate: bool = False,
    internal_verifier: bool = True,
    question_routing: str = "none",
) -> FixedMultiAgent | SupervisorMultiAgent:
    """根据配置创建 Phase 4 agent。"""
    if mode == FixedMultiAgent.mode:
        return FixedMultiAgent(backend=backend, memory_store=memory_store)
    if mode == SupervisorMultiAgent.mode:
        return SupervisorMultiAgent(
            backend=backend,
            retrieval_pipeline=retrieval_pipeline,
            memory_store=memory_store,
            dynamic_routing=dynamic_routing,
            deterministic_answer_gate=deterministic_answer_gate,
            internal_verifier=internal_verifier,
            question_routing=question_routing,
        )
    raise ValueError(f"Unsupported multi-agent mode: {mode}")


def _sum_optional_many(*values: int | None) -> int | None:
    """汇总多次 VLM 调用 token 数；任一侧缺失时保留缺失状态。"""
    if any(value is None for value in values):
        return None
    return sum(int(value) for value in values)


def _sum_response_tokens(responses: list[object | None], attribute: str) -> int | None:
    """只汇总实际执行过的 VLM 调用；被 Supervisor 跳过的 agent 不计入 token。"""
    values = [
        getattr(response, attribute)
        for response in responses
        if response is not None
    ]
    return _sum_optional_many(*values)


def _normalize_selected_tools(
    selected_tools: list[str],
    fallback: list[str],
) -> list[str]:
    """把 Supervisor 输出规整成 canonical tool names；解析失败时走完整可靠链路。"""
    canonical_tools = {
        "vision agent": "Vision Agent",
        "retrieval agent": "Retrieval Agent",
        "reasoning agent": "Reasoning Agent",
        "verifier agent": "Verifier Agent",
        "answer agent": "Answer Agent",
    }
    normalized: list[str] = []
    for tool in selected_tools:
        key = " ".join(str(tool).strip().lower().split())
        canonical = canonical_tools.get(key)
        if canonical and canonical not in normalized:
            normalized.append(canonical)
    if normalized:
        if "Answer Agent" not in normalized:
            normalized.append("Answer Agent")
        return normalized
    return list(fallback)


def _tool_selected(selected_tools: list[str], tool_name: str) -> bool:
    """判断 Supervisor 是否选择了某个 agent/tool。"""
    return tool_name in selected_tools


def _planned_route_from_tools(
    selected_tools: list[str],
    can_retrieve: bool,
) -> list[str]:
    """把 Supervisor 的工具选择转换成本次应执行的 agent route。"""
    route = ["supervisor_agent"]
    if _tool_selected(selected_tools, "Vision Agent"):
        route.append("vision_agent")
    if _tool_selected(selected_tools, "Retrieval Agent") and can_retrieve:
        route.append("retrieval_agent")
    if _tool_selected(selected_tools, "Reasoning Agent") or (
        _tool_selected(selected_tools, "Verifier Agent") and can_retrieve
    ):
        route.append("reasoning_agent")
    if _tool_selected(selected_tools, "Verifier Agent") and can_retrieve:
        route.append("verifier_agent")
    route.append("answer_agent")
    return route


def _heuristic_selected_tools(
    sample: VQARADSample,
    supervisor_tools: list[str],
) -> list[str]:
    """用问题类型约束工具选择，避免简单视觉题被 RAG/Verifier 干扰。

    这不是训练出来的 router，而是 Phase4/5 的可解释调参开关。它只减少不必要工具，
    不会在 Verifier 之后触发重规划。
    """
    if _is_simple_visual_question(sample):
        return ["Vision Agent", "Answer Agent"]
    if _is_visual_reasoning_question(sample):
        return ["Vision Agent", "Reasoning Agent", "Answer Agent"]
    if _needs_external_medical_knowledge(sample):
        return _ensure_tools(
            supervisor_tools,
            [
                "Vision Agent",
                "Retrieval Agent",
                "Reasoning Agent",
                "Verifier Agent",
                "Answer Agent",
            ],
        )
    return _ensure_tools(
        [tool for tool in supervisor_tools if tool != "Verifier Agent"],
        ["Vision Agent", "Reasoning Agent", "Answer Agent"],
    )


def _is_simple_visual_question(sample: VQARADSample) -> bool:
    """识别适合 Vision + Answer 的简单 closed visual question。"""
    question_type = sample.question_type.upper()
    answer_type = sample.answer_type.upper()
    return answer_type == "CLOSED" and question_type in {
        "PRES",
        "MODALITY",
        "PLANE",
        "ORGAN",
        "COUNT",
        "COLOR",
    }


def _is_visual_reasoning_question(sample: VQARADSample) -> bool:
    """识别需要图像观察和轻量推理、但通常不需要外部知识的问题。"""
    return sample.question_type.upper() in {"POS", "SIZE", "ATTRIB"}


def _needs_external_medical_knowledge(sample: VQARADSample) -> bool:
    """识别更可能需要外部医学知识的问题。"""
    question = sample.question.lower()
    if sample.question_type.upper() in {"ABN", "OTHER"}:
        return True
    knowledge_markers = (
        "diagnosis",
        "diagnostic",
        "disease",
        "cause",
        "represent",
        "compatible",
        "suggest",
        "consistent with",
        "abnormality",
        "abnormal",
        "pathology",
    )
    return any(marker in question for marker in knowledge_markers)


def _ensure_tools(tools: list[str], fallback: list[str]) -> list[str]:
    """确保工具列表非空且包含 Answer Agent。"""
    selected = [tool for tool in tools if tool in SupervisorMultiAgent.expected_selected_tools]
    if not selected:
        selected = list(fallback)
    if "Answer Agent" not in selected:
        selected.append("Answer Agent")
    return selected


def _fixed_supervisor_route(
    can_retrieve: bool,
    internal_verifier: bool,
) -> list[str]:
    """生成非动态路由下的 Supervisor 固定执行链路。"""
    route = ["supervisor_agent", "vision_agent"]
    if can_retrieve:
        route.append("retrieval_agent")
    route.append("reasoning_agent")
    if can_retrieve and internal_verifier:
        route.append("verifier_agent")
    route.append("answer_agent")
    return route


def _build_answer_gate(
    claim_statuses: list[dict],
    verification_status: str = "",
    enabled: bool = True,
) -> dict:
    """根据结构化 claim status 生成确定性答案门控。

    HYPOTHESIS 只表示候选解释，不触发硬拦截；UNSUPPORTED / CONTRADICTED 会阻止
    Answer Agent 把对应 claim 当作最终事实输出。
    """
    if not enabled:
        return {
            "decision": "DISABLED",
            "forced_prediction": None,
            "blocked_claims": [],
            "reason": "Deterministic answer gate is disabled for this experiment.",
        }

    blocked_statuses = {"UNSUPPORTED", "CONTRADICTED"}
    blocked_claims = [
        {
            "claim": str(item.get("claim", "")).strip(),
            "status": normalize_claim_status(str(item.get("status", ""))),
            "source_agent": str(item.get("source_agent", "")),
        }
        for item in claim_statuses
        if normalize_claim_status(str(item.get("status", ""))) in blocked_statuses
        and str(item.get("claim", "")).strip()
    ]
    normalized_verification = verification_status.strip().upper()
    if normalized_verification in blocked_statuses and not blocked_claims:
        blocked_claims.append(
            {
                "claim": "final reasoning claim",
                "status": normalized_verification,
                "source_agent": "verifier_agent",
            }
        )

    if not blocked_claims:
        return {
            "decision": "ALLOW",
            "forced_prediction": None,
            "blocked_claims": [],
            "reason": "No UNSUPPORTED or CONTRADICTED claim was found.",
        }

    most_severe_status = (
        "CONTRADICTED"
        if any(item["status"] == "CONTRADICTED" for item in blocked_claims)
        else "UNSUPPORTED"
    )
    return {
        "decision": f"RESTRICT_{most_severe_status}",
        "forced_prediction": "uncertain",
        "blocked_claims": blocked_claims,
        "reason": (
            "At least one claim needed by the final answer is unsupported or "
            "contradicted by verification."
        ),
    }


def _format_answer_gate(answer_gate: dict) -> str:
    """生成写入 shared state 的 gate 摘要。"""
    blocked_claim_lines = [
        f"- [{item['status']}] {item['claim']} (source={item.get('source_agent', '')})"
        for item in answer_gate.get("blocked_claims", [])
    ]
    blocked_claims = "\n".join(blocked_claim_lines) or "None"
    forced_prediction = answer_gate.get("forced_prediction") or "None"
    return (
        f"Answer Gate: {answer_gate.get('decision', 'ALLOW')}\n"
        f"Blocked Claims:\n{blocked_claims}\n"
        f"Forced Prediction: {forced_prediction}\n"
        f"Instruction: do not assert unsupported or contradicted claims."
    )


def _apply_answer_gate(prediction: str, answer_gate: dict) -> str:
    """在模型生成后执行硬门控，保证最终 prediction 不绕过 Verifier。"""
    forced_prediction = answer_gate.get("forced_prediction")
    if forced_prediction:
        return str(forced_prediction)
    return prediction


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
