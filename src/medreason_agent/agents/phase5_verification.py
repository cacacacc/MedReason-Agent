"""Phase 5 Verification + Self-Reflection agent。

本模块不重新执行 Vision / Retrieval / Supervisor。它先复用 Supervisor Multi-Agent
生成 candidate reasoning + answer，然后只在 candidate 之后追加 process-level check：
Self-Reflection 由原 Reasoning Agent 自检；Separate Verifier 由独立 Verifier prompt
检查，revision 仍由 Reasoning Agent 完成。
"""

from __future__ import annotations

from dataclasses import dataclass

from medreason_agent.agents.memory import PersistentAgentMemoryStore
from medreason_agent.agents.multi_agent import MultiAgentResult, SupervisorMultiAgent
from medreason_agent.agents.rag_agents import RetrievalPipeline
from medreason_agent.models.vlm import VLMBackend, VLMRequest
from medreason_agent.prompts.multi_agent import extract_final_answer
from medreason_agent.prompts.phase5_verification import (
    build_outcome_verifier_prompt,
    build_self_reflection_prompt,
    build_self_revision_prompt,
    build_separate_verifier_prompt,
    build_verifier_revision_prompt,
    parse_process_feedback,
)
from medreason_agent.prompts.rag import format_evidence_block

PHASE5_MODES = (
    "supervisor_no_critic",
    "self_reflection",
    "outcome_verifier",
    "separate_verifier",
)


@dataclass(frozen=True)
class Phase5VerificationResult:
    """单条样本的 Phase 5 输出。"""

    candidate: MultiAgentResult
    prediction: str
    raw_output: str
    verification_mode: str
    process_feedback_output: str
    process_feedback: dict
    revision_output: str
    revision_applied: bool
    selective_verification_applied: bool
    tool_calls: list[dict]
    input_tokens: int | None
    output_tokens: int | None


class Phase5VerificationAgent:
    """Phase 5 后验验证 / 自反思实验执行器。"""

    def __init__(
        self,
        backend: VLMBackend,
        retrieval_pipeline: RetrievalPipeline | None = None,
        memory_store: PersistentAgentMemoryStore | None = None,
        verification_mode: str = "supervisor_no_critic",
        dynamic_routing: bool = False,
        deterministic_answer_gate: bool = False,
        selective_verification: str = "all",
        question_routing: str = "none",
    ) -> None:
        if verification_mode not in PHASE5_MODES:
            raise ValueError(f"Unsupported Phase 5 verification_mode: {verification_mode}")
        self.backend = backend
        self.verification_mode = verification_mode
        self.selective_verification = selective_verification
        self.candidate_agent = SupervisorMultiAgent(
            backend=backend,
            retrieval_pipeline=retrieval_pipeline,
            memory_store=memory_store,
            dynamic_routing=dynamic_routing,
            deterministic_answer_gate=deterministic_answer_gate,
            internal_verifier=False,
            question_routing=question_routing,
        )

    def run(
        self,
        sample,
        max_new_tokens: int,
        experiment_id: str = "",
    ) -> Phase5VerificationResult:
        """生成 candidate 后执行 Phase 5 process-level verification。"""
        candidate = self.candidate_agent.run(
            sample=sample,
            max_new_tokens=max_new_tokens,
            experiment_id=experiment_id,
        )
        if self.verification_mode == "supervisor_no_critic":
            return self._without_post_critic(candidate)

        if not self._should_verify(candidate):
            return self._without_post_critic(candidate, selective_skipped=True)

        if self.verification_mode == "self_reflection":
            return self._run_self_reflection(sample, candidate, max_new_tokens)
        if self.verification_mode == "outcome_verifier":
            return self._run_outcome_verifier(sample, candidate, max_new_tokens)
        return self._run_separate_verifier(sample, candidate, max_new_tokens)

    def _without_post_critic(
        self,
        candidate: MultiAgentResult,
        selective_skipped: bool = False,
    ) -> Phase5VerificationResult:
        """不追加 candidate 后 Critic，作为 Phase 5 ablation baseline。"""
        return Phase5VerificationResult(
            candidate=candidate,
            prediction=candidate.prediction,
            raw_output=candidate.raw_output,
            verification_mode=self.verification_mode,
            process_feedback_output="",
            process_feedback={
                "decision": "SKIP" if selective_skipped else "NONE",
                "issues": [],
                "revision_instruction": "",
                "raw": {},
            },
            revision_output="",
            revision_applied=False,
            selective_verification_applied=False,
            tool_calls=[],
            input_tokens=0,
            output_tokens=0,
        )

    def _run_self_reflection(
        self,
        sample,
        candidate: MultiAgentResult,
        max_new_tokens: int,
    ) -> Phase5VerificationResult:
        """由同一个 Reasoning Agent 自检并按需修正一次。"""
        prompt = build_self_reflection_prompt(
            question=sample.question,
            vision_findings=candidate.agent_outputs.get("vision", ""),
            retrieved_evidence=format_evidence_block(candidate.retrieved_evidence),
            reasoning_trace=candidate.reasoning_output,
            candidate_answer=candidate.prediction,
            confidence=candidate.confidence,
        )
        reflection = self._generate(sample, prompt, max_new_tokens)
        feedback = parse_process_feedback(reflection.raw_output, keep_decision="KEEP")
        revision = None
        prediction = candidate.prediction
        if feedback["decision"] == "REVISE":
            revision = self._generate(
                sample,
                build_self_revision_prompt(
                    question=sample.question,
                    vision_findings=candidate.agent_outputs.get("vision", ""),
                    retrieved_evidence=format_evidence_block(candidate.retrieved_evidence),
                    reasoning_trace=candidate.reasoning_output,
                    candidate_answer=candidate.prediction,
                    feedback=reflection.raw_output,
                    confidence=candidate.confidence,
                ),
                max_new_tokens,
            )
            prediction = extract_final_answer(revision.raw_output, question=sample.question)

        return Phase5VerificationResult(
            candidate=candidate,
            prediction=prediction,
            raw_output=revision.raw_output if revision is not None else candidate.raw_output,
            verification_mode=self.verification_mode,
            process_feedback_output=reflection.raw_output,
            process_feedback=feedback,
            revision_output=revision.raw_output if revision is not None else "",
            revision_applied=revision is not None,
            selective_verification_applied=True,
            tool_calls=[
                {"tool": "vlm_generate", "stage": "self_reflection"},
                *(
                    [{"tool": "vlm_generate", "stage": "self_revision"}]
                    if revision is not None
                    else []
                ),
            ],
            input_tokens=_sum_optional(
                reflection.input_tokens,
                _optional_token(revision, "input_tokens"),
            ),
            output_tokens=_sum_optional(
                reflection.output_tokens,
                _optional_token(revision, "output_tokens"),
            ),
        )

    def _run_separate_verifier(
        self,
        sample,
        candidate: MultiAgentResult,
        max_new_tokens: int,
    ) -> Phase5VerificationResult:
        """由独立 Verifier 检查，Reasoning Agent 根据 feedback 修正一次。"""
        prompt = build_separate_verifier_prompt(
            question=sample.question,
            vision_findings=candidate.agent_outputs.get("vision", ""),
            retrieved_evidence=format_evidence_block(candidate.retrieved_evidence),
            reasoning_trace=candidate.reasoning_output,
            candidate_answer=candidate.prediction,
            confidence=candidate.confidence,
        )
        verifier = self._generate(sample, prompt, max_new_tokens)
        feedback = parse_process_feedback(verifier.raw_output, keep_decision="PASS")
        revision = None
        prediction = candidate.prediction
        if feedback["decision"] == "REVISE":
            revision = self._generate(
                sample,
                build_verifier_revision_prompt(
                    question=sample.question,
                    vision_findings=candidate.agent_outputs.get("vision", ""),
                    retrieved_evidence=format_evidence_block(candidate.retrieved_evidence),
                    reasoning_trace=candidate.reasoning_output,
                    candidate_answer=candidate.prediction,
                    feedback=verifier.raw_output,
                    confidence=candidate.confidence,
                ),
                max_new_tokens,
            )
            prediction = extract_final_answer(revision.raw_output, question=sample.question)

        return Phase5VerificationResult(
            candidate=candidate,
            prediction=prediction,
            raw_output=revision.raw_output if revision is not None else candidate.raw_output,
            verification_mode=self.verification_mode,
            process_feedback_output=verifier.raw_output,
            process_feedback=feedback,
            revision_output=revision.raw_output if revision is not None else "",
            revision_applied=revision is not None,
            selective_verification_applied=True,
            tool_calls=[
                {"tool": "vlm_generate", "stage": "separate_process_verifier"},
                *(
                    [{"tool": "vlm_generate", "stage": "verifier_guided_revision"}]
                    if revision is not None
                    else []
                ),
            ],
            input_tokens=_sum_optional(
                verifier.input_tokens,
                _optional_token(revision, "input_tokens"),
            ),
            output_tokens=_sum_optional(
                verifier.output_tokens,
                _optional_token(revision, "output_tokens"),
            ),
        )

    def _run_outcome_verifier(
        self,
        sample,
        candidate: MultiAgentResult,
        max_new_tokens: int,
    ) -> Phase5VerificationResult:
        """只看 Question + Candidate Answer 的 outcome-level verifier。"""
        prompt = build_outcome_verifier_prompt(
            question=sample.question,
            candidate_answer=candidate.prediction,
            confidence=candidate.confidence,
        )
        verifier = self._generate(sample, prompt, max_new_tokens)
        feedback = parse_process_feedback(verifier.raw_output, keep_decision="PASS")
        revision = None
        prediction = candidate.prediction
        if feedback["decision"] == "REVISE":
            revision = self._generate(
                sample,
                build_verifier_revision_prompt(
                    question=sample.question,
                    vision_findings=candidate.agent_outputs.get("vision", ""),
                    retrieved_evidence=format_evidence_block(candidate.retrieved_evidence),
                    reasoning_trace=candidate.reasoning_output,
                    candidate_answer=candidate.prediction,
                    feedback=verifier.raw_output,
                    confidence=candidate.confidence,
                ),
                max_new_tokens,
            )
            prediction = extract_final_answer(revision.raw_output, question=sample.question)

        return Phase5VerificationResult(
            candidate=candidate,
            prediction=prediction,
            raw_output=revision.raw_output if revision is not None else candidate.raw_output,
            verification_mode=self.verification_mode,
            process_feedback_output=verifier.raw_output,
            process_feedback=feedback,
            revision_output=revision.raw_output if revision is not None else "",
            revision_applied=revision is not None,
            selective_verification_applied=True,
            tool_calls=[
                {"tool": "vlm_generate", "stage": "outcome_verifier"},
                *(
                    [{"tool": "vlm_generate", "stage": "outcome_guided_revision"}]
                    if revision is not None
                    else []
                ),
            ],
            input_tokens=_sum_optional(
                verifier.input_tokens,
                _optional_token(revision, "input_tokens"),
            ),
            output_tokens=_sum_optional(
                verifier.output_tokens,
                _optional_token(revision, "output_tokens"),
            ),
        )

    def _should_verify(self, candidate: MultiAgentResult) -> bool:
        """实现 selective verification 策略。主实验默认 all。"""
        if self.selective_verification == "all":
            return True
        if self.selective_verification == "none":
            return False
        if self.selective_verification == "unsupported_only":
            return any(
                item.get("status") in {"UNSUPPORTED", "CONTRADICTED"}
                for item in candidate.claim_statuses
            )
        if self.selective_verification == "risk_rule":
            return _has_candidate_risk(candidate)
        raise ValueError(f"Unsupported selective_verification: {self.selective_verification}")

    def _generate(self, sample, prompt: str, max_new_tokens: int):
        """Phase 5 的额外调用仍走同一个 frozen VLM backend。"""
        return self.backend.generate(
            VLMRequest(
                image_path=str(sample.absolute_image_path),
                question=sample.question,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
            )
        )


def _optional_token(response, attribute: str) -> int | None:
    """读取可选 VLMResponse token 字段。"""
    if response is None:
        return 0
    return getattr(response, attribute)


def _sum_optional(*values: int | None) -> int | None:
    """汇总 token；任一真实调用缺 token 时返回 None。"""
    if any(value is None for value in values):
        return None
    return sum(int(value) for value in values)


def _has_candidate_risk(candidate: MultiAgentResult) -> bool:
    """规则版 selective verification gate，不触发上游重跑。"""
    if candidate.confidence is not None and float(candidate.confidence) < 0.6:
        return True
    if candidate.claim_verification_status in {"UNSUPPORTED", "CONTRADICTED"}:
        return True
    if any(
        item.get("status") in {"UNSUPPORTED", "CONTRADICTED"}
        for item in candidate.claim_statuses
    ):
        return True
    if (
        _route_contains(candidate.agent_route, "retrieval_agent")
        and not candidate.retrieved_evidence
    ):
        return True
    reasoning = candidate.reasoning_output.lower()
    if any(marker in reasoning for marker in ("uncertain", "insufficient", "limited", "cannot")):
        return True
    if candidate.answer_gate.get("decision") in {"REVISED", "FALLBACK"}:
        return True
    return bool(
        candidate.shared_state.get("state_compression", {}).get("compression_ratio", 0) > 0.9
    )


def _route_contains(route: list[str], stage: str) -> bool:
    """检查 candidate route 是否包含某个阶段。"""
    return stage in {str(item) for item in route}
