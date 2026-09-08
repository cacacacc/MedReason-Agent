"""Multi-Agent 共享状态与 State Compression。

Phase 4 的 agent 不再只把上一轮 raw text 直接塞给下一轮，而是共同读写一个
`SharedAgentState`。状态里保留完整 agent 输出用于审计，同时生成压缩上下文给下游
agent 使用，减少 prompt 膨胀和未验证假设污染。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from medreason_agent.evaluation.claim_status import normalize_claim_status

STATE_COMPRESSION_METHOD = "section_extract_v1"

_SECTION_PATTERN_TEMPLATE = (
    r"{name}\s*:\s*(?P<body>.*?)(?:\n[A-Z][A-Za-z ]{{1,40}}:|\Z)"
)


@dataclass
class AgentStateMessage:
    """单个 agent 写入 shared state 的消息。"""

    agent_name: str
    raw_output: str
    compressed_output: str
    output_chars: int
    compressed_chars: int


@dataclass
class SharedAgentState:
    """Phase 4 单样本级共享状态。

    这个状态只在一个 VQA sample 内生效，不跨样本持久化，因此属于 short-term
    memory。它记录 agent 之间共享的关键事实、假设、证据和压缩上下文。
    """

    question: str
    max_chars_per_agent: int = 700
    max_evidence_items: int = 3
    max_evidence_chars: int = 220
    messages: list[AgentStateMessage] = field(default_factory=list)
    claim_statuses: list[dict[str, Any]] = field(default_factory=list)
    retrieved_evidence: list[dict[str, Any]] = field(default_factory=list)
    selected_tools: list[str] = field(default_factory=list)
    persistent_memories: list[dict[str, Any]] = field(default_factory=list)

    def set_selected_tools(self, selected_tools: list[str]) -> None:
        """记录 Supervisor 选择的工具。"""
        self.selected_tools = list(selected_tools)

    def add_agent_output(
        self,
        agent_name: str,
        raw_output: str,
        claim_statuses: list[dict[str, Any]] | None = None,
    ) -> None:
        """把 agent 输出写入共享状态，并同步压缩摘要和 claim statuses。"""
        compressed_output = compress_agent_output(
            agent_name=agent_name,
            raw_output=raw_output,
            max_chars=self.max_chars_per_agent,
        )
        self.messages.append(
            AgentStateMessage(
                agent_name=agent_name,
                raw_output=raw_output,
                compressed_output=compressed_output,
                output_chars=len(raw_output),
                compressed_chars=len(compressed_output),
            )
        )
        if claim_statuses:
            self.extend_claim_statuses(claim_statuses)

    def add_retrieved_evidence(self, evidence_records: list[dict], stage: str) -> None:
        """把 Retrieval Agent 的 evidence 写入共享状态。"""
        for record in evidence_records:
            self.retrieved_evidence.append(dict(record, evidence_stage=stage))

    def add_persistent_memories(self, memories: list[dict[str, Any]]) -> None:
        """把跨运行 memory 检索结果写入当前 shared state。"""
        self.persistent_memories = list(memories)

    def extend_claim_statuses(self, claim_statuses: list[dict[str, Any]]) -> None:
        """合并 claim statuses，并按 claim/status/source 去重。"""
        seen = {
            (
                str(item.get("claim", "")).lower(),
                normalize_claim_status(str(item.get("status", ""))),
                str(item.get("source_agent", "")),
            )
            for item in self.claim_statuses
        }
        for item in claim_statuses:
            claim = str(item.get("claim", "")).strip()
            if not claim:
                continue
            status = normalize_claim_status(str(item.get("status", "")))
            source_agent = str(item.get("source_agent", ""))
            key = (claim.lower(), status, source_agent)
            if key in seen:
                continue
            seen.add(key)
            normalized = dict(item, claim=claim, status=status)
            if source_agent:
                normalized["source_agent"] = source_agent
            self.claim_statuses.append(normalized)

    def compressed_context(self) -> str:
        """生成给下游 agent 读取的短上下文。"""
        parts = [f"Question: {self.question}"]
        if self.selected_tools:
            parts.append("Selected Tools: " + ", ".join(self.selected_tools))

        if self.persistent_memories:
            memory_lines = [
                f"[Memory {index}] score={memory.get('score', '')}; "
                f"sample={memory.get('sample_id', '')}\n"
                + str(memory.get("memory_text", ""))
                for index, memory in enumerate(self.persistent_memories, start=1)
            ]
            parts.append("Persistent Memory:\n" + "\n\n".join(memory_lines))

        evidence_summary = self._compressed_evidence()
        if evidence_summary:
            parts.append("Retrieved Evidence:\n" + evidence_summary)

        if self.claim_statuses:
            claim_lines = [
                f"- [{item['status']}] {item['claim']}"
                + (
                    f" (source={item['source_agent']})"
                    if item.get("source_agent")
                    else ""
                )
                for item in self.claim_statuses
            ]
            parts.append("Claim Statuses:\n" + "\n".join(claim_lines))

        if self.messages:
            message_lines = [
                f"{message.agent_name}: {message.compressed_output}"
                for message in self.messages
                if message.compressed_output
            ]
            parts.append("Agent Memory:\n" + "\n".join(message_lines))

        return "\n\n".join(parts)

    def to_record(self) -> dict[str, Any]:
        """把 shared state 转成可写入 JSONL 的结构化记录。"""
        raw_chars = sum(message.output_chars for message in self.messages)
        compressed_chars = sum(message.compressed_chars for message in self.messages)
        compression_ratio = (
            round(compressed_chars / raw_chars, 6)
            if raw_chars > 0
            else None
        )
        return {
            "state_compression": {
                "method": STATE_COMPRESSION_METHOD,
                "max_chars_per_agent": self.max_chars_per_agent,
                "max_evidence_items": self.max_evidence_items,
                "max_evidence_chars": self.max_evidence_chars,
                "raw_agent_output_chars": raw_chars,
                "compressed_agent_output_chars": compressed_chars,
                "compression_ratio": compression_ratio,
            },
            "selected_tools": list(self.selected_tools),
            "persistent_memories": list(self.persistent_memories),
            "claim_statuses": list(self.claim_statuses),
            "retrieved_evidence": list(self.retrieved_evidence),
            "messages": [
                {
                    "agent_name": message.agent_name,
                    "raw_output": message.raw_output,
                    "compressed_output": message.compressed_output,
                    "output_chars": message.output_chars,
                    "compressed_chars": message.compressed_chars,
                }
                for message in self.messages
            ],
            "compressed_context": self.compressed_context(),
        }

    def _compressed_evidence(self) -> str:
        """压缩 evidence，只保留前几条标题、阶段和短文本。"""
        lines: list[str] = []
        for index, evidence in enumerate(
            self.retrieved_evidence[: self.max_evidence_items],
            start=1,
        ):
            title = str(evidence.get("title", "")).strip() or "Untitled"
            stage = str(evidence.get("evidence_stage", "")).strip() or "retrieval"
            text = " ".join(str(evidence.get("text", "")).split())
            if len(text) > self.max_evidence_chars:
                text = text[: self.max_evidence_chars].rstrip() + "..."
            lines.append(f"[{index}] stage={stage}; title={title}; text={text}")
        return "\n".join(lines)


def compress_agent_output(agent_name: str, raw_output: str, max_chars: int = 700) -> str:
    """对 agent raw output 做规则版 State Compression。"""
    sections_by_agent = {
        "supervisor_agent": ["Selected Tools", "Rationale"],
        "vision_agent": ["Observation", "Uncertainty"],
        "reasoning_agent": [
            "Reasoning",
            "Claims",
            "Claim Statuses",
            "Preliminary Answer",
            "Unsupported Assumptions",
        ],
        "critic_agent": [
            "Critique",
            "Critic Decision",
            "Claim Statuses",
            "Unsupported Medical Claims",
        ],
        "verifier_agent": [
            "Verification",
            "Verification Status",
            "Claim Statuses",
            "Unsupported Medical Claims",
        ],
        "answer_agent": ["Conclusion", "Final Answer"],
    }
    selected_sections = sections_by_agent.get(agent_name, [])
    compressed_parts = [
        section
        for section_name in selected_sections
        if (section := extract_section(raw_output, section_name))
    ]
    raw_compact = " ".join(raw_output.split())
    compressed = " | ".join(compressed_parts) or raw_compact
    if len(compressed) > len(raw_compact):
        compressed = raw_compact
    if len(compressed) > max_chars:
        return compressed[:max_chars].rstrip() + "..."
    return compressed


def extract_section(output: str, section_name: str) -> str:
    """从自由文本中抽取指定 section 的短内容。"""
    pattern = re.compile(
        _SECTION_PATTERN_TEMPLATE.format(name=re.escape(section_name)),
        flags=re.IGNORECASE | re.DOTALL,
    )
    match = pattern.search(output.strip())
    if not match:
        return ""
    body = " ".join(match.group("body").split())
    return f"{section_name}: {body}" if body else ""
