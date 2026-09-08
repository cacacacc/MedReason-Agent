"""Phase 4 agent 协作指标。

这些指标是第一版规则评估，目的是让 Phase 4 先形成可复现的结构化结果。
后续可以用 GPT-based evaluator 或人工标注替换 hallucination / reasoning 评分。
"""

from __future__ import annotations

from collections import Counter

from medreason_agent.evaluation.claim_status import has_unsupported_or_contradicted_claim


def reasoning_score(record: dict) -> float:
    """根据关键 agent 输出是否完整给出 0-1 的规则版 Reasoning Score。"""
    agent_outputs = record.get("agent_outputs", {})
    checks = [
        bool(str(agent_outputs.get("vision", "")).strip()),
        "Reasoning:" in str(agent_outputs.get("reasoning", "")),
        "Claims:" in str(agent_outputs.get("reasoning", "")),
        bool(str(record.get("prediction", "")).strip()),
    ]
    return sum(checks) / len(checks)


def has_potential_hallucination(record: dict) -> bool:
    """启发式判断是否存在潜在 hallucination。

    如果 critic/verifier 明确拒绝、矛盾，或者 unsupported medical claims 不为 None，
    就记为潜在 hallucination。它不是最终医学事实判断，只用于早期误差分析。
    """
    critic_decision = str(record.get("critic_decision", "")).upper()
    verification_status = str(record.get("claim_verification_status", "")).upper()
    combined_outputs = "\n".join(
        str(value) for value in record.get("agent_outputs", {}).values()
    ).lower()

    unsupported_claims = (
        "unsupported medical claims:" in combined_outputs
        and "unsupported medical claims: none" not in combined_outputs
    )
    unsupported_assumptions = (
        "unsupported assumptions:" in combined_outputs
        and "unsupported assumptions: none" not in combined_outputs
    )
    return (
        critic_decision in {"REVISE", "REJECT"}
        or verification_status in {"UNSUPPORTED", "CONTRADICTED"}
        or has_unsupported_or_contradicted_claim(record)
        or unsupported_claims
        or unsupported_assumptions
    )


def tool_selection_accuracy(record: dict) -> float | None:
    """计算 Supervisor 工具选择是否等于 expected tools。

    如果记录里没有工具选择字段，则回退比较实际 route。
    """
    expected_tools = record.get("expected_selected_tools")
    selected_tools = record.get("selected_tools")
    if expected_tools is not None and selected_tools is not None:
        if not expected_tools:
            return None
        return 1.0 if selected_tools == expected_tools else 0.0

    expected_route = record.get("expected_agent_route")
    selected_route = record.get("agent_route")
    if not expected_route:
        return None
    return 1.0 if selected_route == expected_route else 0.0


def summarize_agent_metrics(records: list[dict]) -> dict[str, float | int | None]:
    """汇总 Phase 4 agent 指标。"""
    total = len(records)
    if total == 0:
        return {
            "mean_reasoning_score": 0.0,
            "hallucination_rate": 0.0,
            "mean_tool_selection_accuracy": None,
        }

    reasoning_scores = [reasoning_score(record) for record in records]
    hallucination_flags = [has_potential_hallucination(record) for record in records]
    tool_scores = [
        score for record in records if (score := tool_selection_accuracy(record)) is not None
    ]
    compression_ratios = [
        float(ratio)
        for record in records
        if (ratio := record.get("state_compression", {}).get("compression_ratio")) is not None
    ]
    memory_counts = [len(record.get("memory_records", [])) for record in records]
    gate_counts = Counter(
        str(record.get("answer_gate", {}).get("decision", "MISSING") or "MISSING")
        for record in records
    )
    return {
        "mean_reasoning_score": sum(reasoning_scores) / total,
        "hallucination_rate": sum(hallucination_flags) / total,
        "mean_tool_selection_accuracy": (
            sum(tool_scores) / len(tool_scores) if tool_scores else None
        ),
        "mean_state_compression_ratio": (
            sum(compression_ratios) / len(compression_ratios)
            if compression_ratios
            else None
        ),
        "mean_persistent_memory_hits": sum(memory_counts) / total,
        "answer_gate_counts": dict(sorted(gate_counts.items())),
    }
