"""错误归因规则。

第一版使用可解释的启发式规则，把错误样本归到用户定义的六类 error type。
这不是最终医学判读，而是用于 Phase 3 / Phase 4 的自动误差分析和人工复核抽样。
"""

from __future__ import annotations

from collections import Counter
from typing import Any

ERROR_TYPES = (
    "Perception Error",
    "Retrieval Error",
    "Reasoning Error",
    "Verification Error",
    "Routing Error",
    "State Error",
)


def attribute_error(record: dict[str, Any], evidence_threshold: float = 0.2) -> dict[str, Any]:
    """对单条 prediction record 做错误归因。

    正确样本不归因。错误样本会返回一个 primary error type，同时保留所有触发的
    signals，方便后续人工复核。
    """
    if bool(record.get("correct")):
        return {
            "primary_error_type": "",
            "candidate_error_types": [],
            "signals": {},
            "requires_human_review": False,
        }

    signals = {
        "routing_error": _has_routing_error(record),
        "state_error": _has_state_error(record),
        "retrieval_error": _has_retrieval_error(record, evidence_threshold),
        "verification_error": _has_verification_error(record),
        "perception_error": _has_perception_error(record),
    }

    candidate_error_types: list[str] = []
    if signals["routing_error"]:
        candidate_error_types.append("Routing Error")
    if signals["state_error"]:
        candidate_error_types.append("State Error")
    if signals["retrieval_error"]:
        candidate_error_types.append("Retrieval Error")
    if signals["verification_error"]:
        candidate_error_types.append("Verification Error")
    if signals["perception_error"]:
        candidate_error_types.append("Perception Error")
    if not candidate_error_types:
        candidate_error_types.append("Reasoning Error")

    return {
        "primary_error_type": candidate_error_types[0],
        "candidate_error_types": candidate_error_types,
        "signals": signals,
        "requires_human_review": _requires_human_review(candidate_error_types),
    }


def summarize_error_attribution(records: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总错误归因分布。"""
    incorrect_records = [record for record in records if not bool(record.get("correct"))]
    primary_counts: Counter[str] = Counter()
    candidate_counts: Counter[str] = Counter()
    human_review_count = 0

    for record in incorrect_records:
        attribution = record.get("error_attribution") or attribute_error(record)
        primary = str(attribution.get("primary_error_type", "")) or "UNKNOWN"
        primary_counts[primary] += 1
        for error_type in attribution.get("candidate_error_types", []):
            candidate_counts[str(error_type)] += 1
        if attribution.get("requires_human_review"):
            human_review_count += 1

    return {
        "error_type_counts": {
            error_type: primary_counts.get(error_type, 0)
            for error_type in ERROR_TYPES
        },
        "candidate_error_type_counts": {
            error_type: candidate_counts.get(error_type, 0)
            for error_type in ERROR_TYPES
        },
        "unknown_error_count": primary_counts.get("UNKNOWN", 0),
        "human_review_error_count": human_review_count,
        "num_incorrect": len(incorrect_records),
    }


def _has_routing_error(record: dict[str, Any]) -> bool:
    """判断 Supervisor 是否选错工具或实际 agent route 偏离期望。"""
    expected_tools = record.get("expected_selected_tools")
    selected_tools = record.get("selected_tools")
    if expected_tools and selected_tools is not None and selected_tools != expected_tools:
        return True

    expected_route = record.get("expected_agent_route")
    agent_route = record.get("agent_route")
    if expected_route and agent_route and agent_route != expected_route:
        return True
    return False


def _has_state_error(record: dict[str, Any]) -> bool:
    """判断 shared state 是否缺失、压缩异常或和顶层字段不一致。"""
    shared_state = record.get("shared_state")
    if shared_state is None:
        return False
    if not isinstance(shared_state, dict) or not shared_state:
        return True

    compressed_context = str(shared_state.get("compressed_context", "")).strip()
    messages = shared_state.get("messages", [])
    if not compressed_context or not messages:
        return True

    compression_ratio = record.get("state_compression", {}).get("compression_ratio")
    if compression_ratio is not None and float(compression_ratio) > 1.0:
        return True

    top_claims = record.get("claim_statuses", [])
    state_claims = shared_state.get("claim_statuses", [])
    if top_claims and not state_claims:
        return True
    return False


def _has_retrieval_error(record: dict[str, Any], evidence_threshold: float) -> bool:
    """判断检索是否缺失或证据质量过低。"""
    retrieved_evidence = record.get("retrieved_evidence", [])
    evidence_quality = record.get("evidence_quality", {})
    has_retrieval_route = _route_contains(record, "retriever") or _route_contains(
        record,
        "retrieval_agent",
    )
    if not has_retrieval_route and not retrieved_evidence:
        return False
    if not retrieved_evidence:
        return True

    score = record.get("evidence_quality_score")
    if score is None:
        score = evidence_quality.get("evidence_quality_score")
    return score is not None and float(score) < evidence_threshold


def _has_verification_error(record: dict[str, Any]) -> bool:
    """判断 Verifier 是否明显支持了错误答案，或验证状态和最终输出冲突。"""
    verification_status = str(record.get("claim_verification_status", "")).upper()
    if verification_status == "SUPPORTED":
        return True
    if verification_status in {"UNSUPPORTED", "CONTRADICTED"}:
        prediction = str(record.get("prediction", "")).strip().lower()
        verified_claim = str(record.get("verified_claim", "")).strip().lower()
        return bool(prediction and verified_claim and prediction == verified_claim)
    return False


def _has_perception_error(record: dict[str, Any]) -> bool:
    """判断是否可能是图像观察阶段错误。

    自动系统无法真正看懂图像是否被看错，因此这里只标注明显的 perception 信号：
    Vision Agent 缺失观察，或没有任何 OBSERVED claim。
    """
    agent_outputs = record.get("agent_outputs", {})
    vision_output = str(agent_outputs.get("vision", "")).strip()
    has_vision_route = _route_contains(record, "vision_agent")
    if has_vision_route and not vision_output:
        return True

    if has_vision_route:
        claim_statuses = record.get("claim_statuses", [])
        has_observed_claim = any(
            str(item.get("status", "")).upper() == "OBSERVED"
            for item in claim_statuses
        )
        if not has_observed_claim:
            return True
    return False


def _route_contains(record: dict[str, Any], stage: str) -> bool:
    """检查 agent route 或 tool calls 是否包含某个阶段。"""
    route = [str(item) for item in record.get("agent_route", [])]
    tool_stages = [
        str(item.get("stage", ""))
        for item in record.get("tool_calls", [])
        if isinstance(item, dict)
    ]
    return stage in route or stage in tool_stages


def _requires_human_review(candidate_error_types: list[str]) -> bool:
    """多信号冲突或 perception/verification 问题更需要人工复核。"""
    return len(candidate_error_types) > 1 or any(
        error_type in {"Perception Error", "Verification Error"}
        for error_type in candidate_error_types
    )
