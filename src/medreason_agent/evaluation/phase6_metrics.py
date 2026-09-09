"""Phase 6 adaptive routing metrics."""

from __future__ import annotations

from collections import Counter


def summarize_phase6_metrics(records: list[dict]) -> dict[str, object]:
    """汇总 route distribution、fallback rate 和每条路线的 accuracy。"""
    total = len(records)
    if total == 0:
        return {
            "route_distribution": {},
            "complexity_distribution": {},
            "fallback_rate": 0.0,
            "route_accuracy": {},
            "mean_agent_calls": 0.0,
        }

    route_counts = Counter(str(record.get("phase6_route", "")) for record in records)
    complexity_counts = Counter(
        str(record.get("phase6_routing_decision", {}).get("complexity", ""))
        for record in records
    )
    fallback_count = sum(bool(record.get("phase6_fallback_applied")) for record in records)
    route_accuracy = {}
    for route, count in sorted(route_counts.items()):
        routed = [record for record in records if record.get("phase6_route") == route]
        route_accuracy[route] = {
            "num_samples": count,
            "accuracy": sum(bool(record.get("correct")) for record in routed) / count,
        }

    agent_call_counts = [len(record.get("tool_calls", [])) for record in records]
    return {
        "route_distribution": dict(sorted(route_counts.items())),
        "complexity_distribution": dict(sorted(complexity_counts.items())),
        "fallback_rate": fallback_count / total,
        "fallback_count": fallback_count,
        "route_accuracy": route_accuracy,
        "mean_agent_calls": sum(agent_call_counts) / total,
    }
