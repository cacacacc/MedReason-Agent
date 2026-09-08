"""Claim Status 结构化工具。

这个模块把模型输出里的医学信息拆成可审计的 claim record，并给每条信息标注状态。
核心目的：防止系统把 `HYPOTHESIS` 当成已经由图像或外部证据确认的事实。
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

CLAIM_STATUSES = (
    "OBSERVED",
    "SUPPORTED",
    "HYPOTHESIS",
    "UNSUPPORTED",
    "CONTRADICTED",
)
VERIFICATION_CLAIM_STATUSES = ("SUPPORTED", "UNSUPPORTED", "CONTRADICTED")

_CLAIM_STATUSES_SECTION_PATTERN = re.compile(
    r"claim\s+statuses?\s*:\s*(?P<body>.*?)(?:\n[A-Z][A-Za-z ]{1,40}:|\Z)",
    flags=re.IGNORECASE | re.DOTALL,
)
_JSON_OBJECT_PATTERN = re.compile(r"\{.*?\}")
_STATUS_FIRST_PATTERN = re.compile(
    r"^(?P<status>OBSERVED|SUPPORTED|HYPOTHESIS|UNSUPPORTED|CONTRADICTED)\s*[:|-]\s*(?P<claim>.+)$",
    flags=re.IGNORECASE,
)
_STATUS_LAST_PATTERN = re.compile(
    r"^(?P<claim>.+?)\s*(?:\||\(|\[)\s*(?P<status>OBSERVED|SUPPORTED|HYPOTHESIS|UNSUPPORTED|CONTRADICTED)\s*[\])]*$",
    flags=re.IGNORECASE,
)


def normalize_claim_status(status: str, default: str = "HYPOTHESIS") -> str:
    """把模型输出的状态标准化为固定枚举。

    解析失败时默认回到 `HYPOTHESIS`，因为未知状态不能被当成事实。
    """
    normalized = status.strip().upper()
    if normalized in CLAIM_STATUSES:
        return normalized
    return default


def build_claim_record(
    claim: str,
    status: str,
    source_agent: str = "",
    evidence_ids: list[str] | None = None,
) -> dict[str, Any]:
    """构造统一的 claim status record。"""
    record: dict[str, Any] = {
        "claim": claim.strip(),
        "status": normalize_claim_status(status),
    }
    if source_agent:
        record["source_agent"] = source_agent
    if evidence_ids:
        record["evidence_ids"] = evidence_ids
    return record


def parse_claim_status_lines(output: str, source_agent: str = "") -> list[dict[str, Any]]:
    """从模型输出解析 `Claim Statuses:` 段落。

    推荐格式是 JSONL：
    `{"claim": "right lung opacity", "status": "OBSERVED"}`

    为了兼容大模型格式漂移，也支持 `SUPPORTED: claim` 和
    `claim | SUPPORTED` 这类轻量文本格式。
    """
    body = _extract_claim_status_body(output)
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for line in body.splitlines():
        line = line.strip().lstrip("-*0123456789. )").strip()
        if not line:
            continue

        parsed = _parse_json_claim_status(line) or _parse_text_claim_status(line)
        if parsed is None:
            continue

        claim = parsed["claim"].strip()
        if not claim:
            continue
        status = normalize_claim_status(parsed["status"])
        key = (claim.lower(), status)
        if key in seen:
            continue
        seen.add(key)
        records.append(
            build_claim_record(
                claim=claim,
                status=status,
                source_agent=str(parsed.get("source_agent") or source_agent),
                evidence_ids=parsed.get("evidence_ids"),
            )
        )
    return records


def claims_to_status_records(
    claims: list[str],
    status: str,
    source_agent: str = "",
) -> list[dict[str, Any]]:
    """把旧版 `generated_claims` 转成带状态的结构化记录。"""
    return [
        build_claim_record(claim=claim, status=status, source_agent=source_agent)
        for claim in claims
        if claim.strip()
    ]


def summarize_claim_statuses(records: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总 prediction records 中的 claim status 分布。"""
    counts: Counter[str] = Counter()
    for record in records:
        for claim_record in record.get("claim_statuses", []):
            counts[normalize_claim_status(str(claim_record.get("status", "")))] += 1

    return {
        "num_claim_status_records": sum(counts.values()),
        "claim_status_counts": {
            status: counts.get(status, 0)
            for status in CLAIM_STATUSES
        },
    }


def has_unsupported_or_contradicted_claim(record: dict[str, Any]) -> bool:
    """判断结构化 claims 中是否包含未支持或被矛盾的医学信息。"""
    return any(
        normalize_claim_status(str(claim_record.get("status", "")))
        in {"UNSUPPORTED", "CONTRADICTED"}
        for claim_record in record.get("claim_statuses", [])
    )


def _extract_claim_status_body(output: str) -> str:
    """优先解析 `Claim Statuses:` 段落；没有该段落时回退全文扫描。"""
    match = _CLAIM_STATUSES_SECTION_PATTERN.search(output.strip())
    if not match:
        return output
    return match.group("body")


def _parse_json_claim_status(line: str) -> dict[str, Any] | None:
    """解析单行 JSON claim status。"""
    match = _JSON_OBJECT_PATTERN.search(line)
    if not match:
        return None
    try:
        item = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(item, dict):
        return None
    if "claim" not in item or "status" not in item:
        return None
    return {
        "claim": str(item["claim"]),
        "status": str(item["status"]),
        "source_agent": str(item.get("source_agent", "")),
        "evidence_ids": item.get("evidence_ids"),
    }


def _parse_text_claim_status(line: str) -> dict[str, str] | None:
    """解析非 JSON 的简写 claim status 行。"""
    match = _STATUS_FIRST_PATTERN.match(line) or _STATUS_LAST_PATTERN.match(line)
    if not match:
        return None
    return {
        "claim": match.group("claim").strip(),
        "status": match.group("status").strip(),
    }
