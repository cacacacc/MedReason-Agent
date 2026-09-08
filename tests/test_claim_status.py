from medreason_agent.evaluation.claim_status import (
    build_claim_record,
    claims_to_status_records,
    has_unsupported_or_contradicted_claim,
    normalize_claim_status,
    parse_claim_status_lines,
    summarize_claim_statuses,
)


def test_parse_claim_status_jsonl_section() -> None:
    output = (
        "Reasoning: ok\n"
        "Claim Statuses:\n"
        '{"claim": "right lung opacity", "status": "OBSERVED"}\n'
        '{"claim": "possible pneumonia", "status": "HYPOTHESIS"}\n'
        "Unsupported Assumptions: None"
    )

    assert parse_claim_status_lines(output, source_agent="reasoning_agent") == [
        {
            "claim": "right lung opacity",
            "status": "OBSERVED",
            "source_agent": "reasoning_agent",
        },
        {
            "claim": "possible pneumonia",
            "status": "HYPOTHESIS",
            "source_agent": "reasoning_agent",
        },
    ]


def test_parse_claim_status_text_fallbacks() -> None:
    output = "Claim Statuses:\n- SUPPORTED: opacity\n- pneumonia | CONTRADICTED"

    assert parse_claim_status_lines(output) == [
        {"claim": "opacity", "status": "SUPPORTED"},
        {"claim": "pneumonia", "status": "CONTRADICTED"},
    ]


def test_unknown_status_defaults_to_hypothesis() -> None:
    assert normalize_claim_status("unclear") == "HYPOTHESIS"
    assert build_claim_record("possible edema", "unclear") == {
        "claim": "possible edema",
        "status": "HYPOTHESIS",
    }


def test_claim_status_summary_and_hallucination_flag() -> None:
    records = [
        {
            "claim_statuses": [
                {"claim": "finding", "status": "OBSERVED"},
                {"claim": "diagnosis", "status": "UNSUPPORTED"},
            ]
        }
    ]

    summary = summarize_claim_statuses(records)

    assert summary["num_claim_status_records"] == 2
    assert summary["claim_status_counts"]["OBSERVED"] == 1
    assert summary["claim_status_counts"]["UNSUPPORTED"] == 1
    assert has_unsupported_or_contradicted_claim(records[0])


def test_claims_to_status_records_keeps_hypotheses_explicit() -> None:
    assert claims_to_status_records(["possible pneumonia"], "HYPOTHESIS") == [
        {"claim": "possible pneumonia", "status": "HYPOTHESIS"}
    ]
