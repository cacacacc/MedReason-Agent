"""Compare Direct VLM and CoT baseline outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from medreason_agent.analysis.compare_predictions import (
    compare_records,
    load_jsonl,
    write_comparison_outputs,
)
from medreason_agent.paths import resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Direct VLM and CoT predictions.")
    parser.add_argument(
        "--direct",
        type=Path,
        default=Path("Results/exp01_direct_vlm/predictions.jsonl"),
    )
    parser.add_argument(
        "--cot",
        type=Path,
        default=Path("Results/exp02_cot/predictions.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Results/compare_direct_vs_cot"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    comparison = compare_records(
        direct_records=load_jsonl(resolve_project_path(args.direct)),
        cot_records=load_jsonl(resolve_project_path(args.cot)),
    )
    write_comparison_outputs(resolve_project_path(args.output_dir), comparison)

    summary = {key: value for key, value in comparison.items() if key != "comparisons"}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
