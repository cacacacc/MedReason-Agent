"""对比 Direct VLM 和 CoT baseline 输出。

在 Phase 2 生成 CoT predictions 后运行这个脚本。它会把 CoT 输出和 Phase 1
Direct VLM predictions 对齐比较，并写出 helped/hurt 分析文件。
"""

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
    """解析 Direct-vs-CoT 对比所需的输入和输出路径。"""
    parser = argparse.ArgumentParser(description="对比 Direct VLM 和 CoT predictions。")
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
    # 从磁盘读取两组结果，这样不用重新跑昂贵的 VLM 推理也能重复分析。
    comparison = compare_records(
        direct_records=load_jsonl(resolve_project_path(args.direct)),
        cot_records=load_jsonl(resolve_project_path(args.cot)),
    )
    write_comparison_outputs(resolve_project_path(args.output_dir), comparison)

    # terminal 只打印紧凑 summary；完整逐样本记录写入 `comparisons.jsonl`。
    summary = {key: value for key, value in comparison.items() if key != "comparisons"}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
