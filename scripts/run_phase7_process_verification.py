"""运行 Phase 7 Process-Level Verification 实验。

Phase 7 复用 Phase 5 的 candidate/post-verification runner，但实验语义不同：
Phase 5 关注 verifier 是否改进最终答案；Phase 7 关注 verifier 是否能定位
process-level reasoning error，并报告 precision / recall / F1。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from medreason_agent.paths import resolve_project_path  # noqa: E402
from scripts.run_phase5_verification import run  # noqa: E402


def parse_args() -> argparse.Namespace:
    """解析 Phase 7 命令行参数。"""
    parser = argparse.ArgumentParser(description="运行 Phase 7 process verification。")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/experiments/"
            "exp07_separate_verifier_supervisor_multi_agent_qwen_7b_4090d_pmc10k_300.yaml"
        ),
    )
    parser.add_argument("--backend", choices=["mock", "qwen2_5_vl"], default=None)
    return parser.parse_args()


def main() -> int:
    """执行 Phase 7 实验。"""
    args = parse_args()
    run(resolve_project_path(args.config), backend_name=args.backend)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
