"""Check whether the MedReason-Agent experiment environment is ready."""

import importlib.util
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_PATHS = [
    "README.md",
    "MISSION.md",
    "RESOURCES.md",
    "Docs/research_questions.md",
    "Docs/experiment_protocol.md",
    "Docs/error_taxonomy.md",
    "configs/base.yaml",
    "configs/experiments/exp00_majority.yaml",
    "configs/experiments/exp01_direct_vlm.yaml",
    "Data/Raw",
    "Data/Processed",
    "Results",
    "Experiments",
    "src/medreason_agent",
]

BASE_PACKAGES = [
    "numpy",
    "pandas",
    "PIL",
    "pydantic",
    "yaml",
    "sklearn",
    "tqdm",
]

OPTIONAL_PACKAGES = {
    "agent": ["langgraph", "langchain_core"],
    "rag": ["sentence_transformers", "qdrant_client"],
    "vlm": ["torch", "transformers", "accelerate", "qwen_vl_utils"],
}


def package_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def main() -> int:
    missing_paths = [path for path in REQUIRED_PATHS if not (ROOT / path).exists()]
    missing_base_packages = [pkg for pkg in BASE_PACKAGES if not package_available(pkg)]
    optional_status = {
        group: {pkg: package_available(pkg) for pkg in packages}
        for group, packages in OPTIONAL_PACKAGES.items()
    }

    report = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "project_root": str(ROOT),
        "missing_paths": missing_paths,
        "missing_base_packages": missing_base_packages,
        "optional_packages": optional_status,
    }

    print(json.dumps(report, indent=2, ensure_ascii=False))

    if missing_paths:
        print("FAIL: Required project paths are missing.")
        return 1

    if missing_base_packages:
        print("WARN: Base packages are missing. Install with: pip install -r requirements/dev.txt")
        return 0

    print("PASS: Base experiment environment is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
