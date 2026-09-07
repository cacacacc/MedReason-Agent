"""项目路径辅助函数。

实验脚本经常需要读配置、数据和结果文件。这里统一根据当前包的位置找到项目根目录，
避免每个脚本都手写容易出错的相对路径或绝对路径。
"""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """根据当前文件位置返回仓库根目录。"""
    return Path(__file__).resolve().parents[2]


def resolve_project_path(relative_path: str | Path) -> Path:
    """把项目相对路径转换成绝对路径。"""
    return project_root() / Path(relative_path)


def required_project_paths() -> dict[str, Path]:
    """返回实验环境必须存在的关键目录。

    `scripts/check_environment.py` 会用这些路径判断仓库结构是否完整。
    """
    root = project_root()
    return {
        "configs": root / "configs",
        "data_raw": root / "Data" / "Raw",
        "data_processed": root / "Data" / "Processed",
        "experiments": root / "Experiments",
        "results": root / "Results",
        "docs": root / "Docs",
    }
