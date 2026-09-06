"""Project path helpers for scripts and experiments."""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Return the repository root based on this package location."""
    return Path(__file__).resolve().parents[2]


def resolve_project_path(relative_path: str | Path) -> Path:
    """Resolve a project-relative path to an absolute path."""
    return project_root() / Path(relative_path)


def required_project_paths() -> dict[str, Path]:
    """Return paths that must exist for the experiment environment."""
    root = project_root()
    return {
        "configs": root / "configs",
        "data_raw": root / "Data" / "Raw",
        "data_processed": root / "Data" / "Processed",
        "experiments": root / "Experiments",
        "results": root / "Results",
        "docs": root / "Docs",
    }
