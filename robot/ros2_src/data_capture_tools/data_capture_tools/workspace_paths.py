"""Resolve repository assets without depending on a developer home directory."""

from __future__ import annotations

import os
from pathlib import Path


def workspace_root() -> Path:
    """Return the generated ``<repository>/workspace`` directory."""
    configured = os.environ.get("FT_WORKSPACE_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    repository_candidate = Path(__file__).resolve().parents[4]
    if (repository_candidate / "robot").is_dir():
        return (repository_candidate / "workspace").resolve()
    for candidate in (Path.cwd().resolve(), *Path.cwd().resolve().parents):
        if candidate.name == "workspace" and (candidate / "ur3_robotiq_ros2").is_dir():
            return candidate
    raise RuntimeError(
        "Cannot locate <repository>/workspace; export FT_WORKSPACE_ROOT"
    )


def resolve_workspace_path(value: str | Path) -> Path:
    """Resolve an absolute path or a path below ``<repository>/workspace``."""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (workspace_root() / path).resolve()
