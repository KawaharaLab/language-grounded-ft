"""Resolve generated-workspace sibling paths without host-specific prefixes."""

from __future__ import annotations

import os
from pathlib import Path


def workspace_root() -> Path:
    """Return ``FT_WORKSPACE_ROOT`` or infer it from ``workspace/PatchTST``."""
    configured = os.environ.get("FT_WORKSPACE_ROOT")
    return Path(configured).expanduser().resolve() if configured else Path(__file__).resolve().parents[2]


def workspace_path(relative: str) -> Path:
    """Resolve a path relative to the materialized workspace root."""
    return workspace_root() / relative


def resolve_recorded_path(value: str | os.PathLike[str]) -> Path:
    """Translate paths recorded on the experiment host into this workspace.

    Historical CSVs and checkpoint metadata intentionally remain auditable, so
    loaders translate their old prefixes at the filesystem boundary.
    """
    path = Path(value)
    raw = str(path)
    mappings = (
        ("/home/user/Genesis", "Genesis"),
        ("/mnt/nvme1/user/data/Genesis", "Genesis/data"),
        ("/home/user/PatchTST", "artifacts/PatchTST"),
        ("/mnt/nvme1/user/data/PatchTST", "artifacts/PatchTST"),
        ("/home/user/ur3_robotiq_ros2", "ur3_robotiq_ros2"),
    )
    for old_prefix, local_prefix in mappings:
        if raw == old_prefix or raw.startswith(old_prefix + "/"):
            suffix = raw[len(old_prefix):].lstrip("/")
            return workspace_path(local_prefix) / suffix
    return path
