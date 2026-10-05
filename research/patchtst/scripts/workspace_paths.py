"""Resolve paths shared by PatchTST replay scripts."""

from __future__ import annotations

import os
from pathlib import Path


def workspace_root() -> Path:
    configured = os.environ.get("FT_WORKSPACE_ROOT")
    return Path(configured).expanduser().resolve() if configured else Path(__file__).resolve().parents[2]


def workspace_path(relative: str) -> Path:
    return workspace_root() / relative
