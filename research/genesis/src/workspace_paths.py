"""Portable paths for the bootstrapped language-grounded F/T workspace."""

from __future__ import annotations

import os
from pathlib import Path


_GENESIS_FROM_SOURCE = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(
    os.environ.get("FT_WORKSPACE_ROOT", _GENESIS_FROM_SOURCE.parent)
).expanduser().resolve()
GENESIS_ROOT = Path(
    os.environ.get("FT_GENESIS_ROOT", WORKSPACE_ROOT / "Genesis")
).expanduser().resolve()
PATCHTST_ROOT = Path(
    os.environ.get("FT_PATCHTST_ROOT", WORKSPACE_ROOT / "PatchTST")
).expanduser().resolve()
ROBOT_ROOT = Path(
    os.environ.get("FT_ROBOT_ROOT", WORKSPACE_ROOT / "ur3_robotiq_ros2")
).expanduser().resolve()
GENESIS_DATA_ROOT = Path(
    os.environ.get("FT_GENESIS_DATA_ROOT", GENESIS_ROOT / "data")
).expanduser().resolve()


def workspace_python(repo: Path = PATCHTST_ROOT) -> str:
    """Return an overrideable Python used by historical cross-repository jobs."""

    return os.environ.get("FT_PYTHON", str(repo / ".venv" / "bin" / "python"))


def text_model() -> str:
    """Return a local SentenceTransformer path or a resolvable model identifier."""

    return os.environ.get("FT_TEXT_MODEL", "sentence-transformers/all-mpnet-base-v2")


def text_model_revision() -> str:
    """Return the audited all-mpnet-base-v2 revision used by paper artifacts."""

    return os.environ.get(
        "FT_TEXT_MODEL_REVISION", "e8c3b32edf5434bc2275fc9bab85f82640a19130"
    )
