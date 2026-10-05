#!/usr/bin/env python3
"""Run the authoritative four-head RQ1 CPU audit from a portable workspace."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
from typing import Any


def translate(value: Any, workspace: Path) -> Any:
    if isinstance(value, dict):
        return {
            translate(key, workspace): translate(item, workspace)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [translate(item, workspace) for item in value]
    if not isinstance(value, str):
        return value
    mappings = (
        ("/home/user/PatchTST", workspace / "artifacts/PatchTST"),
        ("/mnt/nvme1/user/data/PatchTST", workspace / "artifacts/PatchTST"),
        ("/home/user/Genesis", workspace / "Genesis"),
        ("/mnt/nvme1/user/data/Genesis", workspace / "Genesis/data"),
    )
    for prefix, replacement in mappings:
        if value == prefix or value.startswith(prefix + "/"):
            return str(replacement / value[len(prefix):].lstrip("/"))
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=os.environ.get("FT_WORKSPACE_ROOT"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    workspace = (args.workspace or repo / "workspace").resolve()
    audit_source = workspace / "Genesis/src/experiments/audit_rq1_surface_supervised.py"
    experiment = workspace / "artifacts/PatchTST/results/rq1_language_composition/rq1_20260916_235337"
    out = args.out or workspace / "reproduced/rq1_surface_supervised"
    required = [audit_source, experiment / "experiment.json"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("Missing imported replay inputs:\n  " + "\n  ".join(missing))

    spec = importlib.util.spec_from_file_location("rq1_surface_audit", audit_source)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Cannot import audit module: {audit_source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original_read_json = module.read_json
    module.ROOT = experiment
    module.DEFAULT_OUT = out
    module.read_json = lambda path: translate(original_read_json(path), workspace)
    module.audit(out)


if __name__ == "__main__":
    main()
