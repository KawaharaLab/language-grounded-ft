"""Offline contract tests for the portable RQ1 replay entry point."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


REPO = Path(__file__).resolve().parents[1]
HELPER = REPO / "env/learning-replay-rq1.py"
SPEC = importlib.util.spec_from_file_location("learning_replay_rq1", HELPER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class LearningReplayContractTest(unittest.TestCase):
    def test_recorded_paths_translate_to_portable_workspace(self):
        workspace = Path("/portable/workspace")
        value = {
            "checkpoint": "/home/user/PatchTST/results/rq1/run/best.pt",
            "source": "/mnt/nvme1/user/data/Genesis/train_08142026/index.csv",
            "labels": ["unchanged"],
            "/home/user/PatchTST/results/run/code/common.py": "digest",
        }
        self.assertEqual(
            MODULE.translate(value, workspace),
            {
                "checkpoint": "/portable/workspace/artifacts/PatchTST/results/rq1/run/best.pt",
                "source": "/portable/workspace/Genesis/data/train_08142026/index.csv",
                "labels": ["unchanged"],
                "/portable/workspace/artifacts/PatchTST/results/run/code/common.py": "digest",
            },
        )

    def test_replay_helper_is_syntax_valid(self):
        compile(HELPER.read_text(encoding="utf-8"), str(HELPER), "exec")


if __name__ == "__main__":
    unittest.main()
