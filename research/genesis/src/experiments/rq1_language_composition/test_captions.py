"""CPU-only semantic and lexical checks for the frozen RQ1 caption bank."""

from __future__ import annotations

import copy
import csv
import importlib.util
import json
import os
import re
import unittest
from pathlib import Path


_HERE = Path(__file__).resolve().parent
_SPEC = importlib.util.spec_from_file_location("rq1_caption_module", _HERE / "captions.py")
captions = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(captions)
_WORKSPACE = Path(os.environ.get("FT_WORKSPACE_ROOT", Path(__file__).resolve().parents[4])).expanduser().resolve()
_GENESIS = Path(os.environ.get("FT_GENESIS_ROOT", _WORKSPACE / "Genesis")).expanduser().resolve()
_TRAIN_CSV = _GENESIS / (
    "data/train_08142026/"
    "train_08142026_terrain_floor_wobble_uniform_descent_contact_clean_slip5mms_balanced.csv"
)


class CaptionBankTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rows = {}
        cls.n_rows = 0
        with _TRAIN_CSV.open(newline="") as stream:
            for row in csv.DictReader(stream):
                cls.n_rows += 1
                # Keep the full source's semantic distinctions in the small
                # test input; the generator separately validates ambiguous IDs.
                key = tuple(row[k] for k in ("emb_index", "annotation", *captions.SEMANTIC_FIELDS))
                rows[key] = row
        cls.rows = list(rows.values())
        cls.bank = captions.build_caption_bank(cls.rows)

    def test_exact_current_dataset_and_original_language(self):
        self.assertEqual(self.n_rows, 437244)
        self.assertEqual(len(self.bank["canonical"]), 250)
        for row in self.rows:
            self.assertEqual(self.bank["canonical"][int(row["emb_index"])], row["annotation"])

    def test_all_8000_captions_pass_audit(self):
        audit = captions.audit_caption_bank(self.bank)
        self.assertEqual(audit["checked_diverse_captions"], 8000)
        self.assertEqual(audit["unique_diverse_captions"], 8000)
        for key in ("forbidden_match_count", "duplicate_within_id_count", "weight_mismatch_count", "surface_loss_count"):
            self.assertEqual(audit[key], 0)

    def test_reproducible_regardless_of_source_row_order(self):
        reordered = captions.build_caption_bank(reversed(self.rows))
        self.assertEqual(self.bank, reordered)
        self.assertEqual(json.loads(json.dumps(self.bank)), self.bank)

    def test_ban_covers_inflections_compounds_and_negation(self):
        forbidden = (
            "grasp", "grasped", "grasping", "release", "releasing", "released",
            "hold", "holding", "held", "lift", "lifting", "lifted",
            "place", "placing", "replacement", "press", "pressing", "pressure",
            "slip", "slipping", "slippage", "stable", "unstable", "stability",
            "instability", "drop", "dropped", "dropping", "accidental", "accidentally",
            "The object is not slipping.", "WITHOUT SLIPPING",
        )
        for text in forbidden:
            with self.subTest(text=text):
                self.assertIsNotNone(captions.FORBIDDEN_PATTERN.search(text))
        for text in ("secure", "steady", "grip", "raise", "slide", "put", "rest"):
            self.assertIsNone(captions.FORBIDDEN_PATTERN.search(text))

    def test_unknown_weight_and_interaction_not_invented(self):
        for index, state in enumerate(self.bank["metadata"]["semantic_states"]):
            for text in self.bank["diverse"][index]:
                if not state["weight"]:
                    self.assertNotRegex(text, r"\b(?:light|heavy)\b")
                if not state["interaction"]:
                    self.assertNotRegex(text, r"\b(?:secure|steady|sliding|slides|unintentionally)\b")
                if state["support_surface"]:
                    self.assertIn(state["support_surface"], text)

    def test_transitions_release_location_and_failure_semantics(self):
        for index, state in enumerate(self.bank["metadata"]["semantic_states"]):
            for text in self.bank["diverse"][index]:
                if " then " in state["action"]:
                    self.assertIn(" and then ", text)
                if state["action"] == "release midair":
                    self.assertRegex(text, r"midair|in the air|away from any supporting surface|before it reaches any support")
                if state["action"] == "release":
                    self.assertRegex(text, r"support|surface")
                    self.assertNotIn("midair", text)
                if state["interaction"] == "accidental drop":
                    self.assertIn("unintentionally", text)
                    self.assertIn("falls", text)
                    self.assertRegex(text, r"tries to|trying to|attempting to")
                    self.assertNotIn("let go", text)

    def test_audit_rejects_leakage_in_complete_caption(self):
        bank = copy.deepcopy(self.bank)
        bank["diverse"][0][1] += " No slipping."
        with self.assertRaisesRegex(ValueError, "Forbidden vocabulary"):
            captions.audit_caption_bank(bank)

    def test_audit_rejects_same_id_different_state(self):
        changed = dict(self.rows[0])
        changed["weight"] = "light" if changed["weight"] != "light" else "heavy"
        with self.assertRaisesRegex(ValueError, "multiple semantic states"):
            captions.build_caption_bank(self.rows + [changed])

    def test_eval_suites_frozen_complete_and_unseen(self):
        suites = captions.build_eval_suites()
        self.assertEqual(list(suites), ["canonical", "bare_words", *[f"paraphrase_{i}" for i in range(1, 6)]])
        self.assertEqual(suites["canonical"]["interaction"]["stable"], "The object has stable contact.")
        for suite in suites.values():
            self.assertEqual(tuple(suite["action"]), captions.ACTION_CLASSES)
            self.assertEqual(tuple(suite["weight"]), captions.WEIGHT_CLASSES)
            self.assertEqual(tuple(suite["interaction"]), captions.INTERACTION_CLASSES)
        train = {text for variants in self.bank["diverse"] for text in variants}
        train_sentences = {part.strip().rstrip(".") for text in train for part in re.split(r"[.;] ", text)}
        new = []
        for key, suite in suites.items():
            if not key.startswith("paraphrase_"):
                continue
            self.assertEqual(suite["weight"], suites["canonical"]["weight"])
            for axis in ("action", "interaction"):
                for text in suite[axis].values():
                    self.assertNotIn(text, train)
                    self.assertNotIn(text.rstrip("."), train_sentences)
                    new.append(text)
        self.assertEqual(len(new), 45)
        self.assertEqual(len(set(new)), 45)

    def test_all_joint_candidates_have_one_known_action_and_weight(self):
        joint = captions.build_composition_prototypes()
        expected = [f"{a}|{w}" for a in captions.ACTION_CLASSES for w in captions.WEIGHT_CLASSES]
        self.assertEqual(list(joint), expected)
        self.assertEqual(joint["place|heavy"], "Placing a heavy object.")
        self.assertEqual(joint["place|light"], "Placing a light object.")
        self.assertEqual(len(set(joint.values())), 12)


if __name__ == "__main__":
    unittest.main()
