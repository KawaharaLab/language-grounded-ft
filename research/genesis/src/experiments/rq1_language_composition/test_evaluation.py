"""CPU checks for readout fairness, fixed classes, and paired uncertainty."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate
import report
from captions import build_composition_prototypes


def prediction_frame(truth, prediction, objects=None):
    n = len(truth)
    if objects is None:
        objects = [f"object_{i // 2}" for i in range(n)]
    return pd.DataFrame({
        "union_index": np.arange(n),
        "csv_path": [f"/fixture/episode_{i}.csv" for i in range(n)],
        "start": np.arange(n) * 10,
        "object_id": objects,
        "episode_id": [f"episode_{i}" for i in range(n)],
        "true": truth,
        "pred": prediction,
        "correct": np.asarray(truth) == np.asarray(prediction),
    })


class EvaluationTests(unittest.TestCase):
    def test_release_is_probability_sum_and_transitions_are_not_candidates(self):
        raw_classes = ["grasp", "grasp then lift", "hold", "lift", "place",
                       "place then release", "press", "press then release",
                       "release", "release midair"]
        # Each release subtype loses to grasp individually, but the required
        # merged release candidate wins; enormous transition scores are ignored.
        probabilities = torch.tensor([[0.3, 10., .02, .03, .01, 11., .01, 12., .2, .2]])
        merged = evaluate.merge_action_logits(probabilities.log(), raw_classes)
        expected = torch.tensor([[.3, .02, .03, .01, .01, .4]])
        torch.testing.assert_close(merged.exp(), expected)
        self.assertEqual(evaluate.ACTION_CLASSES[merged.argmax(1).item()], "release")

    def test_joint_independent_heads_can_predict_an_unseen_pair(self):
        actions = list(evaluate.ACTION_CLASSES)
        weights = list(evaluate.WEIGHT_CLASSES)
        pairs = list(build_composition_prototypes())
        action = torch.tensor([[0., 1., 2., 5., 3., 4.]])
        weight = torch.tensor([[3., 0.]])
        scores = evaluate.joint_factorized_scores(action, weight, actions, weights, pairs)
        self.assertEqual(pairs[scores.argmax(1).item()], "place|heavy")
        expected = torch.stack([
            torch.softmax(action, 1)[:, actions.index(pair.split("|")[0])]
            * torch.softmax(weight, 1)[:, weights.index(pair.split("|")[1])]
            for pair in pairs], 1)
        torch.testing.assert_close(scores.exp(), expected)
        torch.testing.assert_close(scores.exp().sum(1), torch.ones(1))

    def test_explicit_candidate_order_not_sorted_or_assumed(self):
        outputs = {"weight": torch.tensor([[2., 7.]])}
        scores = evaluate.supervised_scores(outputs, {"weight": ["light", "heavy"]},
                                             "weight", ["heavy", "light"])
        torch.testing.assert_close(scores, torch.tensor([[7., 2.]]))

    def test_classification_f1_keeps_absent_candidate_class(self):
        values = evaluate.classification_metrics(np.array(["a", "a"]), np.array(["a", "a"]), ["a", "b"])
        self.assertEqual(values["accuracy"], 1.)
        self.assertEqual(values["macro_f1"], .5)
        self.assertEqual(values["confusion_matrix"], [[2, 0], [0, 0]])

    def test_composition_separates_action_weight_and_joint_correctness(self):
        frame = prediction_frame(
            ["place|heavy", "place|heavy", "hold|light", "hold|light"],
            ["place|heavy", "place|light", "lift|light", "hold|heavy"],
        )
        metrics = evaluate.composition_metrics(frame, "place|heavy")
        self.assertEqual(metrics["joint_accuracy"], .25)
        self.assertEqual(metrics["action_accuracy"], .75)
        self.assertEqual(metrics["weight_accuracy"], .5)
        self.assertEqual(metrics["cell_macro_accuracy"], .25)
        unseen = evaluate.composition_metrics(frame, "place|heavy", "unseen")
        self.assertEqual(unseen["samples"], 2)
        self.assertEqual(unseen["joint_accuracy"], .5)
        self.assertEqual(unseen["action_accuracy"], 1.)
        self.assertIsNone(evaluate.composition_metrics(frame, "press|light", "unseen"))

    def test_bootstrap_metric_point_agrees_with_evaluator(self):
        pairs = list(build_composition_prototypes())
        frame = prediction_frame(
            ["place|heavy", "place|heavy", "hold|light", "hold|light"],
            ["place|heavy", "place|light", "lift|light", "hold|heavy"],
        )
        confusion = np.zeros((12, 12), dtype=np.int64)
        for row in frame.itertuples():
            confusion[pairs.index(row.true), pairs.index(row.pred)] += 1
        values = report._metric_arrays(confusion, pairs, composition=True)
        direct = evaluate.composition_metrics(frame, "place|heavy")
        for name, value in values.items():
            self.assertAlmostEqual(float(value), direct[name])

    def test_paired_bootstrap_identity_zero_and_reproducible(self):
        frame = prediction_frame(["a", "a", "b", "b"], ["a", "b", "b", "a"])
        same = [frame.copy() for _ in range(3)]
        rows, audit, draws = report.paired_object_bootstrap(same, same, ["a", "b"], "identity", replicates=100)
        again, audit_again, draws_again = report.paired_object_bootstrap(same, same, ["a", "b"], "identity", replicates=100)
        self.assertEqual(rows, again)
        self.assertEqual(audit, audit_again)
        np.testing.assert_array_equal(draws["multiplicities"], draws_again["multiplicities"])
        self.assertTrue(np.all(draws["multiplicities"].sum(1) == 2))
        for row in rows:
            self.assertEqual(row["difference"], 0.)
            self.assertEqual(row["lower"], 0.)
            self.assertEqual(row["upper"], 0.)

    def test_paired_bootstrap_averages_seed_metrics_not_predictions(self):
        truth = ["a", "a", "b", "b"]
        correct = prediction_frame(truth, truth)
        opposite = prediction_frame(truth, ["b", "b", "a", "a"])
        left = [correct, correct, opposite]
        right = [opposite, opposite, opposite]
        rows, _, _ = report.paired_object_bootstrap(left, right, ["a", "b"], "seed_mean", replicates=100)
        acc = next(row for row in rows if row["metric"] == "accuracy")
        self.assertAlmostEqual(acc["left"], 2 / 3)
        self.assertEqual(acc["right"], 0.)
        self.assertAlmostEqual(acc["difference"], 2 / 3)

    def test_paired_bootstrap_refuses_misaligned_rows_or_missing_seed(self):
        frame = prediction_frame(["a", "a", "b", "b"], ["a", "b", "b", "a"])
        good = [frame.copy() for _ in range(3)]
        bad = [frame.iloc[::-1].copy(), frame.copy(), frame.copy()]
        with self.assertRaises(AssertionError):
            report.paired_object_bootstrap(good, bad, ["a", "b"], "misaligned", replicates=5)
        with self.assertRaisesRegex(ValueError, "three predeclared seeds"):
            report.paired_object_bootstrap(good[:2], good, ["a", "b"], "missing_seed", replicates=5)


if __name__ == "__main__":
    unittest.main()
