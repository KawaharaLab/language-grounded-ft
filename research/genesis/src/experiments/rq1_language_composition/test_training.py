"""CPU checks for the scientific training contracts (no full training jobs)."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from common import (AXES, CONFIG, HYPER, build_model, load_model,
                    lr_for_epoch, multipositive_loss, supervised_loss)
from prepare_data import CLASSES
from train import encoder_hash, loss_for, population_stats


class TrainingContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_identical_encoder_initialization_across_objectives(self):
        for seed in (42, 43, 44):
            torch.manual_seed(seed)
            contrastive = build_model("contrastive", CLASSES)
            torch.manual_seed(seed)
            supervised = build_model("supervised", CLASSES)
            self.assertEqual(encoder_hash(contrastive), encoder_hash(supervised))
            sample = torch.randn(2, 80, 12)
            contrastive.eval(); supervised.eval()
            torch.testing.assert_close(contrastive.encoder(sample), supervised.encoder(sample), rtol=0, atol=0)
            torch.testing.assert_close(contrastive(sample), F.normalize(supervised.encoder(sample), dim=-1, eps=1e-8))

    def test_published_mlp_architecture_and_initial_weights(self):
        workspace = Path(os.environ.get("FT_WORKSPACE_ROOT", Path(__file__).resolve().parents[4])).expanduser().resolve()
        patchtst = Path(os.environ.get("FT_PATCHTST_ROOT", workspace / "PatchTST")).expanduser().resolve()
        source = patchtst / "results/paper_classification_multiseed/paper_classification_3seed_20260909/code/PatchTST_self_supervised/baseline_models.py"
        specification = importlib.util.spec_from_file_location("rq1_archived_baseline", source)
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        torch.manual_seed(42)
        archived = module.FlatMLPContrastive(**CONFIG).eval()
        torch.manual_seed(42)
        current = build_model("contrastive", CLASSES).eval()
        for name, tensor in archived.state_dict().items():
            torch.testing.assert_close(tensor, current.encoder.state_dict()[name], rtol=0, atol=0)
        sample = torch.randn(3, 80, 12)
        torch.testing.assert_close(archived(sample), current(sample), rtol=1e-6, atol=1e-7)

    def test_population_stats_use_only_selected_rows_and_count_duplicates(self):
        rng = np.random.default_rng(42)
        raw = rng.normal(size=(24, 80, 12)).astype(np.float32)
        raw[:, :, 11] = 3.25
        raw[23] = 1e5  # A held-out row must never affect normalization.
        indices = np.r_[np.arange(22), 5, 5, 10]
        selected = raw[indices].astype(np.float64)
        expected_mean = selected.mean((0, 1))
        expected_std = np.maximum(selected.std((0, 1), ddof=0), 1e-6)
        mean, std, count = population_stats(torch.from_numpy(raw), torch.from_numpy(indices))
        np.testing.assert_allclose(mean.numpy(), expected_mean, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(std.numpy(), expected_std, rtol=1e-6, atol=1e-8)
        self.assertEqual(count, len(indices) * 80)
        self.assertAlmostEqual(float(std[-1]), 1e-6, places=10)

    def test_population_stats_multiblock_matches_numpy(self):
        rng = np.random.default_rng(7)
        raw = rng.normal(size=(1040, 80, 12)).astype(np.float32)
        indices = np.arange(1033)
        mean, std, count = population_stats(torch.from_numpy(raw), torch.from_numpy(indices))
        selected = raw[indices].astype(np.float64)
        np.testing.assert_allclose(mean.numpy(), selected.mean((0, 1)), atol=1e-7)
        np.testing.assert_allclose(std.numpy(), selected.std((0, 1)), rtol=1e-6)
        self.assertEqual(count, 1033 * 80)

    def test_unique_positive_loss_is_symmetric_diagonal_infonce(self):
        torch.manual_seed(12)
        force = torch.randn(7, 16, requires_grad=True)
        text = torch.randn(7, 16)
        labels = torch.arange(7)
        logits = F.normalize(force, dim=1) @ F.normalize(text, dim=1).T / HYPER["temperature"]
        reference = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        actual = multipositive_loss(force, text, labels)
        torch.testing.assert_close(actual, reference)
        actual.backward()
        self.assertTrue(torch.isfinite(force.grad).all())

    def test_same_semantic_group_never_becomes_a_negative(self):
        torch.manual_seed(19)
        force = torch.randn(6, 16, requires_grad=True)
        text = torch.randn(6, 16)
        loss = multipositive_loss(force, text, torch.zeros(6, dtype=torch.long))
        # Transposed/noncontiguous logsumexp can differ by roundoff only.
        torch.testing.assert_close(loss, torch.tensor(0.), rtol=0, atol=1e-6)
        loss.backward()
        torch.testing.assert_close(force.grad, torch.zeros_like(force), rtol=0, atol=1e-6)

    def test_multipositive_loss_uses_sum_of_positive_probabilities(self):
        torch.manual_seed(33)
        force = torch.randn(5, 9)
        text = torch.randn(5, 9)
        labels = torch.tensor([0, 1, 0, 2, 1])
        logits = (F.normalize(force, dim=1).double() @ F.normalize(text, dim=1).double().T) / HYPER["temperature"]
        same = labels[:, None] == labels[None, :]
        left = -(logits.softmax(1) * same).sum(1).log().mean()
        right = -(logits.T.softmax(1) * same.T).sum(1).log().mean()
        reference = (left + right) / 2
        actual = multipositive_loss(force, text, labels)
        torch.testing.assert_close(actual.double(), reference, rtol=2e-6, atol=1e-6)
        permutation = torch.tensor([3, 1, 4, 2, 0])
        shuffled = multipositive_loss(force[permutation], text[permutation], labels[permutation])
        torch.testing.assert_close(actual, shuffled)

    def test_supervision_masks_unobservable_targets_and_averages_active_heads(self):
        torch.manual_seed(7)
        logits = {axis: torch.randn(4, len(CLASSES[axis]), requires_grad=True) for axis in AXES}
        labels = torch.tensor([[0, -1, 2, -1], [1, 0, -1, -1], [2, 1, 1, -1], [3, -1, 0, -1]])
        weights = {axis: torch.linspace(.7, 1.3, len(CLASSES[axis])) for axis in AXES}
        loss = supervised_loss(logits, labels, weights)
        expected = torch.stack([
            F.cross_entropy(logits["action"], labels[:, 0], weight=weights["action"]),
            F.cross_entropy(logits["weight"][[1, 2]], labels[[1, 2], 1], weight=weights["weight"]),
            F.cross_entropy(logits["interaction"][[0, 2, 3]], labels[[0, 2, 3], 2], weight=weights["interaction"]),
        ]).mean()
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertIsNone(logits["support_surface"].grad)
        torch.testing.assert_close(logits["weight"].grad[[0, 3]], torch.zeros((2, 2)), rtol=0, atol=0)
        with self.assertRaises(ValueError):
            supervised_loss(logits, torch.full_like(labels, -1), weights)

    def test_caption_variant_gather_uses_semantic_id_and_variant(self):
        model = torch.nn.Identity()
        values = F.normalize(torch.randn(4, 9), dim=1)
        ids = torch.tensor([2, 1, 0, 2])
        variant_ids = torch.tensor([3, 0, 2, 1])
        bank = F.normalize(torch.randn(3, 4, 9), dim=-1)
        labels = torch.full((4, 4), -1)
        actual = loss_for(model, {"kind": "contrastive"}, values, ids, labels, bank, {}, variant_ids)
        expected = multipositive_loss(values, bank[ids, variant_ids], ids)
        torch.testing.assert_close(actual, expected)

    def test_checkpoint_model_roundtrip_and_lr_endpoints(self):
        torch.manual_seed(4)
        sample = torch.randn(2, 80, 12)
        for kind in ("contrastive", "supervised"):
            original = build_model(kind, CLASSES).eval()
            restored = load_model({"kind": kind, "model": original.state_dict()}, CLASSES, device="cpu")
            first, second = original(sample), restored(sample)
            if isinstance(first, dict):
                for axis in AXES:
                    torch.testing.assert_close(first[axis], second[axis], rtol=0, atol=0)
            else:
                torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertEqual(lr_for_epoch(10), HYPER["peak_lr"])
        self.assertAlmostEqual(lr_for_epoch(400), HYPER["min_lr"])
        self.assertGreater(lr_for_epoch(1), 0)


if __name__ == "__main__":
    unittest.main()
