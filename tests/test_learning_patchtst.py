"""Fast, offline checks for the materialized PatchTST research overlay."""

from __future__ import annotations

import io
import os
from pathlib import Path
import tempfile
import sys
from unittest import mock
import unittest

import torch
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = Path(os.environ.get("FT_WORKSPACE_ROOT", REPO_ROOT / "workspace"))
PATCHTST = WORKSPACE / "PatchTST"
if not PATCHTST.exists():
    # This fallback lets contributors test the overlay before materialization.
    PATCHTST = REPO_ROOT / "research" / "patchtst"
    sys.path.insert(0, str(REPO_ROOT / "third_party" / "patchtst" / "PatchTST_self_supervised"))
sys.path.insert(0, str(PATCHTST / "PatchTST_self_supervised"))

from baseline_models import StandardCNNContrastive  # noqa: E402
from patchtst_bert import ForceContrastiveDataset  # noqa: E402
from src.callback.patch_mask import create_patch  # noqa: E402
from token_fusion_encoder import RawTokenFusionEncoder  # noqa: E402
from workspace_paths import resolve_recorded_path  # noqa: E402


class PatchTSTSmokeTest(unittest.TestCase):
    def setUp(self):
        self.previous_root = os.environ.get("FT_WORKSPACE_ROOT")
        os.environ["FT_WORKSPACE_ROOT"] = str(WORKSPACE)

    def tearDown(self):
        if self.previous_root is None:
            os.environ.pop("FT_WORKSPACE_ROOT", None)
        else:
            os.environ["FT_WORKSPACE_ROOT"] = self.previous_root

    def test_learning_model_window_and_checkpoint_roundtrip(self):
        torch.manual_seed(7)
        windows = torch.randn(2, 80, 12)

        patches, patch_count = create_patch(windows, patch_len=10, stride=10)
        self.assertEqual(patches.shape, (2, 8, 12, 10))
        self.assertEqual(patch_count, 8)

        model = StandardCNNContrastive(
            nvars=12, base_channels=8, depth=2, kernel_size=3,
            dropout=0.0, out_embed_dim=16,
        ).eval()
        with torch.no_grad():
            expected = model(windows)
        self.assertEqual(expected.shape, (2, 16))
        torch.testing.assert_close(expected.norm(dim=-1), torch.ones(2), atol=1e-5, rtol=1e-5)

        buffer = io.BytesIO()
        torch.save({"model": model.state_dict(), "window": 80}, buffer)
        buffer.seek(0)
        restored = StandardCNNContrastive(12, 8, 2, 3, 0.0, 16).eval()
        checkpoint = torch.load(buffer, map_location="cpu", weights_only=False)
        restored.load_state_dict(checkpoint["model"], strict=True)
        self.assertEqual(checkpoint["window"], 80)
        with torch.no_grad():
            torch.testing.assert_close(restored(windows), expected)


    def test_learning_legacy_paths_resolve_inside_workspace(self):
        self.assertEqual(
            resolve_recorded_path("/home/user/Genesis/data/train_08142026/index.csv"),
            WORKSPACE / "Genesis/data/train_08142026/index.csv",
        )
        self.assertEqual(
            resolve_recorded_path("/mnt/nvme1/user/data/PatchTST/saved_models/model.pth"),
            WORKSPACE / "artifacts/PatchTST/saved_models/model.pth",
        )
        self.assertEqual(
            resolve_recorded_path("/home/user/PatchTST/results/run/summary.json"),
            WORKSPACE / "artifacts/PatchTST/results/run/summary.json",
        )

    def test_tiny_token_fusion_training_step(self):
        torch.manual_seed(11)
        model = RawTokenFusionEncoder(
            nvars=3, context_points=20, patch_len=5, stride=5,
            n_layers=1, n_heads=2, d_model=8, d_ff=16,
            dropout=0.0, head_dropout=0.0, token_fusion_layers=1,
            token_fusion_heads=2, token_fusion_ff_mult=2,
            out_embed_dim=4, normalize_output=False,
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        before = model.proj[-1].weight.detach().clone()
        output = model(torch.randn(2, 20, 3))
        self.assertEqual(output.shape, (2, 4))
        output.square().mean().backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))
        optimizer.step()
        self.assertFalse(torch.equal(before, model.proj[-1].weight.detach()))

    def test_actual_recorded_csv_and_embedding_load_inside_workspace(self):
        train = WORKSPACE / "Genesis/data/train_08142026"
        source_index = train / (
            "train_08142026_terrain_floor_wobble_uniform_descent_"
            "contact_clean_slip5mms_balanced.csv"
        )
        if not source_index.exists():
            self.skipTest("0814 training artifacts have not been imported")
        row = pd.read_csv(source_index, nrows=1)
        self.assertTrue(str(row.iloc[0].csv_path).startswith("/home/user/Genesis/"))
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Path(temporary)
            row.to_csv(fixture / "one.csv", index=False)
            (fixture / "bert_emb").symlink_to(train / "bert_emb_terrain_floor_wobble_uniform_descent_contact_clean_slip5mms")
            real_read_csv = pd.read_csv
            real_torch_load = torch.load

            def deny_original_csv(path, *args, **kwargs):
                self.assertFalse(str(path).startswith(("/home/user/", "/mnt/nvme1/user/data/")))
                return real_read_csv(path, *args, **kwargs)

            def deny_original_tensor(path, *args, **kwargs):
                self.assertFalse(str(path).startswith(("/home/user/", "/mnt/nvme1/user/data/")))
                return real_torch_load(path, *args, **kwargs)

            with mock.patch("pandas.read_csv", side_effect=deny_original_csv), mock.patch(
                "torch.load", side_effect=deny_original_tensor
            ):
                dataset = ForceContrastiveDataset(
                    str(fixture), "one.csv", seq_len=80,
                    emb_dir="bert_emb", drop_zero_force=False,
                )
                force, embedding = next(iter(torch.utils.data.DataLoader(dataset, batch_size=1)))
            self.assertEqual(force.shape, (1, 80, 12))
            self.assertEqual(embedding.ndim, 2)
            self.assertTrue(torch.isfinite(force).all())
            self.assertTrue(torch.isfinite(embedding).all())

    def test_imported_lora_checkpoint_inference_when_available(self):
        checkpoint_path = WORKSPACE / (
            "artifacts/PatchTST/results/terminal_insertion_supervised/"
            "terminal_insertion_supervised_20260913_172852/pretrained_adaptation/"
            "supervised/lora/seed_42/n_001/best.pth"
        )
        if not checkpoint_path.exists():
            self.skipTest("RQ3 LoRA artifact has not been imported")
        from train_terminal_insertion_probe import (
            EncoderClassifier, apply_attention_qv_lora, scratch_tokenfusion_encoder,
        )

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        encoder = scratch_tokenfusion_encoder()
        apply_attention_qv_lora(
            encoder,
            int(checkpoint["run_config"]["lora_rank"]),
            float(checkpoint["run_config"]["lora_alpha"]),
        )
        model = EncoderClassifier(encoder, encoder.output_dim).eval()
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        mean = torch.tensor(checkpoint["normalization"]["mean"]).view(1, 1, -1)
        scale = torch.tensor(checkpoint["normalization"]["scale"]).view(1, 1, -1)
        with torch.no_grad():
            logit = model((torch.zeros(1, 80, 12) - mean) / scale)
        self.assertEqual(logit.shape, (1,))
        self.assertTrue(torch.isfinite(logit).all())

    def test_primary_contrastive_lora_checkpoint_inference_when_available(self):
        artifact = WORKSPACE / "artifacts/PatchTST"
        base_path = artifact / (
            "saved_models/terrain_contrastive/"
            "terrain_floor_wobble_uniform_descent_contact_clean_slip5mms_20260824_132352/"
            "terrain_floor_wobble_uniform_descent_slip5mms_tokenfusion_seed42/"
            "terrain_floor_wobble_uniform_descent_slip5mms_tokenfusion_seed42_best.pth"
        )
        adapted_path = artifact / (
            "results/terminal_insertion_adaptation_efficiency/"
            "terminal_insertion_adaptation_efficiency_enter_minus030s_20260827_150802/"
            "lora_r4/seed_42/n_005/best.pth"
        )
        if not base_path.exists() or not adapted_path.exists():
            self.skipTest("primary RQ3 contrastive-LoRA artifacts have not been imported")
        from train_terminal_insertion_probe import (
            EncoderClassifier, apply_attention_qv_lora,
            build_pretrained_encoder, load_checkpoint,
        )

        checkpoint = load_checkpoint(adapted_path)
        encoder = build_pretrained_encoder("contrastive", load_checkpoint(base_path))
        apply_attention_qv_lora(encoder, 4, 8.0)
        model = EncoderClassifier(encoder, encoder.output_dim).eval()
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        self.assertEqual(checkpoint["run_config"]["train_episode_count"], 5)
        self.assertEqual(checkpoint["run_config"]["seed"], 42)
        mean = torch.tensor(checkpoint["normalization"]["mean"]).view(1, 1, -1)
        scale = torch.tensor(checkpoint["normalization"]["scale"]).view(1, 1, -1)
        with torch.no_grad():
            logit = model((torch.zeros(1, 80, 12) - mean) / scale)
        self.assertEqual(logit.shape, (1,))
        self.assertTrue(torch.isfinite(logit).all())


if __name__ == "__main__":
    unittest.main()
