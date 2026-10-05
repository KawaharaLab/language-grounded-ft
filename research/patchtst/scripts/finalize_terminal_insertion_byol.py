#!/usr/bin/env python3
"""Audit a newly added pretrained insertion baseline and plot diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score
import torch
import umap


SEEDS = (42, 43, 44, 45, 46)
COUNTS = (1, 2, 5, 10, 25, 50, 100)
UMAP_COUNTS = (1, 5, 100)


def plot_diagnostics(directory: Path, predictions: pd.DataFrame, features, title: str) -> None:
    figure, axis = plt.subplots(figsize=(5.2, 4.4), constrained_layout=True)
    matrix = confusion_matrix(predictions.label, predictions.prediction, labels=[0, 1])
    ConfusionMatrixDisplay(matrix, display_labels=["Before insertion", "After insertion"]).plot(
        ax=axis, colorbar=False, cmap="Blues", values_format="d"
    )
    axis.set_title(title)
    figure.savefig(directory / "confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close(figure)
    if features is None:
        return
    coordinates = umap.UMAP(
        n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42, n_jobs=1
    ).fit_transform(features)
    figure, axis = plt.subplots(figsize=(5.6, 4.5), constrained_layout=True)
    for label, name, color in ((0, "Before insertion", "#0072B2"), (1, "After insertion", "#D55E00")):
        selected = predictions.label.to_numpy() == label
        axis.scatter(*coordinates[selected].T, s=14, alpha=0.7, c=color, label=name)
    incorrect = predictions.label.to_numpy() != predictions.prediction.to_numpy()
    axis.scatter(*coordinates[incorrect].T, marker="x", s=28, c="black", label="Misclassified")
    axis.set(xlabel="UMAP 1", ylabel="UMAP 2", title=title)
    axis.legend(fontsize=9)
    figure.savefig(directory / "umap.png", dpi=200, bbox_inches="tight")
    plt.close(figure)
    table = predictions.copy()
    table["umap_1"], table["umap_2"] = coordinates.T
    table.to_csv(directory / "umap_coordinates.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model", choices=("byol", "supervised"), default="byol")
    arguments = parser.parse_args()
    model_name = arguments.model
    display = {"byol": "BYOL", "supervised": "Supervised"}[model_name]
    root = arguments.run_dir.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    reference = Path(manifest["reference_results"])
    sys.path.insert(0, str(root / "code/PatchTST_self_supervised"))
    from train_terminal_insertion_probe import (
        EncoderClassifier, apply_attention_qv_lora, build_pretrained_encoder, encode_features,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for saved LoRA checkpoint verification")
    checkpoint = torch.load(manifest["checkpoint"], map_location="cpu", weights_only=False)
    dataset_manifest = json.loads((root / "dataset/dataset_manifest.json").read_text())
    dataset_path = root / "dataset/terminal_insertion_windows.npz"
    dataset_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    assert dataset_hash == dataset_manifest["dataset_sha256"]
    reference_checkpoint = torch.load(root / "scratch/seed_42/n_100/best.pth", map_location="cpu", weights_only=False)
    assert reference_checkpoint["manifest"] == dataset_manifest
    del reference_checkpoint
    with np.load(root / "dataset/terminal_insertion_windows.npz", allow_pickle=False) as dataset:
        force = (dataset["force"] - dataset["mean"]) / dataset["scale"]
        labels = dataset["label"].astype(int)
        episode_indices = dataset["episode_index"]
        episode_ids = dataset["episode_ids"].astype(str)
        relative_sec = dataset["relative_to_reference_sec"]
    test_episodes = set(dataset_manifest["splits"]["test"])
    test_indices = np.flatnonzero([episode_ids[index] in test_episodes for index in episode_indices])
    with np.load(root / f"pretrained_frozen/{model_name}/fixed_features.npz", allow_pickle=False) as cache:
        fixed_features = cache["features"]
    artifacts = [f"# {display} terminal-insertion artifacts", "",
                 "All 70 conditions include test predictions and confusion matrices. UMAP is generated for the prespecified seed 42, N = 1, 5, 100 in both adaptation modes; no test-based condition selection. UMAP is descriptive, not an independent test statistic.", ""]
    audit_rows = []
    device = torch.device("cuda")
    for mode in ("frozen_linear", "lora"):
        base = root / (f"pretrained_frozen/{model_name}" if mode == "frozen_linear" else f"pretrained_adaptation/{model_name}/lora")
        for seed in SEEDS:
            for count in COUNTS:
                directory = base / f"seed_{seed}/n_{count:03d}"
                predictions = pd.read_csv(directory / "test_predictions.csv")
                metrics = json.loads((directory / "metrics.json").read_text())
                assert metrics["model"] == model_name
                np.testing.assert_array_equal(predictions.window_index, test_indices)
                np.testing.assert_array_equal(predictions.label, labels[test_indices])
                np.testing.assert_array_equal(predictions.episode_id, episode_ids[episode_indices[test_indices]])
                np.testing.assert_allclose(predictions.relative_to_reference_sec, relative_sec[test_indices])
                expected_ids = np.random.default_rng(seed).permutation(dataset_manifest["splits"]["train"]).tolist()[:count]
                best = torch.load(directory / "best.pth", map_location="cpu", weights_only=False)
                if mode == "frozen_linear":
                    config = json.loads((directory / "config.json").read_text())
                    assert config["selected_episode_ids"] == expected_ids
                    standardized = (fixed_features[test_indices] - np.asarray(config["feature_mean"], dtype=np.float32)) / np.maximum(np.asarray(config["feature_std"], dtype=np.float32), 1e-6)
                    head = torch.nn.Linear(768, 1).eval()
                    head.load_state_dict(best["state_dict"], strict=True)
                    with torch.no_grad():
                        probabilities = torch.sigmoid(head(torch.from_numpy(standardized))).numpy().reshape(-1)
                    features = standardized
                else:
                    assert metrics["selected_train_episode_ids"] == expected_ids
                    assert metrics["trainable_parameters"] == 8961
                    assert best["run_config"]["seed"] == seed
                    encoder = build_pretrained_encoder(model_name, checkpoint).to(device)
                    apply_attention_qv_lora(encoder, rank=4, alpha=8.0)
                    model = EncoderClassifier(encoder, 768).to(device)
                    model.load_state_dict(best["state_dict"], strict=True)
                    model.eval()
                    features = encode_features(encoder, torch.from_numpy(force[test_indices].astype(np.float32)), 256, 0, device)
                    with torch.no_grad():
                        probabilities = torch.sigmoid(model.classifier(features.to(device))).cpu().numpy().reshape(-1)
                    features = features.numpy()
                    del model, encoder
                np.testing.assert_allclose(probabilities, predictions.probability_after, atol=2e-5, rtol=2e-5)
                np.testing.assert_array_equal(predictions.prediction, predictions.probability_after.to_numpy() >= 0.5)
                np.testing.assert_array_equal(predictions.prediction, probabilities >= 0.5)
                macro_f1 = f1_score(predictions.label, predictions.prediction, average="macro")
                np.testing.assert_allclose(macro_f1, metrics["test_window"]["macro_f1"], atol=1e-12)
                np.testing.assert_allclose(accuracy_score(predictions.label, predictions.prediction), metrics["test_window"]["accuracy"], atol=1e-12)
                title = f"{display} {mode.replace('_', ' ')}; seed {seed}, N={count}"
                selected_umap = seed == 42 and count in UMAP_COUNTS
                plot_diagnostics(directory, predictions, features if selected_umap else None, title)
                audit_rows.append({"adaptation": mode, "seed": seed, "episode_count": count,
                                   "test_windows": len(predictions), "accuracy": accuracy_score(predictions.label, predictions.prediction),
                                   "macro_f1": macro_f1, "best_epoch": metrics["best_epoch"]})
                artifacts.append(f"- {mode}, seed {seed}, N={count}: [confusion matrix]({directory / 'confusion_matrix.png'})" +
                                 (f" / [UMAP]({directory / 'umap.png'})" if selected_umap else ""))
                print(f"Verified {mode} seed={seed} N={count}: F1={macro_f1:.4f}", flush=True)

    for stem in ("pretrained_linear_vs_scratch_end_to_end", "pretrained_lora_vs_end_to_end"):
        original = pd.read_csv(reference / f"{stem}_all_seed_results.csv")
        revised = pd.read_csv(root / f"{stem}_all_seed_results.csv")
        retained = set(original.model) - {"jepa", model_name}
        keys = ["model", "seed", "episode_count"]
        original = original[original.model.isin(retained)].sort_values(keys).reset_index(drop=True)
        retained_rows = revised[revised.model.isin(retained)].sort_values(keys).reset_index(drop=True)
        pd.testing.assert_frame_equal(original, retained_rows, check_dtype=False, atol=1e-12, rtol=1e-12)
        assert len(revised) == 35 * (len(retained) + 1) and set(revised.model) == retained | {model_name}
    pd.DataFrame(audit_rows).to_csv(root / f"{model_name}_verified_results.csv", index=False)
    lines = [f"# {display} terminal insertion: five-seed results", "",
             f"{display} pretraining seed 42, validation-best epoch {manifest['checkpoint_epoch']}. Downstream seeds 42–46; same seven nested episode subsets, corrected timing (Enter − 0.30 s), train/val/test episodes 100/5/20, fixed threshold 0.5. Macro F1 is reported as percentages, mean ± sample standard deviation (ddof=1); these are NOT five independent pretraining runs.", "",
             f"| Labeled episodes | {display} frozen Macro F1 (%) | {display} LoRA Macro F1 (%) | Scratch Macro F1 (%) |", "| ---: | ---: | ---: | ---: |"]
    frozen = pd.read_csv(root / "pretrained_linear_vs_scratch_end_to_end_aggregate.csv")
    lora = pd.read_csv(root / "pretrained_lora_vs_end_to_end_aggregate.csv")
    for count in COUNTS:
        values = []
        for table, method in ((frozen, model_name), (lora, model_name), (lora, "scratch")):
            row = table[(table.model == method) & (table.episode_count == count)].iloc[0]
            values.append(f"{100 * row.test_macro_f1_mean:.2f} ± {100 * row.test_macro_f1_std:.2f}")
        lines.append(f"| {count} | " + " | ".join(values) + " |")
    lines += ["", f"All retained comparison rows exactly match the historical comparison (tolerance 1e-12). New {display} predictions were recomputed from saved best checkpoints and checked against CSV and reported Macro F1. Neither evaluation thresholds nor epoch choices were tuned on test results.", "",
              "Input normalization uses all 100 training episodes for every N. The frozen probe additionally standardizes embedding coordinates using only the selected N training episodes, exactly as in the existing frozen baselines; this remains an affine linear classifier. LoRA uses unstandardized encoder features, as in the existing LoRA baselines.", "",
              f"See ARTIFACTS.md for diagnostics. PDF/PNG comparisons include {display} and exclude JEPA; old runs and figures are preserved."]
    (root / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    (root / "ARTIFACTS.md").write_text("\n".join(artifacts) + "\n")
    (root / "finalize_terminal_insertion_byol.py").write_bytes(Path(__file__).read_bytes())
    (root / "validation.json").write_text(json.dumps({
        "conditions": 70, "prediction_recomputation": "passed", "reference_rows_unchanged": True,
        "dataset_manifest_matches_original_scratch": True, "dataset_sha256": dataset_hash,
        "umap_conditions": 6, "finalizer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }, indent=2) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
