#!/usr/bin/env python3
"""Replay test probabilities with K-consecutive-positive stop policies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import ListedColormap

from train_terminal_insertion_probe import (
    EncoderClassifier,
    LightweightMLP,
    LinearProbe,
    PRETRAINED_MODELS,
    build_pretrained_encoder,
    encode_features,
    load_checkpoint,
    make_loader,
    predict_logits,
    scratch_tokenfusion_encoder,
    set_seed,
)


MODEL_ORDER = [
    "scratch_tokenfusion",
    "scratch_mlp",
    "vae",
    "masked_reconstruction",
    "vicreg",
    "contrastive",
    "jepa",
]

DISPLAY_NAMES = {
    "scratch_tokenfusion": "Scratch TokenFusion",
    "scratch_mlp": "Scratch MLP",
    "vae": "VAE",
    "masked_reconstruction": "Masked reconstruction",
    "vicreg": "VICReg",
    "contrastive": "Contrastive",
    "jepa": "JEPA",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-consecutive", type=int, default=5)
    return parser.parse_args()


def first_consecutive_trigger(
    times: np.ndarray, probabilities: np.ndarray, threshold: float, required: int
) -> float | None:
    run_length = 0
    for time_sec, probability in zip(times, probabilities):
        run_length = run_length + 1 if probability >= threshold else 0
        if run_length >= required:
            return float(time_sec)
    return None


def percentile(values: pd.Series, q: float) -> float | None:
    finite = values.dropna().to_numpy(dtype=np.float64)
    return float(np.percentile(finite, q)) if len(finite) else None


def markdown_table(table: pd.DataFrame) -> str:
    headers = [str(column) for column in table.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in table.itertuples(index=False, name=None):
        cells = []
        for value in row:
            if isinstance(value, float):
                cells.append("" if pd.isna(value) else f"{value:.4f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def short_episode_label(episode_id: str) -> str:
    run, episode = episode_id.split("/")
    return f"{run[9:15]}/{episode.replace('episode_', 'e')}"


def load_test_probabilities(
    run_dir: Path,
    normalized_tensor: torch.Tensor,
    labels: np.ndarray,
    test_indices: np.ndarray,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> dict[str, np.ndarray]:
    probabilities = {}
    targets = torch.from_numpy(labels[test_indices].astype(np.float32))
    for model_name in MODEL_ORDER:
        best = torch.load(
            run_dir / model_name / "best.pth", map_location="cpu", weights_only=False
        )
        if model_name in PRETRAINED_MODELS:
            encoder_checkpoint = load_checkpoint(Path(best["encoder_checkpoint"]))
            encoder = build_pretrained_encoder(model_name, encoder_checkpoint).to(device)
            for parameter in encoder.parameters():
                parameter.requires_grad = False
            features = encode_features(
                encoder,
                normalized_tensor[test_indices],
                batch_size,
                num_workers,
                device,
            )
            model = LinearProbe(int(getattr(encoder, "output_dim", 768))).to(device)
            del encoder
        elif model_name == "scratch_tokenfusion":
            encoder = scratch_tokenfusion_encoder()
            model = EncoderClassifier(encoder, encoder.output_dim).to(device)
            features = normalized_tensor[test_indices]
        else:
            model = LightweightMLP().to(device)
            features = normalized_tensor[test_indices]
        model.load_state_dict(best["state_dict"], strict=True)
        loader = make_loader(
            features,
            targets,
            batch_size,
            False,
            num_workers,
            42,
        )
        logits = predict_logits(model, loader, device)
        probabilities[model_name] = torch.sigmoid(torch.from_numpy(logits)).numpy()
        del model, features, loader
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"inference complete: {model_name}", flush=True)
    return probabilities


def main() -> None:
    args = parse_args()
    if args.max_consecutive < 1:
        raise ValueError("--max-consecutive must be at least 1")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    set_seed(42)
    device = torch.device(args.device)
    run_dir = args.run_dir.resolve()
    output_dir = run_dir / "consecutive_trigger_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(
        (run_dir / "dataset" / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    episode_metadata = {
        int(episode["episode_index"]): episode for episode in manifest["episodes"]
    }
    test_order = {
        episode_id: index for index, episode_id in enumerate(manifest["splits"]["test"])
    }
    with np.load(
        run_dir / "dataset" / "terminal_insertion_windows.npz", allow_pickle=False
    ) as dataset:
        force = dataset["force"].astype(np.float32)
        labels = dataset["label"].astype(np.int64)
        window_episode_indices = dataset["episode_index"].astype(np.int64)
        relative_key = (
            "relative_to_reference_sec"
            if "relative_to_reference_sec" in dataset.files
            else "relative_to_enter_sec"
        )
        relative_sec = dataset[relative_key].astype(np.float64)
        episode_ids = dataset["episode_ids"].astype(str).tolist()
        mean = dataset["mean"].astype(np.float32)
        scale = dataset["scale"].astype(np.float32)
    normalized = (force - mean.reshape(1, 1, -1)) / scale.reshape(1, 1, -1)
    normalized_tensor = torch.from_numpy(normalized)
    test_episode_numbers = {
        index
        for index, episode_id in enumerate(episode_ids)
        if episode_id in set(manifest["splits"]["test"])
    }
    test_indices = np.flatnonzero(
        np.asarray(
            [index in test_episode_numbers for index in window_episode_indices],
            dtype=bool,
        )
    )
    probabilities = load_test_probabilities(
        run_dir,
        normalized_tensor,
        labels,
        test_indices,
        device,
        args.batch_size,
        args.num_workers,
    )

    window_rows = []
    policy_rows = []
    test_episode_indices = window_episode_indices[test_indices]
    test_relative_sec = relative_sec[test_indices]
    for model_name in MODEL_ORDER:
        metrics = json.loads(
            (run_dir / model_name / "metrics.json").read_text(encoding="utf-8")
        )
        threshold = float(metrics["selected_threshold"])
        model_probabilities = probabilities[model_name]
        for local_index, dataset_index in enumerate(test_indices):
            episode_index = int(window_episode_indices[dataset_index])
            window_rows.append(
                {
                    "model": model_name,
                    "episode_index": episode_index,
                    "episode_id": episode_metadata[episode_index]["episode_id"],
                    "relative_to_enter_sec": float(relative_sec[dataset_index]),
                    "relative_to_reference_sec": float(relative_sec[dataset_index]),
                    "label": int(labels[dataset_index]),
                    "probability": float(model_probabilities[local_index]),
                    "threshold": threshold,
                    "positive": bool(model_probabilities[local_index] >= threshold),
                }
            )
        for episode_index in sorted(np.unique(test_episode_indices).tolist()):
            mask = test_episode_indices == episode_index
            times = test_relative_sec[mask]
            episode_probabilities = model_probabilities[mask]
            order = np.argsort(times)
            times = times[order]
            episode_probabilities = episode_probabilities[order]
            metadata = episode_metadata[int(episode_index)]
            for consecutive in range(1, args.max_consecutive + 1):
                trigger_sec = first_consecutive_trigger(
                    times, episode_probabilities, threshold, consecutive
                )
                outcome = (
                    "missed"
                    if trigger_sec is None
                    else "early_stop"
                    if trigger_sec < 0.0
                    else "success"
                )
                policy_rows.append(
                    {
                        "model": model_name,
                        "model_display": DISPLAY_NAMES[model_name],
                        "consecutive": consecutive,
                        "episode_index": int(episode_index),
                        "episode_id": metadata["episode_id"],
                        "test_order": test_order[metadata["episode_id"]],
                        "threshold": threshold,
                        "outcome": outcome,
                        "early_stop": outcome == "early_stop",
                        "success": outcome == "success",
                        "missed": outcome == "missed",
                        "trigger_relative_sec": trigger_sec,
                        "detection_delay_sec": (
                            trigger_sec if outcome == "success" else None
                        ),
                    }
                )

    pd.DataFrame(window_rows).to_csv(
        output_dir / "test_window_predictions.csv", index=False
    )
    policies = pd.DataFrame(policy_rows).sort_values(
        ["model", "consecutive", "test_order"]
    )
    policies.to_csv(output_dir / "consecutive_trigger_by_episode.csv", index=False)

    summary_rows = []
    for model_name in MODEL_ORDER:
        for consecutive in range(1, args.max_consecutive + 1):
            group = policies[
                (policies["model"] == model_name)
                & (policies["consecutive"] == consecutive)
            ]
            delays = group.loc[group["success"], "detection_delay_sec"]
            early_times = group.loc[group["early_stop"], "trigger_relative_sec"]
            summary_rows.append(
                {
                    "model": model_name,
                    "consecutive": consecutive,
                    "episodes": len(group),
                    "early_stop_episodes": int(group["early_stop"].sum()),
                    "early_stop_rate": float(group["early_stop"].mean()),
                    "success_episodes": int(group["success"].sum()),
                    "success_rate": float(group["success"].mean()),
                    "missed_episodes": int(group["missed"].sum()),
                    "missed_rate": float(group["missed"].mean()),
                    "first_early_stop_sec_median": percentile(early_times, 50),
                    "delay_sec_mean": float(delays.mean()) if len(delays) else None,
                    "delay_sec_min": float(delays.min()) if len(delays) else None,
                    "delay_sec_p25": percentile(delays, 25),
                    "delay_sec_median": percentile(delays, 50),
                    "delay_sec_p75": percentile(delays, 75),
                    "delay_sec_p95": percentile(delays, 95),
                    "delay_sec_max": float(delays.max()) if len(delays) else None,
                }
            )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "consecutive_trigger_summary.csv", index=False)

    # K=1 must exactly reproduce the existing episode-level evaluation.
    for model_name in MODEL_ORDER:
        expected = json.loads(
            (run_dir / model_name / "metrics.json").read_text(encoding="utf-8")
        )["test_sequence"]
        actual = summary[
            (summary["model"] == model_name) & (summary["consecutive"] == 1)
        ].iloc[0]
        if int(actual["early_stop_episodes"]) != int(expected["early_stop_episodes"]):
            raise AssertionError(f"K=1 early-stop mismatch for {model_name}")
        if int(actual["missed_episodes"]) != 0:
            raise AssertionError(f"unexpected K=1 miss for {model_name}")

    colors = plt.cm.tab10(np.linspace(0, 1, len(MODEL_ORDER)))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for color, model_name in zip(colors, MODEL_ORDER):
        model_summary = summary[summary["model"] == model_name]
        axes[0].plot(
            model_summary["consecutive"],
            model_summary["early_stop_rate"] * 100,
            marker="o",
            color=color,
            label=DISPLAY_NAMES[model_name],
        )
        axes[0].plot(
            model_summary["consecutive"],
            model_summary["missed_rate"] * 100,
            marker="x",
            linestyle="--",
            color=color,
            alpha=0.8,
        )
        axes[1].plot(
            model_summary["consecutive"],
            model_summary["delay_sec_median"] * 1000,
            marker="o",
            color=color,
            label=DISPLAY_NAMES[model_name],
        )
        axes[1].fill_between(
            model_summary["consecutive"],
            model_summary["delay_sec_p25"] * 1000,
            model_summary["delay_sec_p75"] * 1000,
            color=color,
            alpha=0.1,
        )
    axes[0].set_title("Episode outcome vs. required consecutive positives")
    axes[0].set_xlabel("Required consecutive positive outputs")
    axes[0].set_ylabel("Episode rate [%]")
    axes[0].set_xticks(range(1, args.max_consecutive + 1))
    axes[0].grid(alpha=0.25)
    axes[0].text(0.02, 0.97, "solid/circle: early stop\ndashed/x: missed", transform=axes[0].transAxes, va="top")
    axes[1].set_title("Successful-stop detection delay")
    axes[1].set_xlabel("Required consecutive positive outputs")
    axes[1].set_ylabel("Median delay [ms] (band: IQR)")
    axes[1].set_xticks(range(1, args.max_consecutive + 1))
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8, loc="upper left")
    fig.savefig(output_dir / "consecutive_trigger_tradeoff.png", dpi=180)
    plt.close(fig)

    k3 = policies[policies["consecutive"] == 3]
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(16, 9),
        gridspec_kw={"height_ratios": [1.1, 1.3]},
        constrained_layout=True,
    )
    outcome_codes = {"success": 0, "early_stop": 1, "missed": 2}
    heat = np.zeros((len(MODEL_ORDER), len(test_order)), dtype=np.int64)
    labels_for_plot = [None] * len(test_order)
    for row in k3.itertuples(index=False):
        heat[MODEL_ORDER.index(row.model), row.test_order] = outcome_codes[row.outcome]
        labels_for_plot[row.test_order] = short_episode_label(row.episode_id)
    axes[0].imshow(
        heat,
        aspect="auto",
        interpolation="nearest",
        cmap=ListedColormap(["#d7f0df", "#ef8a8a", "#f6d365"]),
        vmin=0,
        vmax=2,
    )
    axes[0].set_yticks(range(len(MODEL_ORDER)), [DISPLAY_NAMES[m] for m in MODEL_ORDER])
    axes[0].set_xticks(
        range(len(labels_for_plot)), labels_for_plot, rotation=55, ha="right", fontsize=8
    )
    axes[0].set_title("K=3 test outcomes (green=success, red=early stop, yellow=missed)")
    outcome_text = ["OK", "EARLY", "MISS"]
    for row_index in range(heat.shape[0]):
        for column_index in range(heat.shape[1]):
            axes[0].text(
                column_index,
                row_index,
                outcome_text[heat[row_index, column_index]],
                ha="center",
                va="center",
                fontsize=6.2,
            )

    k3_delays = [
        k3.loc[
            (k3["model"] == model) & k3["success"], "detection_delay_sec"
        ].dropna().to_numpy()
        for model in MODEL_ORDER
    ]
    positions = np.arange(1, len(MODEL_ORDER) + 1)
    axes[1].boxplot(k3_delays, positions=positions, vert=False, widths=0.55)
    rng = np.random.default_rng(42)
    for position, values in zip(positions, k3_delays):
        axes[1].scatter(
            values,
            position + rng.uniform(-0.12, 0.12, size=len(values)),
            s=24,
            alpha=0.8,
            color="#023047",
        )
    # boxplot was constructed in seconds; convert its x tick labels to ms by
    # applying a display-only formatter while retaining exact stored values.
    axes[1].xaxis.set_major_formatter(lambda value, _position: f"{value * 1000:.0f}")
    axes[1].set_yticks(positions, [DISPLAY_NAMES[m] for m in MODEL_ORDER])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("K=3 detection delay [ms]")
    axes[1].set_title("K=3 delay distribution for successful episodes")
    axes[1].grid(axis="x", alpha=0.25)
    fig.savefig(output_dir / "k3_episode_distributions.png", dpi=180)
    plt.close(fig)

    k1 = summary[summary["consecutive"] == 1].set_index("model")
    k3_summary = summary[summary["consecutive"] == 3].copy()
    k3_summary["early_stop_reduction_pp_vs_k1"] = k3_summary.apply(
        lambda row: 100
        * (k1.loc[row["model"], "early_stop_rate"] - row["early_stop_rate"]),
        axis=1,
    )
    report_columns = [
        "model",
        "early_stop_episodes",
        "early_stop_rate",
        "success_episodes",
        "missed_episodes",
        "delay_sec_p25",
        "delay_sec_median",
        "delay_sec_p75",
        "delay_sec_p95",
        "early_stop_reduction_pp_vs_k1",
    ]
    report = [
        "# Consecutive-positive stop policy",
        "",
        f"Timing reference: {manifest.get('timing_reference', 'recorded Enter key stamp')}; recorded Enter correction: -{float(manifest.get('enter_reaction_delay_sec', 0.0)):.3f} s.",
        "",
        "A K=3 stop is emitted at the third consecutive positive 0.1-second-stride output. The first trigger determines the episode outcome.",
        "",
        "## K=3 test result",
        "",
        markdown_table(k3_summary[report_columns]),
        "",
        "All delay statistics are conditional on a successful first trigger after the corrected timing reference.",
        "",
    ]
    (output_dir / "README.md").write_text("\n".join(report), encoding="utf-8")
    print(k3_summary[report_columns].to_string(index=False))
    print(f"\nWrote consecutive-trigger analysis to {output_dir}")


if __name__ == "__main__":
    main()
