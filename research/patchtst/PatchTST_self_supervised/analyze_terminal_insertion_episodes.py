#!/usr/bin/env python3
"""Export episode-level early-stop and detection-delay test distributions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap


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
    return parser.parse_args()


def percentile(series: pd.Series, q: float) -> float | None:
    values = series.dropna().to_numpy(dtype=np.float64)
    return float(np.percentile(values, q)) if len(values) else None


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


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = run_dir / "episode_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(
        (run_dir / "dataset" / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    timing_reference = manifest.get("timing_reference", "recorded Enter key stamp")
    reaction_delay = float(manifest.get("enter_reaction_delay_sec", 0.0))
    reference_label = (
        f"corrected insertion (Enter - {reaction_delay:.1f} s)"
        if reaction_delay
        else "Enter"
    )
    episode_metadata = {
        int(episode["episode_index"]): episode for episode in manifest["episodes"]
    }
    test_order = {
        episode_id: index for index, episode_id in enumerate(manifest["splits"]["test"])
    }

    rows = []
    thresholds = {}
    for model in MODEL_ORDER:
        metrics_path = run_dir / model / "metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError(metrics_path)
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        thresholds[model] = float(metrics["selected_threshold"])
        for record in metrics["test_sequence"]["per_episode"]:
            metadata = episode_metadata[int(record["episode_index"])]
            if metadata["split"] != "test":
                raise ValueError(f"non-test episode in test metrics: {metadata['episode_id']}")
            early_stop = bool(record["early_stop"])
            detected = bool(record["detected_after_enter"])
            delay = record["detection_delay_sec"]
            rows.append(
                {
                    "model": model,
                    "model_display": DISPLAY_NAMES[model],
                    "episode_index": int(record["episode_index"]),
                    "episode_id": metadata["episode_id"],
                    "test_order": test_order[metadata["episode_id"]],
                    "threshold": thresholds[model],
                    "early_stop": early_stop,
                    "first_early_stop_sec": record["first_early_stop_sec"],
                    "detected_after_enter": detected,
                    "detection_delay_sec_all": delay,
                    "detection_delay_sec_safe": (
                        delay if not early_stop and detected else None
                    ),
                    "max_before_probability": record["max_before_probability"],
                    "max_after_probability": record["max_after_probability"],
                }
            )

    episodes = pd.DataFrame(rows).sort_values(["model", "test_order"])
    expected_rows = len(MODEL_ORDER) * len(manifest["splits"]["test"])
    if len(episodes) != expected_rows:
        raise ValueError(f"expected {expected_rows} rows, found {len(episodes)}")
    episodes.to_csv(output_dir / "test_episode_metrics.csv", index=False)
    episode_keys = ["test_order", "episode_id"]
    early_stop_matrix = (
        episodes.pivot(index=episode_keys, columns="model", values="early_stop")
        .reset_index()
        .sort_values("test_order")
    )
    delay_matrix = (
        episodes.pivot(
            index=episode_keys, columns="model", values="detection_delay_sec_safe"
        )
        .reset_index()
        .sort_values("test_order")
    )
    ordered_matrix_columns = ["test_order", "episode_id", *MODEL_ORDER]
    early_stop_matrix[ordered_matrix_columns].to_csv(
        output_dir / "early_stop_by_episode.csv", index=False
    )
    delay_matrix[ordered_matrix_columns].to_csv(
        output_dir / "safe_detection_delay_by_episode.csv", index=False
    )

    summary_rows = []
    for model in MODEL_ORDER:
        group = episodes[episodes["model"] == model]
        safe = group[~group["early_stop"]]
        safe_detected = safe[safe["detected_after_enter"]]
        early_times = group.loc[group["early_stop"], "first_early_stop_sec"]
        delays = safe_detected["detection_delay_sec_safe"]
        summary_rows.append(
            {
                "model": model,
                "test_episodes": len(group),
                "early_stop_episodes": int(group["early_stop"].sum()),
                "early_stop_rate": float(group["early_stop"].mean()),
                "first_early_stop_sec_median": percentile(early_times, 50),
                "safe_episodes": len(safe),
                "safe_missed_episodes": int((~safe["detected_after_enter"]).sum()),
                "delay_count": len(delays.dropna()),
                "delay_sec_mean": float(delays.mean()) if len(delays.dropna()) else None,
                "delay_sec_std": float(delays.std(ddof=1)) if len(delays.dropna()) > 1 else 0.0,
                "delay_sec_min": float(delays.min()) if len(delays.dropna()) else None,
                "delay_sec_p25": percentile(delays, 25),
                "delay_sec_median": percentile(delays, 50),
                "delay_sec_p75": percentile(delays, 75),
                "delay_sec_p90": percentile(delays, 90),
                "delay_sec_p95": percentile(delays, 95),
                "delay_sec_max": float(delays.max()) if len(delays.dropna()) else None,
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "test_episode_distribution_summary.csv", index=False)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(16, 9),
        gridspec_kw={"height_ratios": [1.1, 1.3]},
        constrained_layout=True,
    )

    heat = np.zeros((len(MODEL_ORDER), len(test_order)), dtype=np.int64)
    labels = [None] * len(test_order)
    for row in episodes.itertuples(index=False):
        model_index = MODEL_ORDER.index(row.model)
        heat[model_index, row.test_order] = int(row.early_stop)
        labels[row.test_order] = short_episode_label(row.episode_id)
    axes[0].imshow(
        heat,
        aspect="auto",
        interpolation="nearest",
        cmap=ListedColormap(["#d7f0df", "#ef8a8a"]),
        vmin=0,
        vmax=1,
    )
    axes[0].set_yticks(range(len(MODEL_ORDER)), [DISPLAY_NAMES[m] for m in MODEL_ORDER])
    axes[0].set_xticks(range(len(labels)), labels, rotation=55, ha="right", fontsize=8)
    axes[0].set_title(
        f"Test episodes: pre-{reference_label} early stop (red=STOP, green=none)"
    )
    for row_index in range(heat.shape[0]):
        for column_index in range(heat.shape[1]):
            axes[0].text(
                column_index,
                row_index,
                "STOP" if heat[row_index, column_index] else "OK",
                ha="center",
                va="center",
                fontsize=6.5,
                color="#6b0000" if heat[row_index, column_index] else "#14532d",
            )

    delay_values = [
        episodes.loc[
            (episodes["model"] == model) & episodes["detection_delay_sec_safe"].notna(),
            "detection_delay_sec_safe",
        ].to_numpy()
        for model in MODEL_ORDER
    ]
    positions = np.arange(1, len(MODEL_ORDER) + 1)
    boxes = axes[1].boxplot(
        delay_values,
        positions=positions,
        vert=False,
        widths=0.55,
        showmeans=True,
        patch_artist=True,
        meanprops={"marker": "D", "markerfacecolor": "black", "markeredgecolor": "black", "markersize": 4},
        medianprops={"color": "#111827", "linewidth": 1.6},
    )
    for box in boxes["boxes"]:
        box.set_facecolor("#8ecae6")
        box.set_alpha(0.75)
    rng = np.random.default_rng(42)
    for position, values in zip(positions, delay_values):
        jitter = rng.uniform(-0.12, 0.12, size=len(values))
        axes[1].scatter(values, position + jitter, s=24, alpha=0.8, color="#023047")
    axes[1].axvline(0.0, color="#6b7280", linestyle="--", linewidth=1)
    axes[1].set_yticks(positions, [DISPLAY_NAMES[m] for m in MODEL_ORDER])
    axes[1].invert_yaxis()
    axes[1].set_xlabel(f"First post-{reference_label} detection delay [s]")
    axes[1].set_title("Detection delay for episodes without an early stop")
    axes[1].grid(axis="x", alpha=0.25)
    fig.savefig(output_dir / "test_episode_distributions.png", dpi=180)
    plt.close(fig)

    report_columns = [
        "model",
        "early_stop_episodes",
        "early_stop_rate",
        "safe_episodes",
        "safe_missed_episodes",
        "delay_sec_min",
        "delay_sec_p25",
        "delay_sec_median",
        "delay_sec_p75",
        "delay_sec_p95",
        "delay_sec_max",
    ]
    report = [
        "# Episode-level test distributions",
        "",
        f"Timing reference: {timing_reference}; recorded Enter correction: -{reaction_delay:.3f} s.",
        "",
        f"An early stop means at least one positive prediction before {reference_label}. Detection-delay statistics include only episodes with no early stop; an episode stopped before the reference has no operational post-reference delay.",
        "",
        markdown_table(summary[report_columns]),
        "",
        "See `early_stop_by_episode.csv` and `safe_detection_delay_by_episode.csv` for the 20 episode matrices, `test_episode_metrics.csv` for the full long-form records, and `test_episode_distributions.png` for the distributions.",
        "",
    ]
    (output_dir / "README.md").write_text("\n".join(report), encoding="utf-8")

    print(summary[report_columns].to_string(index=False))
    print(f"\nWrote episode analysis to {output_dir}")


if __name__ == "__main__":
    main()
