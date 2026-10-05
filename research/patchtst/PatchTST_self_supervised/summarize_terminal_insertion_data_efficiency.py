#!/usr/bin/env python3
"""Aggregate multi-seed frozen-probe terminal-insertion data-efficiency runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_ORDER = [
    "contrastive",
    "masked_reconstruction",
    "vae",
    "vicreg",
    "jepa",
    "raw_linear",
]

PRETRAINED_MODEL_ORDER = [
    "contrastive",
    "masked_reconstruction",
    "vae",
    "vicreg",
    "jepa",
]

DISPLAY_NAMES = {
    "contrastive": "Contrastive",
    "masked_reconstruction": "Masked reconstruction",
    "vae": "VAE",
    "vicreg": "VICReg",
    "jepa": "JEPA",
    "raw_linear": "Raw linear baseline",
}

METRICS = [
    "test_macro_f1",
    "test_balanced_accuracy",
    "test_auroc",
    "test_auprc",
    "test_k1_early_stop_rate",
    "test_k1_delay_median_sec",
    "test_k3_early_stop_rate",
    "test_k3_missed_rate",
    "test_k3_delay_median_sec",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


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


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    frames = []
    missing = []
    for model in MODEL_ORDER:
        path = run_dir / model / "results.csv"
        if not path.exists():
            missing.append(model)
            continue
        frames.append(pd.read_csv(path))
    if missing:
        raise RuntimeError(f"missing model results: {missing}")
    seeds = pd.concat(frames, ignore_index=True)
    expected_counts = {1, 2, 5, 10, 25, 50, 100}
    expected_seeds = {42, 43, 44, 45, 46}
    for model in MODEL_ORDER:
        subset = seeds[seeds["model"] == model]
        if set(subset["episode_count"]) != expected_counts:
            raise ValueError(f"incomplete episode counts for {model}")
        if set(subset["seed"]) != expected_seeds:
            raise ValueError(f"incomplete seeds for {model}")
    seeds.to_csv(run_dir / "all_seed_results.csv", index=False)

    aggregate = seeds.groupby(["model", "episode_count"], as_index=False)[METRICS].agg(
        ["mean", "std"]
    )
    aggregate.columns = [
        "_".join(item).rstrip("_") if isinstance(item, tuple) else str(item)
        for item in aggregate.columns
    ]
    aggregate.to_csv(run_dir / "aggregate.csv", index=False)

    colors = plt.cm.tab10(np.linspace(0, 1, len(MODEL_ORDER)))
    panels = [
        ("test_macro_f1", "Test Macro F1", 1.0),
        ("test_auroc", "Test AUROC", 1.0),
        ("test_k3_early_stop_rate", "K=3 early-stop rate", 100.0),
        ("test_k3_delay_median_sec", "K=3 median delay [ms]", 1000.0),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    for axis, (metric, title, multiplier) in zip(axes.flat, panels):
        for color, model in zip(colors, MODEL_ORDER):
            group = aggregate[aggregate["model"] == model].sort_values("episode_count")
            x = group["episode_count"].to_numpy(dtype=float)
            mean = group[f"{metric}_mean"].to_numpy(dtype=float) * multiplier
            std = group[f"{metric}_std"].fillna(0).to_numpy(dtype=float) * multiplier
            axis.plot(x, mean, marker="o", color=color, label=DISPLAY_NAMES[model])
            axis.fill_between(x, mean - std, mean + std, color=color, alpha=0.12)
        axis.set_xscale("log")
        axis.set_xticks(sorted(expected_counts), [str(value) for value in sorted(expected_counts)])
        axis.set_xlabel("Labeled training episodes")
        axis.set_title(title)
        axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8, loc="lower right")
    axes[1, 0].set_ylabel("Episode rate [%]")
    fig.savefig(run_dir / "data_efficiency_curves.png", dpi=180)
    plt.close(fig)

    # Paper-ready comparison of pretrained frozen encoders only.  The line is
    # the five-seed mean and both the band and capped bars show +/- one sample
    # standard deviation.  Raw-feature and full-finetuning conditions are
    # intentionally absent from this figure.
    pretrained_colors = {
        "contrastive": "#0072B2",
        "masked_reconstruction": "#D55E00",
        "vae": "#009E73",
        "vicreg": "#CC79A7",
        "jepa": "#E69F00",
    }
    pretrained_markers = {
        "contrastive": "o",
        "masked_reconstruction": "s",
        "vae": "^",
        "vicreg": "D",
        "jepa": "P",
    }
    fig, axis = plt.subplots(figsize=(9.2, 5.6), constrained_layout=True)
    pretrained_rows = []
    for model in PRETRAINED_MODEL_ORDER:
        group = aggregate[aggregate["model"] == model].sort_values("episode_count")
        x = group["episode_count"].to_numpy(dtype=float)
        mean = group["test_macro_f1_mean"].to_numpy(dtype=float)
        std = group["test_macro_f1_std"].fillna(0).to_numpy(dtype=float)
        color = pretrained_colors[model]
        axis.fill_between(x, mean - std, mean + std, color=color, alpha=0.13, linewidth=0)
        axis.errorbar(
            x,
            mean,
            yerr=std,
            marker=pretrained_markers[model],
            markersize=5,
            linewidth=2,
            elinewidth=1,
            capsize=3,
            color=color,
            label=DISPLAY_NAMES[model],
        )
        exported = group[
            ["model", "episode_count", "test_macro_f1_mean", "test_macro_f1_std"]
        ].copy()
        exported["num_seeds"] = len(expected_seeds)
        pretrained_rows.append(exported)

    axis.set_xscale("log")
    x_ticks = sorted(expected_counts)
    axis.set_xticks(x_ticks, [str(value) for value in x_ticks])
    axis.set_ylim(0.65, 1.0)
    axis.set_xlabel("Number of labeled training episodes")
    axis.set_ylabel("Test Macro F1")
    axis.grid(alpha=0.25, which="both")
    axis.legend(loc="lower right", frameon=True, ncol=2)
    axis.text(
        0.01,
        0.02,
        r"Mean $\pm$ 1 SD across 5 seeds",
        transform=axis.transAxes,
        fontsize=9,
        color="0.35",
    )
    fig.savefig(run_dir / "pretrained_data_efficiency_macro_f1_5seed.png", dpi=300)
    fig.savefig(run_dir / "pretrained_data_efficiency_macro_f1_5seed.pdf")
    plt.close(fig)
    pd.concat(pretrained_rows, ignore_index=True).to_csv(
        run_dir / "pretrained_data_efficiency_macro_f1_5seed.csv", index=False
    )

    at_100 = aggregate[aggregate["episode_count"] == 100].copy()
    at_100["model"] = at_100["model"].map(DISPLAY_NAMES)
    report_columns = [
        "model",
        "test_macro_f1_mean",
        "test_macro_f1_std",
        "test_auroc_mean",
        "test_k3_early_stop_rate_mean",
        "test_k3_missed_rate_mean",
        "test_k3_delay_median_sec_mean",
    ]
    report = [
        "# Frozen linear-probe data efficiency",
        "",
        "Every pretrained encoder is frozen. Its output is standardized using only the selected labeled training episodes, followed by an identical single Linear head. Input normalization remains fixed from the original 100-episode training split. Bands in the figure are one standard deviation across five seeds.",
        "",
        "`raw_linear` is a separately labeled raw-feature control, not a pretrained encoder.",
        "",
        "## 100-episode endpoint",
        "",
        markdown_table(at_100[report_columns]),
        "",
    ]
    (run_dir / "README.md").write_text("\n".join(report), encoding="utf-8")
    print(at_100[report_columns].to_string(index=False))
    print(f"\nWrote aggregate results to {run_dir}")


if __name__ == "__main__":
    main()
