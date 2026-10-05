#!/usr/bin/env python3
"""Create a minimal two-curve terminal-insertion adaptation overview."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


EPISODE_COUNTS = (1, 2, 5, 10, 25, 50, 100)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("aggregate_csv", type=Path)
    parser.add_argument("output_stem", type=Path)
    parser.add_argument("--fig-width", type=float, default=6.4)
    parser.add_argument("--fig-height", type=float, default=2.6)
    parser.add_argument("--y-max", type=float, default=1.0)
    return parser.parse_args()


def read_curves(path: Path) -> dict[str, list[float]]:
    selected = {"scratch": {}, "contrastive": {}}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            model = row["model"]
            if model in selected:
                selected[model][int(row["episode_count"])] = float(
                    row["test_macro_f1_mean"]
                )
    missing = {
        model: [count for count in EPISODE_COUNTS if count not in values]
        for model, values in selected.items()
    }
    if any(missing.values()):
        raise ValueError(f"Missing aggregate points: {missing}")
    return {
        model: [values[count] for count in EPISODE_COUNTS]
        for model, values in selected.items()
    }


def main() -> None:
    args = parse_args()
    curves = read_curves(args.aggregate_csv.expanduser().resolve())
    output_stem = args.output_stem.expanduser().resolve()
    output_stem.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(args.fig_width, args.fig_height))
    figure.patch.set_facecolor((1.0, 1.0, 1.0, 0.0))
    axis.set_facecolor("white")
    axis.plot(
        EPISODE_COUNTS,
        curves["scratch"],
        color="#262626",
        linewidth=2.2,
        linestyle=(0, (2.2, 2.2)),
        marker="x",
        markersize=5.5,
        markeredgewidth=1.6,
        zorder=2,
    )
    axis.plot(
        EPISODE_COUNTS,
        curves["contrastive"],
        color="#0072B2",
        linewidth=2.6,
        marker="o",
        markersize=5.0,
        markeredgewidth=0,
        zorder=3,
    )

    axis.set_xscale("log")
    axis.set_xticks(EPISODE_COUNTS, [str(value) for value in EPISODE_COUNTS])
    if args.y_max <= 1.0:
        axis.set_ylim(0.75, 1.0)
    else:
        axis.set_ylim(0.75, args.y_max)
    axis.set_yticks((0.75, 0.80, 0.85, 0.90, 0.95, 1.00))
    axis.set_xlabel("Labeled episodes", fontsize=11)
    axis.set_ylabel("Macro-F1", fontsize=11)
    axis.tick_params(axis="both", labelsize=9, length=3)
    axis.grid(axis="y", color="0.85", linewidth=0.65)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.margins(x=0.035)

    figure.tight_layout(pad=0.25)
    figure.savefig(
        output_stem.with_suffix(".png"),
        dpi=300,
        facecolor=figure.get_facecolor(),
        edgecolor="none",
    )
    figure.savefig(
        output_stem.with_suffix(".pdf"),
        facecolor=figure.get_facecolor(),
        edgecolor="none",
    )
    plt.close(figure)


if __name__ == "__main__":
    main()
