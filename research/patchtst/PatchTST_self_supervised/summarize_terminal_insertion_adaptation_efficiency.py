#!/usr/bin/env python3
"""Summarize terminal-insertion data and gradient-step efficiency."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONDITIONS = {
    "frozen_linear": "Frozen + linear",
    "lora_r4": "LoRA (r=4)",
    "scratch": "Scratch End-to-End",
    "full_finetune": "Full fine-tuning",
}
EPISODE_COUNTS = (1, 2, 5, 10, 25, 50, 100)
SEEDS = (42, 43, 44, 45, 46)
COLORS = {
    "frozen_linear": "#d62728",
    "lora_r4": "#e377c2",
    "scratch": "#1f77b4",
    "full_finetune": "#17becf",
}

PRETRAINED_MODELS = {
    "contrastive": "Contrastive",
    "masked_reconstruction": "Masked reconstruction",
    "vae": "VAE",
    "vicreg": "VICReg",
    "jepa": "JEPA",
}
COMPARISON_ORDER = ("scratch", *PRETRAINED_MODELS.keys())
COMPARISON_NAMES = {
    "scratch": "Scratch PatchTST (end-to-end)",
    **PRETRAINED_MODELS,
}
COMPARISON_COLORS = {
    "scratch": "#4D4D4D",
    "contrastive": "#0072B2",
    "masked_reconstruction": "#D55E00",
    "vae": "#009E73",
    "vicreg": "#CC79A7",
    "jepa": "#E69F00",
}
COMPARISON_MARKERS = {
    "scratch": "X",
    "contrastive": "o",
    "masked_reconstruction": "s",
    "vae": "^",
    "vicreg": "D",
    "jepa": "P",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def markdown_table(table: pd.DataFrame, decimals: int = 4) -> str:
    lines = [
        "| " + " | ".join(map(str, table.columns)) + " |",
        "| " + " | ".join("---" for _ in table.columns) + " |",
    ]
    for row in table.itertuples(index=False, name=None):
        cells = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                cells.append("" if pd.isna(value) else f"{value:.{decimals}f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def first_episode_target(aggregate: pd.DataFrame, condition: str, target: float) -> str:
    rows = aggregate[
        (aggregate["condition"] == condition)
        & (aggregate["test_macro_f1_mean"] >= target)
    ].sort_values("episode_count")
    return str(int(rows.iloc[0]["episode_count"])) if len(rows) else "未到達"


def first_step_target(curves: pd.DataFrame, condition: str, target: float) -> str:
    rows = curves[
        (curves["condition"] == condition)
        & (curves["val_macro_f1_mean"] >= target)
    ].sort_values("gradient_steps")
    return str(int(rows.iloc[0]["gradient_steps"])) if len(rows) else "未到達"


def first_episode_rate_at_most(
    aggregate: pd.DataFrame, condition: str, target: float
) -> str:
    rows = aggregate[
        (aggregate["condition"] == condition)
        & (aggregate["test_early_stop_rate_mean"] <= target + 1e-12)
    ].sort_values("episode_count")
    return str(int(rows.iloc[0]["episode_count"])) if len(rows) else "未到達"


def load_pretraining_comparison(
    run_dir: Path, adaptation_rows: pd.DataFrame
) -> pd.DataFrame | None:
    """Combine matched frozen probes with the existing end-to-end baseline."""
    scratch = adaptation_rows[adaptation_rows["condition"] == "scratch"].copy()
    scratch["model"] = "scratch"
    columns = [
        "model",
        "seed",
        "episode_count",
        "test_macro_f1",
        "test_auroc",
        "test_early_stop_rate",
    ]
    frames = [scratch[columns]]
    expected_pairs = {(seed, count) for seed in SEEDS for count in EPISODE_COUNTS}

    for model in PRETRAINED_MODELS:
        path = run_dir / "pretrained_frozen" / model / "results.csv"
        if not path.exists():
            return None
        frame = pd.read_csv(path)
        actual_pairs = set(zip(frame["seed"], frame["episode_count"]))
        if actual_pairs != expected_pairs:
            raise ValueError(f"incomplete pretrained results: {path}")
        if set(frame["model"]) != {model}:
            raise ValueError(f"model mismatch in {path}")
        if not np.allclose(frame["threshold"], 0.5):
            raise ValueError(f"non-fixed threshold in {path}")
        frame = frame.rename(
            columns={"test_k1_early_stop_rate": "test_early_stop_rate"}
        )
        frames.append(frame[columns])

        # The nested labeled episode subset must match the scratch baseline for
        # every seed and data-budget point.
        for seed, episode_count in expected_pairs:
            config_path = (
                run_dir
                / "pretrained_frozen"
                / model
                / f"seed_{seed}"
                / f"n_{episode_count:03d}"
                / "config.json"
            )
            scratch_path = (
                run_dir
                / "scratch"
                / f"seed_{seed}"
                / f"n_{episode_count:03d}"
                / "metrics.json"
            )
            config = json.loads(config_path.read_text(encoding="utf-8"))
            scratch_metrics = json.loads(scratch_path.read_text(encoding="utf-8"))
            if config["selected_episode_ids"] != scratch_metrics["selected_train_episode_ids"]:
                raise ValueError(
                    f"episode subset mismatch: {model}, seed={seed}, n={episode_count}"
                )
            if config.get("relative_time_key") != "relative_to_reference_sec":
                raise ValueError(f"uncorrected time reference: {config_path}")

    return pd.concat(frames, ignore_index=True)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    rows = []
    histories = []
    for condition, display in CONDITIONS.items():
        for seed in SEEDS:
            for episode_count in EPISODE_COUNTS:
                condition_dir = (
                    run_dir / condition / f"seed_{seed}" / f"n_{episode_count:03d}"
                )
                metrics_path = condition_dir / "metrics.json"
                history_path = condition_dir / "history.csv"
                if not metrics_path.exists() or not history_path.exists():
                    raise FileNotFoundError(f"incomplete condition: {condition_dir}")
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                window = metrics["test_window"]
                sequence = metrics["test_sequence"]
                confusion = np.asarray(
                    window["confusion_matrix_before_after"], dtype=np.int64
                )
                rows.append(
                    {
                        "condition": condition,
                        "display": display,
                        "seed": seed,
                        "episode_count": episode_count,
                        "train_windows": int(metrics["train_windows"]),
                        "trainable_parameters": int(metrics["trainable_parameters"]),
                        "total_parameters": int(metrics["model_parameters"]),
                        "best_epoch": int(metrics["best_epoch"]),
                        "best_gradient_steps": int(metrics["best_gradient_steps"]),
                        "test_macro_f1": float(window["macro_f1"]),
                        "test_balanced_accuracy": float(window["balanced_accuracy"]),
                        "test_auroc": float(window["auroc"]),
                        "test_auprc": float(window["auprc"]),
                        "test_false_positive_windows": int(confusion[0, 1]),
                        "test_false_negative_windows": int(confusion[1, 0]),
                        "test_early_stop_rate": float(sequence["early_stop_rate"]),
                        "test_missed_rate": float(
                            sequence["missed_after_enter_rate"]
                        ),
                        "test_safe_delay_median_sec": sequence[
                            "detection_delay_sec_median"
                        ],
                    }
                )
                if episode_count == 100 and seed == 42:
                    history = pd.read_csv(history_path)
                    if len(history) != 300:
                        raise ValueError(
                            f"full-data history must have 300 epochs: {history_path}"
                        )
                    history["condition"] = condition
                    history["display"] = display
                    history["seed"] = seed
                    histories.append(history)

    seed_results = pd.DataFrame(rows)
    seed_results.to_csv(run_dir / "all_seed_results.csv", index=False)
    metric_columns = [
        "train_windows",
        "best_epoch",
        "best_gradient_steps",
        "test_macro_f1",
        "test_balanced_accuracy",
        "test_auroc",
        "test_auprc",
        "test_false_positive_windows",
        "test_false_negative_windows",
        "test_early_stop_rate",
        "test_missed_rate",
        "test_safe_delay_median_sec",
    ]
    aggregate = seed_results.groupby(
        ["condition", "display", "episode_count"], as_index=False
    )[metric_columns].agg(["mean", "std"])
    aggregate.columns = [
        "_".join(column).rstrip("_") if isinstance(column, tuple) else column
        for column in aggregate.columns
    ]
    aggregate.to_csv(run_dir / "data_efficiency_aggregate.csv", index=False)

    full_histories = pd.concat(histories, ignore_index=True)
    full_histories.to_csv(run_dir / "gradient_step_histories.csv", index=False)
    curve_metrics = {
        "val_macro_f1_at_fixed_threshold": ["mean", "std"],
        "val_balanced_accuracy_at_fixed_threshold": ["mean", "std"],
        "val_auroc": ["mean", "std"],
        "val_loss": ["mean", "std"],
    }
    curves = full_histories.groupby(
        ["condition", "display", "gradient_steps"], as_index=False
    ).agg(curve_metrics)
    curves.columns = [
        "_".join(column).rstrip("_") if isinstance(column, tuple) else column
        for column in curves.columns
    ]
    curves = curves.rename(
        columns={
            "val_macro_f1_at_fixed_threshold_mean": "val_macro_f1_mean",
            "val_macro_f1_at_fixed_threshold_std": "val_macro_f1_std",
            "val_balanced_accuracy_at_fixed_threshold_mean": "val_balanced_accuracy_mean",
            "val_balanced_accuracy_at_fixed_threshold_std": "val_balanced_accuracy_std",
        }
    )
    curves.to_csv(run_dir / "gradient_step_aggregate.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)
    data_panels = [
        ("test_macro_f1", "Test Macro-F1", 1.0),
        ("test_auroc", "Test ROC-AUC", 1.0),
        ("test_early_stop_rate", "Early-stop episodes [%]", 100.0),
    ]
    for axis, (metric, title, multiplier) in zip(axes, data_panels):
        for condition, display in CONDITIONS.items():
            group = aggregate[aggregate["condition"] == condition].sort_values(
                "episode_count"
            )
            x = group["episode_count"].to_numpy(dtype=float)
            mean = group[f"{metric}_mean"].to_numpy(dtype=float) * multiplier
            std = group[f"{metric}_std"].fillna(0).to_numpy(dtype=float) * multiplier
            axis.plot(
                x, mean, marker="o", color=COLORS[condition], label=display
            )
            axis.fill_between(
                x, mean - std, mean + std, color=COLORS[condition], alpha=0.14
            )
        axis.set_xscale("log")
        axis.set_xticks(EPISODE_COUNTS, [str(value) for value in EPISODE_COUNTS])
        axis.set_xlabel("Labeled training episodes")
        axis.set_title(title)
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8, loc="lower right")
    axes[2].set_ylim(-5, 105)
    fig.savefig(run_dir / "adaptation_data_efficiency.png", dpi=200)
    plt.close(fig)

    comparison_seeds = load_pretraining_comparison(run_dir, seed_results)
    comparison_endpoint_table = None
    if comparison_seeds is not None:
        comparison_seeds.to_csv(
            run_dir / "pretraining_comparison_all_seed_results.csv", index=False
        )
        comparison_metrics = [
            "test_macro_f1",
            "test_auroc",
            "test_early_stop_rate",
        ]
        comparison = comparison_seeds.groupby(
            ["model", "episode_count"], as_index=False
        )[comparison_metrics].agg(["mean", "std"])
        comparison.columns = [
            "_".join(column).rstrip("_") if isinstance(column, tuple) else column
            for column in comparison.columns
        ]
        comparison.to_csv(
            run_dir / "pretraining_comparison_aggregate.csv", index=False
        )

        panels = [
            ("test_macro_f1", "Test Macro-F1", 1.0),
            ("test_auroc", "Test ROC-AUC", 1.0),
            ("test_early_stop_rate", "Early-stop episodes [%]", 100.0),
        ]
        fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)
        for axis, (metric, title, multiplier) in zip(axes, panels):
            for model in COMPARISON_ORDER:
                group = comparison[comparison["model"] == model].sort_values(
                    "episode_count"
                )
                x = group["episode_count"].to_numpy(dtype=float)
                mean = group[f"{metric}_mean"].to_numpy(dtype=float) * multiplier
                std = group[f"{metric}_std"].fillna(0).to_numpy(dtype=float) * multiplier
                color = COMPARISON_COLORS[model]
                axis.fill_between(
                    x, mean - std, mean + std, color=color, alpha=0.11, linewidth=0
                )
                axis.errorbar(
                    x,
                    mean,
                    yerr=std,
                    marker=COMPARISON_MARKERS[model],
                    markersize=4.5,
                    linewidth=2.4 if model == "scratch" else 1.8,
                    linestyle="--" if model == "scratch" else "-",
                    elinewidth=0.9,
                    capsize=2.5,
                    color=color,
                    label=COMPARISON_NAMES[model],
                    zorder=5 if model == "scratch" else 2,
                )
            axis.set_xscale("log")
            axis.set_xticks(EPISODE_COUNTS, [str(value) for value in EPISODE_COUNTS])
            axis.set_xlabel("Labeled training episodes")
            axis.set_title(title)
            axis.grid(alpha=0.25, which="both")
        axes[0].set_ylim(0.4, 1.0)
        axes[0].legend(fontsize=8, loc="lower right", ncol=2)
        axes[2].set_ylim(-5, 105)
        fig.savefig(run_dir / "data_efficiency.png", dpi=300)
        fig.savefig(run_dir / "data_efficiency.pdf")
        plt.close(fig)

        fig, axis = plt.subplots(figsize=(9.2, 5.6), constrained_layout=True)
        for model in COMPARISON_ORDER:
            group = comparison[comparison["model"] == model].sort_values(
                "episode_count"
            )
            x = group["episode_count"].to_numpy(dtype=float)
            mean = group["test_macro_f1_mean"].to_numpy(dtype=float)
            std = group["test_macro_f1_std"].fillna(0).to_numpy(dtype=float)
            color = COMPARISON_COLORS[model]
            axis.fill_between(
                x, mean - std, mean + std, color=color, alpha=0.11, linewidth=0
            )
            axis.errorbar(
                x,
                mean,
                yerr=std,
                marker=COMPARISON_MARKERS[model],
                markersize=5,
                linewidth=2.6 if model == "scratch" else 2,
                linestyle="--" if model == "scratch" else "-",
                elinewidth=1,
                capsize=3,
                color=color,
                label=COMPARISON_NAMES[model],
                zorder=5 if model == "scratch" else 2,
            )
        axis.set_xscale("log")
        axis.set_xticks(EPISODE_COUNTS, [str(value) for value in EPISODE_COUNTS])
        axis.set_ylim(0.4, 1.0)
        axis.set_xlabel("Number of labeled training episodes")
        axis.set_ylabel("Test Macro-F1")
        axis.grid(alpha=0.25, which="both")
        axis.legend(loc="lower right", frameon=True, ncol=2, fontsize=9)
        axis.text(
            0.01,
            0.02,
            r"Mean $\pm$ 1 SD across 5 seeds",
            transform=axis.transAxes,
            fontsize=9,
            color="0.35",
        )
        figure_stem = "pretrained_vs_scratch_patchtst_macro_f1_5seed"
        fig.savefig(run_dir / f"{figure_stem}.png", dpi=300)
        fig.savefig(run_dir / f"{figure_stem}.pdf")
        plt.close(fig)

        comparison_endpoint = comparison[comparison["episode_count"] == 100].copy()
        comparison_endpoint["Condition"] = comparison_endpoint["model"].map(
            COMPARISON_NAMES
        )
        comparison_endpoint["Macro-F1"] = comparison_endpoint.apply(
            lambda row: (
                f"{row.test_macro_f1_mean:.4f} ± {row.test_macro_f1_std:.4f}"
            ),
            axis=1,
        )
        comparison_endpoint_table = comparison_endpoint[["Condition", "Macro-F1"]]
    else:
        print(
            "Pretrained-model curves are incomplete; kept adaptation-only plot.",
            flush=True,
        )

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    gradient_panels = [
        ("val_macro_f1", "Validation Macro-F1", False),
        ("val_loss", "Validation weighted BCE", True),
    ]
    for axis, (metric, title, log_y) in zip(axes, gradient_panels):
        for condition, display in CONDITIONS.items():
            group = curves[curves["condition"] == condition].sort_values(
                "gradient_steps"
            )
            x = group["gradient_steps"].to_numpy(dtype=float)
            mean = group[f"{metric}_mean"].to_numpy(dtype=float)
            std = group[f"{metric}_std"].fillna(0).to_numpy(dtype=float)
            axis.plot(x, mean, color=COLORS[condition], label=display)
            axis.fill_between(
                x,
                np.maximum(mean - std, 1e-8) if log_y else mean - std,
                mean + std,
                color=COLORS[condition],
                alpha=0.14,
            )
        axis.set_xscale("log")
        if log_y:
            axis.set_yscale("log")
        axis.set_xlabel("Cumulative optimizer steps (100 training episodes)")
        axis.set_title(title)
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8, loc="lower right")
    fig.savefig(run_dir / "gradient_step_efficiency.png", dpi=200)
    plt.close(fig)

    endpoint = aggregate[aggregate["episode_count"] == 100].copy()
    endpoint["Condition"] = endpoint["condition"].map(CONDITIONS)
    endpoint["Macro-F1"] = endpoint.apply(
        lambda row: f"{row.test_macro_f1_mean:.4f} ± {row.test_macro_f1_std:.4f}",
        axis=1,
    )
    endpoint["Early-stop"] = endpoint.apply(
        lambda row: (
            f"{100 * row.test_early_stop_rate_mean:.1f} ± "
            f"{100 * row.test_early_stop_rate_std:.1f}%"
        ),
        axis=1,
    )
    endpoint["Best steps"] = endpoint.apply(
        lambda row: (
            f"{row.best_gradient_steps_mean:.0f} ± "
            f"{row.best_gradient_steps_std:.0f}"
        ),
        axis=1,
    )
    endpoint_table = endpoint[["Condition", "Macro-F1", "Early-stop", "Best steps"]]

    efficiency_rows = []
    for condition, display in CONDITIONS.items():
        efficiency_rows.append(
            {
                "条件": display,
                "test Macro-F1平均≥0.95に必要なepisode数": first_episode_target(
                    aggregate, condition, 0.95
                ),
                "平均フライング率≤30%に必要なepisode数": first_episode_rate_at_most(
                    aggregate, condition, 0.30
                ),
                "validation Macro-F1平均≥0.95の最初のstep": first_step_target(
                    curves, condition, 0.95
                ),
                "validation Macro-F1平均≥0.99の最初のstep": first_step_target(
                    curves, condition, 0.99
                ),
            }
        )
    efficiency_table = pd.DataFrame(efficiency_rows)
    report = [
        "# Data efficiency / gradient-step efficiency",
        "",
        "補正済み（Enter - 0.30 s）データを用い、全条件で判定閾値を0.5に固定した。"
        "学習episodeはseedごとのnested subsetで、帯はseed 42–46の標準偏差である。",
        "",
        "## 100-episode endpoint",
        "",
        markdown_table(endpoint_table),
        "",
        "## 到達効率",
        "",
        markdown_table(efficiency_table),
        "",
        "Data efficiencyはvalidation lossが最小のcheckpointをtestで一度評価した結果。"
        "Gradient-step efficiencyは代表seed 42の100-episode学習中のvalidation値であり、"
        "途中のtest値は使用していない。optimizer step数は計算時間やFLOPsを表さない。",
        "",
        "LoRAはrank 4、alpha 8で、全attention blockのQ/Vのみを適応した。",
        "",
    ]
    if comparison_endpoint_table is not None:
        report.extend(
            [
                "## 事前学習方式別の比較",
                "",
                "ScratchはPatchTSTを分類headとともにEnd-to-End学習した。事前学習5方式はencoderを凍結し、選択した学習episodeだけでencoder出力を標準化して同一のLinear headを学習した。全条件で閾値0.5、Enter - 0.30 s補正、seed 42–46の同一nested subsetを用いた。full fine-tuningはこの比較図に含めない。",
                "",
                markdown_table(comparison_endpoint_table),
                "",
            ]
        )
    (run_dir / "README_ja.md").write_text("\n".join(report), encoding="utf-8")
    print(markdown_table(endpoint_table))
    print()
    print(markdown_table(efficiency_table))
    print(f"\nWrote efficiency summary to {run_dir}")


if __name__ == "__main__":
    main()
