#!/usr/bin/env python3
"""Compare pretrained linear/LoRA adaptation with scratch end-to-end training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
# Embed TrueType text instead of Type 3 glyphs in manuscript figures.
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODELS = {
    "contrastive": "Contrastive",
    "masked_reconstruction": "Masked reconstruction",
    "vae": "VAE",
    "vicreg": "VICReg",
    "jepa": "JEPA",
    "byol": "BYOL",
    "supervised": "Supervised",
}
DEFAULT_PRETRAINING = ("contrastive", "masked_reconstruction", "vae", "vicreg", "jepa")
MODEL_ORDER = ("scratch", *MODELS.keys())
MODEL_NAMES = {"scratch": "Scratch End-to-End", **MODELS}
COLORS = {
    "scratch": "#4D4D4D",
    "contrastive": "#0072B2",
    "masked_reconstruction": "#D55E00",
    "vae": "#009E73",
    "vicreg": "#CC79A7",
    "jepa": "#E69F00",
    "byol": "#E69F00",
    "supervised": "#56B4E9",
}
MARKERS = {
    "scratch": "X",
    "contrastive": "o",
    "masked_reconstruction": "s",
    "vae": "^",
    "vicreg": "D",
    "jepa": "P",
    "byol": "P",
    "supervised": "v",
}
EPISODE_COUNTS = (1, 2, 5, 10, 25, 50, 100)
SEEDS = (42, 43, 44, 45, 46)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--pretraining-models", nargs="+", choices=tuple(MODELS), default=DEFAULT_PRETRAINING)
    return parser.parse_args()


def markdown_table(table: pd.DataFrame) -> str:
    lines = [
        "| " + " | ".join(map(str, table.columns)) + " |",
        "| " + " | ".join("---" for _ in table.columns) + " |",
    ]
    for row in table.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(map(str, row)) + " |")
    return "\n".join(lines)


def scratch_metrics_path(run_dir: Path, seed: int, episode_count: int) -> Path:
    return (
        run_dir
        / "scratch"
        / f"seed_{seed}"
        / f"n_{episode_count:03d}"
        / "metrics.json"
    )


def load_scratch(run_dir: Path) -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        for episode_count in EPISODE_COUNTS:
            path = scratch_metrics_path(run_dir, seed, episode_count)
            if not path.exists():
                raise FileNotFoundError(path)
            metrics = json.loads(path.read_text(encoding="utf-8"))
            if metrics.get("model") != "scratch_tokenfusion":
                raise ValueError(f"scratch model mismatch: {path}")
            if not np.isclose(float(metrics["selected_threshold"]), 0.5):
                raise ValueError(f"scratch threshold mismatch: {path}")
            window = metrics["test_window"]
            sequence = metrics["test_sequence"]
            rows.append(
                {
                    "model": "scratch",
                    "model_display": MODEL_NAMES["scratch"],
                    "seed": seed,
                    "episode_count": episode_count,
                    "test_macro_f1": float(window["macro_f1"]),
                    "test_auroc": float(window["auroc"]),
                    "test_early_stop_rate": float(sequence["early_stop_rate"]),
                    "test_missed_rate": float(sequence["missed_after_enter_rate"]),
                    "trainable_parameters": int(metrics["trainable_parameters"]),
                    "best_epoch": int(metrics["best_epoch"]),
                }
            )
    return pd.DataFrame(rows)


def validate_episode_subset(
    run_dir: Path,
    seed: int,
    episode_count: int,
    selected_episode_ids: list[str],
    source: Path,
) -> None:
    scratch = json.loads(
        scratch_metrics_path(run_dir, seed, episode_count).read_text(encoding="utf-8")
    )
    if selected_episode_ids != scratch["selected_train_episode_ids"]:
        raise ValueError(
            f"episode subset differs from scratch for seed={seed}, "
            f"n={episode_count}: {source}"
        )


def load_pretrained_linear(run_dir: Path, models=DEFAULT_PRETRAINING) -> pd.DataFrame:
    rows = []
    expected_pairs = {(seed, count) for seed in SEEDS for count in EPISODE_COUNTS}
    for model in models:
        display = MODELS[model]
        base = run_dir / "pretrained_frozen" / model
        results_path = base / "results.csv"
        if not results_path.exists():
            raise FileNotFoundError(results_path)
        results = pd.read_csv(results_path)
        actual_pairs = set(zip(results["seed"], results["episode_count"]))
        if actual_pairs != expected_pairs or len(results) != len(expected_pairs):
            raise ValueError(f"incomplete pretrained linear results: {results_path}")
        if set(results["model"]) != {model}:
            raise ValueError(f"model mismatch: {results_path}")
        if not np.allclose(results["threshold"], 0.5):
            raise ValueError(f"threshold mismatch: {results_path}")

        for row in results.itertuples(index=False):
            seed = int(row.seed)
            episode_count = int(row.episode_count)
            config_path = (
                base
                / f"seed_{seed}"
                / f"n_{episode_count:03d}"
                / "config.json"
            )
            config = json.loads(config_path.read_text(encoding="utf-8"))
            if config.get("relative_time_key") != "relative_to_reference_sec":
                raise ValueError(f"uncorrected timing reference: {config_path}")
            validate_episode_subset(
                run_dir,
                seed,
                episode_count,
                config["selected_episode_ids"],
                config_path,
            )
            rows.append(
                {
                    "model": model,
                    "model_display": display,
                    "seed": seed,
                    "episode_count": episode_count,
                    "test_macro_f1": float(row.test_macro_f1),
                    "test_auroc": float(row.test_auroc),
                    "test_early_stop_rate": float(row.test_k1_early_stop_rate),
                    "test_missed_rate": float(row.test_k3_missed_rate),
                    "trainable_parameters": 769,
                    "best_epoch": int(row.best_epoch),
                }
            )
    return pd.DataFrame(rows)


def lora_result_dir(run_dir: Path, model: str) -> Path:
    if model == "contrastive":
        return run_dir / "lora_r4"
    return run_dir / "pretrained_adaptation" / model / "lora"


def load_pretrained_lora(run_dir: Path, models=DEFAULT_PRETRAINING) -> pd.DataFrame:
    rows = []
    for model in models:
        display = MODELS[model]
        base = lora_result_dir(run_dir, model)
        for seed in SEEDS:
            for episode_count in EPISODE_COUNTS:
                path = (
                    base
                    / f"seed_{seed}"
                    / f"n_{episode_count:03d}"
                    / "metrics.json"
                )
                if not path.exists():
                    raise FileNotFoundError(path)
                metrics = json.loads(path.read_text(encoding="utf-8"))
                if metrics.get("model") != model:
                    raise ValueError(f"model mismatch: {path}")
                if metrics.get("pretrained_adaptation") != "lora":
                    raise ValueError(f"adaptation mismatch: {path}")
                if not np.isclose(float(metrics["selected_threshold"]), 0.5):
                    raise ValueError(f"threshold mismatch: {path}")
                if metrics.get("timing_reference") != "corrected physical insertion estimate":
                    raise ValueError(f"timing-reference mismatch: {path}")
                if not np.isclose(float(metrics.get("enter_reaction_delay_sec", -1)), 0.3):
                    raise ValueError(f"reaction-delay mismatch: {path}")
                validate_episode_subset(
                    run_dir,
                    seed,
                    episode_count,
                    metrics["selected_train_episode_ids"],
                    path,
                )
                window = metrics["test_window"]
                sequence = metrics["test_sequence"]
                rows.append(
                    {
                        "model": model,
                        "model_display": display,
                        "seed": seed,
                        "episode_count": episode_count,
                        "test_macro_f1": float(window["macro_f1"]),
                        "test_auroc": float(window["auroc"]),
                        "test_early_stop_rate": float(sequence["early_stop_rate"]),
                        "test_missed_rate": float(sequence["missed_after_enter_rate"]),
                        "trainable_parameters": int(metrics["trainable_parameters"]),
                        "best_epoch": int(metrics["best_epoch"]),
                    }
                )
    return pd.DataFrame(rows)


def aggregate_results(rows: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "test_macro_f1",
        "test_auroc",
        "test_early_stop_rate",
        "test_missed_rate",
        "trainable_parameters",
        "best_epoch",
    ]
    aggregate = rows.groupby(
        ["model", "model_display", "episode_count"], as_index=False
    )[metrics].agg(["mean", "std"])
    aggregate.columns = [
        "_".join(column).rstrip("_") if isinstance(column, tuple) else column
        for column in aggregate.columns
    ]
    return aggregate


def plot_macro_f1(
    aggregate: pd.DataFrame,
    output_stem: Path,
) -> None:
    lower = aggregate["test_macro_f1_mean"] - aggregate[
        "test_macro_f1_std"
    ].fillna(0)
    y_min = max(0.0, np.floor((float(lower.min()) - 0.03) * 10) / 10)
    # Sized for a single-column paper figure: explicit, larger typography stays
    # legible after LaTeX scales the PDF to \linewidth.
    fig, axis = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    for model in MODEL_ORDER:
        group = aggregate[aggregate["model"] == model].sort_values("episode_count")
        if group.empty:
            continue
        x = group["episode_count"].to_numpy(dtype=float)
        mean = 100.0 * group["test_macro_f1_mean"].to_numpy(dtype=float)
        std = 100.0 * group["test_macro_f1_std"].fillna(0).to_numpy(dtype=float)
        axis.fill_between(
            x,
            mean - std,
            mean + std,
            color=COLORS[model],
            alpha=0.11,
            linewidth=0,
        )
        axis.errorbar(
            x,
            mean,
            yerr=std,
            marker=MARKERS[model],
            markersize=7,
            linewidth=3.2 if model == "scratch" else 2.6,
            linestyle="--" if model == "scratch" else "-",
            elinewidth=1.4,
            capsize=4,
            color=COLORS[model],
            label=MODEL_NAMES[model],
            zorder=6 if model == "scratch" else 2,
        )
    axis.set_xscale("log")
    axis.set_xticks(EPISODE_COUNTS, [str(value) for value in EPISODE_COUNTS])
    axis.set_ylim(100.0 * y_min, 100.0)
    axis.set_xlabel("Number of labeled training episodes", fontsize=18)
    axis.set_ylabel("Test Macro-F1 (%)", fontsize=18)
    axis.tick_params(axis="both", which="major", labelsize=15)
    axis.grid(alpha=0.25, which="both")
    axis.legend(loc="lower right", frameon=True, ncol=2, fontsize=13)
    fig.savefig(output_stem.with_suffix(".png"), dpi=300)
    fig.savefig(output_stem.with_suffix(".pdf"))
    plt.close(fig)


def endpoint_table(aggregate: pd.DataFrame) -> pd.DataFrame:
    endpoint = aggregate[aggregate["episode_count"] == 100].copy()
    endpoint["Condition"] = endpoint["model"].map(MODEL_NAMES)
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
    order = {model: index for index, model in enumerate(MODEL_ORDER)}
    endpoint["order"] = endpoint["model"].map(order)
    return endpoint.sort_values("order")[["Condition", "Macro-F1", "Early-stop"]]


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    scratch = load_scratch(run_dir)

    linear_rows = pd.concat(
        [scratch, load_pretrained_linear(run_dir, args.pretraining_models)], ignore_index=True
    )
    lora_rows = pd.concat(
        [scratch, load_pretrained_lora(run_dir, args.pretraining_models)], ignore_index=True
    )
    linear_aggregate = aggregate_results(linear_rows)
    lora_aggregate = aggregate_results(lora_rows)

    linear_rows.to_csv(
        run_dir / "pretrained_linear_vs_scratch_end_to_end_all_seed_results.csv",
        index=False,
    )
    linear_aggregate.to_csv(
        run_dir / "pretrained_linear_vs_scratch_end_to_end_aggregate.csv",
        index=False,
    )
    lora_rows.to_csv(
        run_dir / "pretrained_lora_vs_end_to_end_all_seed_results.csv", index=False
    )
    lora_aggregate.to_csv(
        run_dir / "pretrained_lora_vs_end_to_end_aggregate.csv", index=False
    )

    linear_stem = run_dir / "pretrained_linear_vs_scratch_end_to_end_macro_f1_5seed"
    lora_stem = run_dir / "pretrained_lora_vs_end_to_end_macro_f1_5seed"
    plot_macro_f1(
        linear_aggregate,
        linear_stem,
    )
    plot_macro_f1(
        lora_aggregate,
        lora_stem,
    )

    # Keep the established linear-plot filename synchronized with the explicit
    # name. The LoRA filename is overwritten with the corrected comparison.
    plot_macro_f1(
        linear_aggregate,
        run_dir / "pretrained_vs_scratch_patchtst_macro_f1_5seed",
    )

    linear_table = endpoint_table(linear_aggregate)
    lora_table = endpoint_table(lora_aggregate)
    report = [
        "# Pretrained adaptation vs scratch End-to-End",
        "",
        "End-to-Endは、事前学習を使わず分類タスクに特化して全PatchTSTをscratchから学習する条件を指す。事前学習済みencoderのfull fine-tuningは本比較に含めない。Enter - 0.30 sの補正済みデータ、閾値0.5、seed 42--46の同一nested episode subsetを使用し、帯とエラーバーは5 seedの標準偏差を示す。",
        "",
        "## Pretrained frozen encoder + Linear vs scratch End-to-End",
        "",
        markdown_table(linear_table),
        "",
        "## Pretrained encoder + LoRA vs scratch End-to-End",
        "",
        "LoRAは全attention blockのQ/Vへrank 4、alpha 8で適用する。",
        "",
        markdown_table(lora_table),
        "",
    ]
    (run_dir / "PRETRAINED_LORA_VS_END_TO_END.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    print("Linear probing vs scratch End-to-End")
    print(markdown_table(linear_table))
    print("\nLoRA vs scratch End-to-End")
    print(markdown_table(lora_table))
    print(f"\nWrote corrected comparisons to {run_dir}")


if __name__ == "__main__":
    main()
