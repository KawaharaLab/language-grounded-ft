#!/usr/bin/env python3
"""Aggregate terminal-insertion test metrics after every condition finishes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


EXPECTED_MODELS = [
    "contrastive",
    "masked_reconstruction",
    "jepa",
    "vae",
    "vicreg",
    "scratch_tokenfusion",
    "scratch_mlp",
]


def dataframe_to_markdown(table: pd.DataFrame) -> str:
    """Render a compact Markdown table without pandas' optional tabulate package."""
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(
        (args.run_dir / "dataset" / "dataset_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    timing_reference = manifest.get("timing_reference", "recorded Enter key stamp")
    reaction_delay = float(manifest.get("enter_reaction_delay_sec", 0.0))
    rows = []
    missing = []
    for model in EXPECTED_MODELS:
        path = args.run_dir / model / "metrics.json"
        if not path.exists():
            missing.append(model)
            continue
        metrics = json.loads(path.read_text(encoding="utf-8"))
        window = metrics["test_window"]
        sequence = metrics["test_sequence"]
        rows.append(
            {
                "model": model,
                "encoder_frozen": metrics["encoder_frozen"],
                "model_parameters": metrics["model_parameters"],
                "trainable_parameters": metrics["trainable_parameters"],
                "best_epoch": metrics["best_epoch"],
                "best_val_loss": metrics["best_val_loss"],
                "threshold": metrics["selected_threshold"],
                "accuracy": window["accuracy"],
                "balanced_accuracy": window["balanced_accuracy"],
                "macro_f1": window["macro_f1"],
                "after_precision": window["after_precision"],
                "after_recall": window["after_recall"],
                "auroc": window["auroc"],
                "auprc": window["auprc"],
                "early_stop_rate": sequence["early_stop_rate"],
                "missed_after_enter_rate": sequence["missed_after_enter_rate"],
                "detection_delay_sec_median": sequence[
                    "detection_delay_sec_median"
                ],
                "detection_delay_sec_p95": sequence["detection_delay_sec_p95"],
                "wandb_url": metrics["wandb"]["url"],
            }
        )
    if missing:
        raise RuntimeError(f"missing completed conditions: {missing}")

    table = pd.DataFrame(rows).sort_values(
        ["macro_f1", "early_stop_rate", "detection_delay_sec_median"],
        ascending=[False, True, True],
    )
    table.to_csv(args.run_dir / "comparison.csv", index=False)
    display_columns = [
        "model",
        "macro_f1",
        "balanced_accuracy",
        "auroc",
        "auprc",
        "early_stop_rate",
        "missed_after_enter_rate",
        "detection_delay_sec_median",
        "best_epoch",
    ]
    markdown = [
        "# Terminal insertion benchmark",
        "",
        "All test metrics use the validation-selected checkpoint and threshold.",
        f"Timing reference: {timing_reference}; recorded Enter correction: -{reaction_delay:.3f} s.",
        "",
        dataframe_to_markdown(table[display_columns]),
        "",
    ]
    (args.run_dir / "summary.md").write_text("\n".join(markdown), encoding="utf-8")
    print(table[display_columns].to_string(index=False))


if __name__ == "__main__":
    main()
