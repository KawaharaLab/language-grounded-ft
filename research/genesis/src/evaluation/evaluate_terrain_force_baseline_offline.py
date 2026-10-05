"""Evaluate the lift-and-lower terrain rule on indexed 80-step force windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lift_and_lower_baseline import estimate_terrain_from_force_window
from lift_and_lower_object import EXPECTED_FORCE_COLS


CLASSES = ("flat surface", "step", "slope")


def normalize_terrain_label(value) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    if "flat" in text:
        return "flat surface"
    if "step" in text:
        return "step"
    if "slope" in text:
        return "slope"
    return None


def evaluate_index(index_path: Path, seq_len: int = 80) -> tuple[pd.DataFrame, dict]:
    index = pd.read_csv(index_path)
    place = index[index["action"].eq("place")].copy()
    place["true"] = (
        place["label"].map(normalize_terrain_label)
        if "label" in place
        else None
    )
    if "support_surface" in place:
        support_labels = place["support_surface"].map(normalize_terrain_label)
        if "true" not in place or place["true"] is None:
            place["true"] = support_labels
        else:
            place["true"] = place["true"].fillna(support_labels)
    unknown_ground_truth = int(place["true"].isna().sum())
    place = place.dropna(subset=["true"])

    records = []
    skip_counts = {"missing_csv": 0, "short_window": 0, "nonfinite": 0, "zero_force": 0}
    for csv_path, rows in place.groupby("csv_path", sort=False):
        path = Path(csv_path)
        if not path.is_file():
            skip_counts["missing_csv"] += len(rows)
            continue
        force = pd.read_csv(path, usecols=EXPECTED_FORCE_COLS).to_numpy(dtype=np.float32)
        for row in rows.itertuples(index=False):
            start = int(row.start)
            window = force[start : start + seq_len]
            if len(window) != seq_len:
                skip_counts["short_window"] += 1
                continue
            if not np.isfinite(window).all():
                skip_counts["nonfinite"] += 1
                continue
            if np.max(np.abs(window)) <= 1e-8:
                skip_counts["zero_force"] += 1
                continue
            result = estimate_terrain_from_force_window(window)
            features = result["features"]
            records.append(
                {
                    "csv_path": str(path),
                    "start": start,
                    "support_surface": getattr(row, "support_surface", None),
                    "annotation": getattr(row, "annotation", None),
                    "true": row.true,
                    "pred": result["prediction"],
                    "correct": result["prediction"] == row.true,
                    **features,
                }
            )

    predictions = pd.DataFrame(records)
    confusion = pd.crosstab(predictions["true"], predictions["pred"]).reindex(
        index=CLASSES, columns=CLASSES, fill_value=0
    )
    per_class = {}
    for label in CLASSES:
        count = int((predictions["true"] == label).sum())
        correct = int(confusion.loc[label, label])
        per_class[label] = {
            "count": count,
            "correct": correct,
            "recall": correct / count if count else None,
        }
    summary = {
        "index_path": str(index_path),
        "sequence_length": seq_len,
        "place_rows": int(index[index["action"].eq("place")].shape[0]),
        "unknown_ground_truth_rows": unknown_ground_truth,
        "usable_rows": len(predictions),
        "skipped": skip_counts,
        "correct": int(predictions["correct"].sum()),
        "accuracy": float(predictions["correct"].mean()),
        "macro_recall": float(np.mean([item["recall"] for item in per_class.values()])),
        "per_class": per_class,
        "confusion_matrix": {
            truth: {pred: int(confusion.loc[truth, pred]) for pred in CLASSES}
            for truth in CLASSES
        },
    }
    return predictions, summary


def save_confusion(summary: dict, path: Path, title: str) -> None:
    matrix = np.array(
        [[summary["confusion_matrix"][truth][pred] for pred in CLASSES] for truth in CLASSES]
    )
    fig, axis = plt.subplots(figsize=(6.2, 5.2))
    image = axis.imshow(matrix, cmap="Blues")
    for row in range(len(CLASSES)):
        for col in range(len(CLASSES)):
            axis.text(col, row, f"{matrix[row, col]:,}", ha="center", va="center")
    axis.set_xticks(range(len(CLASSES)), CLASSES)
    axis.set_yticks(range(len(CLASSES)), CLASSES)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Ground truth")
    axis.set_title(title)
    fig.colorbar(image, ax=axis)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-index", type=Path, required=True)
    parser.add_argument("--eval-index", type=Path, required=True)
    parser.add_argument("--train-name", default="train")
    parser.add_argument("--eval-name", default="eval")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, default=80)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    combined = {}
    for split, index_path in (
        (args.train_name, args.train_index),
        (args.eval_name, args.eval_index),
    ):
        predictions, summary = evaluate_index(index_path, args.seq_len)
        predictions.to_csv(args.output_dir / f"{split}_predictions.csv", index=False)
        save_confusion(summary, args.output_dir / f"{split}_confusion.png", f"{split}: terrain force baseline")
        combined[split] = summary
        print(
            f"{split}: {summary['correct']}/{summary['usable_rows']} "
            f"accuracy={summary['accuracy']:.4f} macro_recall={summary['macro_recall']:.4f}"
        )

    (args.output_dir / "summary.json").write_text(
        json.dumps(combined, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
