#!/usr/bin/env python3
"""Audit train/eval label compatibility before supervised or linear-probe runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from classification_taxonomy import (
    TERRAIN0814_ACTION_CLASSES,
    TERRAIN0814_INTERACTION_CLASSES,
    TERRAIN0814_WEIGHT_CLASSES,
    canonicalize_classification_labels,
)


EXPECTED = {
    "action": TERRAIN0814_ACTION_CLASSES,
    "weight": TERRAIN0814_WEIGHT_CLASSES,
    "interaction": TERRAIN0814_INTERACTION_CLASSES,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def counts(frame: pd.DataFrame, target: str) -> dict[str, int]:
    return {
        str(label): int(count)
        for label, count in frame[target].value_counts(dropna=False).items()
        if not pd.isna(label)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=Path, required=True)
    parser.add_argument("--eval_dir", type=Path, required=True)
    parser.add_argument("--eval_pattern", required=True)
    parser.add_argument("--independent_eval_csv", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    train_path = args.train_csv.resolve()
    eval_dir = args.eval_dir.resolve()
    train_raw = pd.read_csv(train_path)
    required = {"csv_path", "start", "annotation", *EXPECTED}
    missing = required - set(train_raw.columns)
    if missing:
        raise ValueError(f"Training CSV lacks columns: {sorted(missing)}")
    train = canonicalize_classification_labels(train_raw, "terrain0814_eval")

    report = {
        "label_taxonomy": "terrain0814_eval",
        "train_csv": str(train_path),
        "train_csv_sha256": sha256(train_path),
        "train_rows": int(len(train)),
        "train_unique_windows": int(train[["csv_path", "start"]].drop_duplicates().shape[0]),
        "train_raw_counts": {target: counts(train_raw, target) for target in EXPECTED},
        "train_canonical_counts": {target: counts(train, target) for target in EXPECTED},
        "train_missing_labels": {
            target: int(train[target].isna().sum()) for target in EXPECTED
        },
        "evaluation": {},
    }
    if report["train_unique_windows"] != report["train_rows"]:
        raise ValueError("Training CSV contains duplicate (csv_path, start) windows")

    for target, expected_classes in EXPECTED.items():
        path = eval_dir / args.eval_pattern.format(target=target)
        frame = canonicalize_classification_labels(
            pd.read_csv(path), "terrain0814_eval"
        )
        if frame[target].isna().any():
            raise ValueError(f"{path.name} contains missing {target} labels")
        observed_counts = counts(frame, target)
        if set(observed_counts) != set(expected_classes):
            raise ValueError(
                f"{target} classes differ: observed={sorted(observed_counts)}, "
                f"expected={list(expected_classes)}"
            )
        if len(set(observed_counts.values())) != 1:
            raise ValueError(f"{target} evaluation is not balanced: {observed_counts}")
        if frame[["csv_path", "start"]].duplicated().any():
            raise ValueError(f"{target} evaluation contains duplicate windows")
        index_map = pd.read_csv(eval_dir / f"{target}_index_map.csv")[target].astype(str).tolist()
        if set(index_map) != set(expected_classes):
            raise ValueError(f"{target}_index_map.csv differs from the expected taxonomy")
        accidental_episode_rows = int(
            frame["csv_path"].astype(str).str.contains("accidental_drop", case=False).sum()
        )
        if target == "action" and accidental_episode_rows:
            raise ValueError(
                f"Action evaluation includes {accidental_episode_rows} accidental-drop episode rows"
            )
        report["evaluation"][target] = {
            "csv": str(path),
            "csv_sha256": sha256(path),
            "rows": int(len(frame)),
            "counts": observed_counts,
            "accidental_drop_episode_rows": accidental_episode_rows,
        }

    independent_path = eval_dir / args.independent_eval_csv
    independent = pd.read_csv(independent_path)
    target_presence = independent[list(EXPECTED)].notna().sum(axis=1)
    if not target_presence.eq(1).all():
        raise ValueError("Each independent-eval row must belong to exactly one target")
    expected_independent_rows = sum(value["rows"] for value in report["evaluation"].values())
    if len(independent) != expected_independent_rows:
        raise ValueError(
            f"Independent eval has {len(independent)} rows, expected {expected_independent_rows}"
        )
    report["independent_evaluation"] = {
        "csv": str(independent_path),
        "csv_sha256": sha256(independent_path),
        "rows": int(len(independent)),
        "one_target_per_row": True,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "data_audit.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    lines = [
        "# Supervised / SSL classification data audit",
        "",
        f"- Train rows: {len(train):,} (unique windows: {report['train_unique_windows']:,})",
        "- Taxonomy: `terrain0814_eval`",
        "- Action: `release midair` is merged into `release`; connected actions are masked only for the action head.",
        "- Weight and interaction labels on connected-action rows remain available.",
        "",
        "## Canonical training labels",
        "",
        "| Target | Non-missing class counts | Missing |",
        "| --- | --- | ---: |",
    ]
    for target in EXPECTED:
        lines.append(
            f"| {target} | `{report['train_canonical_counts'][target]}` | "
            f"{report['train_missing_labels'][target]:,} |"
        )
    lines.extend([
        "",
        "## Independent balanced evaluation",
        "",
        "| Target | Rows | Per-class counts | Accidental-drop episode rows |",
        "| --- | ---: | --- | ---: |",
    ])
    for target, value in report["evaluation"].items():
        lines.append(
            f"| {target} | {value['rows']:,} | `{value['counts']}` | "
            f"{value['accidental_drop_episode_rows']:,} |"
        )
    lines.append("")
    (args.output_dir / "data_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print(args.output_dir / "data_audit.md")


if __name__ == "__main__":
    main()
