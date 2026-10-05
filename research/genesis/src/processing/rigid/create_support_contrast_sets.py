#!/usr/bin/env python3
"""Create object/action/interaction/weight-matched support contrast sets."""

from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


GENESIS_ROOT = Path(
    os.environ.get("FT_GENESIS_ROOT", Path(__file__).resolve().parents[3])
).expanduser().resolve()
DATA_ROOT = Path(os.environ.get("FT_GENESIS_DATA_ROOT", GENESIS_ROOT / "data"))
SUPPORT_CLASSES = ["fixed support", "wobbling support"]
SUPPORT_ACTIONS = ["place", "place then release", "press", "press then release"]


def _object_name(csv_path: str) -> str:
    return Path(csv_path).parents[2].name


def _mass_by_csv(paths: list[str]) -> dict[str, float]:
    result: dict[str, float] = {}
    for index, csv_path in enumerate(sorted(set(paths)), start=1):
        try:
            mass = pd.read_csv(csv_path, usecols=["obj_mass"], nrows=1)
            value = float(mass["obj_mass"].iloc[0])
        except (OSError, ValueError, IndexError, pd.errors.ParserError):
            value = float("nan")
        result[csv_path] = value
        if index % 1000 == 0:
            print(f"loaded masses: {index}/{len(set(paths))}", flush=True)
    return result


def _sample_matched(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    selected: list[pd.DataFrame] = []
    strata = ["object_name", "action", "interaction", "physical_weight"]
    for _, group in frame.groupby(strata, dropna=False, sort=True):
        counts = group["support_condition"].value_counts()
        target = min(int(counts.get(value, 0)) for value in SUPPORT_CLASSES)
        if target == 0:
            continue
        for support in SUPPORT_CLASSES:
            pool = group[group["support_condition"].eq(support)]
            indices = rng.choice(pool.index.to_numpy(), size=target, replace=False)
            selected.append(pool.loc[indices])
    if not selected:
        raise RuntimeError("No support-condition strata occur on both sides")
    output = pd.concat(selected, ignore_index=True)
    return output.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input-csv", default="")
    parser.add_argument("--output-csv", default="")
    args = parser.parse_args()

    data_dir = DATA_ROOT / args.dataset
    input_path = data_dir / (
        args.input_csv or f"{args.dataset}_support_contrast_all.csv"
    )
    output_path = data_dir / (
        args.output_csv or f"{args.dataset}_support_contrast_balanced.csv"
    )
    frame = pd.read_csv(input_path, low_memory=False)
    required = {"csv_path", "start", "action", "interaction", "support_condition", "annotation"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    for column in ["action", "interaction", "support_condition"]:
        frame[column] = frame[column].fillna("").astype(str).str.strip().str.lower()
    candidates = frame[
        frame["support_condition"].isin(SUPPORT_CLASSES)
        & frame["action"].isin(SUPPORT_ACTIONS)
        & frame["interaction"].isin(["stable", "slip"])
    ].copy()
    if args.dataset.startswith("eval"):
        # Preserve the current independent evaluation policy: no connected actions.
        candidates = candidates[~candidates["action"].str.contains(" then ", regex=False)]

    candidates["object_name"] = candidates["csv_path"].map(_object_name)
    mass_map = _mass_by_csv(candidates["csv_path"].tolist())
    candidates["physical_mass_kg"] = candidates["csv_path"].map(mass_map)
    before_missing_mass = len(candidates)
    candidates = candidates[np.isfinite(candidates["physical_mass_kg"])].copy()
    removed_missing_mass = before_missing_mass - len(candidates)
    candidates["physical_weight"] = np.where(
        candidates["physical_mass_kg"] >= 0.2, "heavy", "light"
    )

    output = _sample_matched(candidates, args.seed)
    output = output.drop(
        columns=[
            column
            for column in output.columns
            if column == "label" or column.endswith("_emb_index")
        ],
        errors="ignore",
    )
    output.to_csv(output_path, index=False)

    print(f"input candidates: {len(candidates)} (missing mass removed={removed_missing_mass})")
    print(f"saved {len(output)} rows -> {output_path}")
    print(f"objects: {output['object_name'].nunique()}")
    print("support:", output["support_condition"].value_counts().sort_index().to_dict())
    print(
        "support x action:",
        output.groupby(["support_condition", "action"]).size().to_dict(),
    )
    print(
        "support x interaction:",
        output.groupby(["support_condition", "interaction"]).size().to_dict(),
    )
    print(
        "support x physical_weight:",
        output.groupby(["support_condition", "physical_weight"]).size().to_dict(),
    )
    print("unique annotations:", output["annotation"].nunique())


if __name__ == "__main__":
    main()
