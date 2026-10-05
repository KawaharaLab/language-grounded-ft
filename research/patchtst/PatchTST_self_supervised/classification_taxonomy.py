"""Canonical class labels shared by supervised and linear-probe evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd


TERRAIN0814_ACTION_CLASSES = (
    "grasp",
    "hold",
    "lift",
    "place",
    "press",
    "release",
)
TERRAIN0814_WEIGHT_CLASSES = ("heavy", "light")
TERRAIN0814_INTERACTION_CLASSES = ("accidental drop", "slip", "stable")
TAXONOMIES = ("raw", "terrain0814_eval")


def canonicalize_classification_labels(
    frame: pd.DataFrame,
    taxonomy: str,
) -> pd.DataFrame:
    """Return a copy with labels aligned to the independent 0814 evaluation.

    Connected actions do not have a single correct class in the six-way action
    evaluation, so only their action label is masked.  Their weight and
    interaction labels remain available.  ``release midair`` is merged into
    ``release`` exactly as in the evaluation set.
    """
    if taxonomy not in TAXONOMIES:
        raise ValueError(f"Unknown label taxonomy: {taxonomy!r}")
    output = frame.copy()
    if taxonomy == "raw":
        return output

    if "action" in output:
        action = output["action"].astype("string").str.strip().str.lower()
        action = action.replace({"release midair": "release"})
        output["action"] = action.where(action.isin(TERRAIN0814_ACTION_CLASSES), np.nan)

    if "weight" in output:
        weight = output["weight"].astype("string").str.strip().str.lower()
        output["weight"] = weight.where(weight.isin(TERRAIN0814_WEIGHT_CLASSES), np.nan)

    if "interaction" in output:
        interaction = output["interaction"].astype("string").str.strip().str.lower()
        interaction = interaction.replace({"fast slip": "slip", "slow slip": "slip"})
        output["interaction"] = interaction.where(
            interaction.isin(TERRAIN0814_INTERACTION_CLASSES), np.nan
        )
    return output
