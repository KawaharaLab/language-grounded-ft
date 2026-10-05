import glob
import os
import sys
from pathlib import Path

import pandas as pd

from force_window_filter import filter_all_zero_force_windows

MODE = os.environ.get("MODE", "train_08142026")  # "train" or "eval"
DEFAULT_EXCLUDED_RUN_TOKENS = (
    "inclined_50,step_5cm,step_8cm" if "08142026" in MODE else ""
)
EXCLUDED_RUN_TOKENS = {
    value.strip().lower()
    for value in os.environ.get("EXCLUDED_RUN_TOKENS", DEFAULT_EXCLUDED_RUN_TOKENS).split(",")
    if value.strip()
}
REQUIRE_STEP_VARIANT = os.environ.get("REQUIRE_STEP_VARIANT", "").strip().lower()
REQUIRED_WOBBLE_VARIANT = os.environ.get(
    "REQUIRED_WOBBLE_VARIANT", ""
).strip().lower()
GENESIS_ROOT = Path(
    os.environ.get("FT_GENESIS_ROOT", Path(__file__).resolve().parents[3])
).expanduser().resolve()
DATA_DIR = str(Path(os.environ.get("FT_GENESIS_DATA_ROOT", GENESIS_ROOT / "data")) / MODE)
OUTPUT_NAME = os.environ.get("OUTPUT_NAME", MODE)
ANNOTATION_VARIANT = os.environ.get("ANNOTATION_VARIANT", "").strip()
out_path = f"{DATA_DIR}/{OUTPUT_NAME}.csv"


def run_is_excluded(deformation: str) -> bool:
    normalized = deformation.strip().lower()
    if any(token in normalized for token in EXCLUDED_RUN_TOKENS):
        return True
    if (
        REQUIRED_WOBBLE_VARIANT
        and "wobble" in normalized
        and REQUIRED_WOBBLE_VARIANT not in normalized
    ):
        return True
    return bool(
        REQUIRE_STEP_VARIANT
        and "step_" in normalized
        and REQUIRE_STEP_VARIANT not in normalized
    )

def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    if "action" not in df.columns:
        df["action"] = None
    if "interaction" not in df.columns:
        df["interaction"] = None
    if "weight" not in df.columns:
        df["weight"] = None
    if "placement_outcome" not in df.columns:
        df["placement_outcome"] = None

    df["action"] = df["action"].fillna("").astype(str).str.strip()
    df["interaction"] = df["interaction"].fillna("").astype(str).str.strip()
    df["weight"] = df["weight"].fillna("").astype(str).str.strip()
    df["placement_outcome"] = df["placement_outcome"].fillna("").astype(str).str.strip().str.lower()

    def _fallback_action(annotation: str) -> str:
        text = (annotation or "").lower()
        if "accidental drop" in text or "accidentally dropped" in text:
            accidental_action_phrases = {
                "while grasping": "grasp",
                "while lifting": "lift",
                "while holding": "hold",
                "while placing": "place",
                "while pressing": "press",
                "while releasing": "release",
            }
            for phrase, action in accidental_action_phrases.items():
                if phrase in text:
                    return action
            return ""
        if "placing" in text and "then releasing" in text:
            return "place then release"
        if "pressing" in text and "then releasing" in text:
            return "press then release"
        if "releasing" in text and "midair" in text:
            return "release midair"
        if "releasing" in text:
            return "release"
        if "pressing" in text:
            return "press"
        if "bumping" in text:
            return "bump"
        if "placing" in text:
            return "place"
        if "place firmly" in text:
            return "place firmly"
        if "place gently" in text:
            return "place gently"
        if "drop" in text:
            return "drop"
        if "hold" in text:
            return "hold"
        if "grasp" in text:
            return "grasp"
        if "lift" in text:
            return "lift"
        return ""

    def _fallback_interaction(annotation: str) -> str:
        text = (annotation or "").lower()
        if "accidental drop" in text or "accidentally dropped" in text:
            return "accidental drop"
        if "stable" in text:
            return "stable"
        if "slip" in text:
            return "slip"
        return ""

    def _fallback_weight(annotation: str) -> str:
        text = (annotation or "").lower()
        if "light" in text:
            return "light"
        if "heavy" in text:
            return "heavy"
        return ""

    def _fallback_placement_outcome(annotation: str) -> str:
        text = (annotation or "").lower()
        if "topple" in text:
            return "topple"
        if "remain" in text and "upright" in text:
            return "upright"
        return ""

    missing_action = df["action"] == ""
    if missing_action.any():
        df.loc[missing_action, "action"] = df.loc[missing_action, "annotation"].map(_fallback_action)

    missing_interaction = df["interaction"] == ""
    if missing_interaction.any():
        df.loc[missing_interaction, "interaction"] = df.loc[missing_interaction, "annotation"].map(_fallback_interaction)
    df["interaction"] = df["interaction"].replace(
        {
            "slow slip": "slip",
            "fast slip": "slip",
            "slowly slipping": "slip",
            "quickly slipping": "slip",
        }
    )
    # Canonicalize legacy rows that encoded the physical event as an action.
    # Re-annotation supplies the true command; this fallback retains the known
    # lift action and marks otherwise unrecoverable legacy actions as unknown.
    df.loc[df["action"] == "lift then accidental drop", "action"] = "lift"
    df.loc[df["action"] == "accidental drop", "action"] = "unknown"

    missing_weight = df["weight"] == ""
    if missing_weight.any():
        df.loc[missing_weight, "weight"] = df.loc[missing_weight, "annotation"].map(_fallback_weight)

    df["placement_outcome"] = df["placement_outcome"].replace(
        {
            "toppled": "topple",
            "topples after release": "topple",
            "remains upright": "upright",
        }
    )
    missing_placement_outcome = df["placement_outcome"] == ""
    if missing_placement_outcome.any():
        df.loc[missing_placement_outcome, "placement_outcome"] = df.loc[
            missing_placement_outcome, "annotation"
        ].map(_fallback_placement_outcome)

    # Placement outcome is defined only for windows labeled as placement.
    df.loc[~df["action"].isin(["place", "place then release"]), "placement_outcome"] = ""

    # Intentional release/drop categories do not have an ongoing grasp
    # interaction. Accidental drop remains an interaction in its own right.
    df.loc[
        df["action"].isin(["drop", "release", "release midair"]),
        "interaction",
    ] = ""

    def _to_label(row: pd.Series) -> str:
        action = row["action"]
        if action == "lift then accidental drop":
            return "accidental drop"
        if action == "hold":
            return "hold"
        if action in {"place gently"}:
            return "place gently"
        if action in {"place firmly"}:
            return "place firmly"
        if action in {"press", "press then release", "place then release"}:
            return action
        if "grasp" in action:
            return "grasp"
        if action in {
            "lift",
            "place",
            "bump",
            "hold",
            "accidental drop",
            "drop",
            "release",
            "release midair",
        }:
            return action
        return "unknown"

    df["label"] = df.apply(_to_label, axis=1)
    return df


def parse_annotation_path(fp: str) -> tuple[str, str, str]:
    """
    Parse object/material/deformation from:
    .../csv/{obj}/{material}/{deformation}/{obj}_{material}_{deformation}_annotations.csv
    """
    p = Path(fp)
    deformation = p.parent.name
    material = p.parent.parent.name
    obj_name = p.parent.parent.parent.name
    return obj_name, material, deformation


def main() -> int:
    annotation_dir = os.path.join(DATA_DIR, "csv", "*", "Rigid", "*")
    annotation_suffix = (
        f"_annotations_{ANNOTATION_VARIANT}.csv"
        if ANNOTATION_VARIANT
        else "_annotations.csv"
    )
    pattern = os.path.join(annotation_dir, f"*{annotation_suffix}")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No annotation files found with pattern: {pattern}")
        return 1

    dfs = []
    for fp in files:
        df = pd.read_csv(fp)
        obj_name, material, deformation = parse_annotation_path(fp)
        if deformation == "bump" or deformation.startswith("_attempt_"):
            continue
        if run_is_excluded(deformation):
            continue
        df["csv_path"] = f"{DATA_DIR}/csv/{obj_name}/{material}/{deformation}/{obj_name}_{material}_{deformation}.csv"
        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True)
    out = add_labels(out)
    before_filter = len(out)
    out, removed_zero = filter_all_zero_force_windows(out)
    if removed_zero:
        print(f"Removed {removed_zero} all-zero force windows ({before_filter} -> {len(out)}).")
    out.to_csv(out_path, index=False)

    print(f"Wrote {out_path} with {len(out)} rows from {len(dfs)} included files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
