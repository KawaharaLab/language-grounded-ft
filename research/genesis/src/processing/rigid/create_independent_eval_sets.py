import os
from pathlib import Path

import pandas as pd


DATASET = os.environ.get("DATASET", "eval_08142026")
SEED = int(os.environ.get("SEED", "42"))
WEIGHT_PER_CLASS = int(os.environ.get("WEIGHT_PER_CLASS", "3000"))
DEFAULT_EXCLUDE_RUN_TOKENS = (
    "inclined_50,step_5cm,step_8cm" if DATASET == "eval_08142026" else ""
)
EXCLUDE_RUN_TOKENS = [
    value.strip().lower()
    for value in os.environ.get("EXCLUDE_RUN_TOKENS", DEFAULT_EXCLUDE_RUN_TOKENS).split(",")
    if value.strip()
]
GENESIS_ROOT = Path(
    os.environ.get("FT_GENESIS_ROOT", Path(__file__).resolve().parents[3])
).expanduser().resolve()
DATA_DIR = Path(os.environ.get("FT_GENESIS_DATA_ROOT", GENESIS_ROOT / "data")) / DATASET
INPUT_CSV = DATA_DIR / os.environ.get("INPUT_CSV", f"{DATASET}.csv")
OUTPUT_PREFIX = os.environ.get("OUTPUT_PREFIX", DATASET)

ACTION_EXCLUDED_ACTIONS = {"unknown", ""}
ACTION_CLASSES = ["grasp", "hold", "lift", "place", "press", "release", "release midair"]
ACTION_BALANCE_CLASSES = ["grasp", "hold", "lift", "place", "press", "release"]
WEIGHT_CLASSES = ["heavy", "light"]
INTERACTION_CLASSES = ["accidental drop", "slip", "stable"]


def _normalize_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.lower()


def _accidental_drop_episode_mask(frame: pd.DataFrame) -> pd.Series:
    """Return rows whose source run is an accidental_drop_* episode."""
    if "csv_path" not in frame.columns:
        raise ValueError("csv_path is required to exclude accidental_drop_* episodes")
    run_names = frame["csv_path"].fillna("").astype(str).map(
        lambda value: Path(value).parent.name.lower()
    )
    return run_names.str.startswith("accidental_drop_")


def _balanced_sample(
    frame: pd.DataFrame,
    column: str,
    classes: list[str],
    per_class: int | None = None,
) -> tuple[pd.DataFrame, int]:
    counts = frame[column].value_counts()
    missing = [value for value in classes if int(counts.get(value, 0)) == 0]
    if missing:
        raise ValueError(f"No rows available for {column} classes: {missing}")
    maximum = min(int(counts[value]) for value in classes)
    target = maximum if per_class is None else min(per_class, maximum)
    sampled = [
        frame[frame[column] == value].sample(n=target, replace=False, random_state=SEED)
        for value in classes
    ]
    output = pd.concat(sampled, ignore_index=True)
    output = output.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    return output, target


def _write_set(frame: pd.DataFrame, target: str, target_per_class: int) -> None:
    output = frame.drop(
        columns=[column for column in frame.columns if column == "label" or column.endswith("_emb_index")],
        errors="ignore",
    )
    output_path = DATA_DIR / f"{OUTPUT_PREFIX}_{target}_balanced.csv"
    output.to_csv(output_path, index=False)
    counts = output[target].value_counts().sort_index().to_dict()
    print(
        f"{target}: saved {len(output)} rows ({target_per_class}/class) -> {output_path}\n"
        f"  distribution={counts}"
    )


def main() -> None:
    if WEIGHT_PER_CLASS < 1:
        raise ValueError("WEIGHT_PER_CLASS must be at least 1")
    frame = pd.read_csv(INPUT_CSV, low_memory=False)
    for column in ["action", "weight", "interaction"]:
        if column not in frame.columns:
            raise ValueError(f"Missing required column: {column}")
        frame[column] = _normalize_text(frame[column])

    if EXCLUDE_RUN_TOKENS:
        if "csv_path" not in frame.columns:
            raise ValueError("csv_path is required when EXCLUDE_RUN_TOKENS is set")
        run_names = frame["csv_path"].fillna("").astype(str).map(
            lambda value: Path(value).parent.name.lower()
        )
        excluded_run = pd.Series(False, index=frame.index)
        for token in EXCLUDE_RUN_TOKENS:
            excluded_run |= run_names.str.contains(token, regex=False)
        print(
            f"excluded run tokens {EXCLUDE_RUN_TOKENS}: "
            f"{int(excluded_run.sum())}/{len(frame)} rows"
        )
        frame = frame[~excluded_run].copy()

    # Connected actions are generated and retained for training, but are not
    # part of any of the independent single-action evaluation sets.
    base = frame[~frame["action"].str.contains(" then ", regex=False)].copy()
    accidental_drop_episode = _accidental_drop_episode_mask(base)
    print(
        "action: excluded accidental_drop_* episodes: "
        f"{int(accidental_drop_episode.sum())}/{len(base)} candidate rows"
    )

    action_candidates = base[
        ~accidental_drop_episode
        & ~base["interaction"].eq("accidental drop")
        & ~base["action"].isin(ACTION_EXCLUDED_ACTIONS)
        & base["action"].isin(ACTION_CLASSES)
    ].copy()
    action_candidates["__action_balance"] = action_candidates["action"].replace(
        {"release midair": "release"}
    )
    action_set, action_per_class = _balanced_sample(
        action_candidates, "__action_balance", ACTION_BALANCE_CLASSES
    )

    weight_candidates = base[base["weight"].isin(WEIGHT_CLASSES)].copy()
    weight_set, weight_per_class = _balanced_sample(
        weight_candidates,
        "weight",
        WEIGHT_CLASSES,
        per_class=WEIGHT_PER_CLASS,
    )

    interaction_candidates = base[base["interaction"].isin(INTERACTION_CLASSES)].copy()
    interaction_set, interaction_per_class = _balanced_sample(
        interaction_candidates, "interaction", INTERACTION_CLASSES
    )

    action_set = action_set.drop(columns="__action_balance")
    _write_set(action_set, "action", action_per_class)
    _write_set(weight_set, "weight", weight_per_class)
    _write_set(interaction_set, "interaction", interaction_per_class)


if __name__ == "__main__":
    main()
