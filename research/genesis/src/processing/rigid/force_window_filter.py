from pathlib import Path

import numpy as np
import pandas as pd

SEQUENCE_LENGTH = 80
FORCE_FEATURE_COLS = [
    "left_fx",
    "left_fy",
    "left_fz",
    "left_tx",
    "left_ty",
    "left_tz",
    "right_fx",
    "right_fy",
    "right_fz",
    "right_tx",
    "right_ty",
    "right_tz",
]


def force_window_is_all_zero(
    force_df: pd.DataFrame,
    start: int,
    length: int = SEQUENCE_LENGTH,
    force_cols: list[str] | None = None,
) -> bool:
    cols = FORCE_FEATURE_COLS if force_cols is None else force_cols
    missing_cols = [col for col in cols if col not in force_df.columns]
    if missing_cols:
        raise ValueError(f"Missing force columns: {missing_cols}")

    end = start + length
    seg = force_df.iloc[start:end]
    if len(seg) < length:
        return True
    return bool(np.all(seg[cols].to_numpy(dtype=float) == 0.0))


def filter_all_zero_force_windows(
    df: pd.DataFrame,
    sequence_length: int = SEQUENCE_LENGTH,
    force_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, int]:
    required_cols = {"csv_path", "start"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"CSV must contain columns: {sorted(missing_cols)}")

    cols = FORCE_FEATURE_COLS if force_cols is None else force_cols
    keep = pd.Series(True, index=df.index)

    for csv_path, group in df.groupby("csv_path", sort=False):
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"Force CSV not found: {path}")

        force_df = pd.read_csv(path, usecols=cols)
        force_values = force_df.to_numpy(dtype=float)
        for row_idx, start in group["start"].astype(int).items():
            segment = force_values[start : start + sequence_length]
            if len(segment) < sequence_length or np.all(segment == 0.0):
                keep.at[row_idx] = False

    removed = int((~keep).sum())
    return df.loc[keep].copy(), removed
