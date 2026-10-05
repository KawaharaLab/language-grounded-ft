import csv
import json
import os
import time
from multiprocessing import Process
from pathlib import Path

import numpy as np
import pandas as pd

from force_window_filter import SEQUENCE_LENGTH, force_window_is_all_zero

BASE_PATH = Path(
    os.environ.get("FT_GENESIS_ROOT", Path(__file__).resolve().parent.parent.parent.parent)
).expanduser().resolve()
DATA_ROOT = Path(os.environ.get("FT_GENESIS_DATA_ROOT", BASE_PATH / "data")).expanduser().resolve()
DATASET = os.environ.get("DATASET", "train_08142026")
DEFAULT_EXCLUDED_RUN_TOKENS = (
    "inclined_50,step_5cm,step_8cm" if "08142026" in DATASET else ""
)
EXCLUDED_RUN_TOKENS = {
    value.strip().lower()
    for value in os.environ.get(
        "EXCLUDED_ANNOTATION_RUN_TOKENS", DEFAULT_EXCLUDED_RUN_TOKENS
    ).split(",")
    if value.strip()
}
INCLUDED_RUN_TOKENS = {
    value.strip().lower()
    for value in os.environ.get("INCLUDED_ANNOTATION_RUN_TOKENS", "").split(",")
    if value.strip()
}
REQUIRE_STEP_VARIANT = os.environ.get("REQUIRE_STEP_VARIANT", "").strip().lower()
REQUIRED_WOBBLE_VARIANT = os.environ.get(
    "REQUIRED_WOBBLE_VARIANT", ""
).strip().lower()
WINDOW_STRIDE = 10
# WINDOW_STRIDE = 40 if "eval" in DATASET else 10
MAX_ANNOTATION_PROCESSES = int(os.environ.get("MAX_ANNOTATION_PROCESSES", "8"))
if MAX_ANNOTATION_PROCESSES < 1:
    raise ValueError("MAX_ANNOTATION_PROCESSES must be at least 1.")
MAX_FORCE_ABS = 100.0
CONTACT_FORCE_COLS = ["left_fx", "left_fy", "left_fz", "right_fx", "right_fy", "right_fz"]
CONTACT_CONFIRM_STEPS = int(os.environ.get("CONTACT_CONFIRM_STEPS", "5"))
MAX_COM_STEP_M = float(os.environ.get("MAX_COM_STEP_M", "0.05"))
MAX_COM_XY_RADIUS_M = float(os.environ.get("MAX_COM_XY_RADIUS_M", "2.0"))
MIN_COM_Z_M = float(os.environ.get("MIN_COM_Z_M", "-0.1"))
MAX_COM_Z_M = float(os.environ.get("MAX_COM_Z_M", "2.0"))
LEGACY_TILE_Z_TOLERANCE_M = float(os.environ.get("LEGACY_TILE_Z_TOLERANCE_M", "0.008"))
INCLUDE_PLACEMENT_OUTCOME = os.environ.get(
    "INCLUDE_PLACEMENT_OUTCOME", "0"
).lower() in {"1", "true", "yes"}
ANNOTATION_VARIANT = os.environ.get("ANNOTATION_VARIANT", "").strip()
SLIP_SPEED_THRESHOLD_MPS = float(
    os.environ.get("SLIP_SPEED_THRESHOLD_MPS", "0.005")
)
SIMULATION_DT_S = float(os.environ.get("SIMULATION_DT_S", "0.01"))
MIN_INTERACTION_CONTACT_STEPS = int(
    os.environ.get("MIN_INTERACTION_CONTACT_STEPS", "5")
)
if SLIP_SPEED_THRESHOLD_MPS <= 0.0:
    raise ValueError("SLIP_SPEED_THRESHOLD_MPS must be positive.")
if SIMULATION_DT_S <= 0.0:
    raise ValueError("SIMULATION_DT_S must be positive.")
if MIN_INTERACTION_CONTACT_STEPS < 1:
    raise ValueError("MIN_INTERACTION_CONTACT_STEPS must be at least 1.")
PRESS_DESCENT_EPS_M = float(os.environ.get("PRESS_DESCENT_EPS_M", "1e-5"))
PLACE_ACTION = "place"
WOBBLE_SURFACE_STATIC_TOLERANCE_DEG = float(
    os.environ.get("WOBBLE_SURFACE_STATIC_TOLERANCE_DEG", "0.05")
)
if WOBBLE_SURFACE_STATIC_TOLERANCE_DEG < 0.0:
    raise ValueError("WOBBLE_SURFACE_STATIC_TOLERANCE_DEG must be non-negative.")


def _run_is_excluded(deformation: str) -> bool:
    normalized = deformation.strip().lower()
    if INCLUDED_RUN_TOKENS and not any(
        token in normalized for token in INCLUDED_RUN_TOKENS
    ):
        return True
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


def detect_bugs(force_df: pd.DataFrame, start: int, length: int = SEQUENCE_LENGTH) -> bool:
    """
    Exclude segments containing extreme contact-force readings.
    """
    end = start + length
    seg = force_df.iloc[start:end]
    if seg.empty:
        return True
    return bool(np.any(np.abs(seg[CONTACT_FORCE_COLS].to_numpy()) >= MAX_FORCE_ABS))


def _has_bimanual_contact(force_df: pd.DataFrame, start: int, length: int = SEQUENCE_LENGTH) -> bool:
    """
    Keep a window only when both fingers touch the object at least once in the window.
    """
    end = start + length
    seg = force_df.iloc[start:end]
    if seg.empty:
        return False
    if "obj_left_finger" not in seg.columns or "obj_right_finger" not in seg.columns:
        return False

    left = seg["obj_left_finger"].to_numpy(dtype=bool)
    right = seg["obj_right_finger"].to_numpy(dtype=bool)
    return bool(np.any(left) and np.any(right))


def _hold_has_sustained_support(
    action: str | None,
    confirmed_support: np.ndarray,
    start: int,
    length: int = SEQUENCE_LENGTH,
    includes_accidental_drop: bool = False,
) -> bool:
    """Reject passive supported waiting that would otherwise fall back to hold.

    An accidental-drop event keeps its intended action label even if the object
    reaches the support later in the same terminal window; the event is needed
    for the independent interaction task.
    """
    if action != "hold" or includes_accidental_drop:
        return False
    end = min(start + length, len(confirmed_support))
    return bool(np.any(confirmed_support[start:end]))


def detect_rollout_bug(force_df: pd.DataFrame) -> str | None:
    """Return a reason when the complete rollout is physically invalid."""
    available_force_cols = [column for column in CONTACT_FORCE_COLS if column in force_df.columns]
    if available_force_cols:
        forces = force_df[available_force_cols].to_numpy(dtype=float)
        if np.any(np.abs(forces) >= MAX_FORCE_ABS):
            return f"contact-force component >= {MAX_FORCE_ABS:g} N"

    required = ["obj_COM_x", "obj_COM_y", "obj_COM_z"]
    if any(column not in force_df.columns for column in required):
        return "missing COM columns"
    com = force_df[required].to_numpy(dtype=float)
    if not np.isfinite(com).all():
        return "non-finite COM"
    if len(com) > 1 and np.any(np.linalg.norm(np.diff(com, axis=0), axis=1) > MAX_COM_STEP_M):
        return f"COM step displacement > {MAX_COM_STEP_M:g} m"
    if np.any(np.linalg.norm(com[:, :2], axis=1) > MAX_COM_XY_RADIUS_M):
        return f"COM XY radius > {MAX_COM_XY_RADIUS_M:g} m"
    if np.any((com[:, 2] < MIN_COM_Z_M) | (com[:, 2] > MAX_COM_Z_M)):
        return f"COM z outside [{MIN_COM_Z_M:g}, {MAX_COM_Z_M:g}] m"
    return None


def split_for_model(force_df: pd.DataFrame) -> list[dict]:
    """
    Generate fixed-length windows for annotation.
    """
    if detect_rollout_bug(force_df) is not None:
        return []
    start = 0
    windows = []
    while start + SEQUENCE_LENGTH <= len(force_df):
        if (
            not detect_bugs(force_df, start)
            and not force_window_is_all_zero(force_df, start)
            and _has_bimanual_contact(force_df, start)
        ):
            windows.append({"start": start})
        start += WINDOW_STRIDE
    return windows


def _append_terminal_window(
    force_df: pd.DataFrame, windows: list[dict], event_idx: int | None
) -> None:
    """Include an episode-ending event that falls between regular stride starts."""
    if len(force_df) < SEQUENCE_LENGTH or event_idx is None:
        return
    if any(
        int(window["start"]) <= event_idx < int(window["start"]) + SEQUENCE_LENGTH
        for window in windows
    ):
        return
    start = len(force_df) - SEQUENCE_LENGTH
    if any(int(window["start"]) == start for window in windows):
        return
    if (
        not detect_bugs(force_df, start)
        and not force_window_is_all_zero(force_df, start)
        and _has_bimanual_contact(force_df, start)
    ):
        windows.append({"start": start})
        windows.sort(key=lambda window: int(window["start"]))


def _load_metadata(csv_path: str, obj_name: str, material: str, deformation: str) -> dict:
    metadata_path = os.path.join(csv_path, f"{obj_name}_{material}_metadata_{deformation}.json")
    if not os.path.exists(metadata_path):
        return {}
    with open(metadata_path, encoding="utf-8") as f:
        return json.load(f)


def _first_step(steps_df: pd.DataFrame, candidates: set[str]) -> int | None:
    rows = steps_df[steps_df["action"].astype(str).isin(candidates)]
    if rows.empty:
        return None
    return int(rows["step"].min())


def _first_sustained_state(
    series: np.ndarray,
    value: bool,
    start_step: int | None,
    confirm_steps: int = CONTACT_CONFIRM_STEPS,
    require_opposite_before: bool = False,
) -> int | None:
    """Find the first index starting a confirmed run of the requested state."""
    series = np.asarray(series, dtype=bool)
    if series.size < confirm_steps:
        return None
    begin = max(0, int(start_step) if start_step is not None else 0)
    begin = min(begin, len(series) - confirm_steps)
    opposite_seen = bool(np.any(series[:begin] != value))
    for index in range(begin, len(series) - confirm_steps + 1):
        if np.all(series[index : index + confirm_steps] == value):
            if not require_opposite_before or opposite_seen:
                return index
        if series[index] != value:
            opposite_seen = True
    return None


def _legacy_target_contact_series(force_df: pd.DataFrame, metadata: dict) -> np.ndarray:
    """Estimate target-tile contact for datasets whose recorded tile-contact column is broken."""
    required = {"obj_min_x", "obj_min_y", "obj_min_z", "obj_max_x", "obj_max_y"}
    target = metadata.get("target_xy") or {}
    if not required.issubset(force_df.columns) or not {"x", "y"}.issubset(target):
        return np.zeros(len(force_df), dtype=bool)

    tile = metadata.get("target_tile") or {}
    size_xyz = tile.get("size_xyz", (0.20, 0.20, 0.004))
    half_x = 0.5 * float(size_xyz[0])
    half_y = 0.5 * float(size_xyz[1])
    surface_z = float(tile.get("surface_center_z", 0.004))
    target_x, target_y = float(target["x"]), float(target["y"])

    overlaps_xy = (
        (force_df["obj_max_x"].to_numpy(dtype=float) >= target_x - half_x)
        & (force_df["obj_min_x"].to_numpy(dtype=float) <= target_x + half_x)
        & (force_df["obj_max_y"].to_numpy(dtype=float) >= target_y - half_y)
        & (force_df["obj_min_y"].to_numpy(dtype=float) <= target_y + half_y)
    )
    bottom_z = force_df["obj_min_z"].to_numpy(dtype=float)
    near_surface = np.abs(bottom_z - surface_z) <= LEGACY_TILE_Z_TOLERANCE_M
    return overlaps_xy & near_surface


def _target_contact_series(force_df: pd.DataFrame, metadata: dict) -> np.ndarray:
    recorded = (
        force_df["obj_target_tile"].fillna(False).to_numpy(dtype=bool)
        if "obj_target_tile" in force_df.columns
        else np.zeros(len(force_df), dtype=bool)
    )
    if int(metadata.get("contact_logging_version", 0)) >= 2:
        return recorded
    return recorded | _legacy_target_contact_series(force_df, metadata)


def _support_contact_series(force_df: pd.DataFrame, metadata: dict | None = None) -> np.ndarray:
    if force_df.empty:
        return np.array([], dtype=bool)
    tile = _target_contact_series(force_df, metadata or {})
    plane = (
        force_df["obj_plane"].fillna(False).to_numpy(dtype=bool)
        if "obj_plane" in force_df.columns
        else None
    )
    if tile is None and plane is None:
        return np.array([], dtype=bool)
    if tile is None:
        return plane
    if plane is None:
        return tile
    return np.logical_or(tile, plane)


def _first_support_contact_after(
    force_df: pd.DataFrame, metadata: dict, start_step: int | None
) -> int | None:
    return _first_sustained_state(
        _support_contact_series(force_df, metadata),
        True,
        start_step,
        require_opposite_before=True,
    )


def _first_support_detach_after(
    force_df: pd.DataFrame, metadata: dict, start_step: int | None
) -> int | None:
    return _first_sustained_state(
        _support_contact_series(force_df, metadata),
        False,
        start_step,
        require_opposite_before=True,
    )


def _first_obstacle_contact_after(force_df: pd.DataFrame, start_step: int | None) -> int | None:
    if "obj_obstacle" not in force_df.columns:
        return None
    obstacle = force_df["obj_obstacle"].to_numpy(dtype=bool)
    return _first_sustained_state(obstacle, True, start_step)


def _first_release_index(force_df: pd.DataFrame, start_step: int | None = None) -> int | None:
    if force_df.empty:
        return None
    if "obj_left_finger" not in force_df.columns or "obj_right_finger" not in force_df.columns:
        return None

    left = force_df["obj_left_finger"].to_numpy(dtype=bool)
    right = force_df["obj_right_finger"].to_numpy(dtype=bool)
    contact_either = np.logical_or(left, right)

    return _first_sustained_state(
        contact_either,
        False,
        start_step,
        require_opposite_before=True,
    )


def _confirmed_mask(series: np.ndarray, confirm_steps: int = CONTACT_CONFIRM_STEPS) -> np.ndarray:
    """Keep complete runs that persist for at least ``confirm_steps``."""
    series = np.asarray(series, dtype=bool)
    confirmed = np.zeros(series.size, dtype=bool)
    run_start = None
    for index, value in enumerate(series):
        if value and run_start is None:
            run_start = index
        if run_start is not None and (not value or index == len(series) - 1):
            run_end = index if value and index == len(series) - 1 else index - 1
            if run_end - run_start + 1 >= confirm_steps:
                confirmed[run_start : run_end + 1] = True
            run_start = None
    return confirmed


def _release_is_midair(support_contact: np.ndarray, release_event_idx: int | None) -> bool:
    """Classify hand opening by sustained support contact at its start."""
    if release_event_idx is None:
        return False
    index = int(release_event_idx)
    if index < 0 or index >= len(support_contact):
        return True
    confirmed_support = _confirmed_mask(support_contact)
    return not bool(confirmed_support[index])


def _weight_is_observable(force_segment: pd.DataFrame, support_segment: np.ndarray) -> bool:
    """Weight is observable only while both fingers carry an unsupported object."""
    if force_segment.empty or len(support_segment) != len(force_segment):
        return False
    required = {"obj_left_finger", "obj_right_finger"}
    if not required.issubset(force_segment.columns):
        return False
    airborne = _confirmed_mask(~np.asarray(support_segment, dtype=bool))
    bimanual = (
        force_segment["obj_left_finger"].to_numpy(dtype=bool)
        & force_segment["obj_right_finger"].to_numpy(dtype=bool)
    )
    return bool(np.any(airborne & bimanual))


def _pressing_series(
    force_df: pd.DataFrame,
    metadata: dict,
    place_idx: int | None,
    release_command_step: int | None,
    press_command_step: int | None = None,
) -> np.ndarray:
    """Detect active supported pressing, excluding passive post-place waiting."""
    if force_df.empty or place_idx is None:
        return np.zeros(len(force_df), dtype=bool)

    support = _support_contact_series(force_df, metadata)
    finger = np.logical_or(
        force_df["obj_left_finger"].to_numpy(dtype=bool),
        force_df["obj_right_finger"].to_numpy(dtype=bool),
    )
    eef_z = force_df["eef_z"].to_numpy(dtype=float)
    descending = np.r_[False, np.diff(eef_z) < -PRESS_DESCENT_EPS_M]
    commanded_press = np.zeros(len(force_df), dtype=bool)
    if press_command_step is not None:
        command_start = max(0, int(press_command_step))
        command_end = (
            len(force_df)
            if release_command_step is None
            else min(len(force_df), max(command_start, int(release_command_step)))
        )
        commanded_press[command_start:command_end] = True
    pressing = support & finger & (descending | commanded_press)
    pressing[: place_idx + 1] = False
    if release_command_step is not None:
        pressing[max(0, int(release_command_step)) :] = False
    return _confirmed_mask(pressing)


def _first_grasp_contact_index(force_df: pd.DataFrame, start_step: int | None = None) -> int | None:
    if force_df.empty:
        return None
    if "obj_left_finger" not in force_df.columns or "obj_right_finger" not in force_df.columns:
        return None

    left = force_df["obj_left_finger"].to_numpy(dtype=bool)
    right = force_df["obj_right_finger"].to_numpy(dtype=bool)
    bimanual_contact = np.logical_and(left, right)

    begin = max(0, int(start_step) if start_step is not None else 0)
    begin = min(begin, len(bimanual_contact) - 1)
    if bimanual_contact[begin]:
        return begin

    for i in range(begin + 1, len(bimanual_contact)):
        if (not bimanual_contact[i - 1]) and bimanual_contact[i]:
            return i
    return None


def _infer_place_mode(deformation: str, metadata: dict, steps_df: pd.DataFrame) -> str:
    place_mode = str(metadata.get("place_mode") or "").strip()
    if place_mode.startswith("inclined_"):
        return place_mode
    if place_mode.startswith("step_"):
        return place_mode
    if place_mode in {
        "drop",
        "drop_simple",
        "gentle",
        "gentle_simple",
        "firm",
        "firm_simple",
        "bump",
        "inclined",
        "wobble",
    }:
        return place_mode

    if deformation.startswith("early_"):
        tail = deformation.replace("early_", "", 1)
        if tail.startswith("inclined_"):
            return tail
        if tail.startswith("step_"):
            return tail
        if tail in {"drop", "gentle", "firm", "bump", "inclined", "wobble"}:
            return tail

    actions = set(steps_df["action"].astype(str).tolist())
    if "drop" in actions:
        return "drop"
    if "bump_push" in actions:
        return "bump"
    if "gentle_place" in actions:
        return "gentle"
    if "press_place" in actions:
        return "firm"
    return "gentle"


def _is_early_drop_case(deformation: str, metadata: dict, steps_df: pd.DataFrame) -> bool:
    if deformation.startswith("early_"):
        return True
    if bool(metadata.get("terminated_early_drop", False)):
        return True
    return bool(steps_df["action"].astype(str).str.contains("early_drop").any())


def _longest_true_run(values: np.ndarray) -> int:
    """Return the longest consecutive run of true values."""
    values = np.asarray(values, dtype=bool)
    longest = 0
    current = 0
    for value in values:
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _window_duration_s(force_segment: pd.DataFrame) -> float:
    """Return elapsed simulation time represented by a fixed-length window."""
    if len(force_segment) < 2:
        return 0.0
    if "step" in force_segment.columns:
        steps = pd.to_numeric(force_segment["step"], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(steps[[0, -1]]).all() and steps[-1] > steps[0]:
            return float((steps[-1] - steps[0]) * SIMULATION_DT_S)
    return float((len(force_segment) - 1) * SIMULATION_DT_S)


def _interaction_measurements(force_segment: pd.DataFrame) -> dict[str, float | int | bool]:
    """Measure window-average slip in the gripper's tangential plane.

    The observed motion is the object COM relative to the midpoint of the two
    fingertips. A fixed grasp axis is estimated from the median finger-to-finger
    direction while both fingers are in contact; motion along that axis is removed.
    The maximum tangential excursion from the window's initial relative pose is
    divided by the window's elapsed time. Using the complete window retains displacement across brief
    contact interruptions caused by stick-slip while avoiding a noisy one-step
    numerical derivative.
    """
    required = {
        "obj_COM_x",
        "obj_COM_y",
        "obj_COM_z",
        "left_finger_x",
        "left_finger_y",
        "left_finger_z",
        "right_finger_x",
        "right_finger_y",
        "right_finger_z",
        "obj_left_finger",
        "obj_right_finger",
    }
    missing = required - set(force_segment.columns)
    if missing:
        raise ValueError(f"Missing columns for interaction estimation: {sorted(missing)}")

    left_contact = force_segment["obj_left_finger"].to_numpy(dtype=bool)
    right_contact = force_segment["obj_right_finger"].to_numpy(dtype=bool)
    bimanual = left_contact & right_contact
    longest_contact_run = _longest_true_run(bimanual)
    observable = longest_contact_run >= MIN_INTERACTION_CONTACT_STEPS

    result: dict[str, float | int | bool] = {
        "observable": observable,
        "bimanual_contact_steps": int(bimanual.sum()),
        "longest_bimanual_contact_run": int(longest_contact_run),
        "tangential_displacement_m": float("nan"),
        "tangential_speed_mps": float("nan"),
        "window_duration_s": _window_duration_s(force_segment),
    }
    if not observable:
        return result

    com_pos = force_segment[["obj_COM_x", "obj_COM_y", "obj_COM_z"]].to_numpy(
        dtype=float
    )
    left_finger_pos = force_segment[
        ["left_finger_x", "left_finger_y", "left_finger_z"]
    ].to_numpy(dtype=float)
    right_finger_pos = force_segment[
        ["right_finger_x", "right_finger_y", "right_finger_z"]
    ].to_numpy(dtype=float)
    finger_center = 0.5 * (left_finger_pos + right_finger_pos)

    grasp_directions = right_finger_pos[bimanual] - left_finger_pos[bimanual]
    grasp_axis = np.median(grasp_directions, axis=0)
    axis_norm = float(np.linalg.norm(grasp_axis))
    duration_s = float(result["window_duration_s"])
    if not np.isfinite(axis_norm) or axis_norm <= 1e-9 or duration_s <= 0.0:
        return result
    grasp_axis /= axis_norm

    # Ignore the pre-contact approach and any motion after final loss of both
    # contacts. Samples between first and last bimanual contact remain included,
    # so a brief contact interruption during stick-slip does not erase motion.
    contact_indices = np.flatnonzero(bimanual)
    observation = slice(int(contact_indices[0]), int(contact_indices[-1]) + 1)
    relative_position = (com_pos - finger_center)[observation]
    normal_component = np.outer(relative_position @ grasp_axis, grasp_axis)
    tangential_position = relative_position - normal_component
    finite = np.isfinite(tangential_position).all(axis=1)
    tangential_position = tangential_position[finite]
    if len(tangential_position) < 2:
        return result

    displacement_m = float(
        np.max(np.linalg.norm(tangential_position - tangential_position[0], axis=1))
    )
    result["tangential_displacement_m"] = displacement_m
    result["tangential_speed_mps"] = displacement_m / duration_s
    return result


def _estimate_interaction(force_segment: pd.DataFrame) -> str | None:
    measurements = _interaction_measurements(force_segment)
    if not bool(measurements["observable"]):
        return None
    speed_mps = float(measurements["tangential_speed_mps"])
    if not np.isfinite(speed_mps):
        return None
    if speed_mps >= SLIP_SPEED_THRESHOLD_MPS:
        return "slip"
    return "stable"


def _commanded_action_at_step(steps_df: pd.DataFrame, event_step: int | None) -> str | None:
    """Map the latest command before an event to the intended action taxonomy."""
    if event_step is None or steps_df.empty or not {"action", "step"}.issubset(steps_df.columns):
        return None
    rows = steps_df[pd.to_numeric(steps_df["step"], errors="coerce") <= int(event_step)].copy()
    if rows.empty:
        return None
    rows["step"] = pd.to_numeric(rows["step"], errors="coerce")
    rows = rows.dropna(subset=["step"]).sort_values("step")
    command_map = {
        "grasp": "grasp",
        "hold": "hold",
        "lift": "lift",
        "move_to_target": "hold",
        "place": "place",
        "gentle_place": "place",
        "press_place": "press",
        "release": "release",
    }
    for command in reversed(rows["action"].astype(str).tolist()):
        mapped = command_map.get(command)
        if mapped is not None:
            return mapped
    return None


def _determine_action(
    start: int,
    force_segment: pd.DataFrame,
    first_support_contact_idx: int | None,
    first_grasp_contact_idx: int | None,
    first_lift_idx: int | None,
    accidental_release_idx: int | None,
    release_event_idx: int | None,
    release_midair: bool,
    first_obstacle_contact_idx: int | None,
    lift_command_step: int | None,
    early_drop_case: bool,
    place_mode: str,
    pressing: np.ndarray,
    accidental_drop_action: str | None = None,
) -> str | None:
    seg_end = start + len(force_segment) - 1
    includes_grasp_contact = first_grasp_contact_idx is not None and (start <= first_grasp_contact_idx <= seg_end)
    includes_lift = first_lift_idx is not None and (start <= first_lift_idx <= seg_end)
    includes_accidental_release = accidental_release_idx is not None and start <= accidental_release_idx <= seg_end
    includes_release = release_event_idx is not None and start <= release_event_idx <= seg_end
    includes_place = first_support_contact_idx is not None and start <= first_support_contact_idx <= seg_end
    includes_obstacle_contact = (
        first_obstacle_contact_idx is not None and (start <= first_obstacle_contact_idx <= seg_end)
    )
    segment_pressing = pressing[start : seg_end + 1]
    includes_pressing = bool(np.any(segment_pressing))

    # Accidental drop is an interaction, not an action.  Preserve the intended
    # command active at the event so the two can be evaluated independently.
    includes_lift_command = lift_command_step is not None and start <= lift_command_step <= seg_end
    if early_drop_case and includes_accidental_release:
        if accidental_drop_action is not None:
            return accidental_drop_action
        return "lift" if includes_lift_command or includes_lift else "hold"

    if place_mode == "bump" and includes_obstacle_contact:
        return "bump"

    if includes_release:
        if release_midair:
            return "release midair"
        if includes_place and first_support_contact_idx <= release_event_idx:
            return "place then release"
        if includes_pressing:
            return "press then release"
        return "release"

    # A window containing the initial support-contact event is always place,
    # even if downward pressing begins near its end.  This prevents the
    # deliberately unsupported "place then press" composite label.
    if includes_place:
        return PLACE_ACTION

    if includes_pressing:
        return "press"

    # Once placement or intentional release has happened, unsupported waiting,
    # settling and retreat have no action annotation.
    if first_support_contact_idx is not None and first_support_contact_idx < start:
        return None
    if release_event_idx is not None and release_event_idx < start:
        return None

    if includes_grasp_contact and includes_lift:
        return "grasp then lift"
    if includes_grasp_contact:
        return "grasp"
    if includes_lift:
        return "lift"
    return "hold"


def _weight_descriptor(force_segment: pd.DataFrame, observable: bool = True) -> str:
    if not observable or "obj_mass" not in force_segment.columns or force_segment.empty:
        return "an object"
    mass = float(force_segment["obj_mass"].iloc[0])
    if mass >= 0.2:
        return "a heavy object"
    return "a light object"


def _placement_outcome(action: str, metadata: dict) -> str | None:
    if not INCLUDE_PLACEMENT_OUTCOME or action not in {PLACE_ACTION, "place then release"}:
        return None
    toppled = metadata.get("toppled")
    if toppled is True:
        return "topple"
    if toppled is False:
        return "upright"
    return None


def _target_surface_name(place_mode: str) -> str:
    return {
        "inclined_10": "a gentle slope",
        "inclined_20": "a moderate slope",
        "inclined_30": "a steep slope",
        "step_0.5cm": "a small step",
        "step_1.0cm": "a medium step",
        "step_2cm": "a large step",
        "wobble": "a wobbling surface",
    }.get(place_mode, "a fixed flat surface")


def _wobble_surface_name(force_segment: pd.DataFrame) -> str:
    """Describe a wobble deck from its motion within this exact time window."""
    column = "target_surface_angle_deg"
    if column not in force_segment.columns or force_segment.empty:
        # Legacy wobble runs did not record the hinge angle, so they cannot be
        # separated into moving and stationary windows retrospectively.
        return "a wobbling surface"

    angles_deg = pd.to_numeric(force_segment[column], errors="coerce").to_numpy(
        dtype=float
    )
    angles_deg = angles_deg[np.isfinite(angles_deg)]
    if angles_deg.size == 0:
        return "a wobbling surface"

    if float(np.ptp(angles_deg)) > WOBBLE_SURFACE_STATIC_TOLERANCE_DEG:
        return "a wobbling surface"

    representative_angle_deg = float(np.median(np.abs(angles_deg)))
    names_by_angle = {
        0.0: "a flat surface",
        10.0: "a gentle slope",
        20.0: "a moderate slope",
        30.0: "a steep slope",
    }
    nearest_angle_deg = min(
        names_by_angle,
        key=lambda candidate: abs(representative_angle_deg - candidate),
    )
    return names_by_angle[nearest_angle_deg]


def _support_condition(support_surface: str | None) -> str | None:
    """Return a contrastive support label without reusing interaction stability terms."""
    if support_surface in {"a fixed flat surface", "a fixed level support"}:
        return "fixed support"
    if support_surface in {"a wobbling surface", "a wobbling support"}:
        return "wobbling support"
    return None


def _support_surface_for_action(
    action: str,
    interaction: str | None,
    start: int,
    force_df: pd.DataFrame,
    metadata: dict,
    place_mode: str,
    first_support_contact_idx: int | None,
    pressing: np.ndarray,
) -> str | None:
    """Describe the surface actually contacted during place/press actions."""
    if interaction == "accidental drop" or not any(
        component in {"place", "press"} for component in action.split(" then ")
    ):
        return None

    end = min(len(force_df), start + SEQUENCE_LENGTH)
    target = _confirmed_mask(_target_contact_series(force_df, metadata))
    plane = _confirmed_mask(
        force_df["obj_plane"].fillna(False).to_numpy(dtype=bool)
        if "obj_plane" in force_df.columns
        else np.zeros(len(force_df), dtype=bool)
    )

    if action.startswith("place") and first_support_contact_idx is not None:
        event_start = max(0, int(first_support_contact_idx))
        event_end = min(len(force_df), event_start + CONTACT_CONFIRM_STEPS)
        target_contact = bool(np.any(target[event_start:event_end]))
        plane_contact = bool(np.any(plane[event_start:event_end]))
    else:
        active_press = np.asarray(pressing[start:end], dtype=bool)
        target_contact = bool(np.any(target[start:end] & active_press))
        plane_contact = bool(np.any(plane[start:end] & active_press))

    if target_contact:
        if place_mode == "wobble":
            return _wobble_surface_name(force_df.iloc[start:end])
        return _target_surface_name(place_mode)
    if plane_contact:
        return "a fixed flat surface"
    return None


def _build_annotation(
    action: str,
    interaction: str | None,
    force_segment: pd.DataFrame,
    placement_outcome: str | None = None,
    weight_observable: bool = True,
    support_surface: str | None = None,
) -> str:
    obj_phrase = _weight_descriptor(force_segment, observable=weight_observable)
    slip = interaction in {"slip", "slow slip", "fast slip"}

    if interaction == "accidental drop":
        action_phrase = {
            "grasp": "grasping it",
            "lift": "lifting it",
            "grasp then lift": "grasping and then lifting it",
            "hold": "holding it",
            "place": "placing it",
            "place then release": "placing and then releasing it",
            "press": "pressing it",
            "press then release": "pressing and then releasing it",
            "release": "releasing it",
            "release midair": "releasing it in midair",
            "bump": "moving it against an obstacle",
        }.get(action, f"performing {action}")
        return f"Accidentally dropped {obj_phrase} while {action_phrase}."

    if action == "grasp":
        if slip:
            return f"Grasping {obj_phrase}, letting it slip."
        if interaction == "stable":
            return f"Grasping {obj_phrase} with stable contact."
        return f"Grasping {obj_phrase}."

    if action == "lift":
        if slip:
            return f"Lifting {obj_phrase}, letting it slip."
        if interaction == "stable":
            return f"Lifting {obj_phrase} with stable contact."
        return f"Lifting {obj_phrase}."

    if action == "grasp then lift":
        if slip:
            return f"Grasping then lifting {obj_phrase}, letting it slip."
        if interaction == "stable":
            return f"Grasping then lifting {obj_phrase} with stable contact."
        return f"Grasping then lifting {obj_phrase}."

    if action == "hold":
        if slip:
            return f"Holding {obj_phrase}, letting it slip."
        if interaction == "stable":
            return f"Holding {obj_phrase} steadily."
        return f"Holding {obj_phrase}."

    if action == "release":
        return f"Releasing {obj_phrase} from the supporting surface."

    if action == "release midair":
        return f"Releasing {obj_phrase} in midair."

    if action == "bump":
        if slip:
            return f"Bumping {obj_phrase}, letting it slip."
        if interaction == "stable":
            return f"Bumping {obj_phrase} with stable contact."
        return f"Bumping {obj_phrase}."

    if action == PLACE_ACTION:
        surface = support_surface or "a supporting surface"
        if slip:
            annotation = f"Placing {obj_phrase} on {surface}, letting it slip"
        elif interaction == "stable":
            annotation = f"Placing {obj_phrase} on {surface} with secure contact"
        else:
            annotation = f"Placing {obj_phrase} on {surface}"
        if placement_outcome in {"topple", "topples after release"}:
            return f"{annotation}; it topples after release."
        if placement_outcome in {"upright", "remains upright"}:
            return f"{annotation}; it remains upright after release."
        return f"{annotation}."

    if action == "place then release":
        surface = support_surface or "a supporting surface"
        if slip:
            return f"Placing {obj_phrase} on {surface} and then releasing it while it slips."
        if interaction == "stable":
            return (
                f"Placing {obj_phrase} on {surface} with secure contact and then releasing it."
            )
        return f"Placing {obj_phrase} on {surface} and then releasing it."

    if action == "press":
        surface = support_surface or "a supporting surface"
        if slip:
            return f"Pressing {obj_phrase} against {surface}, letting it slip."
        if interaction == "stable":
            return f"Pressing {obj_phrase} against {surface} with secure contact."
        return f"Pressing {obj_phrase} against {surface}."

    if action == "press then release":
        surface = support_surface or "a supporting surface"
        if slip:
            return f"Pressing {obj_phrase} against {surface} and then releasing it, while it slips."
        if interaction == "stable":
            return (
                f"Pressing {obj_phrase} against {surface} with secure contact and then releasing it."
            )
        return f"Pressing {obj_phrase} against {surface} and then releasing it."

    if action == "place gently":
        if slip:
            return f"Placing {obj_phrase} gently, letting it slip."
        return f"Placing {obj_phrase} gently with stable contact."

    if action == "place firmly":
        if slip:
            return f"Placing {obj_phrase} firmly, letting it slip."
        return f"Placing {obj_phrase} firmly with stable contact."

    # Fallback for unexpected labels.
    if interaction:
        return f"Performs '{action}' with {interaction} contact."
    return f"Performs '{action}'."


def _annotation_filename(obj_name: str, material: str, deformation: str) -> str:
    if ANNOTATION_VARIANT:
        return f"{obj_name}_{material}_{deformation}_annotations_{ANNOTATION_VARIANT}.csv"
    return f"{obj_name}_{material}_{deformation}_annotations.csv"


def list_annotation_tasks(all_objects: list[str], material: str = "Rigid") -> list[tuple[str, str]]:
    tasks = []
    for obj_name in all_objects:
        if obj_name == ".DS_Store":
            continue
        object_root = os.path.join(DATA_ROOT, DATASET, "csv", obj_name, material)
        if not os.path.isdir(object_root):
            continue
        for deformation in sorted(os.listdir(object_root)):
            if deformation.startswith("_attempt_"):
                continue
            if _run_is_excluded(deformation):
                continue
            run_dir = os.path.join(object_root, deformation)
            if not os.path.isdir(run_dir):
                continue
            steps_csv = os.path.join(run_dir, f"{obj_name}_{material}_steps_{deformation}.csv")
            force_csv = os.path.join(run_dir, f"{obj_name}_{material}_{deformation}.csv")
            if not (os.path.exists(steps_csv) and os.path.exists(force_csv)):
                continue
            with open(force_csv, newline="") as f:
                reader = csv.reader(f)
                _ = next(reader, None)
                first_data_row = next(reader, None)
            if first_data_row is None:
                continue
            tasks.append((obj_name, deformation))
    return tasks


def main(obj_name: str, csv_path: str, deformation: str, material: str = "Rigid"):
    annotations_df = pd.DataFrame(
        columns=[
            "action",
            "interaction",
            "placement_outcome",
            "support_surface",
            "support_condition",
            "slip_tangential_displacement_m",
            "slip_tangential_speed_mps",
            "interaction_contact_steps",
            "interaction_contact_max_run",
            "start",
            "annotation",
        ]
    )

    steps_csv = os.path.join(csv_path, f"{obj_name}_{material}_steps_{deformation}.csv")
    force_csv = os.path.join(csv_path, f"{obj_name}_{material}_{deformation}.csv")
    if not os.path.exists(steps_csv) or not os.path.exists(force_csv):
        print(f"Skipping {obj_name}/{deformation}: missing steps or force CSV.")
        return

    steps_df = pd.read_csv(steps_csv)
    force_df = pd.read_csv(force_csv)
    metadata = _load_metadata(csv_path, obj_name, material, deformation)

    rollout_bug = detect_rollout_bug(force_df)
    windows = split_for_model(force_df)
    place_start_step = _first_step(steps_df, {"place", "drop", "gentle_place", "press_place", "bump_push"})
    release_command_step = _first_step(steps_df, {"release"})
    press_command_step = _first_step(steps_df, {"press"})
    lift_command_step = _first_step(steps_df, {"lift"})
    first_support_contact_idx = _first_support_contact_after(force_df, metadata, start_step=place_start_step)
    first_obstacle_contact_idx = _first_obstacle_contact_after(force_df, start_step=place_start_step)
    first_grasp_contact_idx = _first_grasp_contact_index(force_df)
    first_lift_idx = _first_support_detach_after(force_df, metadata, start_step=lift_command_step)
    hold_start_step = _first_step(steps_df, {"hold"})
    release_search_start = hold_start_step if hold_start_step is not None else place_start_step
    early_drop_case = _is_early_drop_case(deformation, metadata, steps_df)
    place_mode = _infer_place_mode(deformation, metadata, steps_df)
    if early_drop_case:
        # A failed pickup/transport may hit the floor before the five-frame
        # finger-loss confirmation completes.  That impact belongs to the
        # accidental-drop event and must never be relabeled as intentional place.
        first_support_contact_idx = None
    accidental_release_idx = (
        _first_release_index(force_df, release_search_start) if early_drop_case else None
    )
    accidental_drop_action = _commanded_action_at_step(steps_df, accidental_release_idx)
    if early_drop_case:
        _append_terminal_window(force_df, windows, accidental_release_idx)

    # New rollouts log the beginning of hand opening explicitly.  Confirm that
    # object contact is subsequently lost, then use the command step so the
    # selected window actually contains the opening motion.  Keep a legacy
    # fallback for already-generated, non-early-drop rollouts.
    if release_command_step is not None:
        confirmed_release_idx = _first_release_index(force_df, release_command_step)
        release_event_idx = release_command_step if confirmed_release_idx is not None else None
    elif not early_drop_case:
        confirmed_release_idx = _first_release_index(force_df, place_start_step)
        release_event_idx = confirmed_release_idx
    else:
        confirmed_release_idx = None
        release_event_idx = None

    support_contact = _support_contact_series(force_df, metadata)
    confirmed_support = _confirmed_mask(support_contact)
    release_midair = _release_is_midair(support_contact, release_event_idx)

    pressing = (
        np.zeros(len(force_df), dtype=bool)
        if early_drop_case
        else _pressing_series(
            force_df,
            metadata,
            place_idx=first_support_contact_idx,
            release_command_step=release_event_idx,
            press_command_step=press_command_step,
        )
    )

    if rollout_bug is not None:
        print(f"Excluding invalid rollout {obj_name}/{deformation}: {rollout_bug}")
        # Overwrite any annotation file produced by an older run so that a
        # rollout-level failure cannot leak otherwise-valid windows into the
        # merged dataset.
        output_csv_path = os.path.join(
            DATA_ROOT,
            DATASET,
            "csv",
            obj_name,
            material,
            deformation,
            _annotation_filename(obj_name, material, deformation),
        )
        annotations_df.to_csv(output_csv_path, index=False)
        return

    for row in windows:
        start = int(row["start"])
        if start + SEQUENCE_LENGTH > len(force_df):
            continue

        force_segment = force_df.iloc[start : start + SEQUENCE_LENGTH].reset_index(drop=True)
        if force_segment.isnull().values.any():
            print(f"Skipping {obj_name}/{deformation} segment @ {start}: NaN values.")
            continue
        if detect_bugs(force_df, start):
            continue
        if force_window_is_all_zero(force_df, start):
            continue
        if not _has_bimanual_contact(force_df, start):
            continue

        action = _determine_action(
            start=start,
            force_segment=force_segment,
            first_support_contact_idx=first_support_contact_idx,
            first_grasp_contact_idx=first_grasp_contact_idx,
            first_lift_idx=first_lift_idx,
            accidental_release_idx=accidental_release_idx,
            release_event_idx=release_event_idx,
            release_midair=release_midair,
            first_obstacle_contact_idx=first_obstacle_contact_idx,
            lift_command_step=lift_command_step,
            early_drop_case=early_drop_case,
            place_mode=place_mode,
            pressing=pressing,
            accidental_drop_action=accidental_drop_action,
        )
        if action is None:
            continue
        includes_accidental_drop = (
            early_drop_case
            and accidental_release_idx is not None
            and start <= accidental_release_idx < start + len(force_segment)
        )
        if _hold_has_sustained_support(
            action,
            confirmed_support,
            start,
            len(force_segment),
            includes_accidental_drop=includes_accidental_drop,
        ):
            continue
        if includes_accidental_drop:
            interaction = "accidental drop"
            interaction_measurements = _interaction_measurements(force_segment)
        elif action in {"release", "release midair"}:
            interaction = None
            interaction_measurements = _interaction_measurements(force_segment)
        else:
            interaction_measurements = _interaction_measurements(force_segment)
            if not bool(interaction_measurements["observable"]):
                interaction = None
            else:
                speed_mps = float(interaction_measurements["tangential_speed_mps"])
                interaction = (
                    "slip"
                    if np.isfinite(speed_mps)
                    and speed_mps >= SLIP_SPEED_THRESHOLD_MPS
                    else "stable"
                )
        placement_outcome = _placement_outcome(action, metadata)
        support_surface = _support_surface_for_action(
            action=action,
            interaction=interaction,
            start=start,
            force_df=force_df,
            metadata=metadata,
            place_mode=place_mode,
            first_support_contact_idx=first_support_contact_idx,
            pressing=pressing,
        )
        support_condition = _support_condition(support_surface)
        support_segment = support_contact[start : start + len(force_segment)]
        weight_observable = _weight_is_observable(force_segment, support_segment)
        annotation = _build_annotation(
            action,
            interaction,
            force_segment,
            placement_outcome,
            weight_observable=weight_observable,
            support_surface=support_surface,
        )

        annotations_df.loc[len(annotations_df)] = {
            "action": action,
            "interaction": interaction,
            "placement_outcome": placement_outcome,
            "support_surface": support_surface,
            "support_condition": support_condition,
            "slip_tangential_displacement_m": interaction_measurements[
                "tangential_displacement_m"
            ],
            "slip_tangential_speed_mps": interaction_measurements[
                "tangential_speed_mps"
            ],
            "interaction_contact_steps": interaction_measurements[
                "bimanual_contact_steps"
            ],
            "interaction_contact_max_run": interaction_measurements[
                "longest_bimanual_contact_run"
            ],
            "start": start,
            "annotation": annotation,
        }

    output_csv_path = os.path.join(
        DATA_ROOT,
        DATASET,
        "csv",
        obj_name,
        material,
        deformation,
        _annotation_filename(obj_name, material, deformation),
    )
    annotations_df.to_csv(output_csv_path, index=False)


if __name__ == "__main__":
    folder_path = os.path.join(DATA_ROOT, DATASET, "csv")
    if not os.path.isdir(folder_path):
        raise SystemExit(f"Dataset folder not found: {folder_path}")

    all_objects = os.listdir(folder_path)
    selected_tasks = list_annotation_tasks(all_objects)

    material = "Rigid"
    processes = []
    for obj_name, deformation in selected_tasks:
        picked_up_path = os.path.join(folder_path, obj_name, material, deformation)
        print(f"Processing {obj_name} / {deformation} ...")

        while len(processes) >= MAX_ANNOTATION_PROCESSES:
            processes = [p for p in processes if p.is_alive()]
            time.sleep(0.1)

        p = Process(target=main, args=(obj_name, picked_up_path, deformation, material))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("All processes completed.")
