#!/usr/bin/env python3
"""Prepare the real-robot terminal-insertion linear-probe dataset.

This script intentionally uses ROS 2's system Python because it deserializes the
``geometry_msgs/WrenchStamped`` CDR payloads directly from rosbag2 SQLite files.
It creates an episode-level split and causal 0.8 s windows without touching the
source bags.
"""

from __future__ import annotations
from workspace_paths import workspace_path

import argparse
import csv
import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np
from geometry_msgs.msg import WrenchStamped
from rclpy.serialization import deserialize_message


FORCE_COLUMNS = [
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
TOPICS = ("/force_torque/left", "/force_torque/right")
TRANSFORMS = {
    "identity": np.ones(12, dtype=np.float32),
    # MMS101 raw sensor frame -> Genesis Panda finger-local convention.
    # This is Rz(pi) for both force and moment pseudovectors, on both fingers.
    "sim_local_rz180": np.asarray(
        [-1.0, -1.0, 1.0, -1.0, -1.0, 1.0] * 2, dtype=np.float32
    ),
}
REQUIRED_EVENTS = (
    "gripper_close_complete",
    "enter",
    "post_enter_descent_complete",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare 100 Hz terminal-insertion windows from successful rosbags"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(str(workspace_path("ur3_robotiq_ros2/data/keyboard_descent"))),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-episodes", type=int, default=100)
    parser.add_argument("--val-episodes", type=int, default=5)
    parser.add_argument("--test-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-hz", type=int, default=100)
    parser.add_argument("--window-steps", type=int, default=80)
    parser.add_argument("--window-stride-steps", type=int, default=10)
    parser.add_argument("--post-close-delay-sec", type=float, default=1.0)
    parser.add_argument(
        "--enter-reaction-delay-sec",
        type=float,
        default=0.0,
        help=(
            "Treat physical insertion as occurring this many seconds before the "
            "recorded Enter key event (for human reaction-time correction)."
        ),
    )
    parser.add_argument(
        "--coordinate-transform",
        choices=sorted(TRANSFORMS),
        default="sim_local_rz180",
    )
    parser.add_argument(
        "--max-source-gap-ms",
        type=float,
        default=50.0,
        help="Reject a selected interval if either force stream has a larger gap",
    )
    return parser.parse_args()


def header_stamp_ns(message: WrenchStamped) -> int:
    return int(message.header.stamp.sec) * 1_000_000_000 + int(
        message.header.stamp.nanosec
    )


def wrench_values(message: WrenchStamped) -> list[float]:
    return [
        message.wrench.force.x,
        message.wrench.force.y,
        message.wrench.force.z,
        message.wrench.torque.x,
        message.wrench.torque.y,
        message.wrench.torque.z,
    ]


def read_event_stamps(episode_dir: Path) -> dict[str, int]:
    path = episode_dir / "csv" / "key_events.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result: dict[str, int] = {}
    for event in REQUIRED_EVENTS:
        matches = [row for row in rows if row.get("key") == event]
        if len(matches) != 1:
            raise ValueError(f"{path}: expected one {event!r}, got {len(matches)}")
        result[event] = int(matches[0]["stamp_ns"])
    if not (
        result["gripper_close_complete"]
        < result["enter"]
        < result["post_enter_descent_complete"]
    ):
        raise ValueError(f"{path}: invalid event ordering")
    return result


def read_force_topic(
    connection: sqlite3.Connection, topic_id: int
) -> tuple[np.ndarray, np.ndarray]:
    stamps: list[int] = []
    values: list[list[float]] = []
    query = "SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp"
    for (payload,) in connection.execute(query, (topic_id,)):
        message = deserialize_message(payload, WrenchStamped)
        stamps.append(header_stamp_ns(message))
        values.append(wrench_values(message))
    if len(stamps) < 2:
        raise ValueError(f"force topic id={topic_id} has only {len(stamps)} messages")
    stamp_array = np.asarray(stamps, dtype=np.int64)
    value_array = np.asarray(values, dtype=np.float64)
    order = np.argsort(stamp_array, kind="stable")
    stamp_array = stamp_array[order]
    value_array = value_array[order]
    stamp_array, unique_indices = np.unique(stamp_array, return_index=True)
    return stamp_array, value_array[unique_indices]


def interpolate_stream(
    source_stamps: np.ndarray,
    source_values: np.ndarray,
    grid: np.ndarray,
    max_source_gap_ns: int,
) -> tuple[np.ndarray, float]:
    if source_stamps[0] > grid[0] or source_stamps[-1] < grid[-1]:
        raise ValueError(
            "force stream does not bracket selected interval: "
            f"source=[{source_stamps[0]},{source_stamps[-1]}], "
            f"grid=[{grid[0]},{grid[-1]}]"
        )
    margin = 20_000_000
    selected = source_stamps[
        (source_stamps >= grid[0] - margin) & (source_stamps <= grid[-1] + margin)
    ]
    max_gap_ns = int(np.diff(selected).max(initial=0))
    if max_gap_ns > max_source_gap_ns:
        raise ValueError(
            f"selected force interval has {max_gap_ns / 1e6:.3f} ms source gap "
            f"(limit={max_source_gap_ns / 1e6:.3f} ms)"
        )
    x = source_stamps.astype(np.float64)
    target = grid.astype(np.float64)
    interpolated = np.column_stack(
        [np.interp(target, x, source_values[:, axis]) for axis in range(6)]
    )
    return interpolated.astype(np.float32), max_gap_ns / 1e6


def episode_identifier(episode_dir: Path, root: Path) -> str:
    return episode_dir.relative_to(root).as_posix()


def split_episode_ids(
    episode_ids: list[str], train_count: int, val_count: int, test_count: int, seed: int
) -> dict[str, list[str]]:
    requested = train_count + val_count + test_count
    if len(episode_ids) != requested:
        raise ValueError(
            f"requested {requested} episodes but found {len(episode_ids)} successful episodes"
        )
    permutation = np.random.default_rng(seed).permutation(len(episode_ids))
    shuffled = [episode_ids[int(index)] for index in permutation]
    return {
        "train": shuffled[:train_count],
        "val": shuffled[train_count : train_count + val_count],
        "test": shuffled[train_count + val_count :],
    }


def json_dump(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    if args.sample_hz <= 0 or 1_000_000_000 % args.sample_hz:
        raise ValueError("--sample-hz must exactly divide 1e9")
    if args.window_steps <= 1 or args.window_stride_steps <= 0:
        raise ValueError("invalid window settings")
    if args.enter_reaction_delay_sec < 0:
        raise ValueError("--enter-reaction-delay-sec must be non-negative")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    root = args.data_root.resolve()
    status_paths = sorted(root.glob("*/episode_*/episode_status.json"))
    episode_dirs = []
    for status_path in status_paths:
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") == "success":
            episode_dirs.append(status_path.parent)
    episode_ids = [episode_identifier(path, root) for path in episode_dirs]
    splits = split_episode_ids(
        episode_ids,
        args.train_episodes,
        args.val_episodes,
        args.test_episodes,
        args.seed,
    )
    split_by_episode = {
        episode_id: split for split, ids in splits.items() for episode_id in ids
    }

    period_ns = 1_000_000_000 // args.sample_hz
    post_close_delay_ns = int(round(args.post_close_delay_sec * 1e9))
    enter_reaction_delay_ns = int(round(args.enter_reaction_delay_sec * 1e9))
    max_source_gap_ns = int(round(args.max_source_gap_ms * 1e6))
    coordinate_sign = TRANSFORMS[args.coordinate_transform]

    windows: list[np.ndarray] = []
    labels: list[int] = []
    window_episode_indices: list[int] = []
    window_end_ns: list[int] = []
    window_relative_sec: list[float] = []
    window_relative_to_raw_enter_sec: list[float] = []
    episode_records: list[dict[str, object]] = []
    training_points: list[np.ndarray] = []

    for episode_index, episode_dir in enumerate(episode_dirs):
        episode_id = episode_ids[episode_index]
        events = read_event_stamps(episode_dir)
        raw_enter_ns = events["enter"]
        corrected_insertion_ns = raw_enter_ns - enter_reaction_delay_ns
        interval_start = events["gripper_close_complete"] + post_close_delay_ns
        interval_end = events["post_enter_descent_complete"]
        if not (
            events["gripper_close_complete"]
            < corrected_insertion_ns
            < events["post_enter_descent_complete"]
        ):
            raise ValueError(
                f"{episode_id}: corrected insertion event is outside the valid "
                "episode interval"
            )
        if interval_start >= corrected_insertion_ns:
            raise ValueError(
                f"{episode_id}: post-close delay reaches corrected insertion event"
            )
        grid_end = interval_start + ((interval_end - interval_start) // period_ns) * period_ns
        grid = np.arange(interval_start, grid_end + 1, period_ns, dtype=np.int64)
        if len(grid) < args.window_steps:
            raise ValueError(f"{episode_id}: selected interval has only {len(grid)} samples")

        bag_databases = list((episode_dir / "bag").glob("*/*.db3"))
        if len(bag_databases) != 1:
            raise ValueError(f"{episode_id}: expected one db3, got {len(bag_databases)}")
        with sqlite3.connect(
            f"file:{bag_databases[0]}?mode=ro", uri=True
        ) as connection:
            topic_ids = dict(connection.execute("SELECT name,id FROM topics"))
            streams = []
            source_gaps_ms = {}
            for topic in TOPICS:
                if topic not in topic_ids:
                    raise ValueError(f"{episode_id}: missing topic {topic}")
                source_stamps, source_values = read_force_topic(
                    connection, topic_ids[topic]
                )
                interpolated, max_gap_ms = interpolate_stream(
                    source_stamps,
                    source_values,
                    grid,
                    max_source_gap_ns,
                )
                streams.append(interpolated)
                source_gaps_ms[topic] = max_gap_ms

        synchronized = np.concatenate(streams, axis=1)
        synchronized *= coordinate_sign.reshape(1, -1)
        if not np.isfinite(synchronized).all():
            raise ValueError(f"{episode_id}: non-finite synchronized force values")
        if split_by_episode[episode_id] == "train":
            training_points.append(synchronized)

        episode_window_start = len(windows)
        for end_index in range(
            args.window_steps - 1, len(grid), args.window_stride_steps
        ):
            start_index = end_index - args.window_steps + 1
            windows.append(synchronized[start_index : end_index + 1].copy())
            end_stamp = int(grid[end_index])
            labels.append(int(end_stamp >= corrected_insertion_ns))
            window_episode_indices.append(episode_index)
            window_end_ns.append(end_stamp)
            window_relative_sec.append((end_stamp - corrected_insertion_ns) / 1e9)
            window_relative_to_raw_enter_sec.append((end_stamp - raw_enter_ns) / 1e9)

        episode_labels = labels[episode_window_start:]
        episode_records.append(
            {
                "episode_index": episode_index,
                "episode_id": episode_id,
                "split": split_by_episode[episode_id],
                "events_ns": events,
                "raw_enter_ns": raw_enter_ns,
                "corrected_insertion_ns": corrected_insertion_ns,
                "enter_reaction_delay_sec": args.enter_reaction_delay_sec,
                "selected_start_ns": interval_start,
                "selected_end_ns": int(grid[-1]),
                "selected_samples": int(len(grid)),
                "windows": len(episode_labels),
                "before_windows": int(episode_labels.count(0)),
                "after_windows": int(episode_labels.count(1)),
                "max_selected_source_gap_ms": source_gaps_ms,
            }
        )
        print(
            f"prepared {episode_index + 1:3d}/{len(episode_dirs)} "
            f"{episode_id} split={split_by_episode[episode_id]} "
            f"samples={len(grid)} windows={len(episode_labels)}",
            flush=True,
        )

    force_windows = np.stack(windows).astype(np.float32)
    label_array = np.asarray(labels, dtype=np.int64)
    episode_index_array = np.asarray(window_episode_indices, dtype=np.int32)
    train_values = np.concatenate(training_points, axis=0).astype(np.float64)
    mean = train_values.mean(axis=0).astype(np.float32)
    std = train_values.std(axis=0).astype(np.float32)
    scale = np.maximum(std, 1e-6).astype(np.float32)

    np.savez_compressed(
        output_dir / "terminal_insertion_windows.npz",
        force=force_windows,
        label=label_array,
        episode_index=episode_index_array,
        window_end_ns=np.asarray(window_end_ns, dtype=np.int64),
        relative_to_enter_sec=np.asarray(window_relative_sec, dtype=np.float32),
        relative_to_reference_sec=np.asarray(window_relative_sec, dtype=np.float32),
        relative_to_raw_enter_sec=np.asarray(
            window_relative_to_raw_enter_sec, dtype=np.float32
        ),
        episode_ids=np.asarray(episode_ids),
        mean=mean,
        std=std,
        scale=scale,
        force_columns=np.asarray(FORCE_COLUMNS),
        coordinate_sign=coordinate_sign,
    )

    split_counts: dict[str, dict[str, int]] = {}
    for split, ids in splits.items():
        indices = {
            index for index, episode_id in enumerate(episode_ids) if episode_id in ids
        }
        mask = np.asarray([index in indices for index in episode_index_array])
        counter = Counter(label_array[mask].tolist())
        split_counts[split] = {
            "episodes": len(ids),
            "windows": int(mask.sum()),
            "before_windows": int(counter[0]),
            "after_windows": int(counter[1]),
        }

    dataset_sha256 = hashlib.sha256(
        (output_dir / "terminal_insertion_windows.npz").read_bytes()
    ).hexdigest()
    manifest = {
        "schema_version": 2,
        "data_root": str(root),
        "seed": args.seed,
        "split_unit": "episode",
        "splits": splits,
        "split_counts": split_counts,
        "sample_hz": args.sample_hz,
        "window_steps": args.window_steps,
        "window_duration_sec": args.window_steps / args.sample_hz,
        "window_stride_steps": args.window_stride_steps,
        "window_stride_sec": args.window_stride_steps / args.sample_hz,
        "timing_reference": "corrected physical insertion estimate",
        "raw_timing_event": "recorded Enter key stamp",
        "enter_reaction_delay_sec": args.enter_reaction_delay_sec,
        "corrected_event_definition": (
            "corrected insertion stamp = recorded Enter stamp - reaction delay"
        ),
        "window_label_rule": (
            "after iff causal window end stamp >= corrected insertion stamp"
        ),
        "selected_interval": "gripper_close_complete + delay through post_enter_descent_complete",
        "post_close_delay_sec": args.post_close_delay_sec,
        "force_time_source": "WrenchStamped.header.stamp",
        "synchronization": "linear interpolation to a shared fixed-rate grid",
        "coordinate_transform": args.coordinate_transform,
        "coordinate_sign": coordinate_sign.tolist(),
        "force_columns": FORCE_COLUMNS,
        "normalization": {
            "source": "unique synchronized samples from the 100 training episodes only",
            "count": int(len(train_values)),
            "mean": mean.tolist(),
            "std": std.tolist(),
            "scale": scale.tolist(),
        },
        "dataset_file": "terminal_insertion_windows.npz",
        "dataset_sha256": dataset_sha256,
        "episodes": episode_records,
    }
    json_dump(manifest, output_dir / "dataset_manifest.json")

    axis_audit = {
        "result": "channel order matches; a common Rz(pi) frame rotation is applied",
        "simulation": {
            "source": "Genesis get_links_contact_force/torque([9,10], sensor=True)",
            "frame": "each Panda finger link local frame",
            "order": FORCE_COLUMNS,
            "right_finger_relative_frame": "Rz(pi) in panda.xml",
        },
        "real_robot": {
            "source": "MMS101 bytes 4..21 decoded as Fx,Fy,Fz,Mx,My,Mz",
            "frame": "raw sensor frame; ROS frame_id is a label and no static TF is defined",
            "left_usb_serial": "D20192LJ",
            "right_usb_serial": "DQ014BS7",
            "order": FORCE_COLUMNS,
        },
        "applied_transform": {
            "name": args.coordinate_transform,
            "description": "Rz(pi): negate Fx,Fy,Tx,Ty; preserve Fz,Tz, independently for both sensors",
            "sign": coordinate_sign.tolist(),
        },
        "evidence": {
            "observation": "during closed-gripper descent, raw left/right Fy have the same negative sign and correlation about 0.97; simulation normalization means have positive left/right Fy",
            "interpretation": "the grasp-normal channel and left/right mirroring agree after the common Rz(pi) rotation",
            "limitation": "physical mounting transforms are not encoded in the URDF, so this is a geometry- and data-supported inference rather than a TF-verifiable calibration",
        },
    }
    json_dump(axis_audit, output_dir / "axis_audit.json")
    print(json.dumps(split_counts, indent=2))
    print(f"dataset={output_dir / 'terminal_insertion_windows.npz'}")
    print(f"sha256={dataset_sha256}")


if __name__ == "__main__":
    main()
