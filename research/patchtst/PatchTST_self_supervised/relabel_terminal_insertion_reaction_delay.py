#!/usr/bin/env python3
"""Relabel terminal-insertion windows for an Enter-key reaction delay.

The source dataset already stores every causal force window, its absolute end
stamp, and the raw Enter stamp for every episode.  This permits an exact label
correction without reading or resampling the source rosbags.  Force arrays,
normalization statistics, and episode splits are preserved byte-for-value.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--enter-reaction-delay-sec", type=float, required=True)
    return parser.parse_args()


def dump_json(payload: object, path: Path) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    if args.enter_reaction_delay_sec < 0:
        raise ValueError("--enter-reaction-delay-sec must be non-negative")

    source_dataset = args.source_dataset.resolve()
    source_manifest = args.source_manifest.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dataset = output_dir / "terminal_insertion_windows.npz"

    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    episode_records = manifest["episodes"]
    with np.load(source_dataset, allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}

    required = {"label", "episode_index", "window_end_ns", "episode_ids"}
    missing = sorted(required - arrays.keys())
    if missing:
        raise ValueError(f"source dataset is missing arrays: {missing}")

    old_labels = arrays["label"].astype(np.int64)
    episode_indices = arrays["episode_index"].astype(np.int64)
    window_end_ns = arrays["window_end_ns"].astype(np.int64)
    episode_ids = arrays["episode_ids"].astype(str).tolist()
    if not (len(old_labels) == len(episode_indices) == len(window_end_ns)):
        raise ValueError("window-level source arrays have inconsistent lengths")
    if len(episode_records) != len(episode_ids):
        raise ValueError("manifest episode count does not match dataset")

    delay_ns = int(round(args.enter_reaction_delay_sec * 1e9))
    raw_enter_ns = np.asarray(
        [int(record["events_ns"]["enter"]) for record in episode_records],
        dtype=np.int64,
    )
    corrected_insertion_ns = raw_enter_ns - delay_ns
    reference_by_window = corrected_insertion_ns[episode_indices]
    raw_enter_by_window = raw_enter_ns[episode_indices]
    new_labels = (window_end_ns >= reference_by_window).astype(np.int64)
    relative_to_reference = (
        (window_end_ns - reference_by_window).astype(np.float64) / 1e9
    ).astype(np.float32)
    relative_to_raw_enter = (
        (window_end_ns - raw_enter_by_window).astype(np.float64) / 1e9
    ).astype(np.float32)

    if np.any((old_labels == 1) & (new_labels == 0)):
        raise AssertionError("an earlier reference unexpectedly changed after -> before")
    changed = old_labels != new_labels

    arrays["label"] = new_labels
    # Preserve the legacy key for existing consumers, but make its corrected
    # semantics explicit through the new generic key and manifest.
    arrays["relative_to_enter_sec"] = relative_to_reference
    arrays["relative_to_reference_sec"] = relative_to_reference
    arrays["relative_to_raw_enter_sec"] = relative_to_raw_enter
    np.savez_compressed(output_dataset, **arrays)

    updated = deepcopy(manifest)
    updated["schema_version"] = max(int(manifest.get("schema_version", 1)), 2)
    updated["source_dataset"] = str(source_dataset)
    updated["source_dataset_sha256"] = hashlib.sha256(
        source_dataset.read_bytes()
    ).hexdigest()
    updated["source_manifest"] = str(source_manifest)
    updated["timing_reference"] = "corrected physical insertion estimate"
    updated["raw_timing_event"] = "recorded Enter key stamp"
    updated["enter_reaction_delay_sec"] = args.enter_reaction_delay_sec
    updated["corrected_event_definition"] = (
        "corrected insertion stamp = recorded Enter stamp - reaction delay"
    )
    updated["window_label_rule"] = (
        "after iff causal window end stamp >= corrected insertion stamp"
    )
    updated["legacy_relative_to_enter_sec_semantics"] = (
        "relative to corrected insertion reference in this dataset"
    )
    updated["relabel_changed_windows"] = int(changed.sum())
    updated["force_windows_modified"] = False
    updated["episode_split_modified"] = False

    split_by_episode = {
        episode_id: split
        for split, ids in updated["splits"].items()
        for episode_id in ids
    }
    split_counts: dict[str, dict[str, int]] = {}
    split_changes: dict[str, int] = {}
    for split, ids in updated["splits"].items():
        episode_numbers = {
            index for index, episode_id in enumerate(episode_ids) if episode_id in ids
        }
        mask = np.isin(episode_indices, list(episode_numbers))
        counts = Counter(new_labels[mask].tolist())
        split_counts[split] = {
            "episodes": len(ids),
            "windows": int(mask.sum()),
            "before_windows": int(counts[0]),
            "after_windows": int(counts[1]),
        }
        split_changes[split] = int(changed[mask].sum())
    updated["split_counts"] = split_counts
    updated["relabel_changed_windows_by_split"] = split_changes

    for episode_index, record in enumerate(updated["episodes"]):
        mask = episode_indices == episode_index
        counts = Counter(new_labels[mask].tolist())
        record["raw_enter_ns"] = int(raw_enter_ns[episode_index])
        record["corrected_insertion_ns"] = int(corrected_insertion_ns[episode_index])
        record["enter_reaction_delay_sec"] = args.enter_reaction_delay_sec
        record["before_windows"] = int(counts[0])
        record["after_windows"] = int(counts[1])
        record["relabel_changed_windows"] = int(changed[mask].sum())
        if record["split"] != split_by_episode[record["episode_id"]]:
            raise AssertionError("episode split changed unexpectedly")

    updated["dataset_file"] = output_dataset.name
    updated["dataset_sha256"] = hashlib.sha256(output_dataset.read_bytes()).hexdigest()
    dump_json(updated, output_dir / "dataset_manifest.json")

    audit = {
        "enter_reaction_delay_sec": args.enter_reaction_delay_sec,
        "sample_hz": updated["sample_hz"],
        "equivalent_sample_shift": args.enter_reaction_delay_sec
        * float(updated["sample_hz"]),
        "window_stride_sec": updated["window_stride_sec"],
        "total_windows": int(len(new_labels)),
        "old_before_windows": int((old_labels == 0).sum()),
        "old_after_windows": int((old_labels == 1).sum()),
        "new_before_windows": int((new_labels == 0).sum()),
        "new_after_windows": int((new_labels == 1).sum()),
        "changed_before_to_after_windows": int(changed.sum()),
        "changed_by_split": split_changes,
        "force_windows_identical": True,
        "episode_splits_identical": True,
    }
    dump_json(audit, output_dir / "reaction_delay_relabel_audit.json")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print(f"dataset={output_dataset}")


if __name__ == "__main__":
    main()
