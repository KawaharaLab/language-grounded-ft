"""Freeze the existing post-thinning RQ1 rows, splits, and composition exclusions.

No source CSV, raw force cache, or published evaluation set is modified.  Row
indices always refer to the original 437,244-window CSV / force-cache order.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

WORKSPACE = Path(
    os.environ.get("FT_WORKSPACE_ROOT", Path(__file__).resolve().parents[4])
).expanduser().resolve()
GENESIS = Path(os.environ.get("FT_GENESIS_ROOT", WORKSPACE / "Genesis")).expanduser().resolve()
PATCHTST = Path(os.environ.get("FT_PATCHTST_ROOT", WORKSPACE / "PatchTST")).expanduser().resolve()
SOURCE = GENESIS / "data/train_08142026/train_08142026_terrain_floor_wobble_uniform_descent_contact_clean_slip5mms_balanced.csv"
SOURCE_SHA256 = "b4f38c4b43d68848bb084fef6d8e7a0e58719b491039df0f4ec9d8df704c1570"
CACHE = GENESIS / "paper_review_20260916/split_pilot"
CACHE_AUDIT = GENESIS / "paper_review_20260916/split_replicates"
OFFICIAL = PATCHTST / "results/paper_classification_multiseed/paper_classification_3seed_20260909/prepared_eval"
AXES = ("action", "weight", "interaction")
HEADS = (*AXES, "support_surface")
CLASSES = {
    "action": ["grasp", "grasp then lift", "hold", "lift", "place", "place then release", "press", "press then release", "release", "release midair"],
    "weight": ["heavy", "light"],
    "interaction": ["accidental drop", "slip", "stable"],
    "support_surface": ["a fixed flat surface", "a flat surface", "a gentle slope", "a large step", "a medium step", "a moderate slope", "a small step", "a steep slope", "a wobbling surface"],
}
SEEDS = (42, 43, 44)
WINDOW = 80
FORCE_COLUMNS = [f"{side}_{axis}" for side in ("left", "right") for axis in ("fx", "fy", "fz", "tx", "ty", "tz")]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def progress(phase: str, **kwargs: Any) -> None:
    print(json.dumps({"phase": phase, **kwargs}), flush=True)


def canonical_action(value: str) -> str:
    return "release" if value == "release midair" else value


def recover_raw_action(action: str, annotation: str) -> str:
    if action == "release" and "in midair" in annotation.lower():
        return "release midair"
    return action


def surface_family(value: str) -> str:
    if value in ("a fixed flat surface", "a flat surface"):
        return "flat"
    if value.endswith(" slope"):
        return "slope"
    if value.endswith(" step"):
        return "step"
    if value == "a wobbling surface":
        return "wobble"
    if value == "":
        return "unknown"
    raise ValueError(f"Unexpected support surface: {value!r}")


def enrich_rows(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["row_id"] = np.arange(len(result), dtype=np.int64)
    path_map = {p: str(Path(p).resolve(strict=True)) for p in result.csv_path.unique()}
    result["episode_id"] = result.csv_path.map(path_map)
    result["object_id"] = result.episode_id.map(lambda p: Path(p).parent.parent.parent.name)
    result["is_accidental_drop_episode"] = result.episode_id.map(
        lambda p: any(part.startswith("accidental_drop_") for part in Path(p).parts)
    )
    result["surface_family"] = result.support_surface.map(surface_family)
    assert not result.duplicated(["episode_id", "start"]).any(), "Aliased/duplicate windows in fixed dataset"
    return result


def labels_for(frame: pd.DataFrame) -> np.ndarray:
    labels = np.full((len(frame), len(HEADS)), -1, dtype=np.int64)
    for head_i, head in enumerate(HEADS):
        values = frame["action_raw"] if head == "action" and "action_raw" in frame else frame[head]
        unexpected = set(values) - set(CLASSES[head]) - {""}
        assert not unexpected, (head, unexpected)
        mapping = {name: i for i, name in enumerate(CLASSES[head])}
        labels[:, head_i] = values.map(mapping).fillna(-1).to_numpy(np.int64)
    return labels


def overlap_mask(starts: np.ndarray, forbidden_starts: np.ndarray, width: int = WINDOW) -> np.ndarray:
    """Any shared raw sample between half-open equal-length intervals.

    Adjacent [0,80) and [80,160) do not overlap. Off-grid starts are handled
    identically; no quantization or search outside the fixed source rows.
    """
    forbidden = np.sort(np.asarray(forbidden_starts, dtype=np.int64))
    starts = np.asarray(starts, dtype=np.int64)
    if len(forbidden) == 0:
        return np.zeros(len(starts), dtype=bool)
    last_before_end = np.searchsorted(forbidden, starts + width, side="left") - 1
    valid = last_before_end >= 0
    answer = np.zeros(len(starts), dtype=bool)
    answer[valid] = forbidden[last_before_end[valid]] + width > starts[valid]
    return answer


def read_episode_mass(path: str) -> tuple[str, float, int]:
    mass = pd.read_csv(path, usecols=["obj_mass"])["obj_mass"].to_numpy(np.float64)
    assert len(mass) > 0 and np.isfinite(mass).all() and (mass > 0).all(), f"Invalid physical mass: {path}"
    assert np.all(mass == mass[0]), f"Physical mass changes within episode: {path}"
    return path, float(mass[0]), len(mass)


def resolve_place_masses(frame: pd.DataFrame, output_dir: Path) -> dict[str, float]:
    paths = sorted(frame.loc[frame.action.str.split(" then ").map(lambda a: "place" in a), "episode_id"].unique())
    records = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, (path, mass, length) in enumerate(pool.map(read_episode_mass, paths), 1):
            records.append({"episode_id": path, "physical_mass_kg": mass, "raw_rows": length})
            if i % 1000 == 0 or i == len(paths):
                progress("mass_audit", checked=i, total=len(paths))
    pd.DataFrame(records).to_csv(output_dir / "place_episode_mass_audit.csv", index=False)
    lookup = {r["episode_id"]: r["physical_mass_kg"] for r in records}
    mass = frame.episode_id.map(lookup)
    weight_known = mass.notna() & frame.weight.isin(CLASSES["weight"])
    physical_weight = np.where(mass >= .2, "heavy", "light")
    inconsistent = weight_known & (frame.weight.to_numpy() != physical_weight)
    assert not inconsistent.any(), f"Physical mass/observed weight mismatch at {frame.index[inconsistent].tolist()[:20]}"
    return lookup


def validate_cache(path: Path, shape: tuple[int, int, int], expected_hash: str) -> dict[str, Any]:
    digest = sha256(path)
    assert digest == expected_hash, f"Audited force cache changed: {path}"
    values = np.load(path, mmap_mode="r")
    assert values.shape == shape and values.dtype == np.float32, (path, values.shape, values.dtype)
    for start in range(0, len(values), 4096):
        block = values[start:start + 4096]
        assert np.isfinite(block).all(), f"Nonfinite input block: {path}, {start}"
        assert (np.max(np.abs(block), axis=(1, 2)) > 1e-8).all(), f"Empty input window: {path}, {start}"
    return {"path": str(path), "sha256": digest, "shape": list(shape), "dtype": "float32", "all_finite": True, "all_nonzero": True}


def prepare_eval(output_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    frames: dict[str, pd.DataFrame] = {}
    caches: dict[str, np.ndarray] = {}
    audits: dict[str, Any] = {}
    for axis in AXES:
        source = OFFICIAL / f"eval_08142026_terrain_{axis}_balanced.csv"
        audit_path = CACHE_AUDIT / f"eval_{axis}_cache.json"
        audit = json.loads(audit_path.read_text())
        digest = sha256(source)
        assert digest == audit["source_sha256"], f"Official evaluation CSV changed: {source}"
        frame = pd.read_csv(source).fillna("")
        assert not frame.duplicated(["csv_path", "start"]).any()
        audits[axis] = {"source_csv": str(source), "source_sha256": digest, "cache_audit": str(audit_path), "cache_audit_sha256": sha256(audit_path)}
        audits[axis]["force"] = validate_cache(CACHE / f"eval_{axis}_force.npy", (len(frame), WINDOW, 12), audit["cache_sha256"])
        caches[axis] = np.load(CACHE / f"eval_{axis}_force.npy", mmap_mode="r")
        # The action evaluation CSV alone merges midair into release. Restore
        # the raw truth before comparing records from the other two CSVs.
        frame["action_raw"] = [recover_raw_action(a, t) for a, t in zip(frame.action, frame.annotation)]
        frame["action"] = frame.action.map(canonical_action)
        frame["_source_axis"] = axis
        frame["_source_position"] = np.arange(len(frame), dtype=np.int64)
        frames[axis] = frame

    common = [c for c in frames["action"].columns if not c.endswith("_emb_index") and not c.startswith("_source_")]
    combined = pd.concat([frames[a][common] for a in AXES], ignore_index=True)
    duplicate = combined[combined.duplicated(["csv_path", "start"], keep=False)]
    if len(duplicate):
        distinct = duplicate.groupby(["csv_path", "start"])[common].nunique(dropna=False)
        assert not (distinct > 1).any().any(), f"Conflicting metadata across official sets: {distinct.columns[(distinct > 1).any()].tolist()}"
    union = enrich_rows(combined.drop_duplicates(["csv_path", "start"], keep="first").reset_index(drop=True))
    assert len(union) == 18838, "Unexpected official union size"
    index = {(p, int(s)): int(i) for i, p, s in zip(union.row_id, union.csv_path, union.start)}
    output = np.lib.format.open_memmap(output_dir / "eval_union_force.npy", mode="w+", dtype=np.float32, shape=(len(union), WINDOW, 12))
    occupied = np.zeros(len(union), dtype=bool)
    duplicate_force_checks = 0
    official_classes = {}
    for axis in AXES:
        frame = frames[axis]
        indices = np.array([index[(p, int(s))] for p, s in zip(frame.csv_path, frame.start)], dtype=np.int64)
        prior = occupied[indices]
        if prior.any():
            assert np.array_equal(output[indices[prior]], caches[axis][prior]), f"Duplicate raw inputs conflict: {axis}"
            duplicate_force_checks += int(prior.sum())
        output[indices[~prior]] = caches[axis][~prior]
        occupied[indices] = True
        np.save(output_dir / f"eval_{axis}_indices.npy", indices)
        truth = frame[axis].to_numpy(dtype=str)
        np.save(output_dir / f"eval_{axis}_truth.npy", truth)
        class_frame = pd.read_csv(OFFICIAL / f"{axis}_index_map.csv").sort_values("emb_index")
        official_classes[axis] = class_frame[axis].tolist()
        assert set(truth) == set(official_classes[axis])
        audits[axis]["rows"] = len(frame)
        audits[axis]["truth_counts"] = {str(k): int(v) for k, v in frame[axis].value_counts().items()}
        audits[axis]["indices"] = f"data/eval_{axis}_indices.npy"
        audits[axis]["truth"] = f"data/eval_{axis}_truth.npy"
        if axis == "action":
            assert not union.iloc[indices].is_accidental_drop_episode.any(), "Accidental-drop episode in action evaluation"
    assert occupied.all()
    output.flush()
    del output
    union.to_csv(output_dir / "eval_union_rows.csv", index=False)
    np.save(output_dir / "eval_union_labels.npy", labels_for(union))
    observable = union.weight.isin(CLASSES["weight"])
    single_action = ~union.action.str.contains(" then ", regex=False)
    composition = union.loc[observable & single_action & ~union.is_accidental_drop_episode].copy()
    composition.insert(0, "union_index", composition.row_id.to_numpy())
    composition["singleton_action"] = composition.action
    composition["interaction_stratum"] = composition.interaction.replace("", "unknown")
    composition.to_csv(output_dir / "composition_rows.csv", index=False)
    np.save(output_dir / "composition_indices.npy", composition.union_index.to_numpy(np.int64))
    counts = composition.groupby(["action", "weight", "interaction_stratum", "surface_family"], dropna=False).agg(windows=("row_id", "size"), episodes=("episode_id", "nunique"), objects=("object_id", "nunique")).reset_index()
    counts.to_csv(output_dir / "composition_stratum_counts.csv", index=False)
    for weight in CLASSES["weight"]:
        selected = composition[(composition.action == "place") & (composition.weight == weight)]
        assert len(selected) and set(selected.interaction) >= {"stable", "slip"}, f"No interaction diversity in evaluation place+{weight}"
    return union, {"axes": audits, "official_classes": official_classes, "rows": len(union), "duplicate_force_checks": duplicate_force_checks, "composition_rows": len(composition), "composition_selection": "Deduplicated union of the three fixed evaluation CSVs; singleton action, observed weight, no accidental_drop_* episode; no resampling", "composition_counts": counts.to_dict("records")}


def make_holdouts(frame: pd.DataFrame) -> tuple[dict[str, np.ndarray], pd.DataFrame, dict[str, Any]]:
    includes_place = frame.action.str.split(" then ").map(lambda a: "place" in a).to_numpy()
    starts = frame.start.to_numpy(np.int64)
    groups = frame.groupby("episode_id", sort=False).indices
    reasons = frame[["row_id", "episode_id", "start", "action", "weight", "physical_mass_kg", "physical_weight"]].copy()
    keep_masks = {"full": np.ones(len(frame), dtype=bool)}
    audit: dict[str, Any] = {}
    for weight in CLASSES["weight"]:
        name = f"without_place_{weight}"
        direct = includes_place & (frame.physical_weight.to_numpy() == weight)
        forbidden = np.zeros(len(frame), dtype=bool)
        for positions in groups.values():
            if direct[positions].any():
                forbidden[positions] = overlap_mask(starts[positions], starts[positions][direct[positions]])
        assert np.all(forbidden[direct])
        keep = ~forbidden
        for positions in groups.values():
            if direct[positions].any():
                assert not overlap_mask(starts[positions][keep[positions]], starts[positions][direct[positions]]).any()
        keep_masks[name] = keep
        reasons[name] = np.where(direct, "direct_place_physical_mass", np.where(forbidden, "overlap_with_excluded_place", "retained"))
        audit[name] = {"direct": int(direct.sum()), "direct_observed_weight": int((direct & frame.weight.isin(CLASSES["weight"]).to_numpy()).sum()), "direct_missing_weight": int((direct & ~frame.weight.isin(CLASSES["weight"]).to_numpy()).sum()), "overlap_only": int((forbidden & ~direct).sum()), "retained": int(keep.sum()), "direct_episodes": int(frame.loc[direct, "episode_id"].nunique())}
        assert not (includes_place & (frame.physical_weight.to_numpy() == weight) & keep).any()
    return keep_masks, reasons, audit


def split_counts(frame: pd.DataFrame, labels: np.ndarray, positions: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {"windows": len(positions), "episodes": int(frame.iloc[positions].episode_id.nunique()), "objects": int(frame.iloc[positions].object_id.nunique())}
    for h, head in enumerate(HEADS):
        result[head] = {name: int((labels[positions, h] == i).sum()) for i, name in enumerate(CLASSES[head])}
        result[head]["missing"] = int((labels[positions, h] == -1).sum())
        assert all(result[head][name] > 0 for name in CLASSES[head]), f"An entire supervised class absent from split: {head}, {result[head]}"
    return result


def prepare_data(run_root: Path) -> dict[str, Any]:
    run_root = run_root.resolve()
    output_dir = run_root / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        assert sha256(SOURCE) == previous["source_sha256"] == SOURCE_SHA256
        for relative, digest in previous["output_sha256"].items():
            assert sha256(run_root / relative) == digest, f"Prepared artifact changed: {relative}"
        progress("already_prepared", manifest=str(manifest_path))
        return previous

    progress("validating_source")
    assert sha256(SOURCE) == SOURCE_SHA256, "Requested post-thinning dataset changed"
    frame = pd.read_csv(SOURCE).fillna("")
    assert len(frame) == 437244
    assert set(frame.emb_index) == set(range(250))
    assert frame.groupby("emb_index").annotation.nunique().max() == 1
    assert frame.groupby("annotation").emb_index.nunique().max() == 1
    audited = json.loads((CACHE_AUDIT / "cache_audit.json").read_text())
    row_hash = hashlib.sha256(pd.util.hash_pandas_object(frame[["csv_path", "start", "emb_index", "annotation"]], index=False).values.tobytes()).hexdigest()
    assert row_hash == audited["row_order_sha256"]
    assert audited["source_csv_sha256"] == SOURCE_SHA256
    force_audit = validate_cache(CACHE / "train_force.npy", (len(frame), WINDOW, 12), audited["cache_sha256"])
    frame = enrich_rows(frame)
    labels = labels_for(frame)
    progress("validated_train_cache", rows=len(frame), episodes=frame.episode_id.nunique())
    masses = resolve_place_masses(frame, output_dir)
    frame["physical_mass_kg"] = frame.episode_id.map(masses)
    frame["physical_weight"] = ""
    known = frame.physical_mass_kg.notna()
    frame.loc[known, "physical_weight"] = np.where(frame.loc[known, "physical_mass_kg"] >= .2, "heavy", "light")
    assert frame.loc[frame.action.str.contains("place", regex=False), "physical_mass_kg"].notna().all()
    frame.to_csv(output_dir / "train_rows.csv", index=False)
    np.save(output_dir / "train_labels.npy", labels)
    keep_masks, reasons, holdout_audit = make_holdouts(frame)
    reasons.to_csv(output_dir / "holdout_reasons.csv", index=False)
    episodes = np.array(sorted(frame.episode_id.unique()))
    splits: dict[str, Any] = {}
    for seed in SEEDS:
        order = np.random.default_rng(seed).permutation(len(episodes))
        val_episodes = set(episodes[order[:round(.1 * len(episodes))]])
        is_val = frame.episode_id.isin(val_episodes).to_numpy()
        contents = {}
        counts = {}
        for name, keep in keep_masks.items():
            train = np.flatnonzero(keep & ~is_val)
            val = np.flatnonzero(keep & is_val)
            assert not set(frame.iloc[train].episode_id) & set(frame.iloc[val].episode_id)
            contents[f"{name}_train"] = train
            contents[f"{name}_val"] = val
            counts[name] = {"train": split_counts(frame, labels, train), "val": split_counts(frame, labels, val)}
        np.savez(output_dir / f"split_seed{seed}.npz", **contents)
        splits[str(seed)] = {"path": f"data/split_seed{seed}.npz", "counts": counts, "shared_train_validation_episodes": 0}
    progress("train_splits_prepared", exclusions=holdout_audit)
    eval_frame, eval_audit = prepare_eval(output_dir)
    assert not set(frame.episode_id) & set(eval_frame.episode_id)
    assert not set(frame.object_id) & set(eval_frame.object_id), "Training/test object identifier collision"
    paths = {"train_rows": "data/train_rows.csv", "train_labels": "data/train_labels.npy", "train_force": str(CACHE / "train_force.npy"), "eval_union_rows": "data/eval_union_rows.csv", "eval_union_labels": "data/eval_union_labels.npy", "eval_union_force": "data/eval_union_force.npy", "composition_indices": "data/composition_indices.npy", "composition_rows": "data/composition_rows.csv", "holdout_reasons": "data/holdout_reasons.csv"}
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "run_root": str(run_root), "source_csv": str(SOURCE), "source_sha256": SOURCE_SHA256, "row_order_sha256": row_hash, "source_rows": len(frame), "source_episodes": len(episodes), "source_objects": int(frame.object_id.nunique()), "semantic_ids": 250, "heads": list(HEADS), "classes": CLASSES, "missing_label": -1, "window_samples": WINDOW, "force_columns": FORCE_COLUMNS, "paths": paths, "train_force_audit": force_audit, "train_cache_audit_path": str(CACHE_AUDIT / "cache_audit.json"), "train_cache_audit_sha256": sha256(CACHE_AUDIT / "cache_audit.json"), "physical_mass_audit": {"place_episodes": len(masses), "read": "Full obj_mass column checked positive, finite, exactly constant within each episode; observed heavy/light checked against >=0.2 kg", "other_episode_mass": "Not read; missing mass never substituted from class text"}, "split_rule": "Sorted canonical episode paths, numpy.default_rng(seed).permutation, first round(0.1 * n_episodes) validation. Same base split for all conditions, then same exclusions in train and validation.", "splits": splits, "holdouts": holdout_audit, "holdout_rule": "Direct: any fixed source window containing place as an action component, physical mass in excluded bin, including missing-weight and connected-action windows. Purge all remaining fixed windows sharing any raw sample with a direct window in the same episode; half-open [start,start+80). No episode-wide deletion, row addition, replacement sampling, or rebalancing.", "evaluation": eval_audit, "cross_partition_checks": {"train_eval_shared_episodes": 0, "train_eval_shared_object_ids": 0}, "implementation_sha256": sha256(Path(__file__)), "output_sha256": {str(p.relative_to(run_root)): sha256(p) for p in sorted(output_dir.iterdir()) if p.is_file() and p.name != "manifest.json"}}
    write_json(manifest_path, manifest)
    progress("prepared", manifest=str(manifest_path), composition_rows=eval_audit["composition_rows"])
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    arguments = parser.parse_args()
    prepare_data(arguments.run_dir)
