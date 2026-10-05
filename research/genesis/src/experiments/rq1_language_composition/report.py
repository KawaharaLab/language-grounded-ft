"""Aggregate every predeclared RQ1 run and paired object-cluster uncertainty.

Can be run between stages; missing jobs remain explicitly pending.  It never
selects prompts, methods, subsets, seeds, or checkpoints by evaluation score.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from evaluate import ACTION_CLASSES, WEIGHT_CLASSES, _json, _sha256


SEEDS = (42, 43, 44)
REPLICATES = 2000
BOOTSTRAP_SEED = 20260917
METRICS = ("accuracy", "macro_f1", "joint_accuracy", "action_accuracy", "weight_accuracy", "cell_macro_accuracy")
GROUPS = ("kind", "variant", "trained_subset", "stage", "selection", "task", "suite", "axis",
          "readout", "target", "scope", "stratum_axis", "stratum")
LABELS = {"canonical": "A: original captions", "single": "B: disjoint vocabulary",
          "diverse": "C: diverse disjoint vocabulary"}


def _metric_arrays(matrices: np.ndarray, classes: list[str], composition: bool) -> dict[str, np.ndarray]:
    """Operate on (..., true, predicted) confusion counts, including bootstraps."""
    diagonal = np.diagonal(matrices, axis1=-2, axis2=-1)
    n = matrices.sum(axis=(-2, -1))
    if np.any(n <= 0):
        raise ValueError("Empty bootstrap replicate")
    true_count = matrices.sum(-1)
    denominator = true_count + matrices.sum(-2)
    f1 = np.divide(2 * diagonal, denominator, out=np.zeros_like(diagonal, dtype=float), where=denominator > 0)
    result = {"accuracy": diagonal.sum(-1) / n, "macro_f1": f1.mean(-1)}
    if composition:
        actions = np.asarray([value.split("|")[0] for value in classes])
        weights = np.asarray([value.split("|")[1] for value in classes])
        recalls = np.divide(diagonal, true_count, out=np.zeros_like(diagonal, dtype=float), where=true_count > 0)
        result = {
            "joint_accuracy": result["accuracy"],
            "action_accuracy": (matrices * (actions[:, None] == actions[None, :])).sum((-2, -1)) / n,
            "weight_accuracy": (matrices * (weights[:, None] == weights[None, :])).sum((-2, -1)) / n,
            # Each resample averages truth cells actually observed in it. The
            # report records resamples missing any original observed cell.
            "cell_macro_accuracy": recalls.sum(-1) / (true_count > 0).sum(-1),
        }
    return result


def paired_object_bootstrap(left: list[pd.DataFrame], right: list[pd.DataFrame],
                            classes: list[str], key: str, composition: bool = False,
                            replicates: int = REPLICATES) -> tuple[list[dict], dict, dict]:
    """Paired across methods/conditions and objects, averaging fixed 3 seeds."""
    if len(left) != 3 or len(right) != 3:
        raise ValueError("All three predeclared seeds are required for intervals")
    columns = ["union_index", "csv_path", "start", "object_id", "episode_id", "true"]
    exemplar = left[0][columns].reset_index(drop=True)
    if not len(exemplar):
        raise ValueError("Cannot bootstrap empty evaluation stratum")
    objects = sorted(exemplar.object_id.astype(str).unique())
    object_map = {value: i for i, value in enumerate(objects)}
    class_map = {value: i for i, value in enumerate(classes)}
    oi = exemplar.object_id.astype(str).map(object_map).to_numpy(np.int64)
    yi = exemplar.true.map(class_map).to_numpy(np.int64)
    matrices = []
    for collection in (left, right):
        seed_matrices = []
        for frame in collection:
            pd.testing.assert_frame_equal(frame[columns].reset_index(drop=True), exemplar, check_dtype=False)
            pi = frame.pred.map(class_map).to_numpy(np.int64)
            cm = np.zeros((len(objects), len(classes), len(classes)), dtype=np.int64)
            np.add.at(cm, (oi, yi, pi), 1)
            seed_matrices.append(cm)
        matrices.append(seed_matrices)
    matrices = np.asarray(matrices)  # method, seed, object, true, predicted
    random_seed = BOOTSTRAP_SEED + int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(random_seed)
    draws = rng.multinomial(len(objects), np.full(len(objects), 1 / len(objects)), size=replicates)
    # Batched einsum avoids a large simultaneous model/seed/object tensor.
    point = {name: values.mean(1) for name, values in _metric_arrays(matrices.sum(2), classes, composition).items()}
    bootstrap: dict[str, list[np.ndarray]] = {name: [] for name in point}
    missing_cell_draws = 0
    original_present = matrices[0, 0].sum((0, 2)) > 0
    for start in range(0, replicates, 100):
        cm = np.einsum("bo,msotv->bmstv", draws[start:start + 100], matrices, optimize=True)
        missing_cell_draws += int((cm[:, 0, 0].sum(-1)[:, original_present] == 0).any(1).sum())
        for name, values in _metric_arrays(cm, classes, composition).items():
            bootstrap[name].append(values.mean(2))  # replicate, method; average seed
    rows = []
    for metric in point:
        distribution = np.concatenate(bootstrap[metric])
        difference = distribution[:, 0] - distribution[:, 1]
        lower, upper = np.quantile(difference, [.025, .975])
        rows.append(dict(metric=metric, left=float(point[metric][0]), right=float(point[metric][1]),
                         difference=float(point[metric][0] - point[metric][1]),
                         lower=float(lower), upper=float(upper),
                         left_lower=float(np.quantile(distribution[:, 0], .025)),
                         left_upper=float(np.quantile(distribution[:, 0], .975)),
                         right_lower=float(np.quantile(distribution[:, 1], .025)),
                         right_upper=float(np.quantile(distribution[:, 1], .975)),
                         samples=len(exemplar), objects=len(objects),
                         episodes=int(exemplar.episode_id.nunique()), seeds=3,
                         bootstrap_replicates=replicates))
    audit = dict(key=key, random_seed=random_seed, objects=objects, samples=len(exemplar),
                 classes=classes, missing_original_cell_draws=missing_cell_draws,
                 interpretation="95% percentile object-cluster intervals conditional on these fixed three trained seeds; no seed resampling and no multiplicity correction.")
    resamples = dict(objects=np.asarray(objects), multiplicities=draws)
    return rows, audit, resamples


def aggregate_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    records = []
    for group, part in frame.groupby(list(GROUPS), dropna=False, sort=True):
        if part.seed.duplicated().any():
            raise ValueError(f"Duplicate seed in aggregation group: {group}")
        row = dict(zip(GROUPS, group))
        row.update(seeds=len(part), seed_ids=",".join(map(str, sorted(part.seed.astype(int)))),
                   samples=int(part.samples.iloc[0]), objects=int(part.objects.iloc[0]),
                   episodes=int(part.episodes.iloc[0]))
        if any(part[name].nunique() != 1 for name in ("samples", "objects", "episodes")):
            raise ValueError("Evaluation denominators must be identical across seeds")
        for metric in METRICS:
            if metric in part and part[metric].notna().any():
                values = part[metric].dropna().to_numpy(float)
                if len(values) != len(part):
                    raise ValueError("Metric missing for only some training seeds")
                row[f"{metric}_mean"] = float(values.mean())
                row[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
        records.append(row)
    return pd.DataFrame(records)


def _selection_jobs(design: dict, completed: set[str], kind: str, variant: str,
                    subset: str) -> list[dict] | None:
    selected = []
    for seed in SEEDS:
        matches = [job for job in design["jobs"] if job["kind"] == kind
                   and job.get("variant", "canonical") == variant and job["subset"] == subset and job["seed"] == seed]
        if len(matches) != 1:
            raise ValueError(f"Expected one predeclared job for {kind}/{variant}/{subset}/seed{seed}")
        if matches[0]["id"] not in completed:
            return None
        selected.append(matches[0])
    return selected


def _read_predictions(root: Path, jobs: list[dict], selection: str, stem: str,
                      target: str | None = None, scope: str = "all") -> list[pd.DataFrame]:
    frames = []
    for job in jobs:
        path = root / "jobs" / job["id"] / "evaluation" / selection / f"{stem}_predictions.csv"
        frame = pd.read_csv(path).fillna("")
        if scope == "unseen":
            frame = frame.loc[frame.true == target]
        elif scope == "seen":
            frame = frame.loc[frame.true != target]
        frames.append(frame.reset_index(drop=True))
    return frames


def bootstrap_comparisons(root: Path, design: dict, completed: set[str], output: Path) -> pd.DataFrame:
    output.mkdir(parents=True, exist_ok=True)
    text_metadata = json.loads((root / "text_manifest.json").read_text()) if (root / "text_manifest.json").is_file() else {}
    del text_metadata  # Candidate orders are explicit in the prediction score archives.
    rows, audits = [], []
    frame_cache: dict[tuple, list[pd.DataFrame]] = {}

    def frames_for(jobs: list[dict], selection: str, stem: str, target: str, scope: str) -> list[pd.DataFrame]:
        cache_key = (tuple(job["id"] for job in jobs), selection, stem, target, scope)
        if cache_key not in frame_cache:
            frame_cache[cache_key] = _read_predictions(root, jobs, selection, stem, target, scope)
        return frame_cache[cache_key]

    def compare(left: list[dict] | None, right: list[dict] | None, name: str, stem: str,
                selection: str, task: str, suite: str, axis: str,
                target: str = "none", scope: str = "all") -> None:
        if left is None or right is None:
            return
        lframes = frames_for(left, selection, stem, target, scope)
        rframes = frames_for(right, selection, stem, target, scope)
        if not len(lframes[0]):
            return
        sample = root / "jobs" / left[0]["id"] / "evaluation" / selection / f"{stem}_scores.npz"
        with np.load(sample) as scores:
            classes = scores["classes"].astype(str).tolist()
        key = f"{selection}/{task}/{suite}/{axis}/{target}/{scope}"
        estimates, audit, resamples = paired_object_bootstrap(lframes, rframes, classes, key,
                                                            composition=task == "composition")
        for estimate in estimates:
            rows.append(dict(comparison=name, selection=selection, task=task, suite=suite, axis=axis,
                             target=target, scope=scope, **estimate))
        audit.update(comparison=name, left_jobs=[job["id"] for job in left], right_jobs=[job["id"] for job in right])
        audits.append(audit)
        # Identical key yields identical multiplicities for every paired
        # comparison; save once so re-scoring remains exactly reproducible.
        draw_path = output / f"draws_{hashlib.sha256(key.encode()).hexdigest()[:16]}.npz"
        if not draw_path.exists():
            np.savez_compressed(draw_path, **resamples)

    lexical = {variant: _selection_jobs(design, completed, "contrastive", variant, "full")
               for variant in ("canonical", "single", "diverse")}
    suites = ("canonical", "bare_words", "paraphrase_1", "paraphrase_2", "paraphrase_3", "paraphrase_4", "paraphrase_5")
    for selection in ("best", "last"):
        for suite in suites:
            for axis in ("action", "interaction"):
                for left, right in (("single", "canonical"), ("diverse", "single"), ("diverse", "canonical")):
                    compare(lexical[left], lexical[right], f"{left} minus {right}", f"{suite}_{axis}",
                            selection, "classification", suite, axis)
        supervised_full = _selection_jobs(design, completed, "supervised", "canonical", "full")
        for held_weight in WEIGHT_CLASSES:
            target = f"place|{held_weight}"
            subset = f"without_place_{held_weight}"
            cl = _selection_jobs(design, completed, "contrastive", "canonical", subset)
            sup = _selection_jobs(design, completed, "supervised", "canonical", subset)
            for scope in ("all", "seen", "unseen"):
                for left, right, name in (
                    (cl, sup, "contrastive minus supervised (held-out training)"),
                    (cl, lexical["canonical"], "contrastive held-out minus full"),
                    (sup, supervised_full, "supervised held-out minus full"),
                    (lexical["canonical"], supervised_full, "contrastive minus supervised (full training)"),
                ):
                    compare(left, right, name, "composition_joint", selection, "composition", "canonical",
                            "action_weight", target, scope)
    result = pd.DataFrame(rows)
    result.to_csv(output / "paired_intervals.csv", index=False)
    _json(output / "audit.json", audits)
    return result


def prompt_sensitivity(root: Path, design: dict, completed: set[str], output: Path) -> pd.DataFrame:
    """Fixed five-suite variation; no selection or averaging of predictions."""
    records = []
    for job in design["jobs"]:
        if job["id"] not in completed or job["stage"] != "lexical":
            continue
        for selection in ("best", "last"):
            for axis in ("action", "weight", "interaction"):
                frames = [_read_predictions(root, [job], selection, f"paraphrase_{i}_{axis}")[0]
                          for i in range(1, 6)]
                columns = ["union_index", "csv_path", "start", "true"]
                for frame in frames[1:]:
                    pd.testing.assert_frame_equal(frame[columns], frames[0][columns], check_dtype=False)
                truth = frames[0].true.to_numpy(str)
                predictions = np.stack([frame.pred.to_numpy(str) for frame in frames])
                accuracy = (predictions == truth[None, :]).mean(1)
                folder = root / "jobs" / job["id"] / "evaluation" / selection
                f1 = np.asarray([json.loads((folder / f"paraphrase_{i}_{axis}_metrics.json").read_text())["macro_f1"]
                                 for i in range(1, 6)])
                records.append(dict(job_id=job["id"], variant=job["variant"], seed=job["seed"],
                                    selection=selection, axis=axis, samples=len(truth), suites=5,
                                    accuracy_suite_mean=float(accuracy.mean()), accuracy_suite_min=float(accuracy.min()),
                                    accuracy_suite_max=float(accuracy.max()), accuracy_suite_sd=float(accuracy.std(ddof=1)),
                                    macro_f1_suite_mean=float(f1.mean()), macro_f1_suite_min=float(f1.min()),
                                    macro_f1_suite_max=float(f1.max()), macro_f1_suite_sd=float(f1.std(ddof=1)),
                                    all_five_correct=float((predictions == truth[None, :]).all(0).mean()),
                                    all_five_agree=float((predictions == predictions[:1]).all(0).mean())))
    result = pd.DataFrame(records)
    result.to_csv(output / "prompt_sensitivity_by_seed.csv", index=False)
    if len(result):
        metrics = [name for name in result if name.startswith(("accuracy_", "macro_f1_", "all_five_"))]
        summarized = result.groupby(["variant", "selection", "axis"], sort=True)[metrics].agg(["mean", "std"])
        summarized.columns = ["_".join(pair) for pair in summarized.columns]
        summarized["seeds"] = result.groupby(["variant", "selection", "axis"]).seed.nunique()
        summarized["samples"] = result.groupby(["variant", "selection", "axis"]).samples.first()
        summarized.to_csv(output / "prompt_sensitivity_summary.csv")
    return result


def _format(mean: float, sd: float, digits: int = 2) -> str:
    if pd.isna(mean):
        return "—"
    if pd.isna(sd):
        return f"{100 * mean:.{digits}f} (one seed)"
    return f"{100 * mean:.{digits}f} ± {100 * sd:.{digits}f}"


def _plots(summary: pd.DataFrame, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    best = summary.loc[summary.selection == "best"]
    with plt.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
        for ax, target in zip(axes, ("action", "interaction")):
            for i, variant in enumerate(("canonical", "single", "diverse")):
                rows = best.loc[(best.task == "classification") & (best.stage == "lexical")
                                & (best.suite == "canonical") & (best.axis == target) & (best.variant == variant)]
                if len(rows):
                    row = rows.iloc[0]
                    sd = row.macro_f1_sd if np.isfinite(row.macro_f1_sd) else 0
                    ax.bar(i, 100 * row.macro_f1_mean, yerr=100 * sd, capsize=4, color=("#4477AA", "#EEAA33", "#228833")[i])
            ax.set_xticks(range(3), ["Original", "Disjoint", "Diverse\ndisjoint"])
            ax.set_xlabel(target.capitalize()); ax.set_ylim(0, 100)
        axes[0].set_ylabel("Macro F1 (%)")
        fig.tight_layout(); fig.savefig(output / "lexical_canonical_macro_f1.pdf", bbox_inches="tight"); plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
        for ax, weight in zip(axes, WEIGHT_CLASSES):
            positions = []
            names = []
            for i, (kind, subset, label) in enumerate((
                ("contrastive", "full", "CL full"), ("supervised", "full", "Sup full"),
                ("contrastive", f"without_place_{weight}", "CL held-out"),
                ("supervised", f"without_place_{weight}", "Sup held-out"))):
                rows = best.loc[(best.task == "composition") & (best.readout == "joint")
                                & (best.target == f"place|{weight}") & (best.scope == "unseen")
                                & (best.stratum_axis == "all") & (best.kind == kind)
                                & (best.trained_subset == subset) & (best.variant == "canonical")]
                if len(rows):
                    row = rows.iloc[0]
                    sd = row.joint_accuracy_sd if np.isfinite(row.joint_accuracy_sd) else 0
                    ax.bar(i, 100 * row.joint_accuracy_mean, yerr=100 * sd, capsize=4,
                           color="#4477AA" if kind == "contrastive" else "#EEAA33",
                           alpha=.5 if subset == "full" else 1)
                positions.append(i); names.append(label)
            ax.set_xticks(positions, names, rotation=25, ha="right")
            ax.set_xlabel(f"Unseen place + {weight}"); ax.set_ylim(0, 100)
        axes[0].set_ylabel("Action + weight jointly correct (%)")
        fig.tight_layout(); fig.savefig(output / "composition_unseen_joint_accuracy.pdf", bbox_inches="tight"); plt.close(fig)


def _verify(root: Path, design: dict, completed: set[str], all_done: bool) -> dict:
    design_hash = _sha256(root / "experiment.json")
    text_hash = _sha256(root / "text_embeddings.pt")
    checks = {"experiment.json": design_hash, "text_embeddings.pt": text_hash}
    for path, expected in design.get("source_hashes", {}).items():
        digest = _sha256(Path(path))
        if digest != expected:
            raise ValueError(f"Code changed after design lock: {path}")
        checks[path] = digest
    for job in design["jobs"]:
        if job["id"] not in completed:
            continue
        folder = root / "jobs" / job["id"]
        provenance = json.loads((folder / "evaluation" / "provenance.json").read_text())
        if provenance["protocol_sha256"] != design_hash or provenance["text_bank_sha256"] != text_hash:
            raise ValueError(f"Changed protocol/text bank for {job['id']}")
        for selection in ("best", "last"):
            checkpoint = folder / f"{selection}.pt"
            digest = _sha256(checkpoint)
            if digest != provenance["selections"][selection]["checkpoint_sha256"]:
                raise ValueError(f"Checkpoint changed after evaluation: {checkpoint}")
            checks[str(checkpoint.relative_to(root))] = digest
        if all_done:
            for relative, expected in provenance["artifact_sha256"].items():
                artifact = folder / relative
                digest = _sha256(artifact)
                if digest != expected:
                    raise ValueError(f"Evaluation artifact changed: {artifact}")
                checks[str(artifact.relative_to(root))] = digest
    if all_done:
        manifest_path = root / "data" / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        for relative, expected in manifest["output_sha256"].items():
            digest = _sha256(root / relative)
            if digest != expected:
                raise ValueError(f"Prepared data changed: {relative}")
            checks[relative] = digest
        for name, path, expected in (
            ("source_training_csv", Path(manifest["source_csv"]), manifest["source_sha256"]),
            ("source_training_force", Path(manifest["train_force_audit"]["path"]), manifest["train_force_audit"]["sha256"]),
        ):
            digest = _sha256(path)
            if digest != expected:
                raise ValueError(f"Immutable training input changed: {path}")
            checks[name] = digest
    return {"status": "PASS", "complete_data_hash_audit": all_done, "sha256": checks}


def build_report(run_root: str | Path, with_bootstrap: bool = True) -> dict[str, Any]:
    root = Path(run_root).resolve()
    design = json.loads((root / "experiment.json").read_text())
    output = root / "report"
    output.mkdir(parents=True, exist_ok=True)
    completed = {job["id"] for job in design["jobs"] if (root / "jobs" / job["id"] / "evaluation_complete.json").is_file()}
    for job_id in completed:
        marker = json.loads((root / "jobs" / job_id / "evaluation_complete.json").read_text())
        if marker.get("smoke", False):
            raise ValueError(f"Smoke evaluation cannot enter the experiment report: {job_id}")
    pending = [job["id"] for job in design["jobs"] if job["id"] not in completed]
    if not completed:
        _json(output / "status.json", dict(complete=False, completed=0, planned=len(design["jobs"]), pending=pending))
        (output / "summary.md").write_text("# RQ1 language and composition experiments\n\nNo evaluations completed yet.\n")
        return dict(complete=False, completed=0)
    frames = [pd.read_csv(root / "jobs" / job["id"] / "evaluation" / "metrics.csv")
              for job in design["jobs"] if job["id"] in completed]
    all_rows = pd.concat(frames, ignore_index=True)
    all_rows.to_csv(output / "all_seed_metrics.csv", index=False)
    summary = aggregate_metrics(all_rows)
    summary.to_csv(output / "three_seed_summary.csv", index=False)
    prompt_sensitivity(root, design, completed, output)
    intervals = bootstrap_comparisons(root, design, completed, output / "bootstrap") if with_bootstrap else pd.DataFrame()
    checks = _verify(root, design, completed, not pending)
    _json(output / "integrity_audit.json", checks)
    _plots(summary, output)
    lines = ["# RQ1: vocabulary-disjoint language alignment and held-out compositions", "",
             f"Completed {len(completed)}/{len(design['jobs'])} jobs; each completed job includes best and last evaluations.", "",
             "Best = own-training-validation-selected checkpoint, primary. Last = predeclared sensitivity check. "
             "Reported ± values are sample SD across seeds 42, 43, 44; rows with fewer seeds are explicitly incomplete. "
             "The evaluation set never selects checkpoints, templates, thresholds, or conditions.", "",
             "The source is the fixed, already-thinned 437,244-window RQ1 training dataset. "
             "The three lexical conditions retain the same physical rows and episode split. "
             "Composition conditions remove place-containing windows of the held-out physical mass and every overlapping retained window, "
             "including connected actions and captions without an observed weight. No replacement sampling or rebalancing is performed.", "",
             "## Vocabulary experiment: canonical main evaluation", "",
             "| Condition | Axis | Seeds | Windows | Accuracy (%) | Macro F1 (%) |",
             "|---|---|---:|---:|---:|---:|"]
    primary = summary.loc[(summary.selection == "best") & (summary.task == "classification")
                          & (summary.stage == "lexical") & (summary.suite == "canonical")]
    for _, row in primary.sort_values(["variant", "axis"]).iterrows():
        lines.append(f"| {LABELS[row.variant]} | {row.axis} | {row.seeds} | {row.samples:,} | "
                     f"{_format(row.accuracy_mean, row.accuracy_sd)} | {_format(row.macro_f1_mean, row.macro_f1_sd)} |")
    lines += ["", "The weight axis is a control. Action and interaction are the preregistered lexical targets. "
              "Canonical templates are the original RQ1 templates; bare words and five fixed unused paraphrase suites are secondary. "
              "prompt_sensitivity_by_seed.csv and prompt_sensitivity_summary.csv report the mean, range and SD across the five fixed suites, "
              "the fraction correct for all five, and the fraction of windows with identical predictions for all five. "
              "Agreement alone is not correctness; no predictions are ensembled. "
              "The frozen pretrained language encoder can already know those words: this tests transfer of force-language alignment, not learning unseen words from scratch.", "",
              "## Vocabulary experiment: all secondary prompts", "",
              "| Condition | Suite | Axis | Seeds | Accuracy (%) | Macro F1 (%) |",
              "|---|---|---|---:|---:|---:|"]
    secondary = summary.loc[(summary.selection == "best") & (summary.task == "classification")
                            & (summary.stage == "lexical") & (summary.suite != "canonical")]
    for _, row in secondary.sort_values(["variant", "suite", "axis"]).iterrows():
        lines.append(f"| {LABELS[row.variant]} | {row.suite} | {row.axis} | {row.seeds} | "
                     f"{_format(row.accuracy_mean, row.accuracy_sd)} | {_format(row.macro_f1_mean, row.macro_f1_sd)} |")
    lines += ["", "## Composition: common twelve-candidate main readout", "",
              "Contrastive readout scores all six-action × two-weight joint text prototypes. "
              "The supervised comparator adds log probabilities from its action and weight heads over the same twelve candidates. "
              "Its action head also learns connected actions during training; evaluation excludes connected-action candidates "
              "and sums release and release-midair probability mass. Its interaction and surface heads learn the remaining available caption attributes. "
              "CL factorized prototype readout is retained separately as a diagnostic.", "",
              "Primary composition metrics are joint correctness, separate action/weight correctness, and mean correctness across observed truth cells. "
              "Twelve-class Macro F1 is stored in per-job files only: unobservable cells have no examples and would make it a misleading primary score.", "",
              "| Held-out cell | Training | Method | Evaluation | Seeds | Windows | Joint correct (%) | Action (%) | Weight (%) | Cell mean (%) |",
              "|---|---|---|---|---:|---:|---:|---:|---:|---:|"]
    comp = summary.loc[(summary.selection == "best") & (summary.task == "composition")
                       & (summary.readout == "joint") & (summary.stratum_axis == "all") & (summary.variant == "canonical")]
    for _, row in comp.sort_values(["target", "trained_subset", "kind", "scope"]).iterrows():
        held = "without_" + row.target.replace("|", "_")
        if row.trained_subset not in ("full", held):
            continue
        lines.append(f"| {row.target.replace('|', ' + ')} | {row.trained_subset} | {row.kind} | {row.scope} | "
                     f"{row.seeds} | {row.samples:,} | {_format(row.joint_accuracy_mean, row.joint_accuracy_sd)} | "
                     f"{_format(row.action_accuracy_mean, row.action_accuracy_sd)} | {_format(row.weight_accuracy_mean, row.weight_accuracy_sd)} | "
                     f"{_format(row.cell_macro_accuracy_mean, row.cell_macro_accuracy_sd)} |")
    lines += ["", "Unseen/seen names describe the held-out design cell, including when a full-data reference is shown. "
              "Both symmetric cells are always reported. The same original evaluation-union windows are used for every condition; "
              "observable weight, singleton action, and exclusion of accidental-drop episodes define eligibility. "
              "Interaction and surface-stratified denominators and scores, factorized readout, and all last-checkpoint results are in three_seed_summary.csv.", "",
              "## Paired uncertainty", "",
              "2,000 paired object-cluster bootstrap draws, fixed seeds 42–44, 95% percentile intervals. "
              "Objects, rather than overlapping windows, are the resampling units. The same sampled object multiplicities are applied "
              "to both methods and all three seeds before averaging seed metrics. These intervals condition on the three fitted seeds "
              "and selected evaluation windows. They do not replace the between-seed SD. No multiplicity-adjusted significance claim is made.", "",
              "| Comparison | Task / target | Metric | Difference (pp) | 95% paired object interval (pp) |",
              "|---|---|---|---:|---:|"]
    if len(intervals):
        visible = intervals.loc[(intervals.selection == "best") & (
            ((intervals.task == "classification") & (intervals.suite == "canonical") & (intervals.metric == "macro_f1"))
            | ((intervals.task == "composition") & (intervals.scope == "unseen") & (intervals.metric == "joint_accuracy")))]
        for _, row in visible.iterrows():
            target = row.axis if row.task == "classification" else row.target.replace("|", " + ")
            lines.append(f"| {row.comparison} | {target} | {row.metric} | {100 * row.difference:+.2f} | "
                         f"[{100 * row.lower:+.2f}, {100 * row.upper:+.2f}] |")
    else:
        lines.append("| Pending / omitted in this partial refresh | — | — | — | — |")
    lines += ["", "All prespecified secondary-suite and last-checkpoint intervals are retained in bootstrap/paired_intervals.csv. "
              "The bootstrap audit records object IDs, resampling multiplicities and counts of resamples missing an originally observed cell. "
              "There is no performance-driven prompt selection, threshold adjustment, seen/unseen bias correction, or extra training branch.", "",
              "## Artifacts and completion", "",
              "- all_seed_metrics.csv: every completed seed, checkpoint, suite, readout, and sensitivity stratum.",
              "- three_seed_summary.csv: fixed-condition means and sample SD, with seed counts and all denominators.",
              "- bootstrap/: paired estimates and intervals, draw arrays, and sampling audit.",
              "- ../jobs/<job>/evaluation/<best|last>/: predictions, exact class orders and scores, confusion matrices, per-class results.",
              "- integrity_audit.json: protocol, text bank and checkpoint hashes; full completion also verifies prepared data and original training inputs.", ""]
    if pending:
        lines += ["Pending jobs: " + ", ".join(pending), ""]
    else:
        lines += ["All planned training/evaluation jobs completed; integrity checks passed.", ""]
    (output / "summary.md").write_text("\n".join(lines))
    status = dict(complete=not pending and with_bootstrap, completed=len(completed), planned=len(design["jobs"]),
                  pending=pending, metrics_rows=len(all_rows), aggregate_rows=len(summary),
                  paired_interval_rows=len(intervals), integrity="PASS", bootstrap_included=with_bootstrap)
    _json(output / "status.json", status)
    if status["complete"]:
        _json(root / "all_complete.json", status)
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--no-bootstrap", action="store_true", help="Fast partial refresh; never marks entire experiment complete")
    args = parser.parse_args()
    print(json.dumps(build_report(args.run_root, not args.no_bootstrap), indent=2))


if __name__ == "__main__":
    main()
