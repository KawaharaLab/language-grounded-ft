"""Fixed-protocol RQ1 readouts for vocabulary and held-out composition runs.

No prototype, checkpoint, threshold, or prompt is selected using these results.
The best checkpoint is selected by the training program; last is diagnostic.
All metrics use the original independent evaluation CSV memberships/order.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


AXES = ("action", "weight", "interaction")
ACTION_CLASSES = ("grasp", "hold", "lift", "place", "press", "release")
WEIGHT_CLASSES = ("heavy", "light")
COMPOSITION_TARGETS = ("place|heavy", "place|light")
METADATA = (
    "row_id", "union_index", "csv_path", "start", "episode_id", "object_id",
    "action", "action_raw", "weight", "interaction", "support_surface",
    "surface_family", "interaction_stratum",
)


def _json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _indices(values: np.ndarray, classes: list[str]) -> np.ndarray:
    lookup = {name: i for i, name in enumerate(classes)}
    unknown = set(map(str, values)) - set(lookup)
    if unknown:
        raise ValueError(f"Unknown classes: {sorted(unknown)}")
    return np.asarray([lookup[str(value)] for value in values], dtype=np.int64)


def classification_metrics(truth: np.ndarray, predicted: np.ndarray,
                           classes: list[str]) -> dict[str, Any]:
    """Macro F1 always includes the supplied, fixed candidate class set."""
    truth, predicted = np.asarray(truth, str), np.asarray(predicted, str)
    if len(truth) != len(predicted) or not len(truth):
        raise ValueError("Nonempty, equally sized predictions and truth are required")
    _indices(truth, classes)
    _indices(predicted, classes)
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=classes, zero_division=0)
    return {
        "samples": len(truth), "accuracy": float(np.mean(truth == predicted)),
        "macro_f1": float(f1.mean()), "classes": classes,
        "confusion_matrix": confusion_matrix(truth, predicted, labels=classes).tolist(),
        "per_class": [dict(label=name, precision=float(p), recall=float(r),
                           f1=float(f), support=int(n))
                      for name, p, r, f, n in zip(classes, precision, recall, f1, support)],
    }


def merge_action_logits(logits: torch.Tensor, raw_classes: list[str],
                        candidates: list[str] | None = None) -> torch.Tensor:
    """Condition on singleton actions; release pools both raw release classes.

    Connected-action logits are not candidates in the official six-way task.
    logsumexp is a probability-mass merge, not an average of the two logits.
    """
    candidates = candidates or list(ACTION_CLASSES)
    columns = []
    for label in candidates:
        raw = ("release", "release midair") if label == "release" else (label,)
        positions = [raw_classes.index(value) for value in raw if value in raw_classes]
        if not positions:
            raise ValueError(f"Missing supervised action class {label}")
        selected = logits[:, positions]
        columns.append(torch.logsumexp(selected, dim=1))
    return torch.stack(columns, dim=1)


def supervised_scores(outputs: dict[str, torch.Tensor], raw_classes: dict[str, list[str]],
                      axis: str, candidates: list[str]) -> torch.Tensor:
    if axis == "action":
        return merge_action_logits(outputs[axis], raw_classes[axis], candidates)
    return outputs[axis][:, [raw_classes[axis].index(label) for label in candidates]]


def joint_factorized_scores(action_scores: torch.Tensor, weight_scores: torch.Tensor,
                            action_classes: list[str], weight_classes: list[str],
                            pair_classes: list[str]) -> torch.Tensor:
    log_action = F.log_softmax(action_scores, dim=1)
    log_weight = F.log_softmax(weight_scores, dim=1)
    return torch.stack([
        log_action[:, action_classes.index(pair.split("|")[0])]
        + log_weight[:, weight_classes.index(pair.split("|")[1])]
        for pair in pair_classes], dim=1)


def _metadata(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame[[column for column in METADATA if column in frame]].copy()
    if "union_index" not in output:
        output.insert(0, "union_index", frame.index.to_numpy())
    return output.reset_index(drop=True)


def _save_predictions(folder: Path, stem: str, frame: pd.DataFrame, truth: np.ndarray,
                      scores: np.ndarray, classes: list[str]) -> tuple[pd.DataFrame, dict]:
    scores = np.asarray(scores, dtype=np.float32)
    if scores.shape != (len(frame), len(classes)) or not np.isfinite(scores).all():
        raise ValueError(f"Invalid score array for {stem}: {scores.shape}")
    output = _metadata(frame)
    output["true"] = np.asarray(truth, str)
    output["pred"] = np.asarray(classes)[scores.argmax(1)]
    output["correct"] = output.true == output.pred
    output.to_csv(folder / f"{stem}_predictions.csv", index=False)
    # Candidate scores retain sufficient precision for auditing/re-scoring;
    # explicit class order avoids dependence on implicit label indices.
    np.savez_compressed(folder / f"{stem}_scores.npz", scores=scores,
                        classes=np.asarray(classes), union_index=output.union_index.to_numpy())
    metrics = classification_metrics(output.true.to_numpy(), output.pred.to_numpy(), classes)
    metrics.update(objects=int(output.object_id.nunique()),
                   episodes=int(output.episode_id.nunique()))
    _json(folder / f"{stem}_metrics.json", metrics)
    pd.DataFrame(metrics["confusion_matrix"], index=classes, columns=classes).to_csv(
        folder / f"{stem}_confusion.csv", index_label="true\\pred")
    return output, metrics


def _plot_confusion(metrics: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with plt.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 9}):
        matrix = np.asarray(metrics["confusion_matrix"])
        normalized = np.divide(matrix, matrix.sum(1, keepdims=True),
                               out=np.zeros(matrix.shape, float), where=matrix.sum(1, keepdims=True) > 0)
        fig, ax = plt.subplots(figsize=(max(4, .65 * len(matrix)), max(3.5, .60 * len(matrix))))
        ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
        for i in range(len(matrix)):
            for j in range(len(matrix)):
                ax.text(j, i, f"{normalized[i, j]:.0%}\n({matrix[i, j]})", ha="center", va="center",
                        color="white" if normalized[i, j] > .5 else "black", fontsize=8)
        ax.set_xticks(range(len(matrix)), metrics["classes"], rotation=40, ha="right")
        ax.set_yticks(range(len(matrix)), metrics["classes"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def composition_metrics(predictions: pd.DataFrame, target: str,
                        scope: str = "all") -> dict[str, Any] | None:
    if scope not in ("all", "seen", "unseen"):
        raise ValueError(scope)
    mask = np.ones(len(predictions), bool)
    if scope != "all":
        mask = (predictions.true.to_numpy() == target) == (scope == "unseen")
    frame = predictions.loc[mask]
    if not len(frame):
        return None
    ta = frame.true.str.split("|", regex=False).str[0]
    tw = frame.true.str.split("|", regex=False).str[1]
    pa = frame.pred.str.split("|", regex=False).str[0]
    pw = frame.pred.str.split("|", regex=False).str[1]
    result = {
        "samples": len(frame), "objects": int(frame.object_id.nunique()),
        "episodes": int(frame.episode_id.nunique()), "observed_cells": int(frame.true.nunique()),
        "joint_accuracy": float((frame.true == frame.pred).mean()),
        "action_accuracy": float((ta == pa).mean()), "weight_accuracy": float((tw == pw).mean()),
        "cell_macro_accuracy": float(frame.groupby("true", sort=True).correct.mean().mean()),
    }
    return result


def _composition_readout(folder: Path, name: str, frame: pd.DataFrame,
                         scores: np.ndarray, pair_classes: list[str],
                         common: dict[str, Any]) -> list[dict[str, Any]]:
    truth = (frame.singleton_action.astype(str) + "|" + frame.weight.astype(str)).to_numpy()
    predictions, metrics = _save_predictions(folder, f"composition_{name}", frame, truth, scores, pair_classes)
    rows = []
    for target in COMPOSITION_TARGETS:
        for scope in ("all", "seen", "unseen"):
            values = composition_metrics(predictions, target, scope)
            if values is not None:
                rows.append({**common, "task": "composition", "suite": "canonical",
                             "axis": "action_weight", "readout": name, "target": target,
                             "scope": scope, "stratum_axis": "all", "stratum": "all", **values})
        for stratum_axis in ("interaction_stratum", "surface_family"):
            if stratum_axis not in frame:
                continue
            for stratum in sorted(frame[stratum_axis].astype(str).unique()):
                selected = predictions.loc[frame[stratum_axis].astype(str).to_numpy() == stratum]
                for scope in ("all", "seen", "unseen"):
                    values = composition_metrics(selected, target, scope)
                    if values is not None:
                        rows.append({**common, "task": "composition", "suite": "canonical",
                                     "axis": "action_weight", "readout": name, "target": target,
                                     "scope": scope, "stratum_axis": stratum_axis,
                                     "stratum": stratum, **values})
    per_cell = []
    for label in pair_classes:
        sub = predictions.loc[predictions.true == label]
        per_cell.append(dict(label=label, samples=len(sub),
                             accuracy=float(sub.correct.mean()) if len(sub) else None,
                             objects=int(sub.object_id.nunique()), episodes=int(sub.episode_id.nunique())))
    _json(folder / f"composition_{name}_cells.json", per_cell)
    # The twelve-candidate fixed-label F1 is saved for completeness, but is not
    # the primary composition metric: some physical action/weight cells have
    # no observable ground truth. Primary = joint correctness and cell mean.
    _plot_confusion(metrics, folder / f"composition_{name}_confusion.pdf")
    return rows


@torch.inference_mode()
def evaluate_job(run_root: str | Path, job_id: str, device: str = "cuda",
                 batch_size: int = 2048, allow_smoke: bool = False) -> dict[str, Any]:
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = Path(run_root).resolve()
    design = json.loads((root / "experiment.json").read_text())
    design_hash = _sha256(root / "experiment.json")
    job = next(value for value in design["jobs"] if value["id"] == job_id)
    job_root = root / "jobs" / job_id
    if not (job_root / "training_complete.json").is_file():
        raise RuntimeError(f"Training not complete: {job_id}")
    training = json.loads((job_root / "training_complete.json").read_text())
    if (training["job"] != job or training["design_sha256"] != design_hash
            or (training.get("smoke", False) and not allow_smoke)):
        raise ValueError(f"Training marker/protocol mismatch or smoke checkpoint: {job_id}")
    for path, digest in design.get("source_hashes", {}).items():
        if _sha256(Path(path)) != digest:
            raise ValueError(f"Code changed after design lock: {path}")
    data = root / "data"
    classes = design.get("classes") or json.loads((data / "manifest.json").read_text())["classes"]
    frame = pd.read_csv(data / "eval_union_rows.csv").fillna("")
    force = np.load(data / "eval_union_force.npy", mmap_mode="r")
    if force.shape != (len(frame), 80, 12):
        raise ValueError(f"Evaluation force alignment mismatch: {force.shape}, {len(frame)}")
    text_path = root / "text_embeddings.pt"
    text_bank = torch.load(text_path, map_location="cpu", weights_only=False)
    compose_frame = pd.read_csv(data / "composition_rows.csv").fillna("")
    compose_indices = np.load(data / "composition_indices.npy")
    if len(compose_frame) != len(compose_indices):
        raise ValueError("Composition row/index length mismatch")
    pd.testing.assert_frame_equal(
        frame.iloc[compose_indices][["csv_path", "start"]].reset_index(drop=True),
        compose_frame[["csv_path", "start"]].reset_index(drop=True), check_dtype=False)
    if "union_index" not in compose_frame:
        compose_frame["union_index"] = compose_indices
    # Import here keeps metric functions independently testable/import-safe.
    module_dir = str(Path(__file__).resolve().parent)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    from common import load_model

    results = []
    provenance = dict(job_id=job_id, protocol_sha256=design_hash,
                      text_bank_sha256=_sha256(text_path), smoke=bool(training.get("smoke", False)), selections={})
    for selection in ("best", "last"):
        checkpoint_path = job_root / f"{selection}.pt"
        checkpoint_hash = _sha256(checkpoint_path)
        if checkpoint_hash != training[f"{selection}_sha256"]:
            raise ValueError(f"Checkpoint differs from completed training: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint["design_sha256"] != design_hash or checkpoint["job_id"] != job_id:
            raise ValueError(f"Checkpoint job/design mismatch: {checkpoint_path}")
        for key in ("kind", "variant", "subset", "seed"):
            if checkpoint[key] != job[key]:
                raise ValueError(f"Checkpoint {key} differs from planned job: {checkpoint_path}")
        if checkpoint["epoch"] != training[f"{selection}_epoch"]:
            raise ValueError(f"Checkpoint epoch mismatch: {checkpoint_path}")
        model = load_model(checkpoint, classes, device)
        model.eval()
        mean = torch.as_tensor(checkpoint["normalization"]["mean"], dtype=torch.float32, device=device)
        scale = torch.as_tensor(checkpoint["normalization"]["scale"], dtype=torch.float32, device=device)
        if (mean.shape != (12,) or scale.shape != (12,) or (scale <= 0).any()
                or not torch.isfinite(mean).all() or not torch.isfinite(scale).all()):
            raise ValueError("Invalid training-only normalization")
        batches = []
        for offset in range(0, len(frame), batch_size):
            raw = torch.from_numpy(np.array(force[offset:offset + batch_size], copy=True)).to(device)
            output = model((raw - mean) / scale)
            if job["kind"] == "contrastive":
                batches.append(F.normalize(output, dim=1).cpu())
            else:
                batches.append({axis: values.cpu() for axis, values in output.items()})
        outputs = (torch.cat(batches) if job["kind"] == "contrastive" else
                   {axis: torch.cat([batch[axis] for batch in batches]) for axis in batches[0]})
        del model, batches
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
        folder = job_root / "evaluation" / selection
        folder.mkdir(parents=True, exist_ok=True)
        common = {"job_id": job_id, "kind": job["kind"], "variant": job.get("variant", "canonical"),
                  "trained_subset": job["subset"], "seed": int(job["seed"]), "stage": job["stage"],
                  "selection": selection, "epoch": int(checkpoint["epoch"])}
        suites = list(text_bank["eval"]) if job["kind"] == "contrastive" else ["canonical"]
        for suite in suites:
            for axis in AXES:
                bank = text_bank["eval"][suite][axis]
                candidates = bank["classes"]
                indices = np.load(data / f"eval_{axis}_indices.npy")
                truth = np.load(data / f"eval_{axis}_truth.npy").astype(str)
                selected = frame.iloc[indices].copy()
                selected["union_index"] = indices
                if not np.array_equal(selected[axis].astype(str).to_numpy(), truth):
                    raise ValueError(f"Official truth mismatch: {axis}")
                if job["kind"] == "contrastive":
                    scores = outputs[indices] @ F.normalize(bank["vectors"].float(), dim=1).T
                else:
                    logits = {name: values[indices] for name, values in outputs.items()}
                    scores = supervised_scores(logits, classes, axis, candidates)
                _, metrics = _save_predictions(folder, f"{suite}_{axis}", selected, truth,
                                               scores.numpy(), candidates)
                results.append({**common, "task": "classification", "suite": suite, "axis": axis,
                                "readout": "prototype" if job["kind"] == "contrastive" else "head",
                                "target": "none", "scope": "all", "stratum_axis": "all", "stratum": "all",
                                **{name: metrics[name] for name in ("samples", "objects", "episodes", "accuracy", "macro_f1")}})
                if suite == "canonical":
                    _plot_confusion(metrics, folder / f"{suite}_{axis}_confusion.pdf")
        canonical = text_bank["eval"]["canonical"]
        pair_classes = text_bank["composition"]["classes"]
        if set(pair_classes) != {f"{a}|{w}" for a in ACTION_CLASSES for w in WEIGHT_CLASSES}:
            raise ValueError("Composition must use the same twelve fixed candidates")
        if job["kind"] == "contrastive":
            latent = outputs[compose_indices]
            joint = latent @ F.normalize(text_bank["composition"]["vectors"].float(), dim=1).T
            results.extend(_composition_readout(folder, "joint", compose_frame, joint.numpy(), pair_classes, common))
            axis_scores = {axis: latent @ F.normalize(canonical[axis]["vectors"].float(), dim=1).T
                           for axis in ("action", "weight")}
        else:
            logits = {name: values[compose_indices] for name, values in outputs.items()}
            axis_scores = {axis: supervised_scores(logits, classes, axis, canonical[axis]["classes"])
                           for axis in ("action", "weight")}
        factorized = joint_factorized_scores(axis_scores["action"], axis_scores["weight"],
                                             canonical["action"]["classes"], canonical["weight"]["classes"], pair_classes)
        name = "factorized" if job["kind"] == "contrastive" else "joint"
        results.extend(_composition_readout(folder, name, compose_frame, factorized.numpy(), pair_classes, common))
        provenance["selections"][selection] = dict(checkpoint_sha256=checkpoint_hash, epoch=int(checkpoint["epoch"]))
    output = job_root / "evaluation"
    pd.DataFrame(results).to_csv(output / "metrics.csv", index=False)
    _json(output / "metrics.json", results)
    provenance["artifact_sha256"] = {str(path.relative_to(job_root)): _sha256(path)
                                     for path in sorted(output.rglob("*"))
                                     if path.is_file() and path.name != "provenance.json"}
    _json(output / "provenance.json", provenance)
    _json(job_root / "evaluation_complete.json", dict(rows=len(results), **provenance))
    return {"job_id": job_id, "metrics": str(output / "metrics.csv"), "rows": len(results)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--allow-smoke", action="store_true", help="Isolated preflight only; smoke results cannot enter the experiment report")
    arguments = parser.parse_args()
    print(json.dumps(evaluate_job(arguments.run_root, arguments.job_id, arguments.device,
                                  arguments.batch_size, allow_smoke=arguments.allow_smoke), indent=2))


if __name__ == "__main__":
    main()
