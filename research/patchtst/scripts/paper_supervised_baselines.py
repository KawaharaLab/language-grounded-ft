"""Train normalized RQ1 supervised CNN/MLP baselines with matched seeds."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import os
from pathlib import Path
import queue
import re
import shutil
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch

import paper_classification_multiseed as reference


ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").is_file() and (parent / "PatchTST_self_supervised").is_dir()
)
reference.ROOT = ROOT
REFERENCE = ROOT / "results/paper_classification_multiseed/paper_classification_3seed_20260909"
SEEDS = (42, 43, 44)
MODELS = {
    **reference.MODELS,
    "supervised_cnn": "Supervised CNN",
    "supervised_mlp": "Supervised MLP",
}
MODELS["supervised"] = "Supervised PatchTST"


def checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def prepare(group):
    output = ROOT / "results/paper_supervised_baselines" / group
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing a nonempty group without a manifest: {output}")
    old = json.loads((REFERENCE / "manifest.json").read_text())
    for path, expected in old["hashes"].items():
        if reference.sha256(path) != expected:
            raise ValueError(f"Data differs from RQ1: {path}")
    output.mkdir(parents=True, exist_ok=True)
    eval_dir = output / "prepared_eval"
    shutil.copytree(old["eval_dir"], eval_dir, symlinks=True)
    source_root = ROOT / "PatchTST_self_supervised"
    code_dir = output / "code/PatchTST_self_supervised"
    for source in list(source_root.glob("*.py")) + list((source_root / "src").rglob("*.py")):
        destination = code_dir / source.relative_to(source_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    script_dir = output / "code/scripts"
    script_dir.mkdir(parents=True)
    for filename in ("paper_supervised_baselines.py", "paper_classification_multiseed.py", "run_paper_supervised_baselines.sh"):
        shutil.copy2(ROOT / "scripts" / filename, script_dir / filename)
    (output / "logs").mkdir()
    old_supervised = {run["seed"]: run for run in old["runs"] if run["model"] == "supervised"}
    base_checkpoint = checkpoint(old_supervised[42]["checkpoints"]["best"])
    base_config = {key: value for key, value in base_checkpoint["model_config"].items() if "key" not in key.lower()}
    if base_config["task"] != "multitask" or base_config["label_taxonomy"] != "terrain0814_eval":
        raise ValueError("Expected the release-merged RQ1 multi-task classifier")
    manifest = {
        "group": group,
        "created": datetime.now().isoformat(),
        "output": str(output),
        "eval_dir": str(eval_dir),
        "seeds": list(SEEDS),
        "reference_manifest": str(REFERENCE / "manifest.json"),
        "hashes": dict(old["hashes"]),
        "selection": "maximum validation mean Macro-F1, patience 75, min_delta 1e-4",
        "statistics": "Per-seed metric mean and sample standard deviation (ddof=1)",
        "runs": old["runs"].copy(),
    }
    manifest["hashes"][str(REFERENCE / "manifest.json")] = reference.sha256(REFERENCE / "manifest.json")
    for path in eval_dir.rglob("*"):
        if path.is_file() and not path.is_symlink():
            manifest["hashes"][str(path)] = reference.sha256(path)
    for seed in SEEDS:
        for architecture in ("cnn", "mlp"):
            source_run = next(run for run in old["runs"] if run["model"] == architecture and run["seed"] == 42)
            encoder_config = checkpoint(source_run["checkpoints"]["best"])["model_config"]
            name = f"rq1_supervised_{architecture}_seed{seed}"
            save_root = ROOT / "saved_models/paper_supervised_baselines" / group
            save_dir = save_root / name
            config = {
                **base_config,
                "model_arch": architecture,
                "seed": seed,
                "run_name": name,
                "save_dir": str(save_root),
                "eval_dir": str(eval_dir),
                "wandb_group": group,
                "no_wandb": os.environ.get("WANDB_MODE", "online") == "disabled",
            }
            if architecture == "cnn":
                config.update(cnn_base_channels=encoder_config["base_channels"], cnn_depth=encoder_config["depth"], cnn_kernel_size=encoder_config["kernel_size"], cnn_dropout=encoder_config["dropout"])
            else:
                config.update(mlp_hidden_dim=encoder_config["hidden_dim"], mlp_depth=encoder_config["depth"], mlp_dropout=encoder_config["dropout"])
            run = {
                "model": f"supervised_{architecture}", "seed": seed, "name": name,
                "supervised": True, "new_training": True, "train_config": config,
                "reference_supervised_checkpoint": old_supervised[seed]["checkpoints"]["best"],
                "reference_encoder_checkpoint": source_run["checkpoints"]["best"],
                "checkpoints": {kind: str(save_dir / f"{name}_{suffix}.pth") for kind, suffix in (("best", "best"), ("last", "final"))},
                "train": [sys.executable, "-u", str(code_dir / "patchtst_supervised.py"), *reference.cli_args(config)],
                "train_done": str(output / "logs" / f"{name}.train.done"),
                "eval": {},
            }
            for kind in ("best", "last"):
                target_dir = output / "evaluation" / run["model"] / f"seed{seed}" / kind
                run["eval"][kind] = str(target_dir)
                run[f"evaluate_{kind}"] = [sys.executable, "-u", str(code_dir / "evaluate_supervised_balanced.py"), "--checkpoint", run["checkpoints"][kind], "--eval_dir", str(eval_dir), "--eval_pattern", reference.EVAL_PATTERN, "--output_dir", str(target_dir)]
            manifest["runs"].append(run)
    manifest["code_hashes"] = {str(path): reference.sha256(path) for path in (output / "code").rglob("*") if path.is_file()}
    reference.write_json(manifest_path, manifest)
    reference.write_json(output / "status.json", {"status": "prepared", "new_training_runs": 6, "new_checkpoint_evaluations": 12})
    return manifest


def audit_training(run):
    trained = checkpoint(run["checkpoints"]["best"])
    original = checkpoint(run["reference_supervised_checkpoint"])
    mutable = {"model_arch", "seed", "save_dir", "run_name", "eval_dir", "wandb_project", "wandb_group", "no_wandb"}
    for key, value in original["model_config"].items():
        if key not in mutable and "key" not in key.lower() and trained["model_config"][key] != value:
            raise ValueError(f"Supervised protocol changed: {key}")
    if trained["model_config"]["seed"] != run["seed"] or trained["label_maps"] != original["label_maps"]:
        raise ValueError("Seed or taxonomy differs from RQ1")
    for key in ("mean", "std", "scale", "min", "max"):
        torch.testing.assert_close(trained["normalization"][key], original["normalization"][key], rtol=0, atol=0)
    for key in ("count", "train_samples", "validation_samples", "force_cols"):
        if trained["normalization"][key] != original["normalization"][key]:
            raise ValueError(f"Normalization mismatch: {key}")
    config = trained["model_config"]
    sys.path.insert(0, str(Path(run["train"][2]).parent))
    from classification_taxonomy import canonicalize_classification_labels
    from patchtst_supervised import IGNORE_INDEX
    frame = canonicalize_classification_labels(pd.read_csv(Path(config["train_dir"]) / config["train_csv"]), config["label_taxonomy"])
    if len(frame) != trained["normalization"]["train_samples"] + trained["normalization"]["validation_samples"]:
        raise ValueError("Unexpected skipped training rows")
    mapped = frame.action.map(trained["label_maps"]["action"])
    if (frame.action.notna() & mapped.isna()).any():
        raise ValueError("Unknown non-missing action label")
    targets = mapped.fillna(IGNORE_INDEX).to_numpy(dtype=np.int64)
    _, counts = np.unique(targets, return_counts=True)
    stratify = targets if np.all(counts >= 2) else None
    expected_train, expected_val = train_test_split(np.arange(len(frame)), test_size=config["val_from_train_pct"], random_state=run["seed"], stratify=stratify)
    with np.load(Path(run["checkpoints"]["best"]).parent / "split_indices.npz") as split:
        np.testing.assert_array_equal(split["train"], expected_train)
        np.testing.assert_array_equal(split["validation"], expected_val)
    summary = json.loads((Path(run["checkpoints"]["best"]).parent / "summary.json").read_text())
    final = checkpoint(run["checkpoints"]["last"])
    if trained["epoch"] != summary["best_epoch"] or final["epoch"] != summary["completed_epoch"]:
        raise ValueError("Checkpoint epochs do not match completed training summary")
    if final["model_config"]["seed"] != run["seed"] or set(summary["checkpoints"]) != {"best", "final"}:
        raise ValueError("Missing completed training evaluations or wrong final seed")
    reference.write_json(Path(run["train_done"]).with_suffix(".audit.json"), {
        "seed": run["seed"], "model": run["model"], "protocol_matches_supervised_patchtst": True,
        "normalization_matches_reference_exactly": True, "train_validation_split_matches": True,
        "masked_action_windows": int(np.sum(targets == IGNORE_INDEX)),
        "trainable_params": summary["trainable_params"], "best_epoch": summary["best_epoch"], "completed_epoch": summary["completed_epoch"],
    })


def worker(manifest, jobs, gpu):
    output = Path(manifest["output"])
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUNBUFFERED": "1", "WANDB_MODE": os.environ.get("WANDB_MODE", "online")}
    while True:
        try:
            run = jobs.get_nowait()
        except queue.Empty:
            return
        name = run["name"]
        status_path = output / "logs" / f"{name}.status.json"
        reference.write_json(status_path, {"status": "running", "phase": "training", "gpu": gpu})
        if not Path(run["train_done"]).exists():
            save_dir = Path(run["checkpoints"]["best"]).parent
            finished = (save_dir / "summary.json").is_file() and all(Path(path).is_file() for path in run["checkpoints"].values())
            if finished:
                print(f"REUSE completed training; retry audit: {name}", flush=True)
            elif save_dir.exists():
                raise FileExistsError(f"Partial training exists; use a new GROUP instead of silently restarting: {name}")
            else:
                reference.run_command(run["train"], environment, output / "logs" / f"{name}.train.log")
            audit_training(run)
            Path(run["train_done"]).write_text(datetime.now().isoformat() + "\n")
        for kind in ("best", "last"):
            reference.write_json(status_path, {"status": "running", "phase": f"evaluate_{kind}", "gpu": gpu})
            if not reference.evaluate_complete(run, kind, manifest["eval_dir"]):
                reference.run_command(run[f"evaluate_{kind}"], environment, output / "logs" / f"{name}.{kind}.eval.log")
            if not reference.evaluate_complete(run, kind, manifest["eval_dir"]):
                raise RuntimeError(f"Incomplete evaluation: {name}/{kind}")
        reference.write_json(status_path, {"status": "complete", "gpu": gpu})
        jobs.task_done()


def summarize(manifest):
    output = Path(manifest["output"])
    rows, missing = [], []
    expected = {target: reference.expected_rows(manifest["eval_dir"], target) for target in reference.TARGETS}
    artifacts = ["# RQ1 evaluation artifacts", "", "Best and last; action, weight and interaction. Supervised outputs use classifier heads, not text prototypes.", "", "| Model | Seed | Checkpoint | Target | UMAP | Confusion matrix | Predictions |", "| --- | ---: | --- | --- | --- | --- | --- |"]
    for run in manifest["runs"]:
        for kind in ("best", "last"):
            if not reference.evaluate_complete(run, kind, manifest["eval_dir"]):
                missing.append(f"{run['model']}/seed{run['seed']}/{kind}")
                continue
            for target in reference.TARGETS:
                values = reference.prediction_metrics(run["eval"][kind], run["supervised"], target, expected[target])
                rows.append({"model": run["model"], "seed": run["seed"], "checkpoint": kind, "target": target, **values})
                prediction, umap, confusion = reference.artifacts(run["eval"][kind], run["supervised"], target)
                artifacts.append(f"| {MODELS[run['model']]} | {run['seed']} | {kind} | {target} | [UMAP]({umap}) | [CM]({confusion}) | [CSV]({prediction}) |")
    (output / "ARTIFACTS.md").write_text("\n".join(artifacts) + "\n")
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "per_seed_metrics.csv", index=False)
    if missing:
        reference.write_json(output / "status.json", {"status": "incomplete", "missing_evaluations": missing})
        return False
    old = pd.read_csv(REFERENCE / "per_seed_metrics.csv")
    keys = ["model", "seed", "checkpoint", "target"]
    retained = frame[frame.model.isin(reference.MODELS)].sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(retained[old.columns], old.sort_values(keys).reset_index(drop=True), check_dtype=False, rtol=1e-12, atol=1e-12)
    summary_rows = []
    lines = ["# RQ1 supervised CNN / MLP: three-seed comparison", "", "Seeds 42, 43, 44; percentages, mean ± sample SD (ddof=1). Raw metric CSVs remain on the 0–1 scale.", "Six CNN/MLP training runs are new; four existing model comparisons are reused without changes.", "All supervised models use train-only normalization and the same action-stratified 90/10 window split per seed. Best is selected by validation mean Macro-F1; last is the actual stopping checkpoint.", "The split matches the original RQ1 protocol; it is not an episode-disjoint train/validation split.", "", "| Model | Checkpoint | Action: Accuracy / Macro F1 (%) | Weight: Accuracy / Macro F1 (%) | Interaction: Accuracy / Macro F1 (%) |", "| --- | --- | --- | --- | --- |"]
    for model in MODELS:
        for kind in ("best", "last"):
            cells = []
            for target in reference.TARGETS:
                group = frame[(frame.model == model) & (frame.checkpoint == kind) & (frame.target == target)]
                if sorted(group.seed.tolist()) != list(SEEDS):
                    raise ValueError(f"Expected all three seeds: {model}/{kind}/{target}")
                values = {"model": model, "checkpoint": kind, "target": target}
                for metric in ("accuracy", "macro_f1"):
                    values[f"{metric}_mean"] = group[metric].mean()
                    values[f"{metric}_std"] = group[metric].std(ddof=1)
                summary_rows.append(values)
                cells.append(" / ".join(f"{values[f'{metric}_mean'] * 100:.2f} ± {values[f'{metric}_std'] * 100:.2f}" for metric in ("accuracy", "macro_f1")))
            lines.append(f"| {MODELS[model]} | {kind} | " + " | ".join(cells) + " |")
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    pd.DataFrame(summary_rows).to_csv(output / "aggregate_metrics.csv", index=False)
    reference.write_json(output / "validation.json", {"reference_metrics_unchanged": True, "new_training_runs": 6, "new_checkpoint_evaluations": 12, "new_task_evaluations_with_umap_and_confusion": 36, "seeds": list(SEEDS)})
    reference.write_json(output / "status.json", {"status": "complete", "manuscript_updated": False})
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "summarize"))
    parser.add_argument("--group", required=True)
    parser.add_argument("--gpus", nargs="+", default=["0", "1"])
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.group):
        raise ValueError("Invalid group name")
    if len(set(args.gpus)) != len(args.gpus):
        raise ValueError("GPU IDs must be distinct")
    manifest = prepare(args.group)
    output = Path(manifest["output"])
    if args.command == "prepare":
        print(output / "manifest.json")
        return
    if args.command == "run":
        for path, expected in {**manifest["hashes"], **manifest["code_hashes"]}.items():
            if reference.sha256(path) != expected:
                raise ValueError(f"Input changed since preparation: {path}")
        reference.write_json(output / "status.json", {"status": "running", "gpus": args.gpus, "new_training_runs": 6})
        jobs = queue.Queue()
        for run in manifest["runs"]:
            if run.get("new_training"):
                jobs.put(run)
        failures = []
        with ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
            futures = [executor.submit(worker, manifest, jobs, gpu) for gpu in args.gpus]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as error:
                    failures.append(repr(error))
                    reference.write_json(output / "status.json", {"status": "worker_failed", "errors": failures})
        if failures:
            reference.write_json(output / "status.json", {"status": "failed", "errors": failures})
            raise RuntimeError(failures)
    if not summarize(manifest):
        raise RuntimeError("Evaluation is incomplete; see status.json")
    print(output / "summary.md")


if __name__ == "__main__":
    main()
