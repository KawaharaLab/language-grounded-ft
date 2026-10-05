from workspace_paths import workspace_path
"""Replay manuscript classification runs and publish three-seed statistics."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import torch


ROOT = Path(__file__).resolve().parents[1]
CONTRASTIVE_GROUP = "terrain_floor_wobble_uniform_descent_contact_clean_slip5mms_20260824_132352"
SUPERVISED_GROUP = "tokenfusion_supervised_ssl_terrain0814_uniform_descent_slip5mms_20260825_015515"
REFERENCE_RESULTS = ROOT / "results/terrain_contrastive" / CONTRASTIVE_GROUP
SUPERVISED_RESULTS = ROOT / "results/tokenfusion_supervised_ssl" / SUPERVISED_GROUP
TARGETS = ("action", "weight", "interaction")
MODELS = {
    "tokenfusion": "Token-Fusion PatchTST",
    "cnn": "CNN",
    "mlp": "MLP",
    "supervised": "Supervised Token-Fusion",
}
SEEDS = (42, 43, 44)
EVAL_CSV = "eval_08142026_terrain_independent_balanced.csv"
EVAL_PATTERN = "eval_08142026_terrain_{target}_balanced.csv"
EVAL_ARCH_KEYS = (
    "context_points", "patch_len", "stride", "n_layers", "n_heads", "d_model",
    "d_ff", "head_dropout", "token_fusion_layers", "token_fusion_heads",
    "token_fusion_ff_mult", "out_embed_dim",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def cli_args(config):
    arguments = []
    for key, value in config.items():
        if isinstance(value, bool):
            if value:
                arguments.append(f"--{key}")
        elif value is not None:
            arguments.extend((f"--{key}", str(value)))
    return arguments


def artifacts(directory, supervised, target):
    directory = Path(directory)
    if supervised:
        return (
            directory / f"{target}_pred.csv",
            directory / f"{target}_umap.png",
            directory / f"{target}_confusion_matrix.png",
        )
    return (
        directory / f"pred/{target}_pred.csv",
        directory / f"umap/{target}_umap.png",
        directory / f"heatmap/{target}_heatmap.png",
    )


def expected_rows(eval_dir, target):
    frame = pd.read_csv(Path(eval_dir) / EVAL_PATTERN.format(target=target))
    return frame[["csv_path", "start", target]].rename(columns={target: "true"})


def prediction_metrics(directory, supervised, target, expected):
    paths = artifacts(directory, supervised, target)
    for path in paths:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    frame = pd.read_csv(paths[0])
    keys = ["csv_path", "start", "true"]
    actual = frame[keys].sort_values(keys).reset_index(drop=True)
    desired = expected.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, desired, check_dtype=False)
    if not set(frame["pred"]).issubset(set(expected["true"])):
        raise ValueError(f"Unexpected predicted class in {paths[0]}")
    return {
        "accuracy": accuracy_score(frame["true"], frame["pred"]),
        "macro_f1": f1_score(frame["true"], frame["pred"], average="macro", zero_division=0),
        "samples": len(frame),
    }


def manuscript_sections(text):
    table = next(
        match.group() for match in re.finditer(r"\\begin\{table\*\}.*?\\end\{table\*\}", text, re.S)
        if r"\label{tab:model_comparison}" in match.group()
    )
    training_start = text.index("各モデルは同一の学習データを用い")
    training_end = text.index("評価指標にはAccuracyとMacro F1を用いる．", training_start)
    training_end += len("評価指標にはAccuracyとMacro F1を用いる．")
    results_start = text.index(r"結果をTable~\ref{tab:model_comparison}に示す．")
    results_end = text.index("一方，本評価は有限個", results_start)
    return {"table": table, "training": text[training_start:training_end], "results": text[results_start:results_end]}


def prepare(group, manuscript):
    output = ROOT / "results/paper_classification_multiseed" / group
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    audit = json.loads((SUPERVISED_RESULTS / "data_audit.json").read_text())
    hashes = {audit["train_csv"]: audit["train_csv_sha256"]}
    hashes.update({value["csv"]: value["csv_sha256"] for value in audit["evaluation"].values()})
    for path, expected in hashes.items():
        if sha256(path) != expected:
            raise ValueError(f"Data differs from manuscript seed 42: {path}")
    eval_dir = output / "prepared_eval"
    shutil.copytree(REFERENCE_RESULTS / "prepared_eval_templates_release_merged", eval_dir, symlinks=True, dirs_exist_ok=True)
    templates = json.loads((eval_dir / "prototype_templates.json").read_text())
    if templates["action"]["release"] != "Releasing an object.":
        raise ValueError("The release prototype must be Releasing an object.")
    for target in TARGETS:
        labels = pd.read_csv(eval_dir / f"{target}_index_map.csv")[target].tolist()
        if "release midair" in labels:
            raise ValueError("release midair must be merged into release")
        if any(not templates[target][label].endswith(".") for label in labels):
            raise ValueError(f"Missing sentence template for {target}")
    code_dir = output / "code/PatchTST_self_supervised"
    source_root = ROOT / "PatchTST_self_supervised"
    source_files = list(source_root.glob("*.py")) + list((source_root / "src").rglob("*.py"))
    for source in source_files:
        destination = code_dir / source.relative_to(source_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    manuscript = Path(manuscript).resolve()
    manuscript_text = manuscript.read_text()
    sections = manuscript_sections(manuscript_text)
    shutil.copy2(manuscript, output / "paper.before_multiseed.tex")
    manifest = {
        "group": group, "created": datetime.now().isoformat(), "seeds": list(SEEDS),
        "manuscript": str(manuscript), "manuscript_sections": sections,
        "output": str(output), "eval_dir": str(eval_dir), "hashes": hashes,
        "code_hashes": {str(path): sha256(path) for path in code_dir.rglob("*.py")},
        "statistics": "Per-seed metric mean and sample standard deviation (ddof=1)",
        "selection": {"contrastive": "minimum validation loss", "supervised": "maximum validation mean macro-F1"},
        "runs": [],
    }
    for path in eval_dir.rglob("*"):
        if path.is_file() and not path.is_symlink():
            hashes[str(path)] = sha256(path)
    for model in MODELS:
        supervised = model == "supervised"
        reference_name = "tokenfusion_supervised_multitask_seed42" if supervised else f"terrain_floor_wobble_uniform_descent_slip5mms_{model}_seed42"
        reference_root = ROOT / "saved_models" / ("tokenfusion_supervised" if supervised else "terrain_contrastive") / (SUPERVISED_GROUP if supervised else CONTRASTIVE_GROUP) / reference_name
        reference_checkpoint = reference_root / f"{reference_name}_best.pth"
        checkpoint = torch.load(reference_checkpoint, map_location="cpu", weights_only=False)
        config = checkpoint["model_config" if supervised else "train_args"].copy()
        config = {key: value for key, value in config.items() if "key" not in key.lower()}
        for seed in SEEDS:
            name = f"paper_{model}_seed{seed}"
            save_root = ROOT / "saved_models/paper_classification_multiseed" / group
            save_dir = save_root / name
            run = {"model": model, "seed": seed, "supervised": supervised, "name": name, "eval": {}}
            last_suffix = "final" if supervised else "last"
            if seed == 42:
                run["checkpoints"] = {kind: str(reference_root / f"{reference_name}_{suffix}.pth") for kind, suffix in (("best", "best"), ("last", last_suffix))}
            else:
                run["checkpoints"] = {kind: str(save_dir / f"{name}_{suffix}.pth") for kind, suffix in (("best", "best"), ("last", last_suffix))}
                train_config = {**config, "seed": seed, "run_name": name, "save_dir": str(save_root if supervised else save_dir)}
                if supervised:
                    train_config.update(
                        eval_dir=str(eval_dir), eval_csv=EVAL_CSV, wandb_group=group,
                        no_wandb=os.environ.get("WANDB_MODE", "online") == "disabled",
                    )
                training_file = "patchtst_supervised.py" if supervised else ("patchtst_bert_normalized.py" if model == "tokenfusion" else f"patchtst_{model}.py")
                run["train"] = [sys.executable, "-u", str(code_dir / training_file), *cli_args(train_config)]
                run["train_done"] = str(output / "logs" / f"{name}.train.done")
                run["train_config"] = train_config
            for kind in ("best", "last"):
                target_dir = output / "evaluation" / model / f"seed{seed}" / kind
                target_dir.parent.mkdir(parents=True, exist_ok=True)
                if seed == 42:
                    if supervised:
                        original = SUPERVISED_RESULTS / "supervised_eval" / ("final" if kind == "last" else "best")
                    else:
                        original = ROOT / "data/terrain0814" / f"terrain0814_templates_release_merged_uniform_descent_slip5mms_{reference_name}_{kind}"
                    if not target_dir.exists():
                        target_dir.symlink_to(original, target_is_directory=True)
                run["eval"][kind] = str(target_dir)
                if seed == 42:
                    for target in TARGETS:
                        metrics = prediction_metrics(target_dir, supervised, target, expected_rows(eval_dir, target))
                        if kind == "best" and f"{metrics['accuracy']:.4f}" not in sections["table"]:
                            raise ValueError(f"Seed 42 {model}/{target} differs from the manuscript")
                    continue
                if supervised:
                    command = [sys.executable, "-u", str(code_dir / "evaluate_supervised_balanced.py"), "--checkpoint", run["checkpoints"][kind], "--eval_dir", str(eval_dir), "--eval_pattern", EVAL_PATTERN, "--output_dir", str(target_dir)]
                else:
                    output_name = f"paper3seed_{group}_{model}_seed{seed}_{kind}"
                    actual_output = ROOT / "data/paper3seed" / output_name
                    if not target_dir.exists() and not target_dir.is_symlink():
                        target_dir.symlink_to(actual_output, target_is_directory=True)
                    command = [sys.executable, "-u", str(code_dir / "patchtst_eval_normalized.py"), "--model_path", run["checkpoints"][kind], "--ckpt_type", "contrastive", "--model_arch", "patchtst" if model == "tokenfusion" else model, "--arch", "auto", "--pool", "mean", "--eval_dir", str(eval_dir), "--eval_csv", EVAL_CSV, "--targets", ",".join(TARGETS), "--text_type", "bert", "--zero_force_eps", "1e-8", "--output_name", output_name, *cli_args({key: config[key] for key in EVAL_ARCH_KEYS})]
                run[f"evaluate_{kind}"] = command
            manifest["runs"].append(run)
    (output / "logs").mkdir(exist_ok=True)
    write_json(manifest_path, manifest)
    write_json(output / "status.json", {"status": "prepared", "additional_training_runs": 8})
    return manifest


def run_command(command, environment, path):
    print(f"START {path.name}: {shlex.join(command)}", flush=True)
    with path.open("a") as handle:
        handle.write(f"\n{datetime.now().isoformat()} {shlex.join(command)}\n")
        handle.flush()
        subprocess.run(command, cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT, check=True)
    print(f"DONE {path.name}", flush=True)


def evaluate_complete(run, kind, eval_dir):
    try:
        for target in TARGETS:
            prediction_metrics(run["eval"][kind], run["supervised"], target, expected_rows(eval_dir, target))
    except (FileNotFoundError, AssertionError, ValueError, pd.errors.EmptyDataError):
        return False
    return True


def worker(manifest, seed, gpu):
    output = Path(manifest["output"])
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "WANDB_MODE": os.environ.get("WANDB_MODE", "online"), "PYTHONUNBUFFERED": "1"}
    order = {"tokenfusion": 0, "supervised": 1, "cnn": 2, "mlp": 3}
    jobs = sorted((run for run in manifest["runs"] if run["seed"] == seed), key=lambda run: order[run["model"]])
    for run in jobs:
        name = run["name"]
        complete = Path(run["train_done"])
        if not complete.exists():
            run_command(run["train"], environment, output / "logs" / f"{name}.train.log")
            for checkpoint in run["checkpoints"].values():
                if not Path(checkpoint).is_file():
                    raise FileNotFoundError(checkpoint)
            complete.write_text(datetime.now().isoformat() + "\n")
        for kind in ("best", "last"):
            if not evaluate_complete(run, kind, manifest["eval_dir"]):
                run_command(run[f"evaluate_{kind}"], environment, output / "logs" / f"{name}.{kind}.eval.log")
            if not evaluate_complete(run, kind, manifest["eval_dir"]):
                raise RuntimeError(f"Incomplete evaluation: {name}/{kind}")


def aggregate(manifest):
    output = Path(manifest["output"])
    rows = []
    missing = []
    artifact_lines = ["# Evaluation artifacts", "", "Each link contains all action, weight and interaction outputs.", "", "| Model | Seed | Checkpoint | Predictions / UMAP / Confusion matrix |", "| --- | ---: | --- | --- |"]
    expected = {target: expected_rows(manifest["eval_dir"], target) for target in TARGETS}
    for run in manifest["runs"]:
        for kind in ("best", "last"):
            if not evaluate_complete(run, kind, manifest["eval_dir"]):
                missing.append(f"{run['model']}/seed{run['seed']}/{kind}")
                continue
            for target in TARGETS:
                metrics = prediction_metrics(run["eval"][kind], run["supervised"], target, expected[target])
                rows.append({"model": run["model"], "seed": run["seed"], "checkpoint": kind, "target": target, **metrics})
            artifact_lines.append(f"| {MODELS[run['model']]} | {run['seed']} | {kind} | [{kind}]({run['eval'][kind]}/) |")
    (output / "ARTIFACTS.md").write_text("\n".join(artifact_lines) + "\n")
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "per_seed_metrics.csv", index=False)
    if missing:
        write_json(output / "status.json", {"status": "incomplete", "missing_evaluations": missing})
        return None
    summary = {}
    lines = ["# PAPER classification: three seeds", "", "Seeds: 42, 43, 44. Mean ± sample standard deviation (ddof=1); each seed has equal weight.", "Seed 42 reuses the manuscript experiment. Validation split, initialization, and data order vary with the seed.", "Contrastive best: validation loss. Supervised best: validation mean Macro-F1.", "Sentence prototypes are shared by classification and UMAP; release uses Releasing an object.", "", "| Model | Checkpoint | Action: Accuracy / Macro-F1 | Weight: Accuracy / Macro-F1 | Interaction: Accuracy / Macro-F1 |", "| --- | --- | --- | --- | --- |"]
    for model in MODELS:
        summary[model] = {}
        for kind in ("best", "last"):
            summary[model][kind] = {}
            cells = []
            for target in TARGETS:
                subset = frame[(frame.model == model) & (frame.checkpoint == kind) & (frame.target == target)]
                if sorted(subset.seed.tolist()) != list(SEEDS):
                    raise ValueError(f"Expected exactly three distinct seeds: {model}/{kind}/{target}")
                values = {metric: {"mean": float(subset[metric].mean()), "std": float(subset[metric].std(ddof=1))} for metric in ("accuracy", "macro_f1")}
                summary[model][kind][target] = values
                cells.append(" / ".join(f"{values[metric]['mean']:.4f} ± {values[metric]['std']:.4f}" for metric in ("accuracy", "macro_f1")))
            lines.append(f"| {MODELS[model]} | {kind} | " + " | ".join(cells) + " |")
    write_json(output / "summary.json", summary)
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    return summary


def publish(manifest, summary):
    output = Path(manifest["output"])
    source = manifest["manuscript_sections"]
    table = source["table"]
    table = re.sub(r"\\caption\{[^\n]*\}", lambda match: r"\caption{分類軸ごとに独立に均衡化した評価集合上の分類性能．releaseとrelease midairはreleaseへ統合した．各セルはAccuracy / Macro F1の3 seed（42, 43, 44）における平均$\pm$標準偏差（標本標準偏差，$n-1$分母）を示す．各seedのvalidation best checkpointを評価した．}", table)
    for model, label in MODELS.items():
        cells = []
        for target in TARGETS:
            cells.append(" / ".join(f"${summary[model]['best'][target][metric]['mean']:.4f}\\pm{summary[model]['best'][target][metric]['std']:.4f}$" for metric in ("accuracy", "macro_f1")))
        replacement = f"        {label} & " + " & ".join(cells) + r" \\"
        table, count = re.subn(r"^\s*" + re.escape(label) + r"\s*&[^\n]*", lambda match: replacement, table, flags=re.M)
        if count != 1:
            raise ValueError(f"Expected one table row for {label}")
    findings = []
    for target in TARGETS:
        winner = max((model for model in MODELS if model != "supervised"), key=lambda model: summary[model]["best"][target]["accuracy"]["mean"])
        values = summary[winner]["best"][target]["accuracy"]
        findings.append(f"{target}では{MODELS[winner]}が平均Accuracy ${values['mean']:.4f}\\pm{values['std']:.4f}$で最良となった")
    results = r"結果をTable~\ref{tab:model_comparison}に示す．" + "\n言語対照学習モデル間では，" + "，".join(findings) + "．\nこれらは初期化，学習時のサンプル順序，およびtrain/validation分割を含む3 seedのばらつきを併記した比較である．\n\n"
    training = source["training"] + "\n各モデルをseed 42，43，44で評価し，各seed内で学習用分割のみから正規化統計を算出した．AccuracyおよびMacro F1はseedごとに算出し，その平均と標本標準偏差（分母$n-1$）を報告する．"
    (output / "model_comparison_3seed.tex").write_text(table + "\n")
    (output / "classification_results_3seed.tex").write_text(training + "\n\n" + results)
    manuscript = Path(manifest["manuscript"])
    current = manuscript.read_text()
    replacements = {"table": table, "training": training, "results": results}
    if all(replacement in current for replacement in replacements.values()):
        write_json(output / "status.json", {"status": "complete", "manuscript_updated": str(manuscript)})
        return
    if any(current.count(original) != 1 for original in source.values()):
        write_json(output / "status.json", {"status": "complete_manuscript_changed", "message": "Results and LaTeX snippets saved; manuscript sections changed during training. Merge snippets manually."})
        print("Manuscript sections changed; generated LaTeX is available for review.")
        return
    shutil.copy2(manuscript, output / "paper.before_publish.tex")
    for name, replacement in replacements.items():
        current = current.replace(source[name], replacement, 1)
    manuscript.write_text(current)
    write_json(output / "status.json", {"status": "complete", "manuscript_updated": str(manuscript)})
    print(f"Updated manuscript: {manuscript}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "summarize"))
    parser.add_argument("--group", required=True)
    parser.add_argument("--gpus", nargs=2, default=["0", "1"])
    parser.add_argument("--manuscript", default=str(workspace_path("Genesis/paper.tex")))
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.group):
        raise ValueError("Group must contain only letters, digits, underscores, and hyphens")
    manifest = prepare(args.group, args.manuscript)
    output = Path(manifest["output"])
    if args.command == "prepare":
        aggregate(manifest)
        print(f"Prepared {output / 'manifest.json'}")
        return
    if args.command == "run":
        if len(set(args.gpus)) != 2:
            raise ValueError("Two distinct GPUs are required")
        for path, expected in {**manifest["hashes"], **manifest["code_hashes"]}.items():
            if sha256(path) != expected:
                raise ValueError(f"Input changed after preparation: {path}")
        write_json(output / "status.json", {"status": "running", "gpus": args.gpus, "additional_seeds": [43, 44]})
        failures = []
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(worker, manifest, seed, gpu) for seed, gpu in zip((43, 44), args.gpus)]
            for future in futures:
                try:
                    future.result()
                except Exception as error:
                    failures.append(str(error))
                    print(f"Worker failed: {error}", flush=True)
        if failures:
            write_json(output / "status.json", {"status": "failed", "errors": failures})
            raise RuntimeError("Training/evaluation failed; inspect logs and rerun the same GROUP")
    summary = aggregate(manifest)
    if summary is None:
        raise RuntimeError(f"Three-seed evaluation is incomplete; see {output / 'status.json'}")
    publish(manifest, summary)
    print(f"Results: {output / 'summary.md'}")


if __name__ == "__main__":
    main()
