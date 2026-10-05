#!/usr/bin/env python3
"""Frozen-feature linear-probe data-efficiency experiment for insertion detection."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import wandb

from train_terminal_insertion_probe import (
    PRETRAINED_MODELS,
    build_pretrained_encoder,
    encode_features,
    load_checkpoint,
    predict_logits,
    select_threshold,
    sequence_metrics,
    window_metrics,
)


MODEL_CHOICES = sorted(PRETRAINED_MODELS | {"raw_linear"})


def parse_int_list(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("expected a comma-separated list of positive integers")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", choices=MODEL_CHOICES, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episode-counts", type=parse_int_list, default=parse_int_list("1,2,5,10,25,50,100"))
    parser.add_argument("--seeds", type=parse_int_list, default=parse_int_list("42,43,44,45,46"))
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--fixed-threshold",
        type=float,
        default=None,
        help="Use this probability cutoff instead of validation Macro-F1 tuning.",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wandb-project", default="terminal-insertion-data-efficiency")
    parser.add_argument("--wandb-group", required=True)
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def json_dump(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def make_loader(
    features: torch.Tensor,
    targets: torch.Tensor,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> DataLoader:
    return DataLoader(
        TensorDataset(features, targets),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        generator=torch.Generator().manual_seed(seed),
    )


def first_consecutive_trigger(
    times: np.ndarray, probabilities: np.ndarray, threshold: float, required: int
) -> float | None:
    run_length = 0
    for time_sec, probability in zip(times, probabilities):
        run_length = run_length + 1 if probability >= threshold else 0
        if run_length >= required:
            return float(time_sec)
    return None


def consecutive_policy_metrics(
    probabilities: np.ndarray,
    episode_indices: np.ndarray,
    relative_sec: np.ndarray,
    threshold: float,
    required: int,
) -> dict:
    records = []
    for episode_index in sorted(np.unique(episode_indices).tolist()):
        mask = episode_indices == episode_index
        order = np.argsort(relative_sec[mask])
        times = relative_sec[mask][order]
        probs = probabilities[mask][order]
        trigger_sec = first_consecutive_trigger(times, probs, threshold, required)
        outcome = (
            "missed"
            if trigger_sec is None
            else "early_stop"
            if trigger_sec < 0.0
            else "success"
        )
        records.append(
            {
                "episode_index": int(episode_index),
                "outcome": outcome,
                "trigger_relative_sec": trigger_sec,
            }
        )
    early = [record for record in records if record["outcome"] == "early_stop"]
    success = [record for record in records if record["outcome"] == "success"]
    missed = [record for record in records if record["outcome"] == "missed"]
    delays = [float(record["trigger_relative_sec"]) for record in success]
    return {
        "required_consecutive": required,
        "episodes": len(records),
        "early_stop_episodes": len(early),
        "early_stop_rate": len(early) / len(records),
        "success_episodes": len(success),
        "success_rate": len(success) / len(records),
        "missed_episodes": len(missed),
        "missed_rate": len(missed) / len(records),
        "detection_delay_sec_mean": float(np.mean(delays)) if delays else None,
        "detection_delay_sec_median": float(np.median(delays)) if delays else None,
        "detection_delay_sec_p95": float(np.percentile(delays, 95)) if delays else None,
        "per_episode": records,
    }


def fit_linear_probe(
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    val_features: torch.Tensor,
    val_targets: torch.Tensor,
    args: argparse.Namespace,
    seed: int,
    condition_dir: Path,
) -> tuple[nn.Linear, list[dict], int, float]:
    set_seed(seed)
    device = torch.device(args.device)
    head = nn.Linear(train_features.shape[1], 1).to(device)
    positives = int((train_targets == 1).sum())
    negatives = int((train_targets == 0).sum())
    if positives == 0 or negatives == 0:
        raise ValueError(f"degenerate labels: before={negatives}, after={positives}")
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(negatives / positives, device=device)
    )
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(3, args.patience // 4)
    )
    train_loader = make_loader(
        train_features,
        train_targets,
        args.batch_size,
        True,
        args.num_workers,
        seed,
    )
    val_loader = make_loader(
        val_features,
        val_targets,
        args.batch_size,
        False,
        args.num_workers,
        seed,
    )

    best_path = condition_dir / "best.pth"
    best_val_loss = float("inf")
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        head.train()
        train_total = 0.0
        train_count = 0
        for features, targets in train_loader:
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = head(features).squeeze(-1)
            loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            train_total += float(loss.item()) * len(targets)
            train_count += len(targets)

        head.eval()
        val_total = 0.0
        val_count = 0
        with torch.no_grad():
            for features, targets in val_loader:
                features = features.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                loss = criterion(head(features).squeeze(-1), targets)
                val_total += float(loss.item()) * len(targets)
                val_count += len(targets)
        train_loss = train_total / train_count
        val_loss = val_total / val_count
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        if val_loss < best_val_loss - args.min_delta:
            best_val_loss = val_loss
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "state_dict": head.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                best_path,
            )
        else:
            stale += 1
        scheduler.step(val_loss)
        if args.patience > 0 and stale >= args.patience:
            break

    pd.DataFrame(history).to_csv(condition_dir / "history.csv", index=False)
    best = torch.load(best_path, map_location=device, weights_only=False)
    head.load_state_dict(best["state_dict"], strict=True)
    return head, history, best_epoch, best_val_loss


def probabilities(
    head: nn.Module,
    features: torch.Tensor,
    targets: torch.Tensor,
    args: argparse.Namespace,
    seed: int,
) -> np.ndarray:
    loader = make_loader(
        features,
        targets,
        args.batch_size,
        False,
        args.num_workers,
        seed,
    )
    logits = predict_logits(head, loader, torch.device(args.device))
    return torch.sigmoid(torch.from_numpy(logits)).numpy().reshape(-1)


def main() -> None:
    args = parse_args()
    if args.model in PRETRAINED_MODELS and args.checkpoint is None:
        raise ValueError(f"--checkpoint is required for {args.model}")
    if args.model == "raw_linear" and args.checkpoint is not None:
        raise ValueError("raw_linear does not use --checkpoint")
    if args.fixed_threshold is not None and not 0.0 < args.fixed_threshold < 1.0:
        raise ValueError("--fixed-threshold must be strictly between 0 and 1")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    train_episode_ids = list(manifest["splits"]["train"])
    if max(args.episode_counts) > len(train_episode_ids):
        raise ValueError(
            f"requested {max(args.episode_counts)} episodes from {len(train_episode_ids)}"
        )

    with np.load(args.dataset, allow_pickle=False) as dataset:
        force = dataset["force"].astype(np.float32)
        labels = dataset["label"].astype(np.int64)
        episode_indices = dataset["episode_index"].astype(np.int64)
        relative_key = (
            "relative_to_reference_sec"
            if "relative_to_reference_sec" in dataset.files
            else "relative_to_enter_sec"
        )
        relative_sec = dataset[relative_key].astype(np.float64)
        episode_ids = dataset["episode_ids"].astype(str).tolist()
        input_mean = dataset["mean"].astype(np.float32)
        input_scale = dataset["scale"].astype(np.float32)
    normalized_force = (force - input_mean.reshape(1, 1, -1)) / input_scale.reshape(
        1, 1, -1
    )
    split_episode_sets = {
        split: set(ids) for split, ids in manifest["splits"].items()
    }
    split_indices = {}
    for split in ("train", "val", "test"):
        episode_numbers = {
            index
            for index, episode_id in enumerate(episode_ids)
            if episode_id in split_episode_sets[split]
        }
        split_indices[split] = np.flatnonzero(
            np.asarray(
                [index in episode_numbers for index in episode_indices], dtype=bool
            )
        )

    feature_cache = output_dir / "fixed_features.npz"
    if feature_cache.exists():
        with np.load(feature_cache, allow_pickle=False) as cached:
            fixed_features = cached["features"].astype(np.float32)
        if len(fixed_features) != len(force):
            raise ValueError("cached feature row count does not match dataset")
        print(f"loaded cached features: {feature_cache}", flush=True)
    elif args.model == "raw_linear":
        fixed_features = normalized_force.reshape(len(normalized_force), -1)
        np.savez_compressed(feature_cache, features=fixed_features)
    else:
        set_seed(42)
        encoder_checkpoint = load_checkpoint(args.checkpoint)
        encoder = build_pretrained_encoder(args.model, encoder_checkpoint).to(
            torch.device(args.device)
        )
        for parameter in encoder.parameters():
            parameter.requires_grad = False
        fixed_features = encode_features(
            encoder,
            torch.from_numpy(normalized_force),
            args.batch_size,
            args.num_workers,
            torch.device(args.device),
        ).numpy()
        np.savez_compressed(feature_cache, features=fixed_features)
        del encoder
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    print(
        f"model={args.model} feature_shape={fixed_features.shape} "
        f"finite={np.isfinite(fixed_features).all()}",
        flush=True,
    )

    all_rows = []
    label_tensor = torch.from_numpy(labels.astype(np.float32))
    for seed in args.seeds:
        permutation = np.random.default_rng(seed).permutation(train_episode_ids).tolist()
        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        run = wandb.init(
            project=args.wandb_project,
            group=args.wandb_group,
            name=f"{args.model}_seed{seed}",
            mode=args.wandb_mode,
            dir=str(seed_dir),
            config={
                "model": args.model,
                "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
                "dataset": str(args.dataset.resolve()),
                "manifest": str(args.manifest.resolve()),
                "episode_counts": args.episode_counts,
                "seed": seed,
                "encoder_frozen": True,
                "head": f"Linear({fixed_features.shape[1]}, 1)",
                "feature_standardization": "mean/std from selected labeled training episodes only",
                "input_normalization": "fixed statistics from the 100-episode training split",
                "nested_episode_subsets": True,
                "epochs": args.epochs,
                "patience": args.patience,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "fixed_threshold": args.fixed_threshold,
            },
        )
        try:
            for count_index, episode_count in enumerate(args.episode_counts):
                selected_ids = set(permutation[:episode_count])
                selected_episode_numbers = {
                    index
                    for index, episode_id in enumerate(episode_ids)
                    if episode_id in selected_ids
                }
                train_indices = np.asarray(
                    [
                        index
                        for index in split_indices["train"]
                        if episode_indices[index] in selected_episode_numbers
                    ],
                    dtype=np.int64,
                )
                feature_mean = fixed_features[train_indices].mean(axis=0).astype(np.float32)
                feature_std = fixed_features[train_indices].std(axis=0).astype(np.float32)
                feature_scale = np.maximum(feature_std, 1e-6)
                standardized = {
                    split: torch.from_numpy(
                        (
                            fixed_features[indices] - feature_mean.reshape(1, -1)
                        )
                        / feature_scale.reshape(1, -1)
                    ).float()
                    for split, indices in {
                        "train": train_indices,
                        "val": split_indices["val"],
                        "test": split_indices["test"],
                    }.items()
                }
                condition_dir = seed_dir / f"n_{episode_count:03d}"
                condition_dir.mkdir(parents=True, exist_ok=True)
                json_dump(
                    {
                        "model": args.model,
                        "seed": seed,
                        "episode_count": episode_count,
                        "selected_episode_ids": permutation[:episode_count],
                        "train_windows": len(train_indices),
                        "feature_dimension": fixed_features.shape[1],
                        "feature_mean": feature_mean.tolist(),
                        "feature_std": feature_std.tolist(),
                        "feature_scale_floor": 1e-6,
                        "relative_time_key": relative_key,
                    },
                    condition_dir / "config.json",
                )
                head, history, best_epoch, best_val_loss = fit_linear_probe(
                    standardized["train"],
                    label_tensor[train_indices],
                    standardized["val"],
                    label_tensor[split_indices["val"]],
                    args,
                    seed + episode_count * 1000,
                    condition_dir,
                )
                val_probabilities = probabilities(
                    head,
                    standardized["val"],
                    label_tensor[split_indices["val"]],
                    args,
                    seed,
                )
                val_indices = split_indices["val"]
                if args.fixed_threshold is None:
                    threshold, val_selection = select_threshold(
                        labels[val_indices],
                        val_probabilities,
                        episode_indices[val_indices],
                        relative_sec[val_indices],
                    )
                    threshold_rule = "validation_macro_f1"
                else:
                    threshold = float(args.fixed_threshold)
                    val_selection = {
                        "threshold": threshold,
                        "window": window_metrics(
                            labels[val_indices], val_probabilities, threshold
                        ),
                        "sequence": sequence_metrics(
                            val_probabilities,
                            episode_indices[val_indices],
                            relative_sec[val_indices],
                            threshold,
                        ),
                    }
                    threshold_rule = "fixed"
                test_probabilities = probabilities(
                    head,
                    standardized["test"],
                    label_tensor[split_indices["test"]],
                    args,
                    seed,
                )
                test_indices = split_indices["test"]
                test_window = window_metrics(
                    labels[test_indices], test_probabilities, threshold
                )
                test_sequence_k1 = sequence_metrics(
                    test_probabilities,
                    episode_indices[test_indices],
                    relative_sec[test_indices],
                    threshold,
                )
                test_sequence_k3 = consecutive_policy_metrics(
                    test_probabilities,
                    episode_indices[test_indices],
                    relative_sec[test_indices],
                    threshold,
                    3,
                )
                metrics = {
                    "model": args.model,
                    "seed": seed,
                    "episode_count": episode_count,
                    "train_windows": len(train_indices),
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                    "selected_threshold": threshold,
                    "threshold_rule": threshold_rule,
                    "relative_time_key": relative_key,
                    "validation_selection": val_selection,
                    "test_window": test_window,
                    "test_sequence_k1": test_sequence_k1,
                    "test_sequence_k3": test_sequence_k3,
                    "wandb_url": run.url,
                }
                json_dump(metrics, condition_dir / "metrics.json")
                pd.DataFrame(
                    {
                        "window_index": test_indices,
                        "episode_index": episode_indices[test_indices],
                        "episode_id": [episode_ids[index] for index in episode_indices[test_indices]],
                        "relative_to_reference_sec": relative_sec[test_indices],
                        "label": labels[test_indices],
                        "probability_after": test_probabilities,
                        "prediction": (test_probabilities >= threshold).astype(np.int64),
                    }
                ).to_csv(condition_dir / "test_predictions.csv", index=False)
                row = {
                    "model": args.model,
                    "seed": seed,
                    "episode_count": episode_count,
                    "train_windows": len(train_indices),
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                    "threshold": threshold,
                    "test_macro_f1": test_window["macro_f1"],
                    "test_balanced_accuracy": test_window["balanced_accuracy"],
                    "test_auroc": test_window["auroc"],
                    "test_auprc": test_window["auprc"],
                    "test_k1_early_stop_rate": test_sequence_k1["early_stop_rate"],
                    "test_k1_delay_median_sec": test_sequence_k1[
                        "detection_delay_sec_median"
                    ],
                    "test_k3_early_stop_rate": test_sequence_k3["early_stop_rate"],
                    "test_k3_missed_rate": test_sequence_k3["missed_rate"],
                    "test_k3_delay_median_sec": test_sequence_k3[
                        "detection_delay_sec_median"
                    ],
                    "wandb_url": run.url,
                }
                all_rows.append(row)
                wandb.log(
                    {
                        "episode_count": episode_count,
                        "train_windows": len(train_indices),
                        "best_epoch": best_epoch,
                        "best_val_loss": best_val_loss,
                        "test/macro_f1": test_window["macro_f1"],
                        "test/balanced_accuracy": test_window["balanced_accuracy"],
                        "test/auroc": test_window["auroc"],
                        "test/auprc": test_window["auprc"],
                        "test/k1_early_stop_rate": test_sequence_k1[
                            "early_stop_rate"
                        ],
                        "test/k3_early_stop_rate": test_sequence_k3[
                            "early_stop_rate"
                        ],
                        "test/k3_missed_rate": test_sequence_k3["missed_rate"],
                        "test/k3_delay_median_sec": test_sequence_k3[
                            "detection_delay_sec_median"
                        ],
                    },
                    step=count_index,
                )
                print(
                    f"model={args.model} seed={seed} episodes={episode_count} "
                    f"best={best_val_loss:.6f}@{best_epoch} "
                    f"test_f1={test_window['macro_f1']:.4f} "
                    f"k3_early={test_sequence_k3['early_stop_rate']:.3f}",
                    flush=True,
                )
                del head, standardized
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
            seed_rows = [row for row in all_rows if row["seed"] == seed]
            pd.DataFrame(seed_rows).to_csv(seed_dir / "results.csv", index=False)
            artifact = wandb.Artifact(
                f"terminal-insertion-de-{args.model}-seed{seed}",
                type="evaluation",
            )
            artifact.add_file(seed_dir / "results.csv")
            run.log_artifact(artifact)
        finally:
            run.finish()

    pd.DataFrame(all_rows).to_csv(output_dir / "results.csv", index=False)
    print(f"completed model={args.model}: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
