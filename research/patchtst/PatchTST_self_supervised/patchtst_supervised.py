# Modified for the language-grounded-ft research workspace; see repository history.
from workspace_paths import resolve_recorded_path, workspace_path
import argparse
import json
import math
import os
import random
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

import wandb

from baseline_models import FlatMLPContrastive, StandardCNNContrastive
from force_normalization import (
    compute_axis_stats,
    normalization_checkpoint_payload,
    save_stats_json,
)
from classification_taxonomy import (
    TAXONOMIES,
    TERRAIN0814_ACTION_CLASSES,
    canonicalize_classification_labels,
)
from patchtst_bert import DEFAULT_FORCE_COLS, WarmupCosineScheduler
from token_fusion_encoder import RawTokenFusionEncoder, encoder_config_from_args


TARGET_COLUMNS = ("annotation", "action", "weight", "interaction")
IGNORE_INDEX = -100


def resolve_index_path(data_dir: str, index_csv: str) -> str:
    if os.path.isabs(index_csv):
        return str(resolve_recorded_path(index_csv))
    return os.path.join(data_dir, index_csv)


def resolve_force_path(data_dir: str, csv_path: str) -> Optional[str]:
    candidates = [
        str(resolve_recorded_path(csv_path)),
        os.path.join(data_dir, "csv", csv_path),
        os.path.join(data_dir, csv_path),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def build_label_maps(index_path: str, label_taxonomy: str = "raw") -> Dict[str, Dict[str, int]]:
    frame = canonicalize_classification_labels(pd.read_csv(index_path), label_taxonomy)
    maps = {}
    for column in TARGET_COLUMNS:
        labels = sorted(frame[column].dropna().astype(str).unique().tolist())
        maps[column] = {label: idx for idx, label in enumerate(labels)}
    return maps


class ForceSupervisedDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        index_csv: str,
        label_maps: Dict[str, Dict[str, int]],
        seq_len: int = 80,
        use_cols: Optional[List[str]] = None,
        drop_zero_force: bool = True,
        zero_force_eps: float = 1e-8,
        max_samples: int = 0,
        seed: int = 42,
        active_targets: Optional[List[str]] = None,
        label_taxonomy: str = "raw",
    ):
        super().__init__()
        self.data_dir = data_dir
        self.index_path = resolve_index_path(data_dir, index_csv)
        self.label_maps = label_maps
        self.seq_len = seq_len
        self.use_cols = use_cols or DEFAULT_FORCE_COLS
        self.active_targets = set(active_targets or TARGET_COLUMNS)

        if not os.path.exists(self.index_path):
            raise FileNotFoundError(self.index_path)

        frame = canonicalize_classification_labels(
            pd.read_csv(self.index_path), label_taxonomy
        )
        required = {"csv_path", "start", *TARGET_COLUMNS}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Missing columns {sorted(missing)} in {self.index_path}")
        if max_samples > 0 and len(frame) > max_samples:
            frame = frame.sample(n=max_samples, random_state=seed).reset_index(drop=True)

        force_cache = {}
        for csv_path in frame["csv_path"].unique():
            resolved = resolve_force_path(data_dir, str(csv_path))
            if resolved is not None:
                force_cache[str(csv_path)] = pd.read_csv(
                    resolved, usecols=self.use_cols
                ).values.astype("float32")

        self.force_segments = []
        self.targets = {column: [] for column in TARGET_COLUMNS}
        self.sample_csv_paths = []
        self.sample_starts = []
        skipped_short = 0
        skipped_nonfinite = 0
        skipped_zero_force = 0
        skipped_missing_file = 0

        for row in frame.itertuples(index=False):
            csv_path = str(row.csv_path)
            if csv_path not in force_cache:
                skipped_missing_file += 1
                continue
            start = int(row.start)
            segment = force_cache[csv_path][start : start + seq_len]
            if segment.shape[0] != seq_len:
                skipped_short += 1
                continue
            if not np.isfinite(segment).all():
                skipped_nonfinite += 1
                continue
            if drop_zero_force and np.max(np.abs(segment)) <= zero_force_eps:
                skipped_zero_force += 1
                continue

            encoded = {}
            valid_labels = True
            for column in TARGET_COLUMNS:
                if column not in self.active_targets:
                    encoded[column] = IGNORE_INDEX
                    continue
                value = getattr(row, column)
                if pd.isna(value):
                    encoded[column] = IGNORE_INDEX
                    continue
                label = str(value)
                if label not in label_maps[column]:
                    valid_labels = False
                    break
                encoded[column] = label_maps[column][label]
            if not valid_labels:
                continue

            self.force_segments.append(segment)
            for column in TARGET_COLUMNS:
                self.targets[column].append(encoded[column])
            self.sample_csv_paths.append(csv_path)
            self.sample_starts.append(start)

        if not self.force_segments:
            raise RuntimeError(f"No usable samples in {self.index_path}")

        print(
            f"[ForceSupervisedDataset] usable={len(self.force_segments)} "
            f"missing_file={skipped_missing_file} short={skipped_short} "
            f"nonfinite={skipped_nonfinite} zero_force={skipped_zero_force}"
        )

    def __len__(self):
        return len(self.force_segments)

    def __getitem__(self, index):
        targets = {
            column: torch.tensor(self.targets[column][index], dtype=torch.long)
            for column in TARGET_COLUMNS
        }
        return torch.from_numpy(self.force_segments[index]), targets


class NormalizedSupervisedSubset(Dataset):
    def __init__(self, dataset, indices, mean, scale):
        self.dataset = dataset
        self.indices = list(indices)
        self.mean = torch.as_tensor(mean, dtype=torch.float32).view(1, -1)
        self.scale = torch.as_tensor(scale, dtype=torch.float32).view(1, -1)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        force, targets = self.dataset[self.indices[index]]
        return (force.float() - self.mean) / self.scale, targets


class RawTokenFusionClassifier(nn.Module):
    def __init__(self, task: str, class_counts: Dict[str, int], args, encoder=None):
        super().__init__()
        self.task = task
        if encoder is None:
            config = encoder_config_from_args(args, len(args.force_cols.split(",")))
            encoder = RawTokenFusionEncoder(**{
                key: value for key, value in config.items() if key != "encoder_type"
            })
        self.encoder = encoder
        self.head_dropout = nn.Dropout(args.head_dropout)
        if task == "annotation":
            self.heads = nn.ModuleDict(
                {"annotation": nn.Linear(args.out_embed_dim, class_counts["annotation"])}
            )
        else:
            self.heads = nn.ModuleDict(
                {
                    column: nn.Linear(args.out_embed_dim, class_counts[column])
                    for column in ("action", "weight", "interaction")
                }
            )

    @property
    def active_targets(self):
        if self.task == "annotation":
            return ("annotation",)
        return ("action", "weight", "interaction")

    def forward(self, force_sequence: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.head_dropout(self.encoder(force_sequence))
        return {name: head(features) for name, head in self.heads.items()}


def build_supervised_classifier(task: str, class_counts: Dict[str, int], args):
    model_arch = getattr(args, "model_arch", "tokenfusion")
    nvars = len(args.force_cols.split(","))
    if model_arch == "tokenfusion":
        return RawTokenFusionClassifier(task, class_counts, args)
    if model_arch == "cnn":
        if args.context_points < 2 ** args.cnn_depth:
            raise ValueError("CNN depth would pool the time axis to zero")
        encoder = StandardCNNContrastive(
            nvars=nvars,
            base_channels=args.cnn_base_channels,
            depth=args.cnn_depth,
            kernel_size=args.cnn_kernel_size,
            dropout=args.cnn_dropout,
            out_embed_dim=args.out_embed_dim,
            normalize_output=False,
        )
    elif model_arch == "mlp":
        encoder = FlatMLPContrastive(
            nvars=nvars,
            seq_len=args.context_points,
            hidden_dim=args.mlp_hidden_dim,
            depth=args.mlp_depth,
            dropout=args.mlp_dropout,
            out_embed_dim=args.out_embed_dim,
            normalize_output=False,
        )
    else:
        raise ValueError(f"Unknown supervised architecture: {model_arch}")
    return RawTokenFusionClassifier(task, class_counts, args, encoder=encoder)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def class_weights(
    dataset: ForceSupervisedDataset,
    indices: np.ndarray,
    target: str,
    num_classes: int,
    mode: str,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if mode == "none":
        return None
    values = np.asarray(dataset.targets[target], dtype=np.int64)[indices]
    values = values[values != IGNORE_INDEX]
    counts = np.bincount(values, minlength=num_classes).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError(f"Training split has an empty class for {target}: {counts.tolist()}")
    weights = len(values) / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def compute_loss(
    logits: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    weights: Dict[str, Optional[torch.Tensor]],
    label_smoothing: float,
):
    losses = {}
    for target, target_logits in logits.items():
        labels = targets[target]
        valid = labels != IGNORE_INDEX
        if valid.any():
            losses[target] = F.cross_entropy(
                target_logits[valid],
                labels[valid],
                weight=weights[target],
                label_smoothing=label_smoothing,
            )
    if not losses:
        raise RuntimeError("Batch has no valid supervised targets")
    return torch.stack(list(losses.values())).mean(), losses


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int):
    labels = np.arange(num_classes)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    _, _, micro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="micro",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "samples": int(len(y_true)),
    }


def evaluate(
    model: RawTokenFusionClassifier,
    loader: DataLoader,
    device: torch.device,
    weights: Dict[str, Optional[torch.Tensor]],
    label_smoothing: float,
    class_counts: Dict[str, int],
):
    model.eval()
    loss_total = 0.0
    batches = 0
    truths = {target: [] for target in model.active_targets}
    predictions = {target: [] for target in model.active_targets}

    with torch.no_grad():
        for force_sequence, batch_targets in loader:
            force_sequence = force_sequence.to(device, non_blocking=True)
            batch_targets = {
                target: values.to(device, non_blocking=True)
                for target, values in batch_targets.items()
            }
            logits = model(force_sequence)
            loss, _ = compute_loss(logits, batch_targets, weights, label_smoothing)
            loss_total += loss.item()
            batches += 1

            for target, target_logits in logits.items():
                labels = batch_targets[target]
                valid = labels != IGNORE_INDEX
                truths[target].append(labels[valid].cpu().numpy())
                predictions[target].append(target_logits[valid].argmax(dim=-1).cpu().numpy())

    metrics = {}
    merged_truths = {}
    merged_predictions = {}
    for target in model.active_targets:
        merged_truths[target] = np.concatenate(truths[target])
        merged_predictions[target] = np.concatenate(predictions[target])
        metrics[target] = classification_metrics(
            merged_truths[target], merged_predictions[target], class_counts[target]
        )
    metrics["loss"] = loss_total / max(1, batches)
    metrics["selection_macro_f1"] = float(
        np.mean([metrics[target]["macro_f1"] for target in model.active_targets])
    )
    return metrics, merged_truths, merged_predictions


def flatten_metrics(prefix: str, metrics: dict):
    flattened = {f"{prefix}/loss": metrics["loss"]}
    flattened[f"{prefix}/selection_macro_f1"] = metrics["selection_macro_f1"]
    for target, values in metrics.items():
        if not isinstance(values, dict):
            continue
        for metric_name, value in values.items():
            flattened[f"{prefix}/{target}_{metric_name}"] = value
    return flattened


def save_checkpoint(
    path: str, model, optimizer, epoch, args, label_maps, val_metrics, normalization
):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "task": args.task,
            "model_config": vars(args),
            "label_maps": label_maps,
            "val_metrics": val_metrics,
            "normalization": normalization,
        },
        path,
    )


def save_test_outputs(
    output_dir: str,
    checkpoint_name: str,
    metrics: dict,
    dataset: ForceSupervisedDataset,
    truths: Dict[str, np.ndarray],
    predictions: Dict[str, np.ndarray],
    label_maps: Dict[str, Dict[str, int]],
    active_targets,
):
    metrics_path = os.path.join(output_dir, f"{checkpoint_name}_test_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    frame = pd.DataFrame(
        {
            "csv_path": dataset.sample_csv_paths,
            "start": dataset.sample_starts,
        }
    )
    for target in active_targets:
        valid_indices = np.flatnonzero(np.asarray(dataset.targets[target]) != IGNORE_INDEX)
        inverse_map = {idx: label for label, idx in label_maps[target].items()}
        true_values = np.full(len(dataset), "", dtype=object)
        pred_values = np.full(len(dataset), "", dtype=object)
        true_values[valid_indices] = [inverse_map[int(value)] for value in truths[target]]
        pred_values[valid_indices] = [inverse_map[int(value)] for value in predictions[target]]
        frame[f"{target}_true"] = true_values
        frame[f"{target}_pred"] = pred_values
    frame.to_csv(
        os.path.join(output_dir, f"{checkpoint_name}_test_predictions.csv"), index=False
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Supervised normalized Token-Fusion PatchTST, CNN and MLP baselines"
    )
    parser.add_argument("--task", choices=["annotation", "multitask"], required=True)
    parser.add_argument("--model_arch", choices=["tokenfusion", "cnn", "mlp"], default="tokenfusion")
    parser.add_argument("--cnn_base_channels", type=int, default=81)
    parser.add_argument("--cnn_depth", type=int, default=4)
    parser.add_argument("--cnn_kernel_size", type=int, default=5)
    parser.add_argument("--cnn_dropout", type=float, default=0.2)
    parser.add_argument("--mlp_hidden_dim", type=int, default=429)
    parser.add_argument("--mlp_depth", type=int, default=3)
    parser.add_argument("--mlp_dropout", type=float, default=0.2)
    parser.add_argument(
        "--train_dir", default=str(workspace_path("Genesis/data/train_04272026/"))
    )
    parser.add_argument("--train_csv", default="train_04272026_thin_15pct.csv")
    parser.add_argument(
        "--eval_dir", default=str(workspace_path("Genesis/data/eval_04272026/"))
    )
    parser.add_argument("--eval_csv", default="eval_04272026_thin_15pct.csv")
    parser.add_argument("--context_points", type=int, default=80)
    parser.add_argument("--force_cols", default=",".join(DEFAULT_FORCE_COLS))
    parser.add_argument("--zero_force_eps", type=float, default=1e-8)
    parser.add_argument("--keep_zero_force", action="store_true")
    parser.add_argument(
        "--label_taxonomy",
        choices=TAXONOMIES,
        default="raw",
        help="Canonicalize training/evaluation labels before constructing classifier targets.",
    )
    parser.add_argument("--val_from_train_pct", type=float, default=0.10)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_eval_samples", type=int, default=0)

    parser.add_argument("--patch_len", type=int, default=10)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--n_heads", type=int, default=16)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--d_ff", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--head_dropout", type=float, default=0.2)
    parser.add_argument("--token_fusion_layers", type=int, default=1)
    parser.add_argument("--token_fusion_heads", type=int, default=8)
    parser.add_argument("--token_fusion_ff_mult", type=int, default=4)
    parser.add_argument("--out_embed_dim", type=int, default=768)

    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--peak_lr", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--gradient_clipping", type=float, default=1.0)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--class_weight", choices=["none", "balanced"], default="none")
    parser.add_argument("--early_stopping_patience", type=int, default=40)
    parser.add_argument("--early_stopping_min_delta", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--save_dir", default="saved_models/supervised")
    parser.add_argument("--run_name", default="")
    parser.add_argument("--wandb_project", default="patchtst_supervised")
    parser.add_argument("--wandb_group", default="")
    parser.add_argument("--no_wandb", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    if not 0 < args.val_from_train_pct < 1:
        raise ValueError("--val_from_train_pct must be between 0 and 1")
    if args.model_arch == "tokenfusion" and args.d_model % args.token_fusion_heads != 0:
        raise ValueError("d_model must be divisible by token_fusion_heads")

    run_name = args.run_name or f"supervised_{args.task}_{datetime.now():%Y%m%d_%H%M%S}"
    output_dir = os.path.join(args.save_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} task={args.task} output_dir={output_dir}")

    train_index_path = resolve_index_path(args.train_dir, args.train_csv)
    label_maps = build_label_maps(train_index_path, args.label_taxonomy)
    class_counts = {target: len(mapping) for target, mapping in label_maps.items()}
    expected_multitask_counts = {"weight": 2, "interaction": 3}
    if args.label_taxonomy == "terrain0814_eval":
        expected_multitask_counts["action"] = len(TERRAIN0814_ACTION_CLASSES)
    actual_multitask_counts = {
        target: class_counts[target] for target in expected_multitask_counts
    }
    if actual_multitask_counts != expected_multitask_counts:
        raise ValueError(
            f"Unexpected multitask class counts: {actual_multitask_counts}, "
            f"expected {expected_multitask_counts}"
        )
    print(f"class_counts={class_counts}")

    force_cols = [column.strip() for column in args.force_cols.split(",") if column.strip()]
    args.force_cols = ",".join(force_cols)
    active_targets = (
        ["annotation"]
        if args.task == "annotation"
        else ["action", "weight", "interaction"]
    )
    train_dataset = ForceSupervisedDataset(
        data_dir=args.train_dir,
        index_csv=args.train_csv,
        label_maps=label_maps,
        seq_len=args.context_points,
        use_cols=force_cols,
        drop_zero_force=not args.keep_zero_force,
        zero_force_eps=args.zero_force_eps,
        max_samples=args.max_train_samples,
        seed=args.seed,
        active_targets=active_targets,
        label_taxonomy=args.label_taxonomy,
    )
    test_dataset = ForceSupervisedDataset(
        data_dir=args.eval_dir,
        index_csv=args.eval_csv,
        label_maps=label_maps,
        seq_len=args.context_points,
        use_cols=force_cols,
        drop_zero_force=not args.keep_zero_force,
        zero_force_eps=args.zero_force_eps,
        max_samples=args.max_eval_samples,
        seed=args.seed,
        active_targets=active_targets,
        label_taxonomy=args.label_taxonomy,
    )

    all_indices = np.arange(len(train_dataset))
    stratify_target = "annotation" if args.task == "annotation" else "action"
    stratify_targets = np.asarray(train_dataset.targets[stratify_target])
    _, stratify_counts = np.unique(stratify_targets, return_counts=True)
    stratify = stratify_targets if np.all(stratify_counts >= 2) else None
    if stratify is None:
        print("[warn] validation split is not stratified because a sampled class has fewer than 2 examples")
    train_indices, val_indices = train_test_split(
        all_indices,
        test_size=args.val_from_train_pct,
        random_state=args.seed,
        stratify=stratify,
    )
    np.savez_compressed(
        os.path.join(output_dir, "split_indices.npz"),
        train=train_indices,
        validation=val_indices,
    )
    stats = compute_axis_stats(
        train_dataset.force_segments, train_indices, force_cols
    )
    normalization = normalization_checkpoint_payload(
        stats,
        {
            "data_dir": args.train_dir,
            "index_csv": args.train_csv,
            "seed": args.seed,
            "train_samples": len(train_indices),
            "validation_samples": len(val_indices),
            "context_points": args.context_points,
        },
    )
    save_stats_json(
        normalization, os.path.join(output_dir, f"{run_name}_normalization.json")
    )
    train_subset = NormalizedSupervisedSubset(
        train_dataset, train_indices, stats["mean"], stats["scale"]
    )
    val_subset = NormalizedSupervisedSubset(
        train_dataset, val_indices, stats["mean"], stats["scale"]
    )
    test_subset = NormalizedSupervisedSubset(
        test_dataset, np.arange(len(test_dataset)), stats["mean"], stats["scale"]
    )
    print(
        f"split train={len(train_subset)} val={len(val_subset)} "
        f"test={len(test_dataset)}"
    )

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_subset, shuffle=True, drop_last=False, **loader_kwargs)
    val_loader = DataLoader(val_subset, shuffle=False, drop_last=False, **loader_kwargs)
    test_loader = DataLoader(test_subset, shuffle=False, drop_last=False, **loader_kwargs)

    model = build_supervised_classifier(args.task, class_counts, args).to(device)
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"trainable_params={trainable_params:,}")

    loss_weights = {
        target: class_weights(
            train_dataset,
            train_indices,
            target,
            class_counts[target],
            args.class_weight,
            device,
        )
        for target in model.active_targets
    }
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.peak_lr, weight_decay=args.weight_decay
    )
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        min_lr=args.min_lr,
    )

    wandb_run = None
    if not args.no_wandb:
        try:
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=run_name,
                group=args.wandb_group or None,
                config=vars(args),
            )
        except Exception as error:
            print(f"[warn] wandb disabled: {error}")

    best_path = os.path.join(output_dir, f"{run_name}_best.pth")
    final_path = os.path.join(output_dir, f"{run_name}_final.pth")
    best_score = -math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    completed_epoch = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_total = 0.0
        train_steps = 0

        for force_sequence, batch_targets in train_loader:
            force_sequence = force_sequence.to(device, non_blocking=True)
            batch_targets = {
                target: values.to(device, non_blocking=True)
                for target, values in batch_targets.items()
            }
            optimizer.zero_grad(set_to_none=True)
            logits = model(force_sequence)
            loss, _ = compute_loss(
                logits, batch_targets, loss_weights, args.label_smoothing
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch={epoch}")
            loss.backward()
            if args.gradient_clipping > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clipping)
            optimizer.step()
            train_loss_total += loss.item()
            train_steps += 1

        val_metrics, _, _ = evaluate(
            model,
            val_loader,
            device,
            loss_weights,
            args.label_smoothing,
            class_counts,
        )
        train_loss = train_loss_total / max(1, train_steps)
        val_score = val_metrics["selection_macro_f1"]
        completed_epoch = epoch

        improved = val_score > best_score + args.early_stopping_min_delta
        if improved:
            best_score = val_score
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                best_path,
                model,
                optimizer,
                epoch,
                args,
                label_maps,
                val_metrics,
                normalization,
            )
        else:
            epochs_without_improvement += 1

        log_values = {
            "epoch": epoch,
            "train/loss": train_loss,
            "learning_rate": scheduler.get_last_lr()[0],
            **flatten_metrics("val", val_metrics),
        }
        history.append(log_values)
        pd.DataFrame(history).to_csv(os.path.join(output_dir, "history.csv"), index=False)
        print(
            f"epoch={epoch}/{args.epochs} train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_macro_f1={val_score:.4f} "
            f"best={best_score:.4f}@{best_epoch} lr={scheduler.get_last_lr()[0]:.6g}"
        )
        if wandb_run is not None:
            wandb.log(log_values)
        scheduler.step()

        if (
            args.early_stopping_patience > 0
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            print(
                f"early stopping at epoch={epoch}; "
                f"best_macro_f1={best_score:.4f}@{best_epoch}"
            )
            break

    final_val_metrics, _, _ = evaluate(
        model,
        val_loader,
        device,
        loss_weights,
        args.label_smoothing,
        class_counts,
    )
    save_checkpoint(
        final_path,
        model,
        optimizer,
        completed_epoch,
        args,
        label_maps,
        final_val_metrics,
        normalization,
    )

    summary = {
        "task": args.task,
        "model_arch": args.model_arch,
        "class_counts": class_counts,
        "trainable_params": trainable_params,
        "best_epoch": best_epoch,
        "completed_epoch": completed_epoch,
        "best_validation_macro_f1": best_score,
        "checkpoints": {},
    }
    for checkpoint_name, checkpoint_path in (("best", best_path), ("final", final_path)):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        test_metrics, truths, predictions = evaluate(
            model,
            test_loader,
            device,
            loss_weights,
            args.label_smoothing,
            class_counts,
        )
        summary["checkpoints"][checkpoint_name] = test_metrics
        save_test_outputs(
            output_dir,
            checkpoint_name,
            test_metrics,
            test_dataset,
            truths,
            predictions,
            label_maps,
            model.active_targets,
        )
        print(f"{checkpoint_name} test metrics: {json.dumps(test_metrics, ensure_ascii=False)}")

    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, "label_maps.json"), "w", encoding="utf-8") as file:
        json.dump(label_maps, file, ensure_ascii=False, indent=2)

    if wandb_run is not None:
        wandb.log(flatten_metrics("test_best", summary["checkpoints"]["best"]))
        wandb.log(flatten_metrics("test_final", summary["checkpoints"]["final"]))
        wandb.finish()


if __name__ == "__main__":
    main()
