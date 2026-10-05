#!/usr/bin/env python3
"""Train one terminal-insertion condition and evaluate it exactly once on test."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils import parametrize
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

import wandb

import patchtst_bert
from token_fusion_encoder import RawTokenFusionEncoder, build_token_fusion_encoder, encoder_config_from_args


PRETRAINED_MODELS = {
    "supervised",
    "byol",
    "contrastive",
    "masked_reconstruction",
    "jepa",
    "vae",
    "vicreg",
}
ALL_MODELS = sorted(PRETRAINED_MODELS | {"scratch_tokenfusion", "scratch_mlp"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", choices=ALL_MODELS, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--pretrained-adaptation",
        choices=("frozen_linear", "lora", "full_finetune"),
        default="frozen_linear",
        help="How to adapt a pretrained encoder to the binary task.",
    )
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=float, default=8.0)
    parser.add_argument(
        "--fixed-threshold",
        type=float,
        default=None,
        help="Use this probability cutoff instead of validation Macro-F1 tuning.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--train-episode-count",
        type=int,
        default=0,
        help=(
            "Use a deterministic nested subset of this many training episodes; "
            "zero uses the complete training split."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wandb-project", default="terminal-insertion-linear-probe")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--wandb-group", required=True)
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument(
        "--limit-per-split",
        type=int,
        default=0,
        help="Balanced deterministic subset used only for smoke tests",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_checkpoint(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu", weights_only=False)


def build_contrastive_encoder(checkpoint: dict) -> nn.Module:
    train_args = checkpoint["train_args"]
    encoder = patchtst_bert.PatchTSTContrastive(
        nvars=12,
        context_points=int(train_args.get("context_points", 80)),
        patch_len=int(train_args.get("patch_len", 10)),
        stride=int(train_args.get("stride", 10)),
        n_layers=int(train_args.get("n_layers", 3)),
        n_heads=int(train_args.get("n_heads", 16)),
        d_model=int(train_args.get("d_model", 128)),
        d_ff=int(train_args.get("d_ff", 768)),
        dropout=float(train_args.get("dropout", 0.2)),
        head_dropout=float(train_args.get("head_dropout", 0.2)),
        out_embed_dim=int(train_args.get("out_embed_dim", 768)),
        pool=str(train_args.get("pool", "mean")),
        arch=str(train_args.get("arch", "token_fusion")),
        input_features=str(train_args.get("input_features", "raw")),
        conv_stem_kernel=int(train_args.get("conv_stem_kernel", 5)),
        token_fusion_layers=int(train_args.get("token_fusion_layers", 1)),
        token_fusion_heads=int(train_args.get("token_fusion_heads", 8)),
        token_fusion_ff_mult=int(train_args.get("token_fusion_ff_mult", 4)),
        use_channel_embedding=bool(train_args.get("use_channel_embedding", False)),
    )
    encoder.load_state_dict(checkpoint["model"], strict=True)
    encoder.output_dim = int(train_args.get("out_embed_dim", 768))
    return encoder


def build_pretrained_encoder(model_name: str, checkpoint: dict) -> nn.Module:
    if model_name == "contrastive":
        return build_contrastive_encoder(checkpoint)
    if model_name == "supervised":
        if checkpoint.get("task") != "multitask":
            raise ValueError("The RQ1 supervised multitask checkpoint is required")
        args = SimpleNamespace(**checkpoint["model_config"])
        config = encoder_config_from_args(args, len(args.force_cols.split(",")))
        encoder = build_token_fusion_encoder(config)
        weights = {
            name.removeprefix("encoder."): value
            for name, value in checkpoint["model"].items()
            if name.startswith("encoder.")
        }
        encoder.load_state_dict(weights, strict=True)
        return encoder
    checkpoint_method = checkpoint.get("method")
    if checkpoint_method != model_name:
        raise ValueError(
            f"checkpoint method {checkpoint_method!r} does not match {model_name!r}"
        )
    encoder = build_token_fusion_encoder(checkpoint["encoder_config"])
    encoder.load_state_dict(checkpoint["encoder"], strict=True)
    return encoder


def scratch_tokenfusion_encoder() -> RawTokenFusionEncoder:
    return RawTokenFusionEncoder(
        nvars=12,
        context_points=80,
        patch_len=10,
        stride=10,
        n_layers=3,
        n_heads=16,
        d_model=128,
        d_ff=768,
        dropout=0.2,
        head_dropout=0.2,
        token_fusion_layers=1,
        token_fusion_heads=8,
        token_fusion_ff_mult=4,
        out_embed_dim=768,
        normalize_output=False,
    )


class EncoderClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, output_dim: int):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(output_dim, 1)

    def forward(self, force: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(force)).squeeze(-1)


class LinearProbe(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(input_dim, 1)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.classifier(embedding).squeeze(-1)


class LightweightMLP(nn.Module):
    """A deliberately small 960 -> 64 -> 1 full-window baseline."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(80 * 12, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, force: torch.Tensor) -> torch.Tensor:
        return self.network(force).squeeze(-1)


class LoRAWeight(nn.Module):
    """Low-rank additive parametrization for one Linear weight matrix."""

    def __init__(self, out_features: int, in_features: int, rank: int, alpha: float):
        super().__init__()
        self.scaling = float(alpha) / rank
        self.lora_a = nn.Parameter(torch.empty(rank, in_features))
        self.lora_b = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, original: torch.Tensor) -> torch.Tensor:
        return original + self.scaling * (self.lora_b @ self.lora_a)


class QVLoRAWeight(nn.Module):
    """Q/V-only LoRA for torch MultiheadAttention's packed QKV weight."""

    def __init__(self, embed_dim: int, rank: int, alpha: float):
        super().__init__()
        self.embed_dim = embed_dim
        self.scaling = float(alpha) / rank
        self.q_a = nn.Parameter(torch.empty(rank, embed_dim))
        self.q_b = nn.Parameter(torch.zeros(embed_dim, rank))
        self.v_a = nn.Parameter(torch.empty(rank, embed_dim))
        self.v_b = nn.Parameter(torch.zeros(embed_dim, rank))
        nn.init.kaiming_uniform_(self.q_a, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.v_a, a=math.sqrt(5))

    def forward(self, original: torch.Tensor) -> torch.Tensor:
        q_delta = self.scaling * (self.q_b @ self.q_a)
        k_delta = torch.zeros_like(q_delta)
        v_delta = self.scaling * (self.v_b @ self.v_a)
        return original + torch.cat((q_delta, k_delta, v_delta), dim=0)


def apply_attention_qv_lora(
    encoder: nn.Module, rank: int, alpha: float
) -> list[str]:
    """Freeze the encoder and add LoRA to every backbone/pooler Q and V map."""
    if rank <= 0:
        raise ValueError("--lora-rank must be positive")
    for parameter in encoder.parameters():
        parameter.requires_grad = False

    targets: list[str] = []
    modules = list(encoder.named_modules())
    for name, module in modules:
        if isinstance(module, nn.Linear) and (
            name.endswith("self_attn.W_Q") or name.endswith("self_attn.W_V")
        ):
            adapter = LoRAWeight(
                module.out_features, module.in_features, rank, alpha
            ).to(device=module.weight.device, dtype=module.weight.dtype)
            parametrize.register_parametrization(
                module,
                "weight",
                adapter,
            )
            targets.append(f"{name}.weight")
        elif isinstance(module, nn.MultiheadAttention):
            if module.in_proj_weight is None or not module._qkv_same_embed_dim:
                raise ValueError(f"unsupported MultiheadAttention layout at {name}")
            adapter = QVLoRAWeight(module.embed_dim, rank, alpha).to(
                device=module.in_proj_weight.device,
                dtype=module.in_proj_weight.dtype,
            )
            parametrize.register_parametrization(
                module,
                "in_proj_weight",
                adapter,
            )
            targets.extend(
                (f"{name}.in_proj_weight[Q]", f"{name}.in_proj_weight[V]")
            )
    if not targets:
        raise RuntimeError("no attention Q/V projections were found for LoRA")
    return targets


def keep_frozen_batchnorm_eval(module: nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, nn.modules.batchnorm._BatchNorm):
            child.eval()


def balanced_limit(indices: np.ndarray, labels: np.ndarray, limit: int, seed: int) -> np.ndarray:
    if limit <= 0 or len(indices) <= limit:
        return indices
    rng = np.random.default_rng(seed)
    classes = []
    half = limit // 2
    for label, requested in ((0, limit - half), (1, half)):
        candidates = indices[labels[indices] == label]
        take = min(requested, len(candidates))
        classes.append(rng.choice(candidates, size=take, replace=False))
    selected = np.concatenate(classes)
    if len(selected) < limit:
        remaining = np.setdiff1d(indices, selected, assume_unique=False)
        take = min(limit - len(selected), len(remaining))
        selected = np.concatenate(
            [selected, rng.choice(remaining, size=take, replace=False)]
        )
    rng.shuffle(selected)
    return selected.astype(np.int64)


def make_loader(
    features: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(features, labels),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        generator=generator,
    )


@torch.no_grad()
def encode_features(
    encoder: nn.Module,
    force: torch.Tensor,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> torch.Tensor:
    encoder.eval()
    loader = DataLoader(
        TensorDataset(force),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    outputs = []
    for (batch,) in loader:
        outputs.append(encoder(batch.to(device, non_blocking=True)).cpu())
    return torch.cat(outputs, dim=0)


@torch.no_grad()
def predict_logits(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    outputs = []
    for features, _ in loader:
        outputs.append(model(features.to(device, non_blocking=True)).cpu())
    return torch.cat(outputs).numpy()


def safe_float(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def window_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    predictions = (probabilities >= threshold).astype(np.int64)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "after_precision": float(precision_score(labels, predictions, zero_division=0)),
        "after_recall": float(recall_score(labels, predictions, zero_division=0)),
        "auroc": safe_float(roc_auc_score(labels, probabilities)),
        "auprc": safe_float(average_precision_score(labels, probabilities)),
        "confusion_matrix_before_after": matrix.tolist(),
        "windows": int(len(labels)),
        "before_windows": int((labels == 0).sum()),
        "after_windows": int((labels == 1).sum()),
    }


def sequence_metrics(
    probabilities: np.ndarray,
    episode_indices: np.ndarray,
    relative_sec: np.ndarray,
    threshold: float,
) -> dict:
    records = []
    for episode_index in sorted(np.unique(episode_indices).tolist()):
        mask = episode_indices == episode_index
        times = relative_sec[mask]
        probs = probabilities[mask]
        order = np.argsort(times)
        times = times[order]
        probs = probs[order]
        before_positive = times[(times < 0.0) & (probs >= threshold)]
        after_positive = times[(times >= 0.0) & (probs >= threshold)]
        records.append(
            {
                "episode_index": int(episode_index),
                "early_stop": bool(len(before_positive)),
                "first_early_stop_sec": (
                    float(before_positive[0]) if len(before_positive) else None
                ),
                "detected_after_enter": bool(len(after_positive)),
                "detection_delay_sec": (
                    float(after_positive[0]) if len(after_positive) else None
                ),
                "max_before_probability": (
                    float(probs[times < 0.0].max()) if np.any(times < 0.0) else None
                ),
                "max_after_probability": (
                    float(probs[times >= 0.0].max()) if np.any(times >= 0.0) else None
                ),
            }
        )
    early = [record for record in records if record["early_stop"]]
    missed = [record for record in records if not record["detected_after_enter"]]
    valid_delays = [
        record["detection_delay_sec"]
        for record in records
        if not record["early_stop"] and record["detection_delay_sec"] is not None
    ]
    return {
        "episodes": len(records),
        "early_stop_episodes": len(early),
        "early_stop_rate": len(early) / len(records),
        "missed_after_enter_episodes": len(missed),
        "missed_after_enter_rate": len(missed) / len(records),
        "detected_without_early_stop_episodes": len(valid_delays),
        "detection_delay_sec_mean": (
            float(np.mean(valid_delays)) if valid_delays else None
        ),
        "detection_delay_sec_median": (
            float(np.median(valid_delays)) if valid_delays else None
        ),
        "detection_delay_sec_p95": (
            float(np.percentile(valid_delays, 95)) if valid_delays else None
        ),
        "per_episode": records,
    }


def select_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    episode_indices: np.ndarray,
    relative_sec: np.ndarray,
) -> tuple[float, dict]:
    candidates = np.unique(
        np.concatenate(
            [np.linspace(0.001, 0.999, 999), probabilities.astype(np.float64)]
        )
    )
    best = None
    best_payload = None
    for threshold in candidates:
        metrics = window_metrics(labels, probabilities, float(threshold))
        sequence = sequence_metrics(
            probabilities, episode_indices, relative_sec, float(threshold)
        )
        # Primary selection is validation Macro F1 as specified.  Ties prefer
        # balanced accuracy, fewer unsafe early stops, then a threshold near 0.5.
        key = (
            metrics["macro_f1"],
            metrics["balanced_accuracy"],
            -sequence["early_stop_rate"],
            -abs(float(threshold) - 0.5),
        )
        if best is None or key > best:
            best = key
            best_payload = {
                "threshold": float(threshold),
                "window": metrics,
                "sequence": sequence,
            }
    assert best_payload is not None
    return float(best_payload["threshold"]), best_payload


def json_dump(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if args.model in PRETRAINED_MODELS and args.checkpoint is None:
        raise ValueError(f"--checkpoint is required for {args.model}")
    if args.model not in PRETRAINED_MODELS and args.checkpoint is not None:
        raise ValueError(f"--checkpoint is not used for {args.model}")
    if (
        args.model not in PRETRAINED_MODELS
        and args.pretrained_adaptation != "frozen_linear"
    ):
        raise ValueError("--pretrained-adaptation applies only to pretrained models")
    if args.fixed_threshold is not None and not 0.0 < args.fixed_threshold < 1.0:
        raise ValueError("--fixed-threshold must be strictly between 0 and 1")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    all_train_episode_ids = list(manifest["splits"]["train"])
    if args.train_episode_count < 0:
        raise ValueError("--train-episode-count cannot be negative")
    if args.train_episode_count > len(all_train_episode_ids):
        raise ValueError(
            f"requested {args.train_episode_count} training episodes from "
            f"{len(all_train_episode_ids)} available"
        )
    train_episode_count = args.train_episode_count or len(all_train_episode_ids)
    train_episode_permutation = np.random.default_rng(args.seed).permutation(
        all_train_episode_ids
    ).tolist()
    selected_train_episode_ids = train_episode_permutation[:train_episode_count]
    with np.load(args.dataset, allow_pickle=False) as dataset:
        force = dataset["force"].astype(np.float32)
        labels = dataset["label"].astype(np.int64)
        window_episode_indices = dataset["episode_index"].astype(np.int64)
        relative_key = (
            "relative_to_reference_sec"
            if "relative_to_reference_sec" in dataset.files
            else "relative_to_enter_sec"
        )
        relative_sec = dataset[relative_key].astype(np.float64)
        episode_ids = dataset["episode_ids"].astype(str).tolist()
        mean = dataset["mean"].astype(np.float32)
        scale = dataset["scale"].astype(np.float32)
        force_columns = dataset["force_columns"].astype(str).tolist()
        coordinate_sign = dataset["coordinate_sign"].astype(np.float32)
    normalized = (force - mean.reshape(1, 1, -1)) / scale.reshape(1, 1, -1)
    if not np.isfinite(normalized).all():
        raise ValueError("non-finite normalized force values")
    normalized_tensor = torch.from_numpy(normalized)
    label_tensor = torch.from_numpy(labels.astype(np.float32))

    split_episode_sets = {
        "train": set(selected_train_episode_ids),
        "val": set(manifest["splits"]["val"]),
        "test": set(manifest["splits"]["test"]),
    }
    split_indices = {}
    for split in ("train", "val", "test"):
        episode_numbers = {
            index
            for index, episode_id in enumerate(episode_ids)
            if episode_id in split_episode_sets[split]
        }
        indices = np.flatnonzero(
            np.asarray(
                [index in episode_numbers for index in window_episode_indices],
                dtype=bool,
            )
        )
        split_indices[split] = balanced_limit(
            indices, labels, args.limit_per_split, args.seed + len(split)
        )

    checkpoint = load_checkpoint(args.checkpoint) if args.checkpoint else None
    pretrained_encoder = args.model in PRETRAINED_MODELS
    frozen_encoder = (
        pretrained_encoder and args.pretrained_adaptation == "frozen_linear"
    )
    encoder_base_frozen = pretrained_encoder and args.pretrained_adaptation in {
        "frozen_linear",
        "lora",
    }
    lora_targets: list[str] = []
    if pretrained_encoder:
        encoder = build_pretrained_encoder(args.model, checkpoint).to(device)
        output_dim = int(getattr(encoder, "output_dim", 768))
        if args.pretrained_adaptation == "frozen_linear":
            for parameter in encoder.parameters():
                parameter.requires_grad = False
            encoder.eval()
            encoded = {}
            for split, indices in split_indices.items():
                encoded[split] = encode_features(
                    encoder,
                    normalized_tensor[indices],
                    args.batch_size,
                    args.num_workers,
                    device,
                )
            training_model: nn.Module = LinearProbe(output_dim).to(device)
            feature_tensors = encoded
            frozen_encoder_parameters = sum(
                parameter.numel() for parameter in encoder.parameters()
            )
            default_lr = 2e-3
        else:
            if args.pretrained_adaptation == "lora":
                lora_targets = apply_attention_qv_lora(
                    encoder, args.lora_rank, args.lora_alpha
                )
                default_lr = 5e-4
            else:
                for parameter in encoder.parameters():
                    parameter.requires_grad = True
                default_lr = 1e-4
            training_model = EncoderClassifier(encoder, output_dim).to(device)
            feature_tensors = {
                split: normalized_tensor[indices]
                for split, indices in split_indices.items()
            }
            frozen_encoder_parameters = 0
    elif args.model == "scratch_tokenfusion":
        encoder = scratch_tokenfusion_encoder()
        training_model = EncoderClassifier(encoder, encoder.output_dim).to(device)
        feature_tensors = {
            split: normalized_tensor[indices] for split, indices in split_indices.items()
        }
        frozen_encoder_parameters = 0
        default_lr = 3e-4
    else:
        training_model = LightweightMLP().to(device)
        feature_tensors = {
            split: normalized_tensor[indices] for split, indices in split_indices.items()
        }
        frozen_encoder_parameters = 0
        default_lr = 1e-3

    learning_rate = args.learning_rate or default_lr
    loaders = {
        split: make_loader(
            feature_tensors[split],
            label_tensor[indices],
            args.batch_size,
            split == "train",
            args.num_workers,
            args.seed,
        )
        for split, indices in split_indices.items()
    }
    train_labels = labels[split_indices["train"]]
    positives = int((train_labels == 1).sum())
    negatives = int((train_labels == 0).sum())
    if positives == 0 or negatives == 0:
        raise ValueError(f"training labels are degenerate: before={negatives}, after={positives}")
    pos_weight_value = negatives / positives
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight_value, device=device)
    )
    optimized_parameters = [
        parameter for parameter in training_model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        optimized_parameters, lr=learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(3, args.patience // 4)
    )

    model_parameters = frozen_encoder_parameters + sum(
        parameter.numel() for parameter in training_model.parameters()
    )
    trainable_parameters = sum(
        parameter.numel()
        for parameter in training_model.parameters()
        if parameter.requires_grad
    )
    argument_config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    run_config = {
        **argument_config,
        "dataset": str(args.dataset.resolve()),
        "manifest": str(args.manifest.resolve()),
        "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
        "force_columns": force_columns,
        "coordinate_transform": manifest["coordinate_transform"],
        "coordinate_sign": coordinate_sign.tolist(),
        "normalization_source": "100 real-robot training episodes",
        "selected_train_episode_ids": selected_train_episode_ids,
        "train_episode_count": train_episode_count,
        "train_windows": len(split_indices["train"]),
        "val_windows": len(split_indices["val"]),
        "test_windows": len(split_indices["test"]),
        "pos_weight": pos_weight_value,
        "model_parameters": model_parameters,
        "trainable_parameters": trainable_parameters,
        "encoder_frozen": frozen_encoder,
        "encoder_base_frozen": encoder_base_frozen,
        "pretrained_adaptation": args.pretrained_adaptation,
        "lora_targets": lora_targets,
    }
    run_name = (
        args.model
        if not pretrained_encoder
        else f"{args.model}_{args.pretrained_adaptation}"
    )
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        group=args.wandb_group,
        name=run_name,
        mode=args.wandb_mode,
        config=run_config,
        dir=str(output_dir),
    )

    best_path = output_dir / "best.pth"
    history = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_gradient_steps = 0
    gradient_steps = 0
    stale = 0
    try:
        for epoch in range(1, args.epochs + 1):
            training_model.train()
            if args.pretrained_adaptation == "lora":
                keep_frozen_batchnorm_eval(training_model)
            train_total = 0.0
            train_count = 0
            for features, targets in loaders["train"]:
                features = features.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = training_model(features)
                loss = criterion(logits, targets)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite training loss at epoch {epoch}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(optimized_parameters, 1.0)
                optimizer.step()
                gradient_steps += 1
                train_total += float(loss.item()) * len(targets)
                train_count += len(targets)

            training_model.eval()
            val_total = 0.0
            val_count = 0
            val_epoch_logits = []
            val_epoch_targets = []
            with torch.no_grad():
                for features, targets in loaders["val"]:
                    features = features.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    logits = training_model(features)
                    loss = criterion(logits, targets)
                    val_total += float(loss.item()) * len(targets)
                    val_count += len(targets)
                    val_epoch_logits.append(logits.detach().cpu())
                    val_epoch_targets.append(targets.detach().cpu())
            train_loss = train_total / train_count
            val_loss = val_total / val_count
            val_epoch_probabilities = torch.sigmoid(
                torch.cat(val_epoch_logits)
            ).numpy()
            val_epoch_labels = torch.cat(val_epoch_targets).numpy().astype(np.int64)
            history_threshold = (
                float(args.fixed_threshold)
                if args.fixed_threshold is not None
                else 0.5
            )
            val_epoch_metrics = window_metrics(
                val_epoch_labels, val_epoch_probabilities, history_threshold
            )
            learning_rate_now = optimizer.param_groups[0]["lr"]
            history.append(
                {
                    "epoch": epoch,
                    "gradient_steps": gradient_steps,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_macro_f1_at_fixed_threshold": val_epoch_metrics["macro_f1"],
                    "val_balanced_accuracy_at_fixed_threshold": val_epoch_metrics[
                        "balanced_accuracy"
                    ],
                    "val_auroc": val_epoch_metrics["auroc"],
                    "learning_rate": learning_rate_now,
                }
            )
            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                    "gradient_steps": gradient_steps,
                    "val/macro_f1_at_fixed_threshold": val_epoch_metrics["macro_f1"],
                    "learning_rate": learning_rate_now,
                },
                step=epoch,
            )
            improved = val_loss < best_val_loss - args.min_delta
            if improved:
                best_val_loss = val_loss
                best_epoch = epoch
                best_gradient_steps = gradient_steps
                stale = 0
                torch.save(
                    {
                        "model_name": args.model,
                        "state_dict": training_model.state_dict(),
                        "encoder_checkpoint": (
                            str(args.checkpoint.resolve()) if args.checkpoint else None
                        ),
                        "encoder_frozen": frozen_encoder,
                        "encoder_base_frozen": encoder_base_frozen,
                        "pretrained_adaptation": args.pretrained_adaptation,
                        "epoch": epoch,
                        "gradient_steps": gradient_steps,
                        "val_loss": val_loss,
                        "normalization": {
                            "force_columns": force_columns,
                            "mean": mean,
                            "scale": scale,
                            "coordinate_sign": coordinate_sign,
                        },
                        "manifest": manifest,
                        "run_config": run_config,
                    },
                    best_path,
                )
            else:
                stale += 1
            scheduler.step(val_loss)
            print(
                f"model={args.model} epoch={epoch}/{args.epochs} "
                f"train={train_loss:.6f} val={val_loss:.6f} "
                f"best={best_val_loss:.6f}@{best_epoch} stale={stale}",
                flush=True,
            )
            if args.patience > 0 and stale >= args.patience:
                print(f"early stopping model={args.model} epoch={epoch}", flush=True)
                break

        pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
        best = torch.load(best_path, map_location=device, weights_only=False)
        training_model.load_state_dict(best["state_dict"], strict=True)
        val_logits = predict_logits(training_model, loaders["val"], device)
        val_probabilities = torch.sigmoid(torch.from_numpy(val_logits)).numpy()
        val_indices = split_indices["val"]
        if args.fixed_threshold is None:
            threshold, val_selection = select_threshold(
                labels[val_indices],
                val_probabilities,
                window_episode_indices[val_indices],
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
                    window_episode_indices[val_indices],
                    relative_sec[val_indices],
                    threshold,
                ),
            }
            threshold_rule = "fixed"

        # The test split is first interpreted here, after checkpoint and threshold
        # selection are complete.
        test_logits = predict_logits(training_model, loaders["test"], device)
        test_probabilities = torch.sigmoid(torch.from_numpy(test_logits)).numpy()
        test_indices = split_indices["test"]
        test_window = window_metrics(
            labels[test_indices], test_probabilities, threshold
        )
        test_sequence = sequence_metrics(
            test_probabilities,
            window_episode_indices[test_indices],
            relative_sec[test_indices],
            threshold,
        )
        metrics = {
            "model": args.model,
            "timing_reference": manifest.get("timing_reference", "recorded Enter key stamp"),
            "enter_reaction_delay_sec": float(
                manifest.get("enter_reaction_delay_sec", 0.0)
            ),
            "best_epoch": best_epoch,
            "best_gradient_steps": best_gradient_steps,
            "best_val_loss": best_val_loss,
            "selected_threshold": threshold,
            "threshold_rule": threshold_rule,
            "validation_selection": val_selection,
            "test_window": test_window,
            "test_sequence": test_sequence,
            "model_parameters": model_parameters,
            "trainable_parameters": trainable_parameters,
            "encoder_frozen": frozen_encoder,
            "encoder_base_frozen": encoder_base_frozen,
            "pretrained_adaptation": args.pretrained_adaptation,
            "lora_targets": lora_targets,
            "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
            "train_episode_count": train_episode_count,
            "train_windows": len(split_indices["train"]),
            "selected_train_episode_ids": selected_train_episode_ids,
            "learning_rate": learning_rate,
            "wandb": {
                "project": args.wandb_project,
                "group": args.wandb_group,
                "run_id": run.id if run else None,
                "url": run.url if run else None,
            },
        }
        json_dump(metrics, output_dir / "metrics.json")

        prediction_rows = pd.DataFrame(
            {
                "window_index": test_indices,
                "episode_index": window_episode_indices[test_indices],
                "episode_id": [
                    episode_ids[index]
                    for index in window_episode_indices[test_indices]
                ],
                "relative_to_enter_sec": relative_sec[test_indices],
                "relative_to_reference_sec": relative_sec[test_indices],
                "label": labels[test_indices],
                "probability_after": test_probabilities,
                "prediction": (test_probabilities >= threshold).astype(np.int64),
            }
        )
        prediction_rows.to_csv(output_dir / "test_predictions.csv", index=False)

        summary_values = {
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "selected_threshold": threshold,
            **{f"test/window_{key}": value for key, value in test_window.items() if isinstance(value, (int, float)) and value is not None},
            **{f"test/sequence_{key}": value for key, value in test_sequence.items() if isinstance(value, (int, float)) and value is not None},
        }
        wandb.summary.update(summary_values)
        artifact = wandb.Artifact(
            name=f"terminal-insertion-{args.wandb_group}-{run_name}",
            type="model-evaluation",
            metadata={"model": args.model, "best_epoch": best_epoch},
        )
        for artifact_path in (
            best_path,
            output_dir / "metrics.json",
            output_dir / "history.csv",
            output_dir / "test_predictions.csv",
            args.manifest,
        ):
            artifact.add_file(str(artifact_path))
        run.log_artifact(artifact)
        print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
