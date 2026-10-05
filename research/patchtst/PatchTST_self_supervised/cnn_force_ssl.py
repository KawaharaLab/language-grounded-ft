from workspace_paths import resolve_recorded_path, workspace_path
import argparse
import copy
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import wandb

from baseline_models import StandardCNNContrastive, choose_cnn_width, count_trainable_params
from force_normalization import (
    compute_axis_stats,
    normalization_checkpoint_payload,
    save_stats_json,
)
import patchtst_bert as patchtst


DEFAULT_FORCE_COLS = patchtst.DEFAULT_FORCE_COLS


class ForceWindowDataset(Dataset):
    def __init__(self, data_dir, index_csv, seq_len, force_cols, zero_force_eps=1e-8):
        self.force_segments = []
        index = pd.read_csv(Path(data_dir) / index_csv)
        cache = {}
        skipped = {"missing": 0, "short": 0, "nonfinite": 0, "zero": 0}
        for path in index["csv_path"].unique():
            csv_path = resolve_recorded_path(str(path))
            if not csv_path.is_file():
                csv_path = Path(data_dir) / "csv" / str(path)
            if csv_path.exists():
                cache[path] = pd.read_csv(csv_path, usecols=force_cols).to_numpy(dtype=np.float32)
        for row in index.itertuples(index=False):
            if row.csv_path not in cache:
                skipped["missing"] += 1
                continue
            segment = cache[row.csv_path][int(row.start):int(row.start) + seq_len]
            if len(segment) != seq_len:
                skipped["short"] += 1
                continue
            if not np.isfinite(segment).all():
                skipped["nonfinite"] += 1
                continue
            if np.max(np.abs(segment)) <= zero_force_eps:
                skipped["zero"] += 1
                continue
            self.force_segments.append(segment)
        if not self.force_segments:
            raise RuntimeError("No usable force windows")
        print(f"Loaded {len(self.force_segments)} force windows; skipped={skipped}")

    def __len__(self):
        return len(self.force_segments)

    def __getitem__(self, index):
        return torch.from_numpy(self.force_segments[index])


class NormalizedSubset(Dataset):
    def __init__(self, dataset, indices, mean, scale):
        self.dataset = dataset
        self.indices = list(indices)
        self.mean = torch.as_tensor(mean).float().view(1, -1)
        self.scale = torch.as_tensor(scale).float().view(1, -1)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        force = self.dataset[self.indices[index]].float()
        return (force - self.mean) / self.scale


def masked_view(force, ratio):
    mask = torch.rand(force.shape[0], force.shape[1], 1, device=force.device) < ratio
    return force.masked_fill(mask, 0.0), mask.expand_as(force)


def augmented_view(force, mask_ratio):
    output = force * (1.0 + 0.1 * torch.randn(force.shape[0], 1, force.shape[2], device=force.device))
    output = output + 0.03 * torch.randn_like(output)
    output, _ = masked_view(output, mask_ratio)
    channel_mask = torch.rand(output.shape[0], 1, output.shape[2], device=output.device) < 0.1
    return output.masked_fill(channel_mask, 0.0)


class MaskedUNet(nn.Module):
    def __init__(self, encoder, seq_len, nvars):
        super().__init__()
        self.encoder = encoder
        channels = [block[0].out_channels for block in encoder.features]
        self.seed = nn.Linear(encoder.output.out_features, channels[-1] * math.ceil(seq_len / 2 ** len(channels)))
        decoder = []
        current = channels[-1]
        for skip_channels in reversed(channels[:-1]):
            decoder.append(nn.Conv1d(current + skip_channels, skip_channels, 3, padding=1))
            current = skip_channels
        self.decoder = nn.ModuleList(decoder)
        self.reconstruct = nn.Conv1d(current, nvars, 3, padding=1)
        self.seq_len = seq_len

    def forward(self, force):
        value = force.transpose(1, 2)
        skips = []
        for block in self.encoder.features:
            value = block(value)
            skips.append(value)
        pooled = self.encoder.global_pool(value).squeeze(-1)
        representation = self.encoder.output(pooled)
        value = self.seed(representation).view(value.shape[0], value.shape[1], value.shape[2])
        for layer, skip in zip(self.decoder, reversed(skips[:-1])):
            value = F.interpolate(value, size=skip.shape[-1], mode="linear", align_corners=False)
            value = F.gelu(layer(torch.cat([value, skip], dim=1)))
        value = F.interpolate(value, size=self.seq_len, mode="linear", align_corners=False)
        return self.reconstruct(value).transpose(1, 2), F.normalize(representation, dim=-1)


class VAEModel(nn.Module):
    def __init__(self, encoder, seq_len, nvars, latent_dim):
        super().__init__()
        self.encoder = encoder
        embedding_dim = encoder.output.out_features
        self.mu = nn.Linear(embedding_dim, latent_dim)
        self.logvar = nn.Linear(embedding_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.GELU(),
            nn.Linear(512, seq_len * nvars),
        )
        self.seq_len = seq_len
        self.nvars = nvars

    def forward(self, force):
        representation = self.encoder(force)
        mu = self.mu(representation)
        logvar = self.logvar(representation).clamp(-10, 10)
        latent = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        reconstruction = self.decoder(latent).view(-1, self.seq_len, self.nvars)
        return reconstruction, mu, logvar, representation


class JEPAState(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.target_encoder = copy.deepcopy(encoder)
        self.target_encoder.requires_grad_(False)
        embedding_dim = encoder.output.out_features
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    @torch.no_grad()
    def update_target(self, momentum):
        for target, online in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            target.data.mul_(momentum).add_(online.data, alpha=1.0 - momentum)


class VICRegState(nn.Module):
    def __init__(self, encoder, projection_dim):
        super().__init__()
        self.encoder = encoder
        embedding_dim = encoder.output.out_features
        self.projector = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Linear(512, projection_dim),
        )


def variance_loss(embedding):
    std = torch.sqrt(embedding.var(dim=0) + 1e-4)
    return F.relu(1.0 - std).mean()


def covariance_loss(embedding):
    embedding = embedding - embedding.mean(dim=0)
    covariance = embedding.T @ embedding / max(1, embedding.shape[0] - 1)
    return (covariance.square().sum() - covariance.diagonal().square().sum()) / embedding.shape[1]


def effective_dimension(embeddings):
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    variance = singular_values ** 2
    probability = variance / variance.sum()
    probability = probability[probability > 0]
    effective_rank = float(np.exp(-(probability * np.log(probability)).sum()))
    participation_ratio = float(variance.sum() ** 2 / np.square(variance).sum())
    cumulative = np.cumsum(np.sort(probability)[::-1])
    return {
        "effective_rank": effective_rank,
        "participation_ratio": participation_ratio,
        "dimensions_90pct": int(np.searchsorted(cumulative, 0.90) + 1),
        "dimensions_95pct": int(np.searchsorted(cumulative, 0.95) + 1),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["masked_unet", "jepa", "vae", "vicreg"], required=True)
    parser.add_argument("--data_dir", default=str(workspace_path("Genesis/data/train_04272026/")))
    parser.add_argument("--index_csv", default="train_04272026_thin_15pct.csv")
    parser.add_argument("--context_points", type=int, default=80)
    parser.add_argument("--force_cols", default=",".join(DEFAULT_FORCE_COLS))
    parser.add_argument("--zero_force_eps", type=float, default=1e-8)
    parser.add_argument("--val_from_train_pct", type=float, default=0.10)
    parser.add_argument("--cnn_base_channels", type=int, default=81)
    parser.add_argument("--cnn_depth", type=int, default=4)
    parser.add_argument("--cnn_kernel_size", type=int, default=5)
    parser.add_argument("--cnn_dropout", type=float, default=0.2)
    parser.add_argument("--out_embed_dim", type=int, default=768)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--peak_lr", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--mask_ratio", type=float, default=0.4)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--vae_beta", type=float, default=1e-3)
    parser.add_argument("--jepa_momentum", type=float, default=0.996)
    parser.add_argument("--vicreg_projection_dim", type=int, default=128)
    parser.add_argument("--early_stopping_patience", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--wandb_project", default="cnn_force_ssl")
    parser.add_argument("--wandb_group", default="")
    parser.add_argument("--wandb_entity", default="")
    parser.add_argument(
        "--wandb_mode",
        choices=["online", "offline", "disabled"],
        default="disabled",
        help="Disabled by default so existing scripts keep their previous behavior.",
    )
    parser.add_argument("--wandb_api_key", default=os.getenv("WANDB_API_KEY", ""))
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.wandb_api_key and args.wandb_mode != "disabled":
        wandb.login(key=args.wandb_api_key)
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        group=args.wandb_group or None,
        name=args.run_name,
        config=vars(args),
        mode=args.wandb_mode,
    )
    force_cols = args.force_cols.split(",")
    dataset = ForceWindowDataset(
        args.data_dir,
        args.index_csv,
        args.context_points,
        force_cols,
        args.zero_force_eps,
    )
    total = len(dataset)
    val_size = max(1, round(total * args.val_from_train_pct))
    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(total, generator=generator).tolist()
    train_indices = indices[:-val_size]
    val_indices = indices[-val_size:]
    stats = compute_axis_stats(dataset.force_segments, train_indices, force_cols)
    normalization = normalization_checkpoint_payload(
        stats,
        {"data_dir": args.data_dir, "index_csv": args.index_csv, "seed": args.seed},
    )
    train_data = NormalizedSubset(dataset, train_indices, stats["mean"], stats["scale"])
    val_data = NormalizedSubset(dataset, val_indices, stats["mean"], stats["scale"])
    train_loader = DataLoader(train_data, args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_data, args.batch_size, shuffle=False, num_workers=args.num_workers)

    encoder = StandardCNNContrastive(
        len(force_cols),
        args.cnn_base_channels,
        args.cnn_depth,
        args.cnn_kernel_size,
        args.cnn_dropout,
        args.out_embed_dim,
    )
    encoder_params = count_trainable_params(encoder)
    wandb.config.update({"encoder_params": encoder_params}, allow_val_change=True)
    if args.method == "masked_unet":
        state = MaskedUNet(encoder, args.context_points, len(force_cols))
    elif args.method == "vae":
        state = VAEModel(encoder, args.context_points, len(force_cols), args.latent_dim)
    elif args.method == "jepa":
        state = JEPAState(encoder)
    else:
        state = VICRegState(encoder, args.vicreg_projection_dim)
    state = state.to(device)
    optimizer = torch.optim.AdamW(state.parameters(), lr=args.peak_lr, weight_decay=args.weight_decay)
    scheduler = patchtst.WarmupCosineScheduler(optimizer, args.warmup_epochs, args.epochs, args.min_lr)

    def compute_loss(force, training):
        if args.method == "masked_unet":
            corrupted, mask = masked_view(force, args.mask_ratio)
            reconstruction, _ = state(corrupted)
            return F.mse_loss(reconstruction[mask], force[mask])
        if args.method == "vae":
            reconstruction, mu, logvar, _ = state(force)
            reconstruction_loss = F.mse_loss(reconstruction, force)
            kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).mean()
            return reconstruction_loss + args.vae_beta * kl
        if args.method == "jepa":
            context = augmented_view(force, args.mask_ratio)
            prediction = F.normalize(state.predictor(state.encoder(context)), dim=-1)
            with torch.no_grad():
                state.target_encoder.eval()
                target = state.target_encoder(force)
            loss = 2.0 - 2.0 * (prediction * target).sum(dim=-1).mean()
            return loss + 0.1 * variance_loss(prediction)
        first = state.projector(state.encoder(augmented_view(force, args.mask_ratio)))
        second = state.projector(state.encoder(augmented_view(force, args.mask_ratio)))
        return (
            25.0 * F.mse_loss(first, second)
            + 25.0 * (variance_loss(first) + variance_loss(second))
            + covariance_loss(first)
            + covariance_loss(second)
        )

    output_dir = Path(args.save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_stats_json(normalization, output_dir / f"{args.run_name}_normalization.json")
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    history = []

    def checkpoint(epoch, val_loss):
        return {
            "encoder": state.encoder.state_dict(),
            "ssl_state": state.state_dict(),
            "method": args.method,
            "encoder_config": {
                "nvars": len(force_cols),
                "base_channels": args.cnn_base_channels,
                "depth": args.cnn_depth,
                "kernel_size": args.cnn_kernel_size,
                "dropout": args.cnn_dropout,
                "out_embed_dim": args.out_embed_dim,
            },
            "normalization": normalization,
            "epoch": epoch,
            "valid_loss": val_loss,
            "train_args": vars(args),
        }

    for epoch in range(1, args.epochs + 1):
        state.train()
        if args.method == "jepa":
            state.target_encoder.eval()
        train_total = 0.0
        for force in train_loader:
            force = force.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = compute_loss(force, True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(state.parameters(), 1.0)
            optimizer.step()
            if args.method == "jepa":
                state.update_target(args.jepa_momentum)
            train_total += loss.item()
        state.eval()
        val_total = 0.0
        with torch.no_grad():
            for force in val_loader:
                val_total += compute_loss(force.to(device), False).item()
        train_loss = train_total / len(train_loader)
        val_loss = val_total / len(val_loader)
        history.append({"epoch": epoch, "train_loss": train_loss, "valid_loss": val_loss})
        wandb.log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_loss": val_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "best_valid_loss": min(best_loss, val_loss),
            },
            step=epoch,
        )
        print(f"Epoch {epoch}/{args.epochs} train={train_loss:.6f} val={val_loss:.6f} lr={optimizer.param_groups[0]['lr']:.6g}")
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_epoch = epoch
            stale = 0
            torch.save(checkpoint(epoch, val_loss), output_dir / f"{args.run_name}_best.pth")
        else:
            stale += 1
        scheduler.step()
        if stale >= args.early_stopping_patience:
            print(f"Early stopping at epoch {epoch}; best epoch={best_epoch}")
            break

    torch.save(checkpoint(epoch, val_loss), output_dir / f"{args.run_name}_last.pth")
    pd.DataFrame(history).to_csv(output_dir / f"{args.run_name}_history.csv", index=False)

    best = torch.load(output_dir / f"{args.run_name}_best.pth", map_location=device)
    state.encoder.load_state_dict(best["encoder"])
    state.encoder.eval()
    embeddings = []
    with torch.no_grad():
        for force in val_loader:
            embeddings.append(state.encoder(force.to(device)).cpu().numpy())
    dimensions = effective_dimension(np.concatenate(embeddings))
    dimensions.update({"method": args.method, "samples": len(val_data), "best_epoch": best_epoch, "best_valid_loss": best_loss})
    wandb.summary.update(dimensions)
    with (output_dir / f"{args.run_name}_effective_dimension.json").open("w") as file:
        json.dump(dimensions, file, indent=2)
    with (output_dir / f"{args.run_name}_effective_dimension.md").open("w") as file:
        file.write("# CNN Self-Supervised Effective Dimension\n\n")
        for key, value in dimensions.items():
            file.write(f"- {key}: {value}\n")
    print(dimensions)
    wandb.finish()


if __name__ == "__main__":
    main()
