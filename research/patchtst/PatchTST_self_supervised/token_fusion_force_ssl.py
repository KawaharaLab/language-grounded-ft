from workspace_paths import workspace_path
import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb

import patchtst_bert as patchtst
from byol_force import BYOLState, cosine_ema_momentum
from cnn_force_ssl import (
    ForceWindowDataset,
    JEPAState,
    NormalizedSubset,
    VAEModel,
    VICRegState,
    augmented_view,
    covariance_loss,
    effective_dimension,
    masked_view,
    variance_loss,
)
from force_normalization import (
    compute_axis_stats,
    normalization_checkpoint_payload,
    save_stats_json,
)
from token_fusion_encoder import RawTokenFusionEncoder, encoder_config_from_args


class MaskedReconstructionState(nn.Module):
    def __init__(self, encoder, seq_len, nvars):
        super().__init__()
        self.encoder = encoder
        self.decoder = nn.Sequential(
            nn.Linear(encoder.output_dim, 512),
            nn.GELU(),
            nn.Linear(512, seq_len * nvars),
        )
        self.seq_len = seq_len
        self.nvars = nvars

    def forward(self, force):
        representation = self.encoder(force)
        reconstruction = self.decoder(representation).view(-1, self.seq_len, self.nvars)
        return reconstruction, representation


def parse_args():
    parser = argparse.ArgumentParser(
        description="Self-supervised normalized Raw Token-Fusion PatchTST"
    )
    parser.add_argument(
        "--method", choices=["masked_reconstruction", "jepa", "vae", "vicreg", "byol"], required=True
    )
    parser.add_argument("--data_dir", default=str(workspace_path("Genesis/data/train_04272026/")))
    parser.add_argument("--index_csv", default="train_04272026_thin_15pct.csv")
    parser.add_argument("--context_points", type=int, default=80)
    parser.add_argument("--force_cols", default=",".join(patchtst.DEFAULT_FORCE_COLS))
    parser.add_argument("--zero_force_eps", type=float, default=1e-8)
    parser.add_argument("--val_from_train_pct", type=float, default=0.10)
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
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--peak_lr", type=float, default=2e-3)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--mask_ratio", type=float, default=0.4)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--vae_beta", type=float, default=1e-3)
    parser.add_argument("--jepa_momentum", type=float, default=0.996)
    parser.add_argument("--byol_momentum", type=float, default=0.996)
    parser.add_argument("--byol_projection_dim", type=int, default=256)
    parser.add_argument("--byol_hidden_dim", type=int, default=4096)
    parser.add_argument("--normalization_reference", default="")
    parser.add_argument("--vicreg_projection_dim", type=int, default=128)
    parser.add_argument("--early_stopping_patience", type=int, default=75)
    parser.add_argument("--early_stopping_min_delta", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--wandb_project", default="patchtst_tokenfusion_ssl")
    parser.add_argument("--wandb_group", default="")
    parser.add_argument("--wandb_mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--wandb_api_key", default=os.getenv("WANDB_API_KEY", ""))
    return parser.parse_args()


def main():
    args = parse_args()
    if args.method == "byol":
        cosine_ema_momentum(args.byol_momentum, 0, 1)
        if args.batch_size < 2:
            raise ValueError("BYOL BatchNorm requires batch_size >= 2")
    public_args = {key: value for key, value in vars(args).items() if key != "wandb_api_key"}
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.wandb_api_key and args.wandb_mode != "disabled":
        wandb.login(key=args.wandb_api_key)
    wandb.init(
        project=args.wandb_project,
        name=args.run_name,
        group=args.wandb_group or None,
        mode=args.wandb_mode,
        config=public_args,
    )

    force_cols = [column.strip() for column in args.force_cols.split(",") if column.strip()]
    dataset = ForceWindowDataset(
        args.data_dir,
        args.index_csv,
        args.context_points,
        force_cols,
        args.zero_force_eps,
    )
    val_len = max(1, round(len(dataset) * args.val_from_train_pct))
    permutation = torch.randperm(
        len(dataset), generator=torch.Generator().manual_seed(args.seed)
    ).tolist()
    train_indices = permutation[:-val_len]
    val_indices = permutation[-val_len:]
    stats = compute_axis_stats(dataset.force_segments, train_indices, force_cols)
    normalization = normalization_checkpoint_payload(
        stats,
        {
            "data_dir": args.data_dir,
            "index_csv": args.index_csv,
            "seed": args.seed,
            "train_samples": len(train_indices),
            "validation_samples": len(val_indices),
            "context_points": args.context_points,
        },
    )
    if args.normalization_reference:
        reference = json.loads(Path(args.normalization_reference).read_text())
        for key in ("force_cols", "count", "seed", "train_samples", "validation_samples", "context_points"):
            if normalization[key] != reference[key]:
                raise ValueError(f"Normalization reference mismatch: {key}")
        for key in ("mean", "std", "scale", "min", "max"):
            np.testing.assert_allclose(normalization[key], reference[key], rtol=1e-6, atol=1e-8, err_msg=key)
        print("Computed train-only normalization matches the reference SSL split and statistics", flush=True)
    train_data = NormalizedSubset(dataset, train_indices, stats["mean"], stats["scale"])
    val_data = NormalizedSubset(dataset, val_indices, stats["mean"], stats["scale"])
    train_loader = DataLoader(
        train_data,
        args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_data,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    encoder_config = encoder_config_from_args(args, len(force_cols))
    encoder = RawTokenFusionEncoder(
        **{key: value for key, value in encoder_config.items() if key != "encoder_type"}
    )
    if args.method == "masked_reconstruction":
        state = MaskedReconstructionState(encoder, args.context_points, len(force_cols))
    elif args.method == "vae":
        state = VAEModel(encoder, args.context_points, len(force_cols), args.latent_dim)
    elif args.method == "jepa":
        state = JEPAState(encoder)
    elif args.method == "byol":
        state = BYOLState(encoder, args.byol_projection_dim, args.byol_hidden_dim)
    else:
        state = VICRegState(encoder, args.vicreg_projection_dim)
    state = state.to(device)
    encoder_params = sum(parameter.numel() for parameter in encoder.parameters())
    wandb.config.update({"encoder_params": encoder_params}, allow_val_change=True)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in state.parameters() if parameter.requires_grad],
        lr=args.peak_lr,
        weight_decay=args.weight_decay,
    )
    scheduler = patchtst.WarmupCosineScheduler(
        optimizer, args.warmup_epochs, args.epochs, args.min_lr
    )
    global_step = 0
    total_steps = args.epochs * len(train_loader)
    if args.method == "byol":
        if not len(train_loader):
            raise ValueError("BYOL training split is smaller than one full batch")
        wandb.config.update({
            "byol_trainable_params": sum(parameter.numel() for parameter in state.parameters() if parameter.requires_grad),
            "byol_loss": "sum of symmetric normalized squared distances; no variance penalty",
            "byol_target_batchnorm": "training-view batch statistics; own running buffers; eval-mode validation",
        }, allow_val_change=True)

    def compute_loss(force):
        if args.method == "byol":
            return state(augmented_view(force, args.mask_ratio), augmented_view(force, args.mask_ratio))
        if args.method == "masked_reconstruction":
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
                target = F.normalize(state.target_encoder(force), dim=-1)
            return 2.0 - 2.0 * (prediction * target).sum(dim=-1).mean() + 0.1 * variance_loss(prediction)
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

    def checkpoint(epoch, valid_loss):
        return {
            "encoder": state.encoder.state_dict(),
            "ssl_state": state.state_dict(),
            "method": args.method,
            "encoder_config": encoder_config,
            "normalization": normalization,
            "epoch": epoch,
            "valid_loss": valid_loss,
            "train_args": public_args,
            "global_step": global_step,
        }

    for epoch in range(1, args.epochs + 1):
        state.train()
        if args.method == "jepa":
            state.target_encoder.eval()
        train_total = 0.0
        train_diagnostics = {}
        for force in train_loader:
            force = force.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = compute_loss(force)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(state.parameters(), 1.0)
            optimizer.step()
            if args.method == "jepa":
                state.update_target(args.jepa_momentum)
            elif args.method == "byol":
                momentum = cosine_ema_momentum(args.byol_momentum, global_step, total_steps)
                state.update_target(momentum)
                for key, value in state.metrics.items():
                    train_diagnostics[key] = train_diagnostics.get(key, 0.0) + value
                if global_step == 0:
                    print(f"BYOL first optimizer step completed: loss={loss.item():.6f} momentum={momentum:.6f}", flush=True)
            global_step += 1
            train_total += loss.item()
        state.eval()
        val_total = 0.0
        val_diagnostics = {}
        fork_devices = [torch.cuda.current_device()] if device.type == "cuda" else []
        # Augmentation masks and VAE samples are stochastic.  Reusing the same
        # validation randomness at every epoch makes checkpoint selection
        # reflect model changes instead of a different validation corruption.
        with torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(args.seed + 10_000)
            if device.type == "cuda":
                torch.cuda.manual_seed(args.seed + 10_000)
            with torch.no_grad():
                for force in val_loader:
                    val_total += compute_loss(force.to(device, non_blocking=True)).item()
                    if args.method == "byol":
                        for key, value in state.metrics.items():
                            val_diagnostics[key] = val_diagnostics.get(key, 0.0) + value
        train_loss = train_total / len(train_loader)
        valid_loss = val_total / len(val_loader)
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        if args.method == "byol":
            epoch_metrics["byol_momentum"] = momentum
            for prefix, diagnostics, batches in (
                ("train", train_diagnostics, len(train_loader)),
                ("val", val_diagnostics, len(val_loader)),
            ):
                epoch_metrics.update({f"{prefix}/{key}": float(value / batches) for key, value in diagnostics.items()})
        history.append(epoch_metrics)
        pd.DataFrame(history).to_csv(output_dir / f"{args.run_name}_history.csv", index=False)
        print(
            f"epoch={epoch}/{args.epochs} train={train_loss:.6f} "
            f"val={valid_loss:.6f} lr={optimizer.param_groups[0]['lr']:.6g}"
        )
        wandb.log(epoch_metrics, step=epoch)
        if valid_loss < best_loss - args.early_stopping_min_delta:
            best_loss = valid_loss
            best_epoch = epoch
            stale = 0
            torch.save(checkpoint(epoch, valid_loss), output_dir / f"{args.run_name}_best.pth")
        else:
            stale += 1
        scheduler.step()
        if args.early_stopping_patience > 0 and stale >= args.early_stopping_patience:
            print(f"early stopping at epoch={epoch}; best={best_loss:.6f}@{best_epoch}")
            break

    torch.save(checkpoint(epoch, valid_loss), output_dir / f"{args.run_name}_last.pth")
    pd.DataFrame(history).to_csv(output_dir / f"{args.run_name}_history.csv", index=False)
    best = torch.load(output_dir / f"{args.run_name}_best.pth", map_location=device)
    state.encoder.load_state_dict(best["encoder"])
    state.encoder.eval()
    embeddings = []
    with torch.no_grad():
        for force in val_loader:
            embeddings.append(state.encoder(force.to(device)).cpu().numpy())
    dimensions = effective_dimension(np.concatenate(embeddings))
    dimensions.update(
        {
            "method": args.method,
            "samples": len(val_data),
            "best_epoch": best_epoch,
            "best_valid_loss": best_loss,
            "encoder_params": encoder_params,
        }
    )
    with (output_dir / f"{args.run_name}_effective_dimension.json").open("w") as file:
        json.dump(dimensions, file, indent=2)
    wandb.summary.update(dimensions)
    wandb.finish()


if __name__ == "__main__":
    main()
