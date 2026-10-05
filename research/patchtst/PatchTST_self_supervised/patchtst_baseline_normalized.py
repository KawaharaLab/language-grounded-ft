from workspace_paths import workspace_path
import argparse
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import patchtst_bert as base
from baseline_models import (
    FlatMLPContrastive,
    LSTMContrastive,
    ResidualTCNContrastive,
    StandardCNNContrastive,
    choose_cnn_width,
    choose_lstm_width,
    choose_mlp_width,
    choose_tcn_width,
    count_trainable_params,
)
from force_normalization import (
    compute_axis_stats,
    normalization_checkpoint_payload,
    save_stats_json,
)


class NormalizedSubset(Dataset):
    def __init__(self, dataset, indices, mean, scale):
        self.dataset = dataset
        self.indices = list(indices)
        self.mean = torch.as_tensor(mean, dtype=torch.float32).view(1, -1)
        self.scale = torch.as_tensor(scale, dtype=torch.float32).view(1, -1)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        item = self.dataset[self.indices[index]]
        force = item[0]
        force = (force.float() - self.mean) / self.scale
        return (force, *item[1:])


def parse_args(forced_model_arch=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_arch",
        choices=["cnn", "mlp", "tcn", "lstm"],
        default=forced_model_arch,
    )
    parser.add_argument("--data_dir", default=str(workspace_path("Genesis/data/train_04272026/")))
    parser.add_argument("--index_csv", default="train_04272026_thin_15pct.csv")
    parser.add_argument("--emb_dir", default="bert_emb")
    parser.add_argument("--context_points", type=int, default=80)
    parser.add_argument("--force_cols", default=",".join(base.DEFAULT_FORCE_COLS))
    parser.add_argument("--zero_force_eps", type=float, default=1e-8)
    parser.add_argument("--keep_zero_force", action="store_true")
    parser.add_argument("--val_from_train_pct", type=float, default=0.10)

    parser.add_argument("--cnn_base_channels", type=int, default=0)
    parser.add_argument("--cnn_depth", type=int, default=4)
    parser.add_argument("--cnn_kernel_size", type=int, default=5)
    parser.add_argument("--cnn_dropout", type=float, default=0.2)
    parser.add_argument("--mlp_hidden_dim", type=int, default=0)
    parser.add_argument("--mlp_depth", type=int, default=3)
    parser.add_argument("--mlp_dropout", type=float, default=0.2)
    parser.add_argument("--tcn_channels", type=int, default=0)
    parser.add_argument("--tcn_depth", type=int, default=5)
    parser.add_argument("--tcn_kernel_size", type=int, default=3)
    parser.add_argument("--tcn_dropout", type=float, default=0.2)
    parser.add_argument("--lstm_hidden_dim", type=int, default=0)
    parser.add_argument("--lstm_layers", type=int, default=2)
    parser.add_argument("--lstm_dropout", type=float, default=0.2)
    parser.add_argument("--out_embed_dim", type=int, default=768)

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
    parser.add_argument("--target_params", type=int, default=0)

    parser.add_argument("--epochs", type=int, default=350)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--peak_lr", type=float, default=2e-3)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--gradient_clipping", type=float, default=1.0)
    parser.add_argument("--early_stopping_patience", type=int, default=50)
    parser.add_argument("--early_stopping_min_delta", type=float, default=1e-4)
    parser.add_argument(
        "--loss_type",
        choices=[
            "infonce",
            "infonce_force_to_text",
            "infonce_multi_positive",
            "infonce_multi_positive_numerator",
            "sigmoid",
        ],
        default="infonce",
        help=(
            "Contrastive objective. infonce_multi_positive masks duplicate-annotation "
            "false negatives; infonce_multi_positive_numerator treats all duplicate "
            "annotations as positives."
        ),
    )
    parser.add_argument("--learnable_temp", action="store_true")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--run_name", default="")
    parser.add_argument("--save_dir", default="saved_models/baseline_normalized/")
    parser.add_argument("--wandb_api_key", default=os.getenv("WANDB_API_KEY", ""))

    parser.add_argument("--from_scratch", action="store_true")
    parser.add_argument("--pretrained_path", default="")
    args = parser.parse_args()
    if forced_model_arch is not None:
        args.model_arch = forced_model_arch
    if args.model_arch is None:
        parser.error("--model_arch is required")
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_data_dir(data_dir, index_csv):
    if os.path.isfile(os.path.join(data_dir, index_csv)):
        return data_dir
    legacy_dir = os.path.join(data_dir, "train_04272026")
    if os.path.isfile(os.path.join(legacy_dir, index_csv)):
        return legacy_dir
    raise FileNotFoundError(os.path.join(data_dir, index_csv))


def delta_patchtst_parameter_count(args, nvars):
    reference = base.PatchTSTContrastive(
        nvars=nvars,
        context_points=args.context_points,
        patch_len=args.patch_len,
        stride=args.stride,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_model=args.d_model,
        d_ff=args.d_ff,
        dropout=args.dropout,
        head_dropout=args.head_dropout,
        out_embed_dim=args.out_embed_dim,
        pool="mean",
        arch="delta_token_fusion",
        input_features="raw_delta",
        token_fusion_layers=args.token_fusion_layers,
        token_fusion_heads=args.token_fusion_heads,
        token_fusion_ff_mult=args.token_fusion_ff_mult,
    )
    return count_trainable_params(reference)


def build_model(args, nvars, target_params):
    if args.model_arch == "cnn":
        if args.cnn_base_channels > 0:
            width = args.cnn_base_channels
            model = StandardCNNContrastive(
                nvars,
                width,
                args.cnn_depth,
                args.cnn_kernel_size,
                args.cnn_dropout,
                args.out_embed_dim,
            )
            model_params = count_trainable_params(model)
        else:
            width, model_params = choose_cnn_width(
                nvars,
                args.cnn_depth,
                args.cnn_kernel_size,
                args.cnn_dropout,
                args.out_embed_dim,
                target_params,
            )
            model = StandardCNNContrastive(
                nvars,
                width,
                args.cnn_depth,
                args.cnn_kernel_size,
                args.cnn_dropout,
                args.out_embed_dim,
            )
        config = {
            "base_channels": width,
            "depth": args.cnn_depth,
            "kernel_size": args.cnn_kernel_size,
            "dropout": args.cnn_dropout,
            "out_embed_dim": args.out_embed_dim,
        }
    elif args.model_arch == "mlp":
        if args.mlp_hidden_dim > 0:
            width = args.mlp_hidden_dim
            model = FlatMLPContrastive(
                nvars,
                args.context_points,
                width,
                args.mlp_depth,
                args.mlp_dropout,
                args.out_embed_dim,
            )
            model_params = count_trainable_params(model)
        else:
            width, model_params = choose_mlp_width(
                nvars,
                args.context_points,
                args.mlp_depth,
                args.mlp_dropout,
                args.out_embed_dim,
                target_params,
            )
            model = FlatMLPContrastive(
                nvars,
                args.context_points,
                width,
                args.mlp_depth,
                args.mlp_dropout,
                args.out_embed_dim,
            )
        config = {
            "hidden_dim": width,
            "depth": args.mlp_depth,
            "dropout": args.mlp_dropout,
            "out_embed_dim": args.out_embed_dim,
            "seq_len": args.context_points,
        }
    elif args.model_arch == "tcn":
        if args.tcn_channels > 0:
            width = args.tcn_channels
        else:
            width, _ = choose_tcn_width(
                nvars,
                args.tcn_depth,
                args.tcn_kernel_size,
                args.tcn_dropout,
                args.out_embed_dim,
                target_params,
            )
        model = ResidualTCNContrastive(
            nvars=nvars,
            channels=width,
            depth=args.tcn_depth,
            kernel_size=args.tcn_kernel_size,
            dropout=args.tcn_dropout,
            out_embed_dim=args.out_embed_dim,
        )
        model_params = count_trainable_params(model)
        config = {
            "channels": width,
            "depth": args.tcn_depth,
            "kernel_size": args.tcn_kernel_size,
            "dropout": args.tcn_dropout,
            "out_embed_dim": args.out_embed_dim,
        }
    else:
        if args.lstm_hidden_dim > 0:
            width = args.lstm_hidden_dim
        else:
            width, _ = choose_lstm_width(
                nvars,
                args.lstm_layers,
                args.lstm_dropout,
                args.out_embed_dim,
                target_params,
            )
        model = LSTMContrastive(
            nvars=nvars,
            hidden_dim=width,
            num_layers=args.lstm_layers,
            dropout=args.lstm_dropout,
            out_embed_dim=args.out_embed_dim,
        )
        model_params = count_trainable_params(model)
        config = {
            "hidden_dim": width,
            "num_layers": args.lstm_layers,
            "dropout": args.lstm_dropout,
            "out_embed_dim": args.out_embed_dim,
        }
    return model, config, model_params


def main(forced_model_arch=None):
    args = parse_args(forced_model_arch)
    set_seed(args.seed)
    if not 0.0 < args.val_from_train_pct < 1.0:
        raise ValueError("--val_from_train_pct must be between 0 and 1")
    if args.pretrained_path:
        raise ValueError("The standard CNN/flat MLP baselines do not use pretrained PatchTST weights")

    args.data_dir = resolve_data_dir(args.data_dir, args.index_csv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if args.wandb_api_key:
        base.wandb.login(key=args.wandb_api_key)
    base.wandb.init(
        project=f"{args.model_arch}_bert_normalized",
        name=args.run_name or None,
        config=vars(args),
    )
    run_name = base.wandb.run.name
    os.makedirs(args.save_dir, exist_ok=True)

    force_cols = args.force_cols.split(",") if args.force_cols else base.DEFAULT_FORCE_COLS
    use_annotation_groups = args.loss_type in {
        "infonce_multi_positive",
        "infonce_multi_positive_numerator",
    }
    dataset = base.ForceContrastiveDataset(
        args.data_dir,
        args.index_csv,
        args.context_points,
        force_cols,
        mode="train",
        drop_zero_force=not args.keep_zero_force,
        zero_force_eps=args.zero_force_eps,
        return_group_ids=use_annotation_groups,
        emb_dir=args.emb_dir,
    )
    total_len = len(dataset)
    val_len = max(1, int(round(total_len * args.val_from_train_pct)))
    train_len = total_len - val_len
    if train_len < 1:
        raise ValueError("Training dataset is too small for the requested validation split")

    generator = torch.Generator().manual_seed(args.seed)
    permutation = torch.randperm(total_len, generator=generator).tolist()
    train_indices = permutation[:train_len]
    val_indices = permutation[train_len:]
    stats = compute_axis_stats(dataset.force_segments, train_indices, force_cols)
    normalization = normalization_checkpoint_payload(
        stats,
        metadata={
            "data_dir": args.data_dir,
            "index_csv": args.index_csv,
            "emb_dir": args.emb_dir,
            "seed": args.seed,
            "train_samples": train_len,
            "validation_samples": val_len,
            "context_points": args.context_points,
        },
    )
    stats_path = os.path.join(args.save_dir, f"{run_name}_normalization.json")
    save_stats_json(normalization, stats_path)
    print(f"Train/Val split: train={train_len} val={val_len}")
    print(f"Normalization statistics saved: {stats_path}")

    train_dataset = NormalizedSubset(dataset, train_indices, stats["mean"], stats["scale"])
    val_dataset = NormalizedSubset(dataset, val_indices, stats["mean"], stats["scale"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    target_params = args.target_params or delta_patchtst_parameter_count(args, len(force_cols))
    model, model_config, model_params = build_model(args, len(force_cols), target_params)
    model = model.to(device)
    parameter_error = (model_params - target_params) / target_params
    print(f"Reference delta PatchTST params: {target_params:,}")
    print(
        f"{args.model_arch.upper()} params: {model_params:,} "
        f"(difference={parameter_error:+.2%}, config={model_config})"
    )

    if args.loss_type == "sigmoid":
        criterion = base.SigmoidLoss().to(device)
    else:
        criterion = base.InfoNCELoss(
            learnable=args.learnable_temp,
            symmetric=args.loss_type != "infonce_force_to_text",
            mask_same_annotation=args.loss_type == "infonce_multi_positive",
            multi_positive_numerator=(
                args.loss_type == "infonce_multi_positive_numerator"
            ),
        ).to(device)
    parameter_groups = [{"params": model.parameters(), "lr": args.peak_lr}]
    criterion_params = [parameter for parameter in criterion.parameters() if parameter.requires_grad]
    if criterion_params:
        parameter_groups.append({"params": criterion_params, "lr": args.peak_lr})
    optimizer = torch.optim.Adam(
        parameter_groups,
        lr=args.peak_lr,
        weight_decay=args.weight_decay,
    )
    scheduler = base.WarmupCosineScheduler(
        optimizer,
        args.warmup_epochs,
        args.epochs,
        min_lr=args.min_lr,
    )

    def checkpoint(epoch, valid_loss):
        return {
            "model": model.state_dict(),
            "architecture_version": "standard_v1",
            "model_arch": args.model_arch,
            "model_config": model_config,
            "normalization": normalization,
            "target_patchtst_params": target_params,
            "model_params": model_params,
            "parameter_difference_ratio": parameter_error,
            "epoch": int(epoch),
            "valid_loss": float(valid_loss),
            "train_args": vars(args),
        }

    best_val = float("inf")
    epochs_without_improvement = 0
    best_path = os.path.join(args.save_dir, f"{run_name}_best.pth")
    last_path = os.path.join(args.save_dir, f"{run_name}_last.pth")
    last_epoch = 0
    last_valid_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        train_total = 0.0
        train_steps = 0
        current_lr = optimizer.param_groups[0]["lr"]
        for batch in train_loader:
            if use_annotation_groups:
                force, text_embedding, group_ids = batch
                group_ids = group_ids.to(device)
            else:
                force, text_embedding = batch
                group_ids = None
            force = force.to(device)
            text_embedding = text_embedding.to(device).float()
            text_embedding = text_embedding / (text_embedding.norm(dim=-1, keepdim=True) + 1e-8)
            optimizer.zero_grad(set_to_none=True)
            force_embedding = model(force)
            if isinstance(criterion, base.InfoNCELoss):
                loss = criterion(force_embedding, text_embedding, group_ids=group_ids)
            else:
                loss = criterion(force_embedding, text_embedding)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch={epoch + 1}")
            loss.backward()
            if args.gradient_clipping > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clipping)
            optimizer.step()
            train_total += loss.item()
            train_steps += 1
        if train_steps == 0:
            raise RuntimeError("No training batches were produced; reduce --batch_size")

        model.eval()
        validation_total = 0.0
        validation_steps = 0
        with torch.no_grad():
            for batch in val_loader:
                if use_annotation_groups:
                    force, text_embedding, group_ids = batch
                    group_ids = group_ids.to(device)
                else:
                    force, text_embedding = batch
                    group_ids = None
                force = force.to(device)
                text_embedding = text_embedding.to(device).float()
                text_embedding = text_embedding / (text_embedding.norm(dim=-1, keepdim=True) + 1e-8)
                force_embedding = model(force)
                if isinstance(criterion, base.InfoNCELoss):
                    loss = criterion(force_embedding, text_embedding, group_ids=group_ids)
                else:
                    loss = criterion(force_embedding, text_embedding)
                validation_total += loss.item()
                validation_steps += 1
        train_loss = train_total / train_steps
        valid_loss = validation_total / max(1, validation_steps)

        if valid_loss < best_val - args.early_stopping_min_delta:
            best_val = valid_loss
            epochs_without_improvement = 0
            torch.save(checkpoint(epoch + 1, valid_loss), best_path)
            print(f"  * New best val_loss {best_val:.4f} -> saved {best_path}")
        else:
            epochs_without_improvement += 1

        log_data = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "lr": current_lr,
            "model_params": model_params,
            "target_patchtst_params": target_params,
        }
        print(
            f"Epoch {epoch + 1}/{args.epochs} - train: {train_loss:.4f} "
            f"val: {valid_loss:.4f} lr: {current_lr:.6f}"
        )
        base.wandb.log(log_data)
        scheduler.step()
        last_epoch = epoch + 1
        last_valid_loss = valid_loss

        if (epoch + 1) % max(10, args.epochs // 10) == 0 or epoch + 1 == args.epochs:
            epoch_path = os.path.join(args.save_dir, f"{run_name}_epoch{epoch + 1}.pth")
            torch.save(checkpoint(epoch + 1, valid_loss), epoch_path)
            print("Saved", epoch_path)

        if (
            args.early_stopping_patience > 0
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            print(
                f"Early stopping at epoch {epoch + 1}: no validation-loss improvement "
                f"for {epochs_without_improvement} epochs."
            )
            break

    torch.save(checkpoint(last_epoch, last_valid_loss), last_path)
    print(f"Saved last checkpoint: {last_path} (epoch={last_epoch})")
    base.wandb.finish()


if __name__ == "__main__":
    main()
