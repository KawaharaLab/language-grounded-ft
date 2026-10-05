import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader, Dataset

import patchtst_bert as base
from force_normalization import (
    compute_axis_stats,
    normalization_checkpoint_payload,
    save_stats_json,
)


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=75,
        help="Stop after this many epochs without validation-loss improvement; <=0 disables",
    )
    parser.add_argument(
        "--early_stopping_min_delta",
        type=float,
        default=0.0,
        help="Minimum validation-loss decrease counted as an improvement",
    )
    normalized_args, remaining = parser.parse_known_args(sys.argv[1:])
    original_argv = sys.argv
    try:
        sys.argv = [sys.argv[0], *remaining]
        args = base.parse_args()
    finally:
        sys.argv = original_argv
    args.early_stopping_patience = normalized_args.early_stopping_patience
    args.early_stopping_min_delta = normalized_args.early_stopping_min_delta
    return args


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
        force = (item[0].float() - self.mean) / self.scale
        return (force, *item[1:])


def build_model(args, nvars, device):
    if args.arch.startswith("itransformer"):
        return base.ITransformerContrastive(
            nvars=nvars,
            context_points=args.context_points,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            d_model=args.d_model,
            d_ff=args.d_ff,
            dropout=args.dropout,
            head_dropout=args.head_dropout,
            out_embed_dim=args.out_embed_dim,
            arch=args.arch,
            input_features=args.input_features,
            conv_stem_kernel=args.conv_stem_kernel,
            pool=args.itransformer_pool,
            use_metadata=args.use_channel_embedding,
        ).to(device)
    return base.PatchTSTContrastive(
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
        pool=args.pool,
        arch=args.arch,
        input_features=args.input_features,
        conv_stem_kernel=args.conv_stem_kernel,
        token_fusion_layers=args.token_fusion_layers,
        token_fusion_heads=args.token_fusion_heads,
        token_fusion_ff_mult=args.token_fusion_ff_mult,
        use_channel_embedding=args.use_channel_embedding,
    ).to(device)


def main():
    args = parse_args()
    base.set_seed(args.seed)
    if not 0.0 < args.val_from_train_pct < 1.0:
        raise ValueError("--val_from_train_pct must be between 0 and 1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    project_name = "patchtst_scratch_bert_normalized" if args.from_scratch else "patchtst_bert_normalized"
    base.wandb.login(key=args.wandb_api_key)
    base.wandb.init(project=project_name, name=args.run_name or None, config=vars(args))
    run_name = base.wandb.run.name
    os.makedirs(args.save_dir, exist_ok=True)

    force_cols = args.force_cols.split(",") if args.force_cols else base.DEFAULT_FORCE_COLS
    use_annotation_groups = args.loss_type in {
        "infonce_multi_positive", "infonce_multi_positive_numerator"
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
    generator = torch.Generator().manual_seed(args.seed if args.seed is not None else 42)
    permutation = torch.randperm(total_len, generator=generator).tolist()
    train_indices = permutation[:train_len]
    val_indices = permutation[train_len:]

    stats = compute_axis_stats(dataset.force_segments, train_indices, force_cols)
    normalization = normalization_checkpoint_payload(
        stats,
        metadata={
            "data_dir": args.data_dir,
            "index_csv": args.index_csv,
            "seed": args.seed,
            "train_samples": train_len,
            "validation_samples": val_len,
            "context_points": args.context_points,
        },
    )
    stats_path = os.path.join(args.save_dir, f"{run_name}_normalization.json")
    save_stats_json(normalization, stats_path)
    print(f"Train/Val split from train: train={train_len} val={val_len} (pct={args.val_from_train_pct:.2f})")
    print(f"Normalization statistics saved: {stats_path}")

    train_dataset = NormalizedSubset(dataset, train_indices, stats["mean"], stats["scale"])
    val_dataset = NormalizedSubset(dataset, val_indices, stats["mean"], stats["scale"])
    loader = DataLoader(
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

    nvars = len(force_cols)
    model = build_model(args, nvars, device)
    is_itransformer = args.arch.startswith("itransformer")
    if not args.from_scratch and args.pretrained_path:
        if is_itransformer:
            print("[info] iTransformer variants are not compatible with PatchTST SSL weights; training from scratch.")
        else:
            base.load_backbone_weights(model, args.pretrained_path, device)
    else:
        print("[info] Training from scratch (no pretrained backbone loaded).")

    if args.loss_type == "sigmoid":
        criterion = base.SigmoidLoss().to(device)
        criterion_params = list(criterion.parameters())
    else:
        criterion = base.InfoNCELoss(
            learnable=args.learnable_temp,
            symmetric=args.loss_type != "infonce_force_to_text",
            mask_same_annotation=args.loss_type == "infonce_multi_positive",
            multi_positive_numerator=args.loss_type == "infonce_multi_positive_numerator",
        ).to(device)
        criterion_params = [parameter for parameter in criterion.parameters() if parameter.requires_grad]

    backbone_params = list(model.backbone.parameters())
    backbone_param_ids = {id(parameter) for parameter in backbone_params}
    head_params = [parameter for parameter in model.parameters() if id(parameter) not in backbone_param_ids]
    optimizer = torch.optim.Adam(
        [
            {"params": backbone_params, "lr": args.peak_lr},
            {"params": head_params, "lr": args.peak_lr},
            {"params": criterion_params, "lr": args.peak_lr},
        ],
        lr=args.peak_lr,
        weight_decay=args.weight_decay,
    )
    scheduler = base.WarmupCosineScheduler(
        optimizer,
        args.warmup_epochs,
        args.epochs,
        min_lr=args.min_lr,
    )

    if args.freeze_backbone_epochs > 0:
        for parameter in backbone_params:
            parameter.requires_grad = False
        print(f"Backbone frozen for first {args.freeze_backbone_epochs} epochs.")

    def checkpoint(epoch, valid_loss):
        return {
            "model": model.state_dict(),
            "normalization": normalization,
            "epoch": int(epoch),
            "valid_loss": float(valid_loss),
            "train_args": vars(args),
        }

    best_val = float("inf")
    best_path = os.path.join(args.save_dir, f"{run_name}_best.pth")
    last_path = os.path.join(args.save_dir, f"{run_name}_last.pth")
    epochs_without_improvement = 0
    last_epoch = 0
    last_valid_loss = float("inf")
    for epoch in range(args.epochs):
        model.train()
        if args.freeze_backbone_epochs and epoch == args.freeze_backbone_epochs:
            for parameter in backbone_params:
                parameter.requires_grad = True
            print("Backbone unfrozen.")

        current_lr = optimizer.param_groups[0]["lr"]
        total_loss = 0.0
        steps = 0
        for batch in loader:
            if use_annotation_groups:
                force_seq, text_embedding, group_ids = batch
                group_ids = group_ids.to(device)
            else:
                force_seq, text_embedding = batch
                group_ids = None
            force_seq = force_seq.to(device)
            text_embedding = text_embedding.to(device).float()
            text_embedding = text_embedding / (text_embedding.norm(dim=-1, keepdim=True) + 1e-8)
            optimizer.zero_grad(set_to_none=True)
            force_embedding = model(force_seq)
            if isinstance(criterion, base.InfoNCELoss):
                loss = criterion(force_embedding, text_embedding, group_ids=group_ids)
            else:
                loss = criterion(force_embedding, text_embedding)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch={epoch + 1}, step={steps + 1}")
            loss.backward()
            if args.gradient_clipping > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clipping)
            optimizer.step()
            total_loss += loss.item()
            steps += 1
        if steps == 0:
            raise RuntimeError("No training batches were produced; reduce --batch_size or disable drop_last")
        train_loss = total_loss / steps

        model.eval()
        validation_total = 0.0
        validation_steps = 0
        with torch.no_grad():
            for batch in val_loader:
                if use_annotation_groups:
                    force_seq, text_embedding, group_ids = batch
                    group_ids = group_ids.to(device)
                else:
                    force_seq, text_embedding = batch
                    group_ids = None
                force_seq = force_seq.to(device)
                text_embedding = text_embedding.to(device).float()
                text_embedding = text_embedding / (text_embedding.norm(dim=-1, keepdim=True) + 1e-8)
                force_embedding = model(force_seq)
                if isinstance(criterion, base.InfoNCELoss):
                    loss = criterion(force_embedding, text_embedding, group_ids=group_ids)
                else:
                    loss = criterion(force_embedding, text_embedding)
                validation_total += loss.item()
                validation_steps += 1
        valid_loss = validation_total / max(1, validation_steps)
        improved = valid_loss < best_val - args.early_stopping_min_delta
        if improved:
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
        }
        if isinstance(criterion, base.SigmoidLoss):
            log_data["t_prime"] = criterion.t_prime.item()
            log_data["bias"] = criterion.b.item()
        else:
            log_data["logit_scale"] = criterion.logit_scale_param.exp().item()
        print(
            f"Epoch {epoch + 1}/{args.epochs} - train: {train_loss:.4f} "
            f"val: {valid_loss:.4f} lr: {log_data['lr']:.6f}"
        )
        base.wandb.log(log_data)
        scheduler.step()
        last_epoch = epoch + 1
        last_valid_loss = valid_loss

        if (epoch + 1) % max(10, args.epochs // 10) == 0 or epoch + 1 == args.epochs:
            checkpoint_path = os.path.join(args.save_dir, f"{run_name}_epoch{epoch + 1}.pth")
            torch.save(checkpoint(epoch + 1, valid_loss), checkpoint_path)
            print("Saved", checkpoint_path)

        if (
            args.early_stopping_patience > 0
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            print(
                f"Early stopping at epoch {epoch + 1}: no validation-loss improvement "
                f"for {epochs_without_improvement} epochs (best={best_val:.4f})."
            )
            break

    torch.save(checkpoint(last_epoch, last_valid_loss), last_path)
    print(f"Saved last checkpoint: {last_path} (epoch={last_epoch})")

    base.wandb.finish()


if __name__ == "__main__":
    main()
