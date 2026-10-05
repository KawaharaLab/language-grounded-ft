from workspace_paths import resolve_recorded_path, workspace_path
import argparse
import copy
import json
import os
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import umap
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset, TensorDataset

from baseline_models import StandardCNNContrastive
from classification_taxonomy import TAXONOMIES, canonicalize_classification_labels
from force_normalization import load_stats, normalize_force_array
from token_fusion_encoder import build_token_fusion_encoder


TARGETS = ("label", "action", "weight", "interaction")


class LabeledForceDataset(Dataset):
    def __init__(
        self,
        data_dir,
        index_csv,
        seq_len,
        force_cols,
        zero_force_eps,
        targets=TARGETS,
        label_taxonomy="raw",
    ):
        index = canonicalize_classification_labels(
            pd.read_csv(Path(data_dir) / index_csv), label_taxonomy
        )
        required = {"csv_path", "start", *targets}
        missing = required - set(index.columns)
        if missing:
            raise ValueError(f"Missing columns in {index_csv}: {sorted(missing)}")

        cache = {}
        for csv_name in index["csv_path"].unique():
            csv_path = resolve_recorded_path(str(csv_name))
            if not csv_path.is_file():
                csv_path = Path(data_dir) / "csv" / str(csv_name)
            if csv_path.exists():
                cache[csv_name] = pd.read_csv(csv_path, usecols=force_cols).to_numpy(dtype=np.float32)

        self.force_segments = []
        rows = []
        skipped = {"missing": 0, "short": 0, "nonfinite": 0, "zero": 0}
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
            values = {target: getattr(row, target) for target in targets}
            self.force_segments.append(segment)
            rows.append(
                {
                    "csv_path": str(row.csv_path),
                    "start": int(row.start),
                    **{
                        target: (np.nan if pd.isna(value) else str(value))
                        for target, value in values.items()
                    },
                }
            )
        if not self.force_segments:
            raise RuntimeError(f"No usable samples in {data_dir}/{index_csv}")
        self.metadata = pd.DataFrame(rows)
        print(f"Loaded {len(self.force_segments)} samples from {index_csv}; skipped={skipped}")

    def __len__(self):
        return len(self.force_segments)

    def __getitem__(self, index):
        return torch.from_numpy(self.force_segments[index])


def build_encoder(checkpoint):
    config = checkpoint["encoder_config"]
    if config.get("encoder_type") == "patchtst_raw_token_fusion":
        encoder = build_token_fusion_encoder(config)
    else:
        encoder = StandardCNNContrastive(
            nvars=int(config["nvars"]),
            base_channels=int(config["base_channels"]),
            depth=int(config["depth"]),
            kernel_size=int(config["kernel_size"]),
            dropout=float(config["dropout"]),
            out_embed_dim=int(config["out_embed_dim"]),
        )
    encoder.load_state_dict(checkpoint["encoder"], strict=True)
    encoder.requires_grad_(False)
    encoder.eval()
    return encoder


def extract_embeddings(encoder, force_segments, stats, batch_size, num_workers, device):
    normalized = normalize_force_array(np.stack(force_segments), stats)
    loader = DataLoader(
        torch.from_numpy(normalized),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    embeddings = []
    with torch.no_grad():
        for force in loader:
            embeddings.append(encoder(force.to(device).float()).cpu())
    return torch.cat(embeddings)


def encode_labels(classes, labels):
    class_to_index = {value: index for index, value in enumerate(classes)}
    unknown = sorted(set(str(value) for value in labels) - set(classes))
    if unknown:
        raise ValueError(f"Evaluation labels absent from training data: {unknown}")
    encoded = torch.tensor([class_to_index[str(value)] for value in labels], dtype=torch.long)
    return encoded


def train_linear_head(
    embeddings,
    labels,
    train_indices,
    val_indices,
    num_classes,
    epochs,
    batch_size,
    learning_rate,
    weight_decay,
    class_weight_mode,
    patience,
    seed,
    device,
):
    torch.manual_seed(seed)
    head = nn.Linear(embeddings.shape[1], num_classes).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if class_weight_mode == "balanced":
        counts = torch.bincount(labels[train_indices], minlength=num_classes).float()
        if torch.any(counts == 0):
            raise ValueError(f"Linear-probe training split has an empty class: {counts.tolist()}")
        class_weights = len(train_indices) / (num_classes * counts)
        class_weights = class_weights.to(device)
    else:
        class_weights = None
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    train_loader = DataLoader(
        TensorDataset(embeddings[train_indices], labels[train_indices]),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(
        TensorDataset(embeddings[val_indices], labels[val_indices]),
        batch_size=batch_size,
        shuffle=False,
    )
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    stale = 0
    history = []
    for epoch in range(1, epochs + 1):
        head.train()
        train_total = 0.0
        for batch_embeddings, batch_labels in train_loader:
            batch_embeddings = batch_embeddings.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(head(batch_embeddings), batch_labels)
            loss.backward()
            optimizer.step()
            train_total += loss.item() * len(batch_labels)

        head.eval()
        val_total = 0.0
        val_correct = 0
        with torch.no_grad():
            for batch_embeddings, batch_labels in val_loader:
                batch_embeddings = batch_embeddings.to(device)
                batch_labels = batch_labels.to(device)
                logits = head(batch_embeddings)
                val_total += criterion(logits, batch_labels).item() * len(batch_labels)
                val_correct += (logits.argmax(dim=1) == batch_labels).sum().item()
        train_loss = train_total / len(train_indices)
        val_loss = val_total / len(val_indices)
        val_accuracy = val_correct / len(val_indices)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_loss": val_loss,
                "valid_accuracy": val_accuracy,
            }
        )
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(head.state_dict())
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"epoch={epoch} train_loss={train_loss:.5f} "
                f"val_loss={val_loss:.5f} val_accuracy={val_accuracy:.4f}"
            )
        if patience > 0 and stale >= patience:
            break
    head.load_state_dict(best_state)
    return head, pd.DataFrame(history), best_epoch, best_loss


def predict(head, embeddings, batch_size, device):
    loader = DataLoader(embeddings, batch_size=batch_size, shuffle=False)
    predictions = []
    probabilities = []
    head.eval()
    with torch.no_grad():
        for batch_embeddings in loader:
            logits = head(batch_embeddings.to(device))
            probabilities.append(torch.softmax(logits, dim=1).cpu())
            predictions.append(logits.argmax(dim=1).cpu())
    return torch.cat(predictions).numpy(), torch.cat(probabilities).numpy()


def calculate_metrics(true, predicted):
    precision, recall, f1, _ = precision_recall_fscore_support(
        true,
        predicted,
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(true, predicted)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
    }


def save_confusion_matrix(true, predicted, classes, path, target):
    matrix = confusion_matrix(true, predicted, labels=np.arange(len(classes)))
    figure_width = max(6, 0.8 * len(classes))
    fig, axis = plt.subplots(figsize=(figure_width, figure_width))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        ax=axis,
    )
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title(f"Linear probe: {target}")
    axis.tick_params(axis="x", rotation=45)
    axis.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_embedding_umap(embeddings, labels, classes, path, target):
    reducer = umap.UMAP(random_state=42)
    projected = reducer.fit_transform(np.asarray(embeddings))
    label_to_index = {label: index for index, label in enumerate(classes)}
    colors = [label_to_index[str(label)] for label in labels]
    fig, axis = plt.subplots(figsize=(7, 6))
    scatter = axis.scatter(
        projected[:, 0],
        projected[:, 1],
        c=colors,
        cmap="tab10",
        s=7,
        alpha=0.75,
    )
    handles, _ = scatter.legend_elements()
    axis.legend(handles, classes, bbox_to_anchor=(1.02, 1), loc="upper left")
    axis.set_title(f"Frozen encoder UMAP: {target}")
    axis.set_xticks([])
    axis.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train_dir", default=str(workspace_path("Genesis/data/train_04272026/")))
    parser.add_argument("--train_csv", default="train_04272026_thin_15pct.csv")
    parser.add_argument("--eval_dir", default=str(workspace_path("Genesis/data/eval_04272026/")))
    parser.add_argument("--eval_csv", default="eval_04272026_thin_15pct.csv")
    parser.add_argument("--label_taxonomy", choices=TAXONOMIES, default="raw")
    parser.add_argument(
        "--balanced_eval",
        action="store_true",
        help="Use independent action/weight/interaction-balanced evaluation CSVs and their taxonomy.",
    )
    parser.add_argument(
        "--balanced_eval_pattern",
        default="eval_04272026_{target}_balanced.csv",
    )
    parser.add_argument("--targets", nargs="+", choices=TARGETS, default=list(TARGETS))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--embedding_batch_size", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--class_weight", choices=["none", "balanced"], default="none")
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--zero_force_eps", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    method = checkpoint["method"]
    stats = load_stats(args.checkpoint)
    force_cols = list(stats["force_cols"])
    context_points = int(checkpoint["train_args"]["context_points"])

    train_dataset = LabeledForceDataset(
        args.train_dir,
        args.train_csv,
        context_points,
        force_cols,
        args.zero_force_eps,
        targets=tuple(args.targets),
        label_taxonomy=args.label_taxonomy,
    )
    if args.balanced_eval:
        if "label" in args.targets:
            raise ValueError("The balanced evaluation dataset has no label target")
        eval_datasets = {
            target: LabeledForceDataset(
                args.eval_dir,
                args.balanced_eval_pattern.format(target=target),
                context_points,
                force_cols,
                args.zero_force_eps,
                targets=(target,),
                label_taxonomy=args.label_taxonomy,
            )
            for target in args.targets
        }
        for target in args.targets:
            index_map = Path(args.eval_dir) / f"{target}_index_map.csv"
            if not index_map.is_file():
                raise FileNotFoundError(index_map)
            classes = set(pd.read_csv(index_map)[target].astype(str))
            values = train_dataset.metadata[target].astype("string")
            train_dataset.metadata[target] = values.where(values.isin(classes), np.nan)
    else:
        eval_dataset = LabeledForceDataset(
            args.eval_dir,
            args.eval_csv,
            context_points,
            force_cols,
            args.zero_force_eps,
            targets=tuple(args.targets),
            label_taxonomy=args.label_taxonomy,
        )
        eval_datasets = {target: eval_dataset for target in args.targets}
    val_size = max(1, round(len(train_dataset) * float(checkpoint["train_args"]["val_from_train_pct"])))
    permutation = torch.randperm(
        len(train_dataset),
        generator=torch.Generator().manual_seed(int(checkpoint["train_args"]["seed"])),
    )
    train_indices = permutation[:-val_size]
    val_indices = permutation[-val_size:]
    print(f"Reproduced SSL split: train={len(train_indices)} val={len(val_indices)}")

    encoder = build_encoder(checkpoint).to(device)
    train_embeddings = extract_embeddings(
        encoder,
        train_dataset.force_segments,
        stats,
        args.embedding_batch_size,
        args.num_workers,
        device,
    )
    eval_embeddings_by_target = {
        target: extract_embeddings(
            encoder,
            eval_datasets[target].force_segments,
            stats,
            args.embedding_batch_size,
            args.num_workers,
            device,
        )
        for target in args.targets
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_by_target = {}

    for target in args.targets:
        eval_dataset = eval_datasets[target]
        eval_embeddings = eval_embeddings_by_target[target]
        print(f"\n===== Linear probe: method={method} target={target} =====")
        train_valid = train_dataset.metadata[target].notna().to_numpy()
        eval_valid = eval_dataset.metadata[target].notna().to_numpy()
        target_train_indices = train_indices[torch.from_numpy(train_valid[train_indices.numpy()])]
        target_val_indices = val_indices[torch.from_numpy(train_valid[val_indices.numpy()])]
        eval_indices = np.flatnonzero(eval_valid)
        classes = sorted(set(train_dataset.metadata.loc[target_train_indices.numpy(), target].astype(str)))
        train_labels = torch.full((len(train_dataset),), -1, dtype=torch.long)
        valid_train_rows = np.flatnonzero(train_valid)
        train_labels[valid_train_rows] = encode_labels(
            classes,
            train_dataset.metadata.loc[valid_train_rows, target],
        )
        eval_labels = encode_labels(
            classes,
            eval_dataset.metadata.loc[eval_indices, target],
        )
        print(
            f"target samples: train={len(target_train_indices)} "
            f"val={len(target_val_indices)} eval={len(eval_indices)} classes={len(classes)}"
        )
        head, history, best_epoch, best_loss = train_linear_head(
            train_embeddings,
            train_labels,
            target_train_indices,
            target_val_indices,
            len(classes),
            args.epochs,
            args.batch_size,
            args.learning_rate,
            args.weight_decay,
            args.class_weight,
            args.patience,
            args.seed,
            device,
        )
        predicted, probabilities = predict(head, eval_embeddings[eval_indices], args.batch_size, device)
        metrics = calculate_metrics(eval_labels.numpy(), predicted)
        metrics.update(
            {
                "best_epoch": int(best_epoch),
                "best_valid_loss": float(best_loss),
                "num_classes": len(classes),
                "train_samples": len(target_train_indices),
                "validation_samples": len(target_val_indices),
                "evaluation_samples": len(eval_indices),
                "class_weight": args.class_weight,
            }
        )
        metrics_by_target[target] = metrics
        torch.save(
            {
                "head": head.state_dict(),
                "classes": classes,
                "target": target,
                "method": method,
                "encoder_checkpoint": os.path.abspath(args.checkpoint),
                "best_epoch": best_epoch,
                "best_valid_loss": best_loss,
            },
            output_dir / f"{target}_linear_head.pth",
        )
        history.to_csv(output_dir / f"{target}_history.csv", index=False)
        predictions = eval_dataset.metadata.loc[eval_indices, ["csv_path", "start", target]].rename(
            columns={target: "true"}
        ).reset_index(drop=True)
        predictions["pred"] = [classes[index] for index in predicted]
        for class_index, class_name in enumerate(classes):
            predictions[f"prob_{class_name}"] = probabilities[:, class_index]
        predictions.to_csv(output_dir / f"{target}_pred.csv", index=False)
        save_confusion_matrix(
            eval_labels.numpy(),
            predicted,
            classes,
            output_dir / f"{target}_confusion_matrix.png",
            target,
        )
        save_embedding_umap(
            eval_embeddings[eval_indices].numpy(),
            eval_dataset.metadata.loc[eval_indices, target].astype(str).tolist(),
            classes,
            output_dir / f"{target}_umap.png",
            target,
        )
        print(metrics)

    summary = {
        "method": method,
        "checkpoint": os.path.abspath(args.checkpoint),
        "train_samples": int(len(train_indices)),
        "validation_samples": int(len(val_indices)),
        "evaluation_samples": {
            target: int(len(eval_datasets[target])) for target in args.targets
        },
        "encoder_trainable_parameters": 0,
        "label_taxonomy": args.label_taxonomy,
        "class_weight": args.class_weight,
        "metrics": metrics_by_target,
    }
    with (output_dir / "linear_probe_summary.json").open("w") as file:
        json.dump(summary, file, indent=2)
    with (output_dir / "linear_probe_summary.md").open("w") as file:
        encoder_name = checkpoint["encoder_config"].get("encoder_type", "cnn")
        file.write(f"# SSL Linear Probe: {encoder_name} / {method}\n\n")
        file.write(f"- Encoder checkpoint: `{os.path.abspath(args.checkpoint)}`\n")
        file.write("- Encoder trainable parameters: 0\n")
        file.write(f"- Label taxonomy: `{args.label_taxonomy}`\n")
        file.write(f"- Linear-head class weighting: `{args.class_weight}`\n")
        eval_counts = ", ".join(
            f"{target}={len(eval_datasets[target])}" for target in args.targets
        )
        file.write(
            f"- Train/validation: {len(train_indices)}/{len(val_indices)}\n"
            f"- Evaluation: {eval_counts}\n\n"
        )
        file.write(
            "| Target | Accuracy | Macro Precision | Macro Recall | Macro F1 | "
            "Train | Validation | Evaluation | Best epoch |\n"
        )
        file.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for target in args.targets:
            metrics = metrics_by_target[target]
            file.write(
                f"| {target} | {metrics['accuracy']:.4f} | {metrics['macro_precision']:.4f} | "
                f"{metrics['macro_recall']:.4f} | {metrics['macro_f1']:.4f} | "
                f"{metrics['train_samples']} | {metrics['validation_samples']} | "
                f"{metrics['evaluation_samples']} | {metrics['best_epoch']} |\n"
            )
    print(output_dir / "linear_probe_summary.md")


if __name__ == "__main__":
    main()
