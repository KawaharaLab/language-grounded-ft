from workspace_paths import workspace_path
import argparse
import json
from argparse import Namespace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import umap
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader

from cnn_ssl_linear_probe import LabeledForceDataset
from force_normalization import load_stats, normalize_force_array
from patchtst_supervised import build_supervised_classifier


TARGETS = ("action", "weight", "interaction")


def metrics(true, predicted):
    precision, recall, f1, _ = precision_recall_fscore_support(
        true, predicted, average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(true, predicted)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "samples": int(len(true)),
    }


def save_confusion(true, predicted, classes, path):
    matrix = confusion_matrix(true, predicted, labels=classes)
    fig, axis = plt.subplots(figsize=(max(6, len(classes)), max(6, len(classes))))
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
    axis.set_title(f"Supervised encoder UMAP: {target}")
    axis.set_xticks([])
    axis.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval_dir", default=str(workspace_path("Genesis/data/eval_04272026")))
    parser.add_argument("--eval_pattern", default="eval_04272026_{target}_balanced.csv")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--zero_force_eps", type=float, default=1e-8)
    parser.add_argument("--targets", nargs="+", choices=TARGETS, default=list(TARGETS))
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if checkpoint.get("task") != "multitask":
        raise ValueError("A supervised multitask checkpoint is required")
    model_args = Namespace(**checkpoint["model_config"])
    label_maps = checkpoint["label_maps"]
    class_counts = {target: len(values) for target, values in label_maps.items()}
    model = build_supervised_classifier("multitask", class_counts, model_args)
    model.load_state_dict(checkpoint["model"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    stats = load_stats(args.checkpoint)
    force_cols = list(stats["force_cols"])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_metrics = {}
    skipped_targets = {}

    for target in args.targets:
        eval_classes = pd.read_csv(
            Path(args.eval_dir) / f"{target}_index_map.csv"
        )[target].astype(str).tolist()
        missing_classes = [label for label in eval_classes if label not in label_maps[target]]
        if missing_classes:
            skipped_targets[target] = f"Checkpoint lacks classes: {missing_classes}"
            print(f"[warn] skipping {target}: {skipped_targets[target]}")
            continue
        dataset = LabeledForceDataset(
            args.eval_dir,
            args.eval_pattern.format(target=target),
            int(model_args.context_points),
            force_cols,
            args.zero_force_eps,
            targets=(target,),
        )
        force = normalize_force_array(np.stack(dataset.force_segments), stats)
        loader = DataLoader(
            torch.from_numpy(force),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        selected_indices = [label_maps[target][label] for label in eval_classes]
        predictions = []
        probabilities = []
        embeddings = []
        with torch.no_grad():
            for batch in loader:
                features = model.encoder(batch.to(device).float())
                logits = model.heads[target](model.head_dropout(features))[:, selected_indices]
                embeddings.append(features.cpu())
                probabilities.append(torch.softmax(logits, dim=1).cpu())
                predictions.extend(logits.argmax(dim=1).cpu().tolist())
        embeddings = torch.cat(embeddings).numpy()
        probabilities = torch.cat(probabilities).numpy()
        predicted_labels = [eval_classes[index] for index in predictions]
        true_labels = dataset.metadata[target].astype(str).tolist()
        all_metrics[target] = metrics(true_labels, predicted_labels)

        frame = dataset.metadata[["csv_path", "start", target]].rename(columns={target: "true"})
        frame["pred"] = predicted_labels
        for index, class_name in enumerate(eval_classes):
            frame[f"prob_{class_name}"] = probabilities[:, index]
        frame.to_csv(output_dir / f"{target}_pred.csv", index=False)
        save_confusion(
            true_labels,
            predicted_labels,
            eval_classes,
            output_dir / f"{target}_confusion_matrix.png",
        )
        save_embedding_umap(
            embeddings,
            true_labels,
            eval_classes,
            output_dir / f"{target}_umap.png",
            target,
        )

    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "metrics": all_metrics,
        "skipped_targets": skipped_targets,
    }
    with (output_dir / "balanced_eval_summary.json").open("w") as file:
        json.dump(summary, file, indent=2)
    with (output_dir / "balanced_eval_summary.md").open("w") as file:
        file.write("# Supervised Balanced Evaluation\n\n")
        file.write("| Target | Accuracy | Macro Precision | Macro Recall | Macro F1 | Samples |\n")
        file.write("| --- | ---: | ---: | ---: | ---: | ---: |\n")
        for target in args.targets:
            if target not in all_metrics:
                continue
            value = all_metrics[target]
            file.write(
                f"| {target} | {value['accuracy']:.4f} | {value['macro_precision']:.4f} | "
                f"{value['macro_recall']:.4f} | {value['macro_f1']:.4f} | {value['samples']} |\n"
            )
        if skipped_targets:
            file.write("\n")
            for target, reason in skipped_targets.items():
                file.write(f"- {target}: {reason}\n")
    print(output_dir / "balanced_eval_summary.md")


if __name__ == "__main__":
    main()
