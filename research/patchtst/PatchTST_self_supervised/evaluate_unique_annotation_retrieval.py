#!/usr/bin/env python3
"""Evaluate force-to-text retrieval against a unique annotation gallery.

The gallery is built from an annotation-to-embedding index map.  Each held-out
force window is a query and its exact annotation string identifies the single
relevant gallery item.  Queries whose annotation is absent from the gallery
are excluded from the primary metric and counted as misses in the separately
reported coverage-adjusted metric.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from baseline_models import (
    FlatMLPContrastive,
    LSTMContrastive,
    ResidualTCNContrastive,
    StandardCNNContrastive,
)
from force_normalization import load_stats, normalize_force_array
from workspace_paths import resolve_recorded_path


DEFAULT_FORCE_COLS = [
    "left_fx", "left_fy", "left_fz", "left_tx", "left_ty", "left_tz",
    "right_fx", "right_fy", "right_fz", "right_tx", "right_ty", "right_tz",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="NAME=CHECKPOINT",
        help="Model name and checkpoint path; repeat to compare checkpoints.",
    )
    parser.add_argument("--eval_csv", type=Path, required=True)
    parser.add_argument("--eval_dir", type=Path, required=True)
    parser.add_argument("--gallery_map", type=Path, required=True)
    parser.add_argument("--gallery_emb_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--seq_len", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--top_k", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--zero_force_eps", type=float, default=1e-8)
    return parser.parse_args()


def parse_model_specs(specs: Iterable[str]) -> list[tuple[str, Path]]:
    parsed = []
    seen = set()
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Expected NAME=CHECKPOINT, got: {spec}")
        name, checkpoint = spec.split("=", 1)
        name = name.strip()
        if not name or name in seen:
            raise ValueError(f"Model names must be non-empty and unique: {name!r}")
        path = Path(checkpoint).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        parsed.append((name, path))
        seen.add(name)
    return parsed


def embedding_path(directory: Path, emb_index) -> Path:
    if isinstance(emb_index, (int, np.integer)):
        stem = str(int(emb_index))
    elif isinstance(emb_index, (float, np.floating)) and float(emb_index).is_integer():
        stem = str(int(emb_index))
    else:
        stem = str(emb_index)
    return directory / f"{stem}.pt"


def load_gallery(map_path: Path, embedding_dir: Path):
    gallery = pd.read_csv(map_path)
    required = {"annotation", "emb_index"}
    missing = required - set(gallery.columns)
    if missing:
        raise ValueError(f"Gallery map is missing columns: {sorted(missing)}")
    if gallery["annotation"].isna().any():
        raise ValueError("Gallery contains a missing annotation")
    duplicated = gallery["annotation"].duplicated(keep=False)
    if duplicated.any():
        values = gallery.loc[duplicated, "annotation"].tolist()
        raise ValueError(f"Gallery annotations are not unique: {values[:5]}")

    gallery = gallery.sort_values("emb_index", kind="stable").reset_index(drop=True)
    vectors = []
    for row in gallery.itertuples(index=False):
        path = embedding_path(embedding_dir, row.emb_index)
        if not path.is_file():
            raise FileNotFoundError(path)
        vector = torch.as_tensor(torch.load(path, map_location="cpu"), dtype=torch.float32).flatten()
        if not torch.isfinite(vector).all():
            raise ValueError(f"Non-finite gallery embedding: {path}")
        vectors.append(vector)
    dimensions = {int(vector.numel()) for vector in vectors}
    if len(dimensions) != 1:
        raise ValueError(f"Gallery embedding dimensions differ: {sorted(dimensions)}")
    matrix = torch.stack(vectors)
    matrix = matrix / matrix.norm(dim=1, keepdim=True).clamp_min(1e-8)
    gallery["gallery_index"] = np.arange(len(gallery), dtype=np.int64)
    return gallery, matrix


def resolve_force_csv(csv_path: str, eval_dir: Path) -> Path:
    path = resolve_recorded_path(csv_path)
    if path.is_absolute():
        return path
    direct = eval_dir / path
    if direct.is_file():
        return direct
    return eval_dir / "csv" / path


def load_query_windows(
    eval_csv: Path,
    eval_dir: Path,
    gallery_annotations: set[str],
    force_cols: list[str],
    seq_len: int,
    zero_force_eps: float,
):
    full = pd.read_csv(eval_csv)
    required = {"csv_path", "start", "annotation"}
    missing = required - set(full.columns)
    if missing:
        raise ValueError(f"Evaluation index is missing columns: {sorted(missing)}")

    annotation_present = full["annotation"].notna()
    gallery_present = annotation_present & full["annotation"].isin(gallery_annotations)
    eligible = full.loc[gallery_present]
    missing_gallery = (
        full.loc[annotation_present & ~gallery_present, "annotation"]
        .value_counts()
        .rename_axis("annotation")
        .reset_index(name="query_count")
    )

    segments = np.empty((len(eligible), seq_len, len(force_cols)), dtype=np.float32)
    kept_indices: list[int] = []
    skipped = {"missing_file_or_columns": 0, "short_window": 0, "nonfinite": 0, "zero_force": 0}
    cursor = 0
    grouped = eligible.groupby("csv_path", sort=False)
    total_files = grouped.ngroups
    for file_number, (csv_path, rows) in enumerate(grouped, start=1):
        path = resolve_force_csv(str(csv_path), eval_dir)
        try:
            raw = pd.read_csv(path, usecols=force_cols).to_numpy(dtype=np.float32, copy=True)
        except (FileNotFoundError, ValueError):
            skipped["missing_file_or_columns"] += len(rows)
            continue
        for row_index, start_value in zip(rows.index, rows["start"].to_numpy()):
            start = int(start_value)
            window = raw[start:start + seq_len]
            if len(window) != seq_len:
                skipped["short_window"] += 1
                continue
            if not np.isfinite(window).all():
                skipped["nonfinite"] += 1
                continue
            if np.max(np.abs(window)) <= zero_force_eps:
                skipped["zero_force"] += 1
                continue
            segments[cursor] = window
            kept_indices.append(int(row_index))
            cursor += 1
        if file_number % 250 == 0 or file_number == total_files:
            print(f"[load] files={file_number}/{total_files} usable_windows={cursor}", flush=True)

    if cursor == 0:
        raise RuntimeError("No usable gallery-covered query windows")
    segments = segments[:cursor]
    query_frame = full.loc[kept_indices].copy().reset_index().rename(columns={"index": "eval_row_index"})
    audit = {
        "input_rows": int(len(full)),
        "rows_with_annotation": int(annotation_present.sum()),
        "gallery_covered_rows_before_force_filter": int(gallery_present.sum()),
        "gallery_coverage_before_force_filter": float(gallery_present.sum() / len(full)),
        "usable_gallery_covered_rows": int(cursor),
        "usable_gallery_coverage": float(cursor / len(full)),
        "unique_eval_annotations": int(full.loc[annotation_present, "annotation"].nunique()),
        "covered_unique_eval_annotations": int(eligible["annotation"].nunique()),
        "skipped": skipped,
    }
    return segments, query_frame, missing_gallery, audit


def build_model(checkpoint: dict, nvars: int, seq_len: int):
    architecture = checkpoint.get("model_arch")
    config = checkpoint.get("model_config") or {}
    version = checkpoint.get("architecture_version")
    if version != "standard_v1":
        raise ValueError(f"Unsupported checkpoint architecture_version: {version!r}")
    if architecture == "cnn":
        model = StandardCNNContrastive(nvars=nvars, **config)
    elif architecture == "mlp":
        configured_seq_len = int(config.get("seq_len", seq_len))
        if configured_seq_len != seq_len:
            raise ValueError(f"MLP checkpoint expects seq_len={configured_seq_len}, requested {seq_len}")
        model = FlatMLPContrastive(nvars=nvars, **config)
    elif architecture == "tcn":
        model = ResidualTCNContrastive(nvars=nvars, **config)
    elif architecture == "lstm":
        model = LSTMContrastive(nvars=nvars, **config)
    else:
        raise ValueError(f"Unsupported model_arch: {architecture!r}")
    model.load_state_dict(checkpoint["model"], strict=True)
    return model, architecture


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rank_queries(
    model: torch.nn.Module,
    force_array: np.ndarray,
    correct_indices: np.ndarray,
    gallery_embeddings: torch.Tensor,
    batch_size: int,
    device: torch.device,
    max_k: int,
):
    count = len(force_array)
    ranks = np.empty(count, dtype=np.int16)
    top_indices = np.empty((count, max_k), dtype=np.int16)
    top_similarities = np.empty((count, max_k), dtype=np.float32)
    correct_similarities = np.empty(count, dtype=np.float32)
    gallery_device = gallery_embeddings.to(device)
    model = model.to(device).eval()

    with torch.inference_mode():
        for start in range(0, count, batch_size):
            end = min(start + batch_size, count)
            force = torch.from_numpy(force_array[start:end]).to(device, non_blocking=True)
            embeddings = model(force)
            similarities = embeddings @ gallery_device.T
            target = torch.from_numpy(correct_indices[start:end]).to(device=device, dtype=torch.long)
            correct_similarity = similarities.gather(1, target[:, None]).squeeze(1)
            rank = (similarities > correct_similarity[:, None]).sum(dim=1) + 1
            values, indices = similarities.topk(max_k, dim=1, largest=True, sorted=True)
            ranks[start:end] = rank.cpu().numpy().astype(np.int16)
            top_indices[start:end] = indices.cpu().numpy().astype(np.int16)
            top_similarities[start:end] = values.cpu().numpy().astype(np.float32)
            correct_similarities[start:end] = correct_similarity.cpu().numpy().astype(np.float32)
            if end % (batch_size * 25) == 0 or end == count:
                print(f"[infer] queries={end}/{count}", flush=True)
    return ranks, top_indices, top_similarities, correct_similarities


def metric_values(ranks: np.ndarray, top_k: list[int]) -> dict:
    result = {f"R@{k}": float(np.mean(ranks <= k)) for k in top_k}
    result.update({
        "MRR": float(np.mean(1.0 / ranks.astype(np.float64))),
        "mean_rank": float(np.mean(ranks)),
        "median_rank": float(np.median(ranks)),
    })
    return result


def grouped_metrics(frame: pd.DataFrame, column: str, top_k: list[int]) -> pd.DataFrame:
    rows = []
    values = frame[column].fillna("<missing>").astype(str)
    for value, indices in values.groupby(values, sort=True).groups.items():
        ranks = frame.loc[indices, "rank"].to_numpy()
        row = {column: value, "queries": int(len(ranks)), **metric_values(ranks, top_k)}
        rows.append(row)
    return pd.DataFrame(rows).sort_values([column]).reset_index(drop=True)


def evaluate_one(
    name: str,
    checkpoint_path: Path,
    raw_force: np.ndarray,
    query_frame: pd.DataFrame,
    gallery: pd.DataFrame,
    gallery_embeddings: torch.Tensor,
    audit: dict,
    output_root: Path,
    seq_len: int,
    batch_size: int,
    device: torch.device,
    top_k: list[int],
):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    stats = load_stats(checkpoint_path)
    force_cols = list(stats.get("force_cols", DEFAULT_FORCE_COLS))
    if raw_force.shape[-1] != len(force_cols):
        raise ValueError("Loaded force columns do not match checkpoint normalization")
    model, architecture = build_model(checkpoint, len(force_cols), seq_len)
    normalized_force = normalize_force_array(raw_force, stats)

    annotation_to_index = dict(zip(gallery["annotation"], gallery["gallery_index"]))
    correct_indices = query_frame["annotation"].map(annotation_to_index).to_numpy(dtype=np.int64)
    max_k = min(max(top_k), len(gallery))
    ranks, top_indices, top_similarities, correct_similarities = rank_queries(
        model, normalized_force, correct_indices, gallery_embeddings,
        batch_size, device, max_k,
    )

    predictions = query_frame.copy()
    predictions["correct_gallery_index"] = correct_indices
    predictions["rank"] = ranks
    predictions["correct_similarity"] = correct_similarities
    annotations = gallery["annotation"].to_numpy()
    predictions["top1_annotation"] = annotations[top_indices[:, 0]]
    predictions["top1_similarity"] = top_similarities[:, 0]
    predictions[f"top{max_k}_annotations"] = [
        " || ".join(annotations[indices].tolist()) for indices in top_indices
    ]

    micro = metric_values(ranks, top_k)
    per_annotation = grouped_metrics(predictions, "annotation", top_k)
    macro = {
        key: float(per_annotation[key].mean())
        for key in [*(f"R@{k}" for k in top_k), "MRR", "mean_rank", "median_rank"]
    }
    full_denominator = audit["input_rows"]
    coverage_adjusted = {
        f"R@{k}": float(np.sum(ranks <= k) / full_denominator) for k in top_k
    }
    coverage_adjusted["MRR"] = float(np.sum(1.0 / ranks.astype(np.float64)) / full_denominator)

    model_dir = output_root / name
    model_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(model_dir / "predictions.csv.gz", index=False, compression="gzip")
    per_annotation.to_csv(model_dir / "per_annotation_metrics.csv", index=False)
    for column in ("action", "interaction", "weight", "support_surface", "support_condition"):
        if column in predictions.columns:
            grouped_metrics(predictions, column, top_k).to_csv(
                model_dir / f"per_{column}_metrics.csv", index=False
            )

    metadata = {
        "model_name": name,
        "architecture": architecture,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_valid_loss": float(checkpoint.get("valid_loss", np.nan)),
        "model_config": checkpoint.get("model_config", {}),
        "gallery_size": int(len(gallery)),
        "query_count": int(len(predictions)),
        "micro": micro,
        "macro_by_annotation": macro,
        "coverage_adjusted": coverage_adjusted,
        "query_audit": audit,
    }
    with (model_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
    return metadata


def write_summary(results: list[dict], audit: dict, gallery_size: int, output_path: Path):
    top_k = [int(key.split("@", 1)[1]) for key in results[0]["micro"] if key.startswith("R@")]
    lines = [
        "# Unique annotation gallery retrieval",
        "",
        f"- Gallery: {gallery_size} unique training annotations",
        f"- Evaluation rows: {audit['input_rows']:,}",
        (
            f"- Gallery-covered usable queries: {audit['usable_gallery_covered_rows']:,} "
            f"({audit['usable_gallery_coverage']:.2%})"
        ),
        f"- Unique eval annotations: {audit['unique_eval_annotations']} total / "
        f"{audit['covered_unique_eval_annotations']} covered",
        "- Relevance: exact annotation-string match (one relevant gallery item per query)",
        "",
        "## Results",
        "",
        "| Model | Epoch | Valid loss | "
        + " | ".join(f"micro R@{k}" for k in top_k)
        + " | micro MRR | "
        + " | ".join(f"macro R@{k}" for k in top_k)
        + " | macro MRR |",
        "|---|---:|---:|" + "---:|" * (2 * len(top_k) + 2),
    ]
    for result in results:
        cells = [
            result["model_name"],
            str(result["checkpoint_epoch"]),
            f"{result['checkpoint_valid_loss']:.6f}",
            *(f"{result['micro'][f'R@{k}']:.4f}" for k in top_k),
            f"{result['micro']['MRR']:.4f}",
            *(f"{result['macro_by_annotation'][f'R@{k}']:.4f}" for k in top_k),
            f"{result['macro_by_annotation']['MRR']:.4f}",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend([
        "",
        "## Coverage-adjusted results",
        "",
        "These scores use every eval row as the denominator; annotations absent from the gallery are misses.",
        "",
        "| Model | " + " | ".join(f"R@{k}" for k in top_k) + " | MRR |",
        "|---|" + "---:|" * (len(top_k) + 1),
    ])
    for result in results:
        adjusted = result["coverage_adjusted"]
        cells = [result["model_name"], *(f"{adjusted[f'R@{k}']:.4f}" for k in top_k), f"{adjusted['MRR']:.4f}"]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    model_specs = parse_model_specs(args.model)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    top_k = sorted(set(args.top_k))
    if not top_k or top_k[0] < 1:
        raise ValueError("top_k values must be positive")

    gallery, gallery_embeddings = load_gallery(args.gallery_map, args.gallery_emb_dir)
    if top_k[-1] > len(gallery):
        raise ValueError("top_k cannot exceed gallery size")
    gallery.to_csv(args.output_dir / "gallery.csv", index=False)

    first_stats = load_stats(model_specs[0][1])
    force_cols = list(first_stats.get("force_cols", DEFAULT_FORCE_COLS))
    for name, checkpoint_path in model_specs[1:]:
        stats = load_stats(checkpoint_path)
        if list(stats.get("force_cols", DEFAULT_FORCE_COLS)) != force_cols:
            raise ValueError(f"Force columns differ for model {name}")

    raw_force, query_frame, missing_gallery, audit = load_query_windows(
        args.eval_csv.resolve(), args.eval_dir.resolve(), set(gallery["annotation"]),
        force_cols, args.seq_len, args.zero_force_eps,
    )
    missing_gallery.to_csv(args.output_dir / "annotations_missing_from_gallery.csv", index=False)
    with (args.output_dir / "query_audit.json").open("w", encoding="utf-8") as file:
        json.dump(audit, file, ensure_ascii=False, indent=2)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device} gallery={len(gallery)} queries={len(query_frame)}", flush=True)
    results = []
    for name, checkpoint_path in model_specs:
        print(f"[model] evaluating {name}: {checkpoint_path}", flush=True)
        results.append(evaluate_one(
            name, checkpoint_path, raw_force, query_frame, gallery, gallery_embeddings,
            audit, args.output_dir, args.seq_len, args.batch_size, device, top_k,
        ))
    write_summary(results, audit, len(gallery), args.output_dir / "summary.md")
    print(f"[done] {args.output_dir / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
