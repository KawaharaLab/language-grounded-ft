import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def compute_axis_stats(force_segments, indices, force_cols, std_epsilon=1e-6):
    indices = list(indices)
    if not indices:
        raise ValueError("Cannot compute normalization statistics from an empty training subset")

    nvars = len(force_cols)
    total = np.zeros(nvars, dtype=np.float64)
    total_sq = np.zeros(nvars, dtype=np.float64)
    minimum = np.full(nvars, np.inf, dtype=np.float64)
    maximum = np.full(nvars, -np.inf, dtype=np.float64)
    count = 0

    for index in indices:
        segment = np.asarray(force_segments[index], dtype=np.float64)
        if segment.ndim != 2 or segment.shape[1] != nvars:
            raise ValueError(f"Unexpected force segment shape: {segment.shape}")
        total += segment.sum(axis=0)
        total_sq += np.square(segment).sum(axis=0)
        minimum = np.minimum(minimum, segment.min(axis=0))
        maximum = np.maximum(maximum, segment.max(axis=0))
        count += segment.shape[0]

    mean = total / count
    variance = np.maximum(total_sq / count - np.square(mean), 0.0)
    std = np.sqrt(variance)
    scale = np.maximum(std, std_epsilon)
    return {
        "force_cols": list(force_cols),
        "count": int(count),
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "scale": scale.astype(np.float32),
        "min": minimum.astype(np.float32),
        "max": maximum.astype(np.float32),
        "std_epsilon": float(std_epsilon),
    }


def normalization_checkpoint_payload(stats, metadata=None):
    payload = {
        "force_cols": list(stats["force_cols"]),
        "count": int(stats["count"]),
        "mean": torch.as_tensor(stats["mean"], dtype=torch.float32).cpu(),
        "std": torch.as_tensor(stats["std"], dtype=torch.float32).cpu(),
        "scale": torch.as_tensor(stats["scale"], dtype=torch.float32).cpu(),
        "min": torch.as_tensor(stats["min"], dtype=torch.float32).cpu(),
        "max": torch.as_tensor(stats["max"], dtype=torch.float32).cpu(),
        "std_epsilon": float(stats["std_epsilon"]),
    }
    if metadata:
        payload.update(metadata)
    return payload


def stats_to_jsonable(stats):
    result = {}
    for key, value in stats.items():
        if isinstance(value, torch.Tensor):
            result[key] = value.detach().cpu().tolist()
        elif isinstance(value, np.ndarray):
            result[key] = value.tolist()
        elif isinstance(value, np.generic):
            result[key] = value.item()
        else:
            result[key] = value
    return result


def save_stats_json(stats, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(stats_to_jsonable(stats), file, ensure_ascii=False, indent=2)


def load_stats(source_path):
    source_path = Path(source_path)
    if source_path.suffix.lower() == ".json":
        with source_path.open("r", encoding="utf-8") as file:
            stats = json.load(file)
    else:
        checkpoint = torch.load(source_path, map_location="cpu")
        stats = checkpoint.get("normalization") if isinstance(checkpoint, dict) else None
        if stats is None:
            raise ValueError(f"Checkpoint has no normalization statistics: {source_path}")

    required = {"force_cols", "mean", "std"}
    missing = required - set(stats)
    if missing:
        raise ValueError(f"Normalization statistics are missing keys: {sorted(missing)}")
    result = dict(stats)
    for key in ("mean", "std", "scale", "min", "max"):
        if key in result:
            value = result[key]
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            result[key] = np.asarray(value, dtype=np.float32)
    if "scale" not in result:
        epsilon = float(result.get("std_epsilon", 1e-6))
        result["scale"] = np.maximum(result["std"], epsilon)
    return result


def normalize_force_array(force_array, stats):
    force_array = np.asarray(force_array, dtype=np.float32)
    mean = np.asarray(stats["mean"], dtype=np.float32)
    scale = np.asarray(stats["scale"], dtype=np.float32)
    if force_array.shape[-1] != mean.shape[0]:
        raise ValueError(
            f"Force axis count ({force_array.shape[-1]}) does not match normalization statistics ({mean.shape[0]})"
        )
    return (force_array - mean.reshape(1, 1, -1)) / scale.reshape(1, 1, -1)


def save_stats_table_and_plot(stats, csv_path, plot_path, title):
    table = pd.DataFrame({
        "axis": stats["force_cols"],
        "min": np.asarray(stats["min"]),
        "max": np.asarray(stats["max"]),
        "mean": np.asarray(stats["mean"]),
        "std": np.asarray(stats["std"]),
    })
    csv_path = Path(csv_path)
    plot_path = Path(plot_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_path, index=False)

    x = np.arange(len(table))
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), sharex=True)
    for axis, column, color in zip(
        axes.flat,
        ("min", "max", "mean", "std"),
        ("tab:blue", "tab:red", "tab:green", "tab:orange"),
    ):
        axis.bar(x, table[column], color=color)
        axis.set_title(column)
        axis.grid(axis="y", alpha=0.25)
    for axis in axes[-1]:
        axis.set_xticks(x)
        axis.set_xticklabels(table["axis"], rotation=45, ha="right")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return table


def compute_boxplot_stats(force_segments, indices, force_cols, max_points=1_000_000, seed=42):
    indices = np.asarray(list(indices), dtype=np.int64)
    if indices.size == 0:
        raise ValueError("Cannot compute boxplot statistics from an empty training subset")
    points_per_segment = np.asarray(force_segments[int(indices[0])]).shape[0]
    max_segments = max(1, int(np.ceil(max_points / points_per_segment)))
    if indices.size > max_segments:
        rng = np.random.default_rng(seed)
        indices = rng.choice(indices, size=max_segments, replace=False)
    values = np.concatenate([np.asarray(force_segments[int(index)]) for index in indices], axis=0)
    if values.shape[0] > max_points:
        values = values[:max_points]

    q1, median, q3 = np.quantile(values, [0.25, 0.50, 0.75], axis=0)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    whisker_low = np.empty(values.shape[1], dtype=np.float64)
    whisker_high = np.empty(values.shape[1], dtype=np.float64)
    for axis_index in range(values.shape[1]):
        axis_values = values[:, axis_index]
        lower_values = axis_values[axis_values >= lower_bound[axis_index]]
        upper_values = axis_values[axis_values <= upper_bound[axis_index]]
        whisker_low[axis_index] = lower_values.min()
        whisker_high[axis_index] = upper_values.max()
    return pd.DataFrame({
        "axis": list(force_cols),
        "q1": q1,
        "median": median,
        "q3": q3,
        "whisker_low": whisker_low,
        "whisker_high": whisker_high,
        "sampled_points": int(values.shape[0]),
    })


def save_force_torque_boxplots(boxplot_table, force_plot_path, torque_plot_path, title_prefix):
    groups = {
        "Force": boxplot_table[boxplot_table["axis"].str.contains("_f")],
        "Torque": boxplot_table[boxplot_table["axis"].str.contains("_t")],
    }
    output_paths = {"Force": Path(force_plot_path), "Torque": Path(torque_plot_path)}
    colors = {"Force": "tab:blue", "Torque": "tab:orange"}
    for group_name, table in groups.items():
        if table.empty:
            raise ValueError(f"No {group_name.lower()} axes found in boxplot table")
        output_path = output_paths[group_name]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        box_stats = [
            {
                "label": row.axis,
                "q1": row.q1,
                "med": row.median,
                "q3": row.q3,
                "whislo": row.whisker_low,
                "whishi": row.whisker_high,
                "fliers": [],
            }
            for row in table.itertuples(index=False)
        ]
        fig, axis = plt.subplots(figsize=(10, 6))
        artists = axis.bxp(box_stats, showfliers=False, patch_artist=True)
        for box in artists["boxes"]:
            box.set_facecolor(colors[group_name])
            box.set_alpha(0.65)
        axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        axis.set_title(f"{title_prefix}: {group_name.lower()} axes")
        axis.set_ylabel(group_name)
        axis.grid(axis="y", alpha=0.25)
        axis.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
