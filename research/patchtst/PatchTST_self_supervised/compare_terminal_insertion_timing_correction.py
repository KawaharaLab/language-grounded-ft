#!/usr/bin/env python3
"""Compare original, retimed-only, and retimed/retrained insertion results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from train_terminal_insertion_probe import sequence_metrics, window_metrics


MODEL_ORDER = [
    "scratch_tokenfusion",
    "scratch_mlp",
    "vae",
    "masked_reconstruction",
    "vicreg",
    "contrastive",
    "jepa",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-run", type=Path, required=True)
    parser.add_argument("--corrected-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def markdown_table(frame: pd.DataFrame) -> str:
    lines = [
        "| " + " | ".join(frame.columns) + " |",
        "| " + " | ".join("---" for _ in frame.columns) + " |",
    ]
    for values in frame.itertuples(index=False, name=None):
        cells = []
        for value in values:
            if isinstance(value, float):
                cells.append("" if pd.isna(value) else f"{value:.4f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    original = args.original_run.resolve()
    corrected = args.corrected_run.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    original_manifest = load_json(original / "dataset" / "dataset_manifest.json")
    corrected_manifest = load_json(corrected / "dataset" / "dataset_manifest.json")
    if original_manifest["splits"] != corrected_manifest["splits"]:
        raise ValueError("episode splits differ")

    with np.load(
        original / "dataset" / "terminal_insertion_windows.npz", allow_pickle=False
    ) as old_data, np.load(
        corrected / "dataset" / "terminal_insertion_windows.npz", allow_pickle=False
    ) as new_data:
        if not np.array_equal(old_data["force"], new_data["force"]):
            raise ValueError("force windows differ; timing-only comparison is invalid")
        corrected_labels = new_data["label"].astype(np.int64)
        corrected_episode_indices = new_data["episode_index"].astype(np.int64)
        corrected_relative = new_data["relative_to_reference_sec"].astype(np.float64)

    rows = []
    for model in MODEL_ORDER:
        old_metrics = load_json(original / model / "metrics.json")
        new_metrics = load_json(corrected / model / "metrics.json")
        predictions = pd.read_csv(original / model / "test_predictions.csv").sort_values(
            "window_index"
        )
        indices = predictions["window_index"].to_numpy(dtype=np.int64)
        probabilities = predictions["probability_after"].to_numpy(dtype=np.float64)
        threshold = float(old_metrics["selected_threshold"])
        old_model_corrected_window = window_metrics(
            corrected_labels[indices], probabilities, threshold
        )
        old_model_corrected_sequence = sequence_metrics(
            probabilities,
            corrected_episode_indices[indices],
            corrected_relative[indices],
            threshold,
        )
        rows.append(
            {
                "model": model,
                "original_macro_f1": old_metrics["test_window"]["macro_f1"],
                "old_model_corrected_eval_macro_f1": old_model_corrected_window[
                    "macro_f1"
                ],
                "corrected_retrained_macro_f1": new_metrics["test_window"][
                    "macro_f1"
                ],
                "original_early_stop_rate": old_metrics["test_sequence"][
                    "early_stop_rate"
                ],
                "old_model_corrected_eval_early_stop_rate": old_model_corrected_sequence[
                    "early_stop_rate"
                ],
                "corrected_retrained_early_stop_rate": new_metrics["test_sequence"][
                    "early_stop_rate"
                ],
                "original_delay_sec_median": old_metrics["test_sequence"][
                    "detection_delay_sec_median"
                ],
                "old_model_corrected_eval_delay_sec_median": old_model_corrected_sequence[
                    "detection_delay_sec_median"
                ],
                "corrected_retrained_delay_sec_median": new_metrics["test_sequence"][
                    "detection_delay_sec_median"
                ],
                "old_threshold": threshold,
                "corrected_retrained_threshold": new_metrics["selected_threshold"],
            }
        )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(output / "timing_correction_comparison.csv", index=False)

    x = np.arange(len(comparison))
    width = 0.25
    conditions = [
        ("original", "#8c8c8c"),
        ("old_model_corrected_eval", "#4c78a8"),
        ("corrected_retrained", "#e67e22"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    for offset_index, (prefix, color) in enumerate(conditions):
        offset = (offset_index - 1) * width
        axes[0].bar(
            x + offset,
            comparison[f"{prefix}_macro_f1"],
            width,
            label=prefix.replace("_", " "),
            color=color,
        )
        axes[1].bar(
            x + offset,
            comparison[f"{prefix}_early_stop_rate"] * 100,
            width,
            color=color,
        )
        axes[2].bar(
            x + offset,
            comparison[f"{prefix}_delay_sec_median"] * 1000,
            width,
            color=color,
        )
    labels = [name.replace("_", "\n") for name in comparison["model"]]
    for axis in axes:
        axis.set_xticks(x, labels, fontsize=8)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_title("Window-level Macro F1")
    axes[0].set_ylim(0.75, 1.01)
    axes[0].legend(fontsize=8, loc="lower left")
    axes[1].set_title("Episode early-stop rate")
    axes[1].set_ylabel("Rate [%]")
    axes[1].set_ylim(0, 105)
    axes[2].set_title("Safe detection delay median")
    axes[2].set_ylabel("Delay [ms]")
    fig.savefig(output / "timing_correction_comparison.png", dpi=200)
    plt.close(fig)

    display = comparison[
        [
            "model",
            "original_macro_f1",
            "old_model_corrected_eval_macro_f1",
            "corrected_retrained_macro_f1",
            "original_early_stop_rate",
            "old_model_corrected_eval_early_stop_rate",
            "corrected_retrained_early_stop_rate",
            "old_model_corrected_eval_delay_sec_median",
            "corrected_retrained_delay_sec_median",
        ]
    ]
    audit = load_json(corrected / "dataset" / "reaction_delay_relabel_audit.json")
    consecutive = pd.read_csv(
        corrected
        / "consecutive_trigger_analysis"
        / "consecutive_trigger_summary.csv"
    )
    k4 = consecutive[consecutive["consecutive"] == 4][
        [
            "model",
            "early_stop_episodes",
            "success_episodes",
            "missed_episodes",
            "delay_sec_median",
            "delay_sec_p95",
        ]
    ]
    report = [
        "# Enter反応遅延0.3秒補正の比較",
        "",
        "物理的な挿入基準を `recorded Enter - 0.3 s` とした。力覚窓、正規化、episode splitは変更せず、境界直前の375窓（各episode 3窓）のみbeforeからafterへ変更した。",
        "",
        "- `original`: 旧ラベルで学習・旧基準で評価",
        "- `old_model_corrected_eval`: 旧モデルと旧validation閾値を固定し、補正基準でtestだけ再採点",
        "- `corrected_retrained`: 補正ラベルで再学習し、補正validationで閾値選択、補正testで評価",
        "",
        "## K=1比較",
        "",
        markdown_table(display),
        "",
        "補正基準で旧モデルを再採点するとearly stopは概ね減るが、検出中央値は約0.29–0.43秒になる。補正ラベルで再学習すると窓F1と検出速度は回復する一方、陽性領域が真の境界より前へ広がり、episode early stopが増える。窓単位F1だけでは運用安全性を表せない。",
        "",
        "## 補正後モデルのK=4連続陽性",
        "",
        markdown_table(k4),
        "",
        "Scratch TokenFusionではK=4により20/20成功、early stop 0、miss 0、検出中央値約0.355秒となった。ただしtest 20 episodeで選んだ結果なので、Kの最終選択はvalidation episodeのみで行い、独立testで再確認すべきである。",
        "",
        "## 注意",
        "",
        "0.3秒を全episodeへ一律適用している。実際の反応時間の個人内変動があるため、このラベルは真の接触センサによる時刻ではなく近似である。今後は映像、電気的導通、位置・速度変化など独立信号で挿入時刻を校正するのが望ましい。",
        "",
        f"Relabel audit: {json.dumps(audit, ensure_ascii=False)}",
        "",
    ]
    (output / "README_ja.md").write_text("\n".join(report), encoding="utf-8")
    print(display.to_string(index=False))
    print(f"output={output}")


if __name__ == "__main__":
    main()
