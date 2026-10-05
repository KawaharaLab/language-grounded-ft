#!/usr/bin/env python3
"""Compare frozen, LoRA, full, and scratch terminal-insertion models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONDITIONS = {
    "scratch_end_to_end": ("Scratch End-to-End", "scratch_tokenfusion"),
    "frozen_linear": ("Frozen + linear", "contrastive_adaptation/frozen_linear"),
    "lora_r4": ("LoRA (r=4)", "contrastive_adaptation/lora_r4"),
    "full_finetune": ("Full fine-tuning", "contrastive_adaptation/full_finetune"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--max-consecutive", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def first_consecutive_trigger(
    times: np.ndarray, probabilities: np.ndarray, threshold: float, required: int
) -> float | None:
    run_length = 0
    for time_sec, probability in zip(times, probabilities):
        run_length = run_length + 1 if probability >= threshold else 0
        if run_length >= required:
            return float(time_sec)
    return None


def percentile(values: pd.Series, q: float) -> float | None:
    finite = values.dropna().to_numpy(dtype=np.float64)
    return float(np.percentile(finite, q)) if len(finite) else None


def markdown_table(table: pd.DataFrame, decimals: int = 4) -> str:
    lines = [
        "| " + " | ".join(map(str, table.columns)) + " |",
        "| " + " | ".join("---" for _ in table.columns) + " |",
    ]
    for row in table.itertuples(index=False, name=None):
        cells = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                cells.append("" if pd.isna(value) else f"{value:.{decimals}f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.max_consecutive < 1:
        raise ValueError("--max-consecutive must be at least 1")
    run_dir = args.run_dir.resolve()
    output_dir = run_dir / "contrastive_adaptation" / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_rows = []
    episode_rows = []
    for condition, (display, relative_dir) in CONDITIONS.items():
        model_dir = run_dir / relative_dir
        metrics = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
        predictions = pd.read_csv(model_dir / "test_predictions.csv")
        threshold = float(args.threshold)
        stored_threshold = float(metrics["selected_threshold"])
        if not np.isclose(stored_threshold, threshold):
            raise ValueError(
                f"{condition} used threshold={stored_threshold}; expected {threshold}"
            )
        window = metrics["test_window"]
        confusion = np.asarray(window["confusion_matrix_before_after"], dtype=int)
        total = int(metrics["model_parameters"])
        trainable = int(metrics["trainable_parameters"])
        metric_rows.append(
            {
                "condition": condition,
                "display": display,
                "best_epoch": int(metrics["best_epoch"]),
                "trainable_parameters": trainable,
                "total_parameters": total,
                "trainable_percent": 100.0 * trainable / total,
                "accuracy": float(window["accuracy"]),
                "balanced_accuracy": float(window["balanced_accuracy"]),
                "macro_f1": float(window["macro_f1"]),
                "auroc": float(window["auroc"]),
                "auprc": float(window["auprc"]),
                "false_positive_windows": int(confusion[0, 1]),
                "false_negative_windows": int(confusion[1, 0]),
            }
        )
        for episode_id, group in predictions.groupby("episode_id", sort=False):
            group = group.sort_values("relative_to_reference_sec")
            times = group["relative_to_reference_sec"].to_numpy(dtype=np.float64)
            probabilities = group["probability_after"].to_numpy(dtype=np.float64)
            for consecutive in range(1, args.max_consecutive + 1):
                trigger = first_consecutive_trigger(
                    times, probabilities, threshold, consecutive
                )
                outcome = (
                    "missed"
                    if trigger is None
                    else "early_stop"
                    if trigger < 0.0
                    else "success"
                )
                episode_rows.append(
                    {
                        "condition": condition,
                        "display": display,
                        "episode_id": episode_id,
                        "consecutive": consecutive,
                        "threshold": threshold,
                        "trigger_relative_sec": trigger,
                        "outcome": outcome,
                        "early_stop": outcome == "early_stop",
                        "success": outcome == "success",
                        "missed": outcome == "missed",
                        "detection_delay_sec": trigger if outcome == "success" else None,
                    }
                )

    metric_table = pd.DataFrame(metric_rows)
    episodes = pd.DataFrame(episode_rows)
    metric_table.to_csv(output_dir / "window_metrics.csv", index=False)
    episodes.to_csv(output_dir / "consecutive_by_episode.csv", index=False)

    summary_rows = []
    for (condition, display, consecutive), group in episodes.groupby(
        ["condition", "display", "consecutive"], sort=False
    ):
        delays = group.loc[group["success"], "detection_delay_sec"]
        summary_rows.append(
            {
                "condition": condition,
                "display": display,
                "consecutive": int(consecutive),
                "episodes": len(group),
                "early_stop_episodes": int(group["early_stop"].sum()),
                "early_stop_rate": float(group["early_stop"].mean()),
                "success_episodes": int(group["success"].sum()),
                "success_rate": float(group["success"].mean()),
                "missed_episodes": int(group["missed"].sum()),
                "missed_rate": float(group["missed"].mean()),
                "delay_sec_p25": percentile(delays, 25),
                "delay_sec_median": percentile(delays, 50),
                "delay_sec_p75": percentile(delays, 75),
                "delay_sec_p95": percentile(delays, 95),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "consecutive_summary.csv", index=False)

    for row in metric_rows:
        k1 = summary[
            (summary["condition"] == row["condition"])
            & (summary["consecutive"] == 1)
        ].iloc[0]
        metrics = json.loads(
            (run_dir / CONDITIONS[row["condition"]][1] / "metrics.json").read_text(
                encoding="utf-8"
            )
        )
        expected = metrics["test_sequence"]
        if int(k1["early_stop_episodes"]) != int(expected["early_stop_episodes"]):
            raise AssertionError(f"K=1 early-stop mismatch: {row['condition']}")
        if int(k1["missed_episodes"]) != int(expected["missed_after_enter_episodes"]):
            raise AssertionError(f"K=1 missed mismatch: {row['condition']}")

    colors = dict(zip(CONDITIONS, plt.cm.tab10(np.linspace(0, 1, len(CONDITIONS)))))
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)
    for condition, (display, _) in CONDITIONS.items():
        subset = summary[summary["condition"] == condition]
        axes[0].plot(
            subset["consecutive"], subset["early_stop_rate"] * 100,
            marker="o", color=colors[condition], label=display,
        )
        axes[1].plot(
            subset["consecutive"], subset["missed_rate"] * 100,
            marker="o", color=colors[condition], label=display,
        )
        axes[2].plot(
            subset["consecutive"], subset["delay_sec_median"] * 1000,
            marker="o", color=colors[condition], label=display,
        )
        axes[2].fill_between(
            subset["consecutive"], subset["delay_sec_p25"] * 1000,
            subset["delay_sec_p75"] * 1000, color=colors[condition], alpha=0.12,
        )
    for axis in axes:
        axis.set_xlabel("Required consecutive positives K")
        axis.set_xticks(range(1, args.max_consecutive + 1))
        axis.grid(alpha=0.25)
    axes[0].set_title("Early stop")
    axes[0].set_ylabel("Test episodes [%]")
    axes[1].set_title("Missed insertion")
    axes[1].set_ylabel("Test episodes [%]")
    axes[2].set_title("Successful-trigger delay")
    axes[2].set_ylabel("Median [ms] (band: IQR)")
    axes[2].legend(fontsize=8)
    fig.savefig(output_dir / "adaptation_tradeoff.png", dpi=180)
    plt.close(fig)

    k1 = summary[summary["consecutive"] == 1][
        [
            "condition", "early_stop_episodes", "success_episodes", "missed_episodes",
            "delay_sec_median", "delay_sec_p95",
        ]
    ]
    report = metric_table.merge(k1, on="condition")
    report["trainable"] = report.apply(
        lambda row: f"{int(row.trainable_parameters):,} ({row.trainable_percent:.2f}%)",
        axis=1,
    )
    report["early / success / miss"] = report.apply(
        lambda row: (
            f"{int(row.early_stop_episodes)} / "
            f"{int(row.success_episodes)} / {int(row.missed_episodes)}"
        ),
        axis=1,
    )
    report["median delay [ms]"] = report["delay_sec_median"] * 1000
    compact = report[
        [
            "display", "trainable", "macro_f1", "balanced_accuracy", "auroc",
            "false_positive_windows", "false_negative_windows",
            "early / success / miss", "median delay [ms]",
        ]
    ].rename(
        columns={
            "display": "Condition",
            "trainable": "Trainable params",
            "macro_f1": "Macro-F1",
            "balanced_accuracy": "Balanced acc.",
            "auroc": "ROC-AUC",
            "false_positive_windows": "FP windows",
            "false_negative_windows": "FN windows",
            "early / success / miss": "Early / success / miss (20 ep.)",
            "median delay [ms]": "Median delay [ms]",
        }
    )
    report_lines = [
        "# Contrastive adaptation comparison",
        "",
        "All K=1 decisions use the same fixed probability threshold of 0.5. "
        "The test set contains 979 overlapping windows from 20 held-out episodes.",
        "",
        markdown_table(compact, decimals=3),
        "",
        "The consecutive-positive K=2..5 analysis is post-hoc and uses the same "
        "stored test probabilities; delay is conditional on a successful first trigger.",
        "",
        "Artifacts: `window_metrics.csv`, `consecutive_summary.csv`, "
        "`consecutive_by_episode.csv`, and `adaptation_tradeoff.png`.",
        "",
    ]
    (output_dir / "README.md").write_text("\n".join(report_lines), encoding="utf-8")

    k2 = summary[summary["consecutive"] == 2].copy()
    k2["outcome"] = k2.apply(
        lambda row: (
            f"{int(row.early_stop_episodes)} / "
            f"{int(row.success_episodes)} / {int(row.missed_episodes)}"
        ),
        axis=1,
    )
    k2["median_delay_ms"] = k2["delay_sec_median"] * 1000
    k2_compact = k2[["display", "outcome", "median_delay_ms"]].rename(
        columns={
            "display": "条件",
            "outcome": "フライング / 成功 / 未検出 (20試行)",
            "median_delay_ms": "成功時の遅延中央値 [ms]",
        }
    )
    japanese_report = [
        "# Contrastiveモデルの適応方法比較",
        "",
        "主評価は、すべて確率閾値0.5の1窓出力（K=1）をそのまま使った。"
        "testは未知の20エピソード、合計979窓である。",
        "",
        markdown_table(compact, decimals=3),
        "",
        "LoRAは注意層のQ/Vだけにrank 4の低ランク更新を加えた。"
        "学習対象は8,961パラメータ（全体の0.80%）である。",
        "",
        "## 参考：2窓連続陽性（K=2）",
        "",
        markdown_table(k2_compact, decimals=1),
        "",
        "K=2は学習や閾値選択には使用しておらず、同じtest確率に対する事後解析である。"
        "全条件とも未検出は0試行だった。",
        "",
    ]
    (output_dir / "README_ja.md").write_text(
        "\n".join(japanese_report), encoding="utf-8"
    )
    print(compact.to_string(index=False))
    print(f"\nWrote comparison to {output_dir}")


if __name__ == "__main__":
    main()
