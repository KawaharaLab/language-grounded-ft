#!/usr/bin/env python3
"""Diagnose the terrain-floor+wobble contrastive-training regression."""

from __future__ import annotations
from workspace_paths import workspace_path

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = workspace_path("PatchTST")
GENESIS = workspace_path("Genesis")
TRAIN_ROOT = GENESIS / "data/train_08142026"


def normalized(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    frame = frame.copy()
    for column in columns:
        if column in frame:
            frame[column] = (
                frame[column].fillna("").astype(str).str.strip().str.lower()
            )
    return frame


def family(path: str) -> str:
    value = str(path).lower()
    if "wobble" in value:
        return "wobble"
    if "step_" in value:
        return "step"
    if "inclined_" in value:
        return "slope"
    return "flat"


def prediction_dir(tag: str, model: str) -> Path:
    if tag == "old":
        name = f"terrain0814_common_eval_floor_normal_{model}_seed42_best"
    else:
        name = (
            "terrain0814_templates_release_merged_floor_wobble_"
            f"terrain_floor_wobble_{model}_seed42_best"
        )
    return ROOT / "data/terrain0814" / name / "pred"


def evaluate_predictions(eval_runtime: Path, output: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall_rows: list[dict] = []
    detail_rows: list[dict] = []
    class_rows: list[dict] = []
    for target in ("weight", "interaction"):
        metadata = pd.read_csv(
            eval_runtime / f"eval_08142026_terrain_{target}_balanced.csv",
            low_memory=False,
        )
        metadata["family"] = metadata["csv_path"].map(family)
        for tag in ("old", "new"):
            for model in ("tokenfusion", "cnn", "mlp"):
                prediction = pd.read_csv(prediction_dir(tag, model) / f"{target}_pred.csv")
                joined = prediction.merge(
                    metadata,
                    on=["csv_path", "start"],
                    how="left",
                    suffixes=("_prediction", ""),
                    validate="one_to_one",
                )
                if joined["action"].isna().any():
                    raise RuntimeError(f"Failed to join {tag}/{model}/{target}")
                joined["correct"] = joined["true"].eq(joined["pred"])
                overall_rows.append(
                    {
                        "target": target,
                        "model": model,
                        "training_set": tag,
                        "accuracy": joined["correct"].mean(),
                        "samples": len(joined),
                    }
                )
                for action, group in joined.groupby("action", sort=True):
                    detail_rows.append(
                        {
                            "target": target,
                            "model": model,
                            "training_set": tag,
                            "action": action,
                            "accuracy": group["correct"].mean(),
                            "samples": len(group),
                        }
                    )
                for label, group in joined.groupby("true", sort=True):
                    class_rows.append(
                        {
                            "target": target,
                            "model": model,
                            "training_set": tag,
                            "class": label,
                            "recall": group["correct"].mean(),
                            "samples": len(group),
                        }
                    )
    overall = pd.DataFrame(overall_rows)
    detail = pd.DataFrame(detail_rows)
    classes = pd.DataFrame(class_rows)
    overall.to_csv(output / "fair_overall_metrics.csv", index=False)
    detail.to_csv(output / "fair_per_action_metrics.csv", index=False)
    classes.to_csv(output / "fair_per_class_recall.csv", index=False)
    return overall, detail


def support_metrics(output: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for tag in ("old", "new"):
        for model in ("tokenfusion", "cnn", "mlp"):
            pred = pd.read_csv(
                ROOT
                / "data/wobbleplacesupport"
                / f"wobbleplacesupport_{tag}_{model}_seed42_best"
                / "pred/support_condition_pred.csv"
            )
            correct = pred["true"].eq(pred["pred"])
            rows.append(
                {
                    "model": model,
                    "training_set": tag,
                    "accuracy": correct.mean(),
                    "flat_recall": correct[pred["true"].eq("flat surface")].mean(),
                    "wobbling_recall": correct[
                        pred["true"].eq("wobbling support")
                    ].mean(),
                    "samples": len(pred),
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(output / "support_condition_metrics.csv", index=False)
    return result


def evaluation_overlap(old_runtime: Path, new_runtime: Path, output: Path) -> pd.DataFrame:
    rows = []
    for target in ("action", "weight", "interaction"):
        keys = {}
        for tag, runtime in (("old", old_runtime), ("new", new_runtime)):
            frame = pd.read_csv(
                runtime / f"eval_08142026_terrain_{target}_balanced.csv",
                usecols=["csv_path", "start"],
            )
            keys[tag] = set(zip(frame["csv_path"], frame["start"].astype(int)))
        common = keys["old"] & keys["new"]
        rows.append(
            {
                "target": target,
                "old_rows": len(keys["old"]),
                "new_rows": len(keys["new"]),
                "common_rows": len(common),
                "old_overlap_fraction": len(common) / len(keys["old"]),
                "new_overlap_fraction": len(common) / len(keys["new"]),
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(output / "evaluation_set_overlap.csv", index=False)
    return result


def training_audit(output: Path, contact_sample_size: int) -> tuple[pd.DataFrame, dict]:
    paths = {
        "old": TRAIN_ROOT / "train_08142026_terrain_floor_balanced.csv",
        "new": TRAIN_ROOT / "train_08142026_terrain_floor_wobble_balanced.csv",
    }
    frames = {
        tag: normalized(
            pd.read_csv(path, low_memory=False),
            ["action", "weight", "interaction", "annotation"],
        )
        for tag, path in paths.items()
    }
    summary_rows = []
    hold_rows = []
    for tag, frame in frames.items():
        frame["has_weight"] = frame["weight"].ne("")
        frame["family"] = frame["csv_path"].map(family)
        summary_rows.extend(
            [
                {"training_set": tag, "metric": "rows", "value": len(frame)},
                {
                    "training_set": tag,
                    "metric": "weight_labeled_rows",
                    "value": int(frame["has_weight"].sum()),
                },
                {
                    "training_set": tag,
                    "metric": "wobble_rows",
                    "value": int(frame["family"].eq("wobble").sum()),
                },
            ]
        )
        hold = frame[frame["action"].eq("hold")]
        for (interaction, has_weight, terrain), group in hold.groupby(
            ["interaction", "has_weight", "family"], sort=True
        ):
            hold_rows.append(
                {
                    "training_set": tag,
                    "interaction": interaction,
                    "has_weight": has_weight,
                    "family": terrain,
                    "rows": len(group),
                }
            )

    old_keys = set(zip(frames["old"]["csv_path"], frames["old"]["start"].astype(int)))
    new_keys = set(zip(frames["new"]["csv_path"], frames["new"]["start"].astype(int)))
    key_stats = {
        "old_rows": len(old_keys),
        "new_rows": len(new_keys),
        "common_rows": len(old_keys & new_keys),
        "new_only_rows": len(new_keys - old_keys),
        "new_only_wobble_rows": sum(
            "wobble" in str(path).lower() for path, _ in new_keys - old_keys
        ),
    }

    bad = frames["new"][
        frames["new"]["action"].eq("hold")
        & frames["new"]["interaction"].eq("slip")
        & ~frames["new"]["has_weight"]
    ].copy()
    key_stats.update(
        {
            "new_weightless_hold_slip_rows": len(bad),
            "new_weightless_hold_slip_wobble_rows": int(
                bad["family"].eq("wobble").sum()
            ),
            "new_weightless_hold_slip_accidental_path_rows": int(
                bad["csv_path"].str.lower().str.contains("accidental_drop").sum()
            ),
        }
    )

    sample = bad.sample(n=min(contact_sample_size, len(bad)), random_state=42)
    contact_rows = []
    for csv_path, group in sample.groupby("csv_path", sort=False):
        raw = pd.read_csv(csv_path)
        support_columns = [
            column
            for column in ("obj_plane", "obj_tile", "obj_target_tile")
            if column in raw
        ]
        for row in group.itertuples():
            segment = raw.iloc[int(row.start) : int(row.start) + 80]
            if support_columns:
                support = (
                    segment[support_columns]
                    .fillna(False)
                    .astype(bool)
                    .any(axis=1)
                )
            else:
                support = pd.Series(False, index=segment.index)
            left = (
                segment["obj_left_finger"].fillna(False).astype(bool)
                if "obj_left_finger" in segment
                else pd.Series(False, index=segment.index)
            )
            right = (
                segment["obj_right_finger"].fillna(False).astype(bool)
                if "obj_right_finger" in segment
                else pd.Series(False, index=segment.index)
            )
            contact_rows.append(
                {
                    "csv_path": csv_path,
                    "start": int(row.start),
                    "family": family(csv_path),
                    "support_any": bool(support.any()),
                    "support_all": bool(support.all()),
                    "support_fraction": float(support.mean()),
                    "bimanual_any": bool((left & right).any()),
                    "bimanual_fraction": float((left & right).mean()),
                }
            )
    contact = pd.DataFrame(contact_rows)
    contact.to_csv(output / "weightless_hold_slip_contact_sample.csv", index=False)
    key_stats.update(
        {
            "contact_sample_rows": len(contact),
            "contact_sample_support_any_fraction": contact["support_any"].mean(),
            "contact_sample_support_all_fraction": contact["support_all"].mean(),
            "contact_sample_mean_support_fraction": contact["support_fraction"].mean(),
        }
    )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output / "training_summary.csv", index=False)
    pd.DataFrame(hold_rows).to_csv(output / "hold_training_composition.csv", index=False)
    (output / "training_key_stats.json").write_text(json.dumps(key_stats, indent=2) + "\n")
    return summary, key_stats


def normalization_audit(output: Path, old_group: str, new_group: str) -> pd.DataFrame:
    files = {
        "old": ROOT
        / "saved_models/terrain_contrastive"
        / old_group
        / "terrain_floor_normal_tokenfusion_seed42"
        / "terrain_floor_normal_tokenfusion_seed42_normalization.json",
        "new": ROOT
        / "saved_models/terrain_contrastive"
        / new_group
        / "terrain_floor_wobble_tokenfusion_seed42"
        / "terrain_floor_wobble_tokenfusion_seed42_normalization.json",
    }
    data = {tag: json.loads(path.read_text()) for tag, path in files.items()}
    rows = []
    for index, axis in enumerate(data["old"]["force_cols"]):
        old_std = float(data["old"]["std"][index])
        new_std = float(data["new"]["std"][index])
        rows.append(
            {
                "axis": axis,
                "old_mean": float(data["old"]["mean"][index]),
                "new_mean": float(data["new"]["mean"][index]),
                "old_std": old_std,
                "new_std": new_std,
                "std_change_fraction": new_std / old_std - 1.0,
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(output / "normalization_comparison.csv", index=False)
    return result


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def write_report(
    output: Path,
    support: pd.DataFrame,
    overall: pd.DataFrame,
    per_action: pd.DataFrame,
    overlap: pd.DataFrame,
    training: pd.DataFrame,
    key_stats: dict,
    normalization: pd.DataFrame,
) -> None:
    support_pivot = support.pivot(index="model", columns="training_set", values="accuracy")
    overall_pivot = overall.pivot_table(
        index=["target", "model"], columns="training_set", values="accuracy"
    )
    overall_pivot["delta"] = overall_pivot["new"] - overall_pivot["old"]
    action_pivot = per_action.pivot_table(
        index=["target", "model", "action"],
        columns="training_set",
        values="accuracy",
    )
    action_pivot["delta"] = action_pivot["new"] - action_pivot["old"]
    training_pivot = training.pivot(index="metric", columns="training_set", values="value")
    weight_drop = 1.0 - (
        training_pivot.loc["weight_labeled_rows", "new"]
        / training_pivot.loc["weight_labeled_rows", "old"]
    )
    max_std_change = normalization["std_change_fraction"].abs().max()

    lines = [
        "# Wobble training regression diagnosis",
        "",
        "## Fixed-flat versus wobbling support",
        "",
        "The evaluation contains 754 atomic `place` windows (377 per support), matched for interaction and physical weight. Prototypes are fixed sentence embeddings: `Placing an object on a flat surface.` and `Placing an object on a wobbling support.`.",
        "",
        "| Model | Old terrain-floor | New terrain-floor+wobble | Gain |",
        "| --- | ---: | ---: | ---: |",
    ]
    for model in ("tokenfusion", "cnn", "mlp"):
        old = support_pivot.loc[model, "old"]
        new = support_pivot.loc[model, "new"]
        lines.append(f"| {model} | {pct(old)} | {pct(new)} | {100*(new-old):+.2f} pp |")

    lines.extend(
        [
            "",
            "## Fair old/new comparison on identical evaluation rows",
            "",
            "The original experiment summaries were not directly comparable because their independently sampled evaluation sets overlap by only 8.25% for weight and 23–24% for interaction. Old checkpoints were therefore re-evaluated on the new experiment's exact rows and prototypes.",
            "",
            "| Target | Model | Old | New | Delta |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for (target, model), row in overall_pivot.iterrows():
        lines.append(
            f"| {target} | {model} | {pct(row['old'])} | {pct(row['new'])} | {100*row['delta']:+.2f} pp |"
        )

    lines.extend(
        [
            "",
            "## Where the regression occurs",
            "",
            "Weight regression is concentrated in `hold`; `place` and `lift` are nearly unchanged. Interaction regression is likewise largest in `hold`, while `place` and `press` remain nearly unchanged.",
            "",
            "| Target | Model | Action | Old | New | Delta |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for target in ("weight", "interaction"):
        for model in ("tokenfusion", "cnn", "mlp"):
            for action in ("hold", "lift", "place", "press"):
                index = (target, model, action)
                if index not in action_pivot.index:
                    continue
                row = action_pivot.loc[index]
                lines.append(
                    f"| {target} | {model} | {action} | {pct(row['old'])} | {pct(row['new'])} | {100*row['delta']:+.2f} pp |"
                )

    bad = key_stats["new_weightless_hold_slip_rows"]
    wobble_bad = key_stats["new_weightless_hold_slip_wobble_rows"]
    lines.extend(
        [
            "",
            "## Dataset cause",
            "",
            f"- Weight-labeled training rows fell from {int(training_pivot.loc['weight_labeled_rows', 'old']):,} to {int(training_pivot.loc['weight_labeled_rows', 'new']):,} ({pct(weight_drop)} reduction).",
            "- Weight-labeled `hold` rows fell from 59,792 to 19,980. The new set instead contains 39,774 `hold + slip` rows with no weight annotation; the old balanced set contains zero such rows.",
            f"- A deterministic raw-contact audit of {key_stats['contact_sample_rows']:,} of those rows found support contact in {pct(key_stats['contact_sample_support_any_fraction'])} of windows, with mean support occupancy {pct(key_stats['contact_sample_mean_support_fraction'])}. They are post-contact/support-loaded windows, not ordinary airborne hold-slip examples.",
            f"- Only {wobble_bad:,}/{bad:,} ({pct(wobble_bad/bad)}) of the bad rows are wobble paths. The leak spans flat, slope, and step runs as well, so it is post-contact leakage rather than a wobble-only fault.",
            f"- The new balanced CSV retained only {key_stats['common_rows']:,}/{key_stats['old_rows']:,} old keys. It added {key_stats['new_only_rows']:,} rows, of which only {key_stats['new_only_wobble_rows']:,} are wobble; widespread non-wobble resampling confounded the intended single-pattern addition.",
            "- Interaction errors are specifically a slip failure: on the common evaluation set, token-fusion slip recall drops from 78.61% to 37.12% while accidental-drop and stable recall remain nearly unchanged.",
            f"- Normalization standard deviations changed by at most {pct(max_std_change)}. CNN weight accuracy remains above 90%, so normalization drift is a secondary contributor, not the main cause.",
            "",
            "The balancing MILP minimizes only the light-heavy and slip-stable count differences. It does not maximize the number of labeled weight rows, constrain weight/interaction within each action, preserve the previous selected rows, or penalize the unlabeled `other` weight stratum. Once surface constraints changed the feasible solution, it selected the large pool of supported `hold + slip` windows.",
            "",
            "## Recommended correction",
            "",
            "1. Exclude any `hold` candidate with sustained plane/tile contact. If supported downward motion is present, label it `press`; otherwise leave the post-place waiting window unannotated.",
            "2. For accidental-drop episodes, retain valid pre-event actions and the accidental-drop event window, but exclude post-drop aftermath after sustained support contact.",
            "3. Change train balancing to maximize labeled weight coverage and enforce light/heavy plus slip/stable balance within each atomic action, not only globally.",
            "4. Preserve the old balanced key set where still valid and make the minimum necessary replacements for wobble/surface constraints. This isolates the wobble addition and avoids replacing more than half the baseline set.",
            "5. Regenerate and retrain before judging the final weight/interaction trade-off.",
        ]
    )
    (output / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-group", default="terrain_floor_normal_20260821_011506")
    parser.add_argument("--new-group", default="terrain_floor_wobble_20260822_140617")
    parser.add_argument("--contact-sample-size", type=int, default=2000)
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    output = Path(args.output_dir) if args.output_dir else (
        ROOT / "results/terrain_contrastive" / args.new_group / "wobble_regression_diagnosis"
    )
    output.mkdir(parents=True, exist_ok=True)
    old_runtime = (
        ROOT
        / "results/terrain_contrastive"
        / args.old_group
        / "prepared_eval_templates_release_merged"
    )
    new_runtime = (
        ROOT
        / "results/terrain_contrastive"
        / args.new_group
        / "prepared_eval_templates_release_merged"
    )

    overlap = evaluation_overlap(old_runtime, new_runtime, output)
    overall, per_action = evaluate_predictions(new_runtime, output)
    support = support_metrics(output)
    training, key_stats = training_audit(output, args.contact_sample_size)
    normalization = normalization_audit(output, args.old_group, args.new_group)
    write_report(
        output,
        support,
        overall,
        per_action,
        overlap,
        training,
        key_stats,
        normalization,
    )
    print(f"Wrote diagnosis to {output}")


if __name__ == "__main__":
    main()
