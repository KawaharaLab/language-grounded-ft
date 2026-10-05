import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def parse_metric_summary(path: str) -> pd.DataFrame:
    rows = []
    with open(path, encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if not stripped.startswith("|"):
                continue
            cells = [cell.strip() for cell in stripped.strip("|").split("|")]
            if not cells or cells[0] in {"Metric", "---"} or set(cells[0]) == {"-"}:
                continue
            if len(cells) != 6:
                continue
            try:
                values = [float(value) for value in cells[1:]]
            except ValueError:
                continue
            for target, value in zip(
                ("annotation", "label", "action", "weight", "interaction"), values
            ):
                rows.append({"metric": cells[0], "target": target, "value": value})
    if not rows:
        raise ValueError(f"No metric table found in {path}")
    return pd.DataFrame(rows)


def parse_seed_input(value: str):
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected SEED=PATH")
    seed_text, path = value.split("=", 1)
    try:
        seed = int(seed_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"Invalid seed: {seed_text}") from error
    return seed, path


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate PatchTST evaluation metrics across seeds"
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        type=parse_seed_input,
        help="SEED=/path/to/umap_metrics_summary.md; repeat once per seed",
    )
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    seed_inputs = sorted(args.input)
    seeds = [seed for seed, _ in seed_inputs]
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"Duplicate seeds: {seeds}")

    frames = []
    sources = {}
    for seed, path in seed_inputs:
        resolved = os.path.abspath(path)
        if not os.path.exists(resolved):
            raise FileNotFoundError(resolved)
        frame = parse_metric_summary(resolved)
        frame.insert(0, "seed", seed)
        frames.append(frame)
        sources[str(seed)] = resolved

    combined = pd.concat(frames, ignore_index=True)
    counts = combined.groupby(["metric", "target"])["seed"].nunique()
    if not (counts == len(seeds)).all():
        raise ValueError("Metric tables do not contain the same metric-target pairs")

    summary = (
        combined.groupby(["metric", "target"])["value"]
        .agg(
            mean="mean",
            variance=lambda values: np.var(values, ddof=1),
            std=lambda values: np.std(values, ddof=1),
            minimum="min",
            maximum="max",
        )
        .reset_index()
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_dir / "per_seed_metrics.csv", index=False)
    summary.to_csv(output_dir / "seed_metrics_summary.csv", index=False)

    report = {
        "seeds": seeds,
        "num_seeds": len(seeds),
        "variance": "sample variance (ddof=1)",
        "sources": sources,
        "metrics": summary.to_dict(orient="records"),
    }
    with open(output_dir / "seed_metrics_summary.json", "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    lines = [
        "# Contrastive Delta Token-Fusion PatchTST: Multi-seed Summary",
        "",
        f"Seeds: {', '.join(map(str, seeds))}",
        "",
        "Variance is the sample variance (ddof=1).",
        "",
        "| Metric | Target | Mean | Variance | Std | Min | Max |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.metric} | {row.target} | {row.mean:.6f} | "
            f"{row.variance:.8f} | {row.std:.6f} | {row.minimum:.6f} | {row.maximum:.6f} |"
        )
    (output_dir / "seed_metrics_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(output_dir / "seed_metrics_summary.md")


if __name__ == "__main__":
    main()
