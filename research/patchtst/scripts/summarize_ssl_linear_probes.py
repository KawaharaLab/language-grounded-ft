import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    summaries = []
    for path in sorted(Path(args.root).glob("*/linear_probe_summary.json")):
        with path.open() as file:
            summaries.append(json.load(file))
    if not summaries:
        raise RuntimeError(f"No linear-probe summaries found under {args.root}")

    preferred_targets = ("action", "weight", "interaction")
    targets = tuple(
        target
        for target in preferred_targets
        if all(target in summary.get("metrics", {}) for summary in summaries)
    )
    if not targets:
        raise RuntimeError("Linear-probe summaries have no common classification targets")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as file:
        file.write("# Token-Fusion Self-Supervised Linear-Probe Comparison\n\n")
        file.write("All encoders are frozen; each target uses one independently trained linear layer.\n\n")
        file.write("| Method | Metric | " + " | ".join(target.title() for target in targets) + " |\n")
        file.write("| --- | --- |" + " ---: |" * len(targets) + "\n")
        for summary in summaries:
            for metric, label in (("accuracy", "Accuracy"), ("macro_f1", "Macro F1")):
                values = [summary["metrics"][target][metric] for target in targets]
                file.write(
                    f"| {summary['method']} | {label} | "
                    + " | ".join(f"{value:.4f}" for value in values)
                    + " |\n"
                )
    print(output)


if __name__ == "__main__":
    main()
