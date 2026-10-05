import argparse
import os
import sys

import patchtst_eval as base
from force_normalization import load_stats, normalize_force_array


def parse_normalization_args(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--normalization_stats",
        default="",
        help="Optional normalization JSON or checkpoint; defaults to --model_path",
    )
    args, remaining = parser.parse_known_args(argv)
    return args, remaining


def find_argument(argv, name):
    for index, value in enumerate(argv):
        if value == name and index + 1 < len(argv):
            return argv[index + 1]
        prefix = name + "="
        if value.startswith(prefix):
            return value[len(prefix):]
    raise ValueError(f"Required argument not found: {name}")


def main():
    normalization_args, remaining = parse_normalization_args(sys.argv[1:])
    if "-h" in remaining or "--help" in remaining:
        print("Normalized evaluation additionally accepts:")
        print("  --normalization_stats PATH  normalization JSON/checkpoint (default: --model_path)")
        sys.argv = [sys.argv[0], *remaining]
        base.main()
        return
    model_path = find_argument(remaining, "--model_path")
    stats_path = normalization_args.normalization_stats or model_path
    stats = load_stats(stats_path)
    force_cols = list(stats["force_cols"])
    original_loader = base.load_force_segments_all_rows

    def normalized_loader(eval_dir, index_csv="eval_thin_17pct.csv", cols=None, **kwargs):
        if cols is not None and list(cols) != force_cols:
            raise ValueError(
                f"Evaluation force columns {list(cols)} do not match training columns {force_cols}"
            )
        force_array, eval_df, valid_row_idx, csv_paths, starts = original_loader(
            eval_dir,
            index_csv,
            cols=force_cols,
            **kwargs,
        )
        normalized = normalize_force_array(force_array, stats)
        return normalized, eval_df, valid_row_idx, csv_paths, starts

    base.load_force_segments_all_rows = normalized_loader
    sys.argv = [sys.argv[0], *remaining]
    print(
        f"[normalization] source={os.path.abspath(stats_path)} "
        f"axes={len(force_cols)} count={stats.get('count', 'unknown')}"
    )
    base.main()


if __name__ == "__main__":
    main()
