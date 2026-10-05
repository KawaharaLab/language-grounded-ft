#!/usr/bin/env python3
"""Read-only object-comparison audit of actual successful model windows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from generate_real_model_surface_ft_examples import load_window


def crossing(frame, column, threshold, run=5):
    selected = frame.loc[frame.time_from_estimated_contact_s >= 0].reset_index(drop=True)
    flags = (selected[column].to_numpy() < threshold).astype(int)
    if len(flags) < run:
        return None
    positions = np.flatnonzero(np.convolve(flags, np.ones(run, int), "valid") == run)
    if not len(positions):
        return None
    p = int(positions[0])
    return dict(onset_s=float(selected.time_from_estimated_contact_s.iloc[p]),
                confirmation_s=float(selected.time_from_estimated_contact_s.iloc[p+run-1]))


def metrics(frame):
    result = {}
    pre = frame.loc[frame.time_from_estimated_contact_s < 0]
    for col in ("left_fz", "right_fz", "sum_fz_n"):
        baseline = float(pre[col].median()) if len(pre) >= 5 else None
        item = dict(precontact_median_n=baseline, first10_mean_n=float(frame[col].iloc[:10].mean()),
                    last10_mean_n=float(frame[col].iloc[-10:].mean()), min_n=float(frame[col].min()),
                    max_n=float(frame[col].max()), negative_5=crossing(frame,col,0.,5),
                    negative_1=crossing(frame,col,0.,1))
        if baseline is not None and baseline > .5:
            item["below_90pct_5"] = crossing(frame,col,.9*baseline)
            item["below_10pct_5"] = crossing(frame,col,.1*baseline)
            if item["below_90pct_5"] and item["negative_5"]:
                item["load_change_to_zero_s"] = item["negative_5"]["onset_s"] - item["below_90pct_5"]["onset_s"]
            if item["below_90pct_5"] and item["below_10pct_5"]:
                item["90_to_10pct_s"] = item["below_10pct_5"]["onset_s"] - item["below_90pct_5"]["onset_s"]
        result[col] = item
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    rows = []
    for root in args.roots:
        for path in sorted(root.glob("*/episode_*/model_evaluation.json")):
            ev = json.loads(path.read_text())
            row = dict(episode=str(path.parent), status=ev["status"], outcome=ev.get("outcome"))
            if ev["status"] != "success" or not ev.get("model_triggered"):
                row["excluded"] = "not successful model trial"
                rows.append(row)
                continue
            try:
                frame, info = load_window(path.parent, root.name)
                row.update(info=info, metrics=metrics(frame),
                           window_start_relative_contact_s=float(frame.time_from_estimated_contact_s.iloc[0]),
                           window_end_relative_contact_s=float(frame.time_from_estimated_contact_s.iloc[-1]))
                row["window_csv"] = "__".join(path.parent.parts[-3:]) + ".csv"
                frame.to_csv(args.out/row["window_csv"],index=False)
                print(json.dumps(dict(episode=row["episode"],metrics=row["metrics"])),flush=True)
            except (AssertionError, ValueError, KeyError) as exc:
                row["excluded"] = str(exc) or type(exc).__name__
                print(row["episode"], row["excluded"], file=sys.stderr, flush=True)
            rows.append(row)
    (args.out/"audit.json").write_text(json.dumps(rows,ensure_ascii=False,indent=2)+"\n")


if __name__ == "__main__":
    main()
