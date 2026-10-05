#!/usr/bin/env python3
"""Plot manuscript insertion results as means and error bars, without bands or lines.

Run with ``$FT_PYTHON`` (or the workspace PatchTST virtualenv). Reuse the saved five-seed
aggregates and preserve the manuscript's six-model selection and styling.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = Path(os.environ.get("FT_WORKSPACE_ROOT", ROOT.parent)).expanduser().resolve()
PATCH = Path(os.environ.get("FT_PATCHTST_ROOT", WORKSPACE / "PatchTST")).expanduser().resolve()
SOURCE = PATCH / "PatchTST_self_supervised/summarize_terminal_insertion_pretrained_adaptation.py"
RUN = PATCH / "results/terminal_insertion_supervised/terminal_insertion_supervised_20260913_172852"
STEMS = ("pretrained_linear_vs_scratch_end_to_end", "pretrained_lora_vs_end_to_end")
MODELS = {"scratch", "contrastive", "masked_reconstruction", "vae", "vicreg", "byol"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plot(table: pd.DataFrame, stem: Path, style) -> None:
    lower = table.test_macro_f1_mean - table.test_macro_f1_std.fillna(0)
    y_min = max(0.0, np.floor((float(lower.min()) - 0.03) * 10) / 10)
    fig, axis = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    for model in style.MODEL_ORDER:
        group = table.loc[table.model.eq(model)].sort_values("episode_count")
        if group.empty:
            continue
        assert tuple(group.episode_count) == style.EPISODE_COUNTS
        x = group.episode_count.to_numpy(dtype=float)
        mean = 100.0 * group.test_macro_f1_mean.to_numpy(dtype=float)
        std = 100.0 * group.test_macro_f1_std.to_numpy(dtype=float)
        assert np.isfinite(mean).all() and np.isfinite(std).all()
        bars = axis.errorbar(
            x, mean, yerr=std, marker=style.MARKERS[model], markersize=7,
            linestyle="none", elinewidth=1.4, capsize=4,
            color=style.COLORS[model], label=style.MODEL_NAMES[model],
            zorder=6 if model == "scratch" else 2,
        )
        points, _, bar_collections = bars.lines
        assert points.get_linestyle() == "None"
        np.testing.assert_array_equal(points.get_ydata(), mean)
        expected = np.stack((np.column_stack((x, mean - std)),
                             np.column_stack((x, mean + std))), axis=1)
        np.testing.assert_allclose(bar_collections[0].get_segments(), expected)
    assert not any(isinstance(item, PolyCollection) for item in axis.collections)
    axis.set_xscale("log")
    axis.set_xticks(style.EPISODE_COUNTS, [str(n) for n in style.EPISODE_COUNTS])
    axis.set_ylim(100.0 * y_min, 100.0)
    axis.set_xlabel("Number of labeled training episodes", fontsize=18)
    axis.set_ylabel("Test Macro-F1 (%)", fontsize=18)
    axis.tick_params(axis="both", which="major", labelsize=15)
    axis.grid(alpha=0.25, which="both")
    axis.legend(loc="lower right", frameon=True, ncol=2, fontsize=13)
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    spec = importlib.util.spec_from_file_location("insertion_plot_style", SOURCE)
    style = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(style)
    audit = Path(tempfile.mkdtemp(prefix="terminal_insertion_errorbars_", dir=ROOT / "fig"))
    before = audit / "before"
    before.mkdir()
    records = []
    for stem in STEMS:
        source_csv = RUN / f"{stem}_aggregate.csv"
        source_hash = sha256(source_csv)
        table = pd.read_csv(source_csv)
        selected = table.loc[table.model.isin(MODELS)].copy()
        assert set(selected.model) == MODELS and len(selected) == 42
        target = ROOT / "fig" / f"terminal_insertion_{stem}.pdf"
        shutil.copy2(target, before / target.name)
        generated = audit / target.name
        plot(selected, generated.with_suffix(""), style)
        text = subprocess.check_output(["pdftotext", str(generated), "-"], text=True)
        assert "supervised" not in text.lower()
        assert all(style.MODEL_NAMES[model] in text for model in MODELS)
        fonts = subprocess.check_output(["pdffonts", str(generated)], text=True)
        assert "Type 3" not in fonts and "TrueType" in fonts
        selected.to_csv(audit / source_csv.name, index=False)
        assert sha256(source_csv) == source_hash
        records.append(dict(target=str(target), generated=str(generated),
                            source_csv=str(source_csv), source_sha256=source_hash,
                            old_sha256=sha256(target), new_sha256=sha256(generated),
                            font_report=fonts))
    for record in records:
        assert sha256(Path(record["target"])) == record["old_sha256"]
        shutil.copy2(record["generated"], record["target"])
    shutil.copy2(__file__, audit / Path(__file__).name)
    shutil.copy2(SOURCE, audit / SOURCE.name)
    report = dict(style="mean markers and +/-1 sample SD error bars; no bands or connecting lines",
                  plotted_models=sorted(MODELS), matplotlib_version=matplotlib.__version__,
                  figures=records)
    (audit / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Updated both PDFs. Previews, backups, and verification: {audit}")


if __name__ == "__main__":
    main()
