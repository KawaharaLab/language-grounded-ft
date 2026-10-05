#!/usr/bin/env python3
"""Re-export the two insertion figures with embedded TrueType fonts only.

Use the PatchTST environment to match the original Matplotlib version.
Read existing aggregates without rerunning training or recomputing statistics.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess

import matplotlib as mpl
import pandas as pd
from PIL import Image, ImageChops


ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = Path(os.environ.get("FT_WORKSPACE_ROOT", ROOT.parent)).expanduser().resolve()
PATCH = Path(os.environ.get("FT_PATCHTST_ROOT", WORKSPACE / "PatchTST")).expanduser().resolve()
SOURCE = PATCH / "PatchTST_self_supervised/summarize_terminal_insertion_pretrained_adaptation.py"
RUN = PATCH / "results/terminal_insertion_supervised/terminal_insertion_supervised_20260913_172852"
PAIRS = (
    ("pretrained_linear_vs_scratch_end_to_end", "terminal_insertion_pretrained_linear_vs_scratch_end_to_end"),
    ("pretrained_lora_vs_end_to_end", "terminal_insertion_pretrained_lora_vs_end_to_end"),
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fonts(path):
    return subprocess.check_output(["pdffonts", str(path)], text=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", required=True, type=Path)
    args = parser.parse_args()
    audit = args.audit_dir.resolve()
    assert audit.is_dir()
    before = audit / "before"
    before.mkdir()
    spec = importlib.util.spec_from_file_location("insertion_plot_source", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert mpl.rcParams["pdf.fonttype"] == mpl.rcParams["ps.fonttype"] == 42
    manuscript_hash = sha256(ROOT / "paper.tex")
    rows = []
    for table_stem, figure_stem in PAIRS:
        target = ROOT / "fig" / f"{figure_stem}.pdf"
        csv = RUN / f"{table_stem}_aggregate.csv"
        run_pdf = RUN / f"{table_stem}_macro_f1_5seed.pdf"
        # Stop if the manuscript figure has been manually edited since export.
        assert sha256(target) == sha256(run_pdf), f"Different source figure: {target}"
        shutil.copy2(target, before / target.name)
        shutil.copy2(csv, audit / csv.name)
        original_hash, csv_hash = sha256(target), sha256(csv)
        table = pd.read_csv(csv)
        control_stem = audit / f"{figure_stem}_type3_control"
        with mpl.rc_context({"pdf.fonttype": 3, "ps.fonttype": 3}):
            module.plot_macro_f1(table, control_stem)
        new_stem = audit / figure_stem
        module.plot_macro_f1(table, new_stem)
        # Only PDF font encoding changes. The raster rendering is identical.
        with Image.open(control_stem.with_suffix(".png")) as a, Image.open(new_stem.with_suffix(".png")) as b:
            assert a.size == b.size
            assert ImageChops.difference(a.convert("RGB"), b.convert("RGB")).getbbox() is None
        run_png = RUN / f"{table_stem}_macro_f1_5seed.png"
        with Image.open(run_png) as a, Image.open(new_stem.with_suffix(".png")) as b:
            assert a.size == b.size
            assert ImageChops.difference(a.convert("RGB"), b.convert("RGB")).getbbox() is None, "Plot appearance changed"
        new_pdf = new_stem.with_suffix(".pdf")
        font_report = fonts(new_pdf)
        assert "Type 3" not in font_report
        assert "TrueType" in font_report
        assert all(line.split()[-5] == "yes" for line in font_report.splitlines()[2:] if line.strip()), font_report
        # Compare the figure boxes independently of PDF text encoding.
        def page_size(path):
            info = subprocess.check_output(["pdfinfo", str(path)], text=True)
            return next(line for line in info.splitlines() if line.startswith("Page size:"))
        assert page_size(target) == page_size(new_pdf)
        assert sha256(csv) == csv_hash and sha256(target) == original_hash
        shutil.copy2(new_pdf, target)
        assert sha256(target) == sha256(new_pdf)
        rows.append(dict(target=str(target), aggregate_csv=str(csv), aggregate_sha256=csv_hash,
                         old_sha256=original_hash, new_sha256=sha256(target),
                         before_fonts=fonts(before / target.name), after_fonts=font_report,
                         page_size=page_size(target), png_pixels_identical_to_original=True))
    assert sha256(ROOT / "paper.tex") == manuscript_hash
    shutil.copy2(SOURCE, audit / SOURCE.name)
    shutil.copy2(__file__, audit / Path(__file__).name)
    report = dict(matplotlib_version=mpl.__version__, plot_source=str(SOURCE),
                  plot_source_sha256=sha256(SOURCE), manuscript_unchanged=True, figures=rows,
                  note="Only manuscript PDFs replaced. Original experiment PDFs/CSVs unchanged. "
                       "Whole-paper compilation not tested.")
    (audit / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
