#!/usr/bin/env python3
"""Remove only the Supervised series from the two manuscript insertion PDFs.

Run with ``$FT_PYTHON`` (or the workspace PatchTST virtualenv). Reuse saved aggregates and
the original plotter; retain source results and back up the manuscript PDFs.
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

import matplotlib as mpl
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = Path(os.environ.get('FT_WORKSPACE_ROOT', ROOT.parent)).expanduser().resolve()
PATCH = Path(os.environ.get('FT_PATCHTST_ROOT', WORKSPACE / 'PatchTST')).expanduser().resolve()
SOURCE = PATCH / 'PatchTST_self_supervised/summarize_terminal_insertion_pretrained_adaptation.py'
RUN = PATCH / 'results/terminal_insertion_supervised/terminal_insertion_supervised_20260913_172852'
STEMS = ('pretrained_linear_vs_scratch_end_to_end', 'pretrained_lora_vs_end_to_end')


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def y_min(table: pd.DataFrame) -> float:
    lower = table.test_macro_f1_mean - table.test_macro_f1_std.fillna(0)
    return max(0.0, np.floor((float(lower.min()) - 0.03) * 10) / 10)


def main() -> None:
    spec = importlib.util.spec_from_file_location('insertion_plot_source', SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manuscript_hash = sha256(ROOT / 'paper.tex')
    audit = Path(tempfile.mkdtemp(prefix='terminal_insertion_no_supervised_', dir=ROOT / 'fig'))
    before = audit / 'before'
    before.mkdir()
    records = []
    for stem in STEMS:
        csv = RUN / f'{stem}_aggregate.csv'
        target = ROOT / 'fig' / f'terminal_insertion_{stem}.pdf'
        csv_hash = sha256(csv)
        original_hash = sha256(target)
        table = pd.read_csv(csv)
        selected = table.loc[table.model.ne('supervised')].copy()
        assert len(table) - len(selected) == 7
        expected = {'scratch', 'contrastive', 'masked_reconstruction', 'vae', 'vicreg', 'byol'}
        assert set(selected.model) == expected
        assert all(tuple(group.episode_count) == module.EPISODE_COUNTS
                   for _, group in selected.groupby('model'))
        assert y_min(table) == y_min(selected), 'Removing a series would change the y-axis limits'
        output = audit / target.stem
        with mpl.rc_context({'pdf.fonttype': 42, 'ps.fonttype': 42}):
            module.plot_macro_f1(selected, output)
        generated = output.with_suffix('.pdf')
        text = subprocess.check_output(['pdftotext', str(generated), '-'], text=True)
        assert 'supervised' not in text.lower()
        assert all(module.MODEL_NAMES[model] in text for model in expected)
        fonts = subprocess.check_output(['pdffonts', str(generated)], text=True)
        assert 'Type 3' not in fonts and 'TrueType' in fonts
        assert all(row.split()[-5] == 'yes' for row in fonts.splitlines()[2:] if row.strip())
        assert sha256(csv) == csv_hash and sha256(target) == original_hash
        shutil.copy2(target, before / target.name)
        selected.to_csv(audit / csv.name, index=False)
        records.append(dict(target=str(target), generated=str(generated),
                            source_csv=str(csv), source_sha256=csv_hash,
                            old_sha256=original_hash, new_sha256=sha256(generated),
                            excluded_model='supervised', excluded_rows=7,
                            remaining_models=sorted(expected), font_report=fonts))
    # Replace only after both figures pass the checks.
    for record in records:
        assert sha256(Path(record['target'])) == record['old_sha256']
        shutil.copy2(record['generated'], record['target'])
    assert sha256(ROOT / 'paper.tex') == manuscript_hash
    report = dict(plot_source=str(SOURCE), plot_source_sha256=sha256(SOURCE),
                  manuscript_unchanged=True, aggregates_unchanged=True, figures=records)
    (audit / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'Updated both manuscript PDFs. Backup and verification: {audit}')


if __name__ == '__main__':
    main()
