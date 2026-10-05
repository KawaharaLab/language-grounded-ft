"""Audit and summarize the existing full-pool, four-head MLP for manuscript RQ1.

This does not train or select models, alter the original experiment, or choose
between best and last using YCB. Both saved selections are replayed on CPU.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score


WORKSPACE = Path(
    os.environ.get('FT_WORKSPACE_ROOT', Path(__file__).resolve().parents[3])
).expanduser().resolve()
GENESIS = Path(os.environ.get('FT_GENESIS_ROOT', WORKSPACE / 'Genesis')).expanduser().resolve()
PATCHTST = Path(os.environ.get('FT_PATCHTST_ROOT', WORKSPACE / 'PatchTST')).expanduser().resolve()
ROOT = PATCHTST / 'results/rq1_language_composition/rq1_20260916_235337'
DEFAULT_OUT = GENESIS / 'paper_review_20260916/rq1_surface_supervised'
AXES = ('action', 'weight', 'interaction')
SEEDS = (42, 43, 44)


def read_json(path):
    with Path(path).open() as stream:
        return json.load(stream)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def audit(out):
    torch.set_num_threads(4)
    design = read_json(ROOT / 'experiment.json')
    manifest = read_json(ROOT / 'data/manifest.json')
    design_hash = sha(ROOT / 'experiment.json')
    assert sha(design['source_train']) == design['source_train_sha256']
    for path, digest in design['source_hashes'].items():
        assert sha(path) == digest, path
    sys.path.insert(0, str(ROOT / 'code'))
    from common import build_model, load_model

    classes = design['classes']
    assert list(classes) == [*AXES, 'support_surface']
    assert len(classes['support_surface']) == 9
    source = pd.read_csv(design['source_train']).fillna('')
    train = pd.read_csv(ROOT / 'data/train_rows.csv').fillna('')
    assert len(train) == 437244
    pd.testing.assert_frame_equal(source, train[source.columns], check_dtype=False)
    targets = np.load(ROOT / 'data/train_labels.npy')
    for column, axis in enumerate(classes):
        lookup = {label: i for i, label in enumerate(classes[axis])}
        expected = train[axis].map(lookup).fillna(-1).to_numpy(np.int64)
        np.testing.assert_array_equal(expected, targets[:, column])
    surface_mask = train.support_surface.ne('')
    assert train.loc[surface_mask, 'action'].str.contains('place|press').all()
    assert all(surface in caption for surface, caption in
               zip(train.loc[surface_mask, 'support_surface'], train.loc[surface_mask, 'annotation']))

    union = pd.read_csv(ROOT / 'data/eval_union_rows.csv').fillna('')
    force = np.load(ROOT / 'data/eval_union_force.npy', mmap_mode='r')
    assert force.shape == (len(union), 80, 12)
    assert not set(train.episode_id) & set(union.episode_id)
    assert not set(train.object_id) & set(union.object_id)
    eval_data = {}
    for axis in AXES:
        evidence = manifest['evaluation']['axes'][axis]
        assert sha(evidence['source_csv']) == evidence['source_sha256']
        official = pd.read_csv(evidence['source_csv']).fillna('')
        indices = np.load(ROOT / f'data/eval_{axis}_indices.npy')
        truth = np.load(ROOT / f'data/eval_{axis}_truth.npy').astype(str)
        selected = union.iloc[indices].reset_index(drop=True)
        pd.testing.assert_frame_equal(official[['csv_path', 'start', axis]],
                                      selected[['csv_path', 'start', axis]], check_dtype=False)
        np.testing.assert_array_equal(truth, official[axis].to_numpy())
        if axis == 'action':
            assert not selected.is_accidental_drop_episode.astype(bool).any()
            assert not selected.action.str.contains('then').any()
        eval_data[axis] = (indices, truth, official)

    rows, seed_audits = [], []
    for seed in SEEDS:
        folder = ROOT / f'jobs/supervised_full_seed{seed}'
        completed = read_json(folder / 'training_complete.json')
        init = read_json(folder / 'initialization.json')
        assert completed['job']['subset'] == 'full' and not completed['smoke']
        assert completed['design_sha256'] == design_hash
        split = np.load(ROOT / f'data/split_seed{seed}.npz')
        tr, va = split['full_train'], split['full_val']
        np.testing.assert_array_equal(np.sort(np.r_[tr, va]), np.arange(len(train)))
        assert not set(train.iloc[tr].episode_id) & set(train.iloc[va].episode_id)
        assert init['normalization']['train_windows'] == len(tr)
        assert init['normalization']['observations_per_channel'] == 80 * len(tr)
        for col, axis in enumerate(classes):
            valid = targets[tr, col]
            counts = np.bincount(valid[valid >= 0], minlength=len(classes[axis]))
            np.testing.assert_array_equal(counts, init['train_target_counts'][axis])
            assert (counts > 0).all()
        # Reproduce the historical validation-only selection, not a test argmax.
        history = pd.read_csv(folder / 'history.csv')
        best_loss, best_epoch = float('inf'), 0
        for row in history.itertuples():
            if row.validation_loss < best_loss - design['hyperparameters']['min_delta']:
                best_loss, best_epoch = row.validation_loss, int(row.epoch)
        assert best_epoch == completed['best_epoch']
        torch.manual_seed(seed)
        initial = build_model('supervised', classes).state_dict()
        item = dict(seed=seed, train_windows=len(tr), validation_windows=len(va),
                    shared_episodes=0, shared_objects=len(set(train.iloc[tr].object_id)
                                                        & set(train.iloc[va].object_id)),
                    best_epoch=best_epoch, last_epoch=completed['last_epoch'],
                    elapsed_seconds=completed['elapsed_seconds'],
                    checkpoints={}, surface_weight_change={})
        for selection in ('best', 'last'):
            path = folder / f'{selection}.pt'
            digest = sha(path)
            assert digest == completed[f'{selection}_sha256']
            ck = torch.load(path, map_location='cpu', weights_only=False)
            assert ck['normalization'] == init['normalization']
            assert ck['epoch'] == completed[f'{selection}_epoch']
            surface_key = 'heads.support_surface.weight'
            delta = float((ck['model'][surface_key] - initial[surface_key]).norm())
            assert delta > 0
            model = load_model(ck, classes, 'cpu')
            assert list(model.heads) == list(classes)
            mean = torch.tensor(ck['normalization']['mean'])
            scale = torch.tensor(ck['normalization']['scale'])
            outputs = {axis: [] for axis in AXES}
            with torch.inference_mode():
                for offset in range(0, len(union), 512):
                    x = torch.from_numpy(np.array(force[offset:offset + 512], copy=True))
                    result = model((x - mean) / scale)
                    for axis in AXES:
                        outputs[axis].append(result[axis])
            outputs = {axis: torch.cat(parts) for axis, parts in outputs.items()}
            for axis in AXES:
                indices, truth, official = eval_data[axis]
                stem = folder / f'evaluation/{selection}/canonical_{axis}'
                stored = np.load(str(stem) + '_scores.npz')
                candidates = stored['classes'].astype(str).tolist()
                logits = outputs[axis][indices]
                cols = []
                for label in candidates:
                    names = ('release', 'release midair') if axis == 'action' and label == 'release' else (label,)
                    cols.append(torch.logsumexp(logits[:, [classes[axis].index(name) for name in names]], dim=1))
                scores = torch.stack(cols, dim=1).numpy()
                np.testing.assert_allclose(scores, stored['scores'], atol=5e-4, rtol=1e-4)
                prediction = np.asarray(candidates)[scores.argmax(1)]
                saved = pd.read_csv(str(stem) + '_predictions.csv').fillna('')
                pd.testing.assert_frame_equal(official[['csv_path', 'start']], saved[['csv_path', 'start']], check_dtype=False)
                np.testing.assert_array_equal(truth, saved.true.to_numpy())
                np.testing.assert_array_equal(prediction, saved.pred.to_numpy())
                metric = dict(accuracy=accuracy_score(truth, prediction),
                              macro_f1=f1_score(truth, prediction, labels=candidates, average='macro', zero_division=0))
                original = read_json(str(stem) + '_metrics.json')
                for key, value in metric.items():
                    assert abs(value - original[key]) < 1e-12
                rows.append(dict(seed=seed, selection=selection, axis=axis, windows=len(truth), **metric))
            item['checkpoints'][selection] = dict(path=str(path), sha256=digest)
            item['surface_weight_change'][selection] = delta
            print(f'PASS seed={seed} selection={selection}: CPU replay and all three original RQ1 sets', flush=True)
        seed_audits.append(item)

    metrics = pd.DataFrame(rows)
    summary = metrics.groupby(['selection', 'axis'], sort=False).agg(
        seeds=('seed', 'nunique'), windows=('windows', 'first'),
        accuracy_mean=('accuracy', 'mean'), accuracy_sd=('accuracy', 'std'),
        macro_f1_mean=('macro_f1', 'mean'), macro_f1_sd=('macro_f1', 'std')).reset_index()
    assert summary.seeds.eq(3).all()
    out.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out / 'per_seed_metrics.csv', index=False)
    summary.to_csv(out / 'summary_metrics.csv', index=False)
    evidence = dict(status='passed', updated=datetime.now().astimezone().isoformat(),
                    source_experiment=str(ROOT), design_sha256=design_hash,
                    source_train_sha256=design['source_train_sha256'],
                    source_windows=len(train), classes=classes,
                    surface_labeled_windows=int(surface_mask.sum()),
                    surface_counts=train.loc[surface_mask, 'support_surface'].value_counts().to_dict(),
                    seeds=seed_audits, checks=['source row order and all four encoded targets',
                    'surface in caption and only on place/press-containing actions',
                    'all target classes present in training and surface weights updated',
                    'episode-disjoint full-pool splits; train/validation counts',
                    'validation-only checkpoint selection reconstructed from history',
                    'official evaluation CSV hashes, exact row order and truth',
                    'CPU inference matches every saved prediction for best and last',
                    'accuracy and macro F1 independently recomputed'],
                    normalization_caveat='Training-only indices and checkpoint metadata checked; full raw-force moments not rescanned.',
                    no_new_training=True)
    (out / 'audit.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + '\n')
    lines = ['# RQ1 surface-supervised MLP replacement', '',
             'Reuses all three full-pool models trained on 2026-09-17 (seeds 42/43/44). No new training or test-based checkpoint selection.', '',
             f'Source experiment: `{ROOT}`', '',
             'Shared MLP encoder (1,111,449 parameters), with independent action/weight/interaction/support-surface heads (1,129,905 total parameters).',
             'Targets: 10 raw actions including transitions and release midair; 2 weights; 3 interactions; 9 support descriptions. Missing labels are masked, not assigned to a none class. Class-balanced cross entropy is averaged over active heads.',
             'Official action readout excludes transition candidates and sums release/release-midair probability mass. No surface input is provided at inference.', '',
             '## Fixed original RQ1 test sets', '',
             'Percentages, mean ± sample SD over all three seeds. Best is primary; last is a prespecified sensitivity result.', '',
             '| Selection | Action Macro F1 | Weight Macro F1 | Interaction Macro F1 |',
             '|---|---:|---:|---:|']
    for selection in ('best', 'last'):
        cells = []
        for axis in AXES:
            row = summary[(summary.selection == selection) & (summary.axis == axis)].iloc[0]
            cells.append(f'{100 * row.macro_f1_mean:.2f} ± {100 * row.macro_f1_sd:.2f}')
        lines.append('| ' + selection + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', '## Protocol and scope', '',
              'The unchanged post-thinning source contains 437,244 windows. Approximately 10% of canonical episodes are validation. Both training and validation contain GSO objects; YCB objects are disjoint and test-only. Batch 512, Adam, peak LR 0.002, 10 warmup epochs then cosine decay, max 400 epochs, patience 75, min_delta 0.0001. Each model is selected by its own validation four-head loss.', '',
              'This replaces only the RQ1 supervised reference. Existing contrastive/SSL and RQ2/RQ3 checkpoints are unchanged. Historical contrastive RQ1 models use window validation and diagonal InfoNCE; this supervised reference uses episode validation. It closes the missing-surface supervision gap but is not a surface-only ablation or a completely matched optimization/split experiment.', '',
              'The same-architecture comparison in the original table is therefore MLP, not PatchTST. Preserve historical results; do not relabel their validation scheme.', '',
              'Audit: [audit.json](audit.json). Both selections, full precision: [per_seed_metrics.csv](per_seed_metrics.csv), [summary_metrics.csv](summary_metrics.csv).', '',
              'Reproduce the audit without GPU or training:', '',
              '```bash', f'{os.environ.get("FT_PYTHON", PATCHTST / ".venv/bin/python")} {GENESIS / "src/experiments/audit_rq1_surface_supervised.py"}', '```', '']
    (out / 'summary.md').write_text('\n'.join(lines))
    print(summary.to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    audit(parser.parse_args().out)
