"""Freeze all data, wording, models, and evaluation choices before training."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from common import (ACTION_CLASSES, AXES, CONFIG, DEFAULT_RUN, HYPER, INTERACTION_CLASSES,
                    OFFICIAL, SEEDS, SOURCE, TRAIN, TRAIN_SHA, WEIGHT_CLASSES,
                    make_jobs, now, save_torch, sha256, write_json)
from captions import build_caption_bank, build_eval_suites, build_composition_prototypes

TEXT_MODEL = os.environ.get('FT_TEXT_MODEL', 'sentence-transformers/all-mpnet-base-v2')
TEXT_MODEL_REVISION = os.environ.get(
    'FT_TEXT_MODEL_REVISION', 'e8c3b32edf5434bc2275fc9bab85f82640a19130'
)


def prepare(run_root):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    run_root = Path(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    assert not (run_root / 'experiment.json').exists(), 'The experiment is already frozen'
    assert sha256(TRAIN) == TRAIN_SHA
    if not (run_root / 'data/manifest.json').exists():
        from prepare_data import prepare_data
        prepare_data(run_root)
    data = json.loads((run_root / 'data/manifest.json').read_text())
    frame = pd.read_csv(run_root / 'data/train_rows.csv').fillna('')
    bank = build_caption_bank(frame)
    suites = build_eval_suites()
    composition = build_composition_prototypes()
    assert len(bank['canonical']) == 250 and len(bank['diverse']) == 250
    assert all(len(v) == 32 for v in bank['diverse'])
    assert all(bank['single'][i] == bank['diverse'][i][0] for i in range(250))
    write_json(run_root / 'caption_bank.json', bank)
    write_json(run_root / 'evaluation_texts.json', dict(suites=suites, composition=composition))
    for variant in ('canonical', 'single', 'diverse'):
        captions = bank[variant] if variant != 'diverse' else [text for group in bank[variant] for text in group]
        (run_root / f'train_{variant}_annotations.txt').write_text('\n'.join(sorted(set(captions))) + '\n')
    write_json(run_root / 'pretraining_design.json', dict(created=now(), source_train=str(TRAIN),
        source_train_sha256=TRAIN_SHA, hypotheses=['lexical transfer', 'unseen place/weight combinations'],
        user_confirmed=dict(holdouts=['place|heavy', 'place|light'], main_queries='existing canonical sentences',
                            secondary_queries='bare class words and fixed new paraphrases',
                            classifier='independent raw-action/weight/interaction/support heads', seeds=list(SEEDS)),
        jobs=make_jobs(), hyperparameters=HYPER, model=CONFIG,
        ordering='All lexical training then lexical evaluation, then all composition training and evaluation.',
        training_pool='Only the immutable 437244-row post-thinning CSV; no added/rebalanced rows.',
        contrastive_positive='Same canonical semantic ID in both directions, regardless of paraphrase variant.',
        composition_contrastive_captions='Canonical wording; lexical A reused as full-data reference.',
        supervision='Identical attributes provided; supervised keeps raw transitions and midair plus surface targets.',
        classification_readout='Exclude transition candidates for singleton action test; sum release/midair probabilities.',
        selection='Own validation objective; fixed patience/min_delta; no YCB selection; best primary, last secondary.',
        normalization='Per seed/subset training windows only, full-population axis mean/std, repeated physical samples retain training weights.',
        rng='seed42/43/44 initialization and episode split; force order and dropout same across lexical variants.',
        diversity='32 precomputed captions per meaning, one independent draw per row per training epoch; fixed validation draws.',
        composition_test='Unique union of existing official axis CSVs; singleton action, observed weight, no accidental-drop episode.',
        stats='Accuracy/macroF1; joint composition correctness and cell macro; paired object bootstrap2000 over fixed3seeds.',
        no_time_cap=True))
    text_list = bank['canonical'] + bank['single'] + [text for group in bank['diverse'] for text in group]
    for suite in suites.values():
        for prompts in suite.values():
            text_list += list(prompts.values())
    text_list += list(composition.values())
    unique = sorted(set(text_list))
    print('Encoding', len(unique), 'unique fixed texts', flush=True)
    text_model_path = Path(TEXT_MODEL).expanduser()
    local_only = text_model_path.exists()
    encoder = SentenceTransformer(TEXT_MODEL, device='cuda:0', local_files_only=local_only,
                                  revision=None if local_only else TEXT_MODEL_REVISION)
    encoder.max_seq_length = 512
    vectors = encoder.encode(unique, batch_size=64, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    assert vectors.shape == (len(unique), 768) and np.isfinite(vectors).all()
    lookup = {text: vectors[i] for i, text in enumerate(unique)}
    def tensor(texts):
        return torch.from_numpy(np.stack([lookup[t] for t in texts]))
    train_vectors = dict(canonical=tensor(bank['canonical']), single=tensor(bank['single']),
                         diverse=torch.stack([tensor(group) for group in bank['diverse']]))
    eval_vectors, checks = {}, []
    order = dict(action=ACTION_CLASSES, weight=WEIGHT_CLASSES, interaction=INTERACTION_CLASSES)
    for suite_name, suite in suites.items():
        eval_vectors[suite_name] = {}
        for axis, names in order.items():
            prompts = [suite[axis][name] for name in names]
            embedding = tensor(prompts)
            if suite_name == 'canonical':
                index = pd.read_csv(OFFICIAL / f'{axis}_index_map.csv').set_index(axis)
                old = torch.stack([torch.load(OFFICIAL / f'bert_emb/{axis}/{int(index.loc[name,"emb_index"])}.pt',
                                              map_location='cpu', weights_only=False).reshape(-1) for name in names])
                old = torch.nn.functional.normalize(old, dim=1)
                cos = torch.nn.functional.cosine_similarity(old, embedding, dim=1)
                assert float(cos.min()) > .99999
                checks.append(dict(axis=axis, minimum_cosine=float(cos.min()), max_absolute_error=float((old-embedding).abs().max())))
                embedding = old  # Main evaluation is exactly the archived RQ1 prototype tensor.
            eval_vectors[suite_name][axis] = dict(classes=list(names), texts=prompts, vectors=embedding)
    output = dict(train=train_vectors, eval=eval_vectors,
                  composition=dict(classes=list(composition), texts=list(composition.values()), vectors=tensor(list(composition.values()))))
    save_torch(run_root / 'text_embeddings.pt', output)
    write_json(run_root / 'text_provenance.json', dict(model_path=str(TEXT_MODEL),
        revision=text_model_path.name if local_only else TEXT_MODEL_REVISION,
        max_seq_length=512, unique_texts=len(unique), canonical_checks=checks, dimension=768,
        versions={name: importlib.metadata.version(name) for name in ['torch','numpy','pandas','sentence-transformers','transformers']},
        input_caption_bank_sha256=sha256(run_root / 'caption_bank.json'), eval_texts_sha256=sha256(run_root / 'evaluation_texts.json')))
    code = run_root / 'code'
    code.mkdir(exist_ok=False)
    for path in SOURCE.glob('*.py'):
        shutil.copy2(path, code / path.name)
    inputs = [run_root / 'data/manifest.json', run_root / 'caption_bank.json', run_root / 'evaluation_texts.json',
              run_root / 'text_embeddings.pt', run_root / 'pretraining_design.json']
    inputs += [p for p in (run_root / 'data').glob('*') if p.suffix in {'.csv', '.npy', '.npz'}]
    design = dict(created=now(), run_root=str(run_root), jobs=make_jobs(), classes=data['classes'],
                  hyperparameters=HYPER, model_config=CONFIG, source_train=str(TRAIN), source_train_sha256=TRAIN_SHA,
                  data_manifest=str(run_root / 'data/manifest.json'),
                  source_hashes={str(p):sha256(p) for p in sorted(code.glob('*.py'))},
                  input_hashes={str(p):sha256(p) for p in inputs},
                  stage_order=['lexical','composition'], evaluation_no_test_selection=True)
    write_json(run_root / 'experiment.json', design)
    print('FROZEN', run_root / 'experiment.json', 'jobs', len(design['jobs']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    prepare(args.run_dir)
