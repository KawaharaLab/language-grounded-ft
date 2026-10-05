"""Shared, import-safe definitions for the fixed-pool RQ1 experiments."""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


WORKSPACE = Path(
    os.environ.get('FT_WORKSPACE_ROOT', Path(__file__).resolve().parents[4])
).expanduser().resolve()
GENESIS = Path(os.environ.get('FT_GENESIS_ROOT', WORKSPACE / 'Genesis')).expanduser().resolve()
PATCH = Path(os.environ.get('FT_PATCHTST_ROOT', WORKSPACE / 'PatchTST')).expanduser().resolve()
SOURCE = Path(__file__).resolve().parent
DEFAULT_RUN = PATCH / 'results/rq1_language_composition/rq1_20260916_235337'
TRAIN = GENESIS / 'data/train_08142026/train_08142026_terrain_floor_wobble_uniform_descent_contact_clean_slip5mms_balanced.csv'
TRAIN_SHA = 'b4f38c4b43d68848bb084fef6d8e7a0e58719b491039df0f4ec9d8df704c1570'
TRAIN_FORCE = GENESIS / 'paper_review_20260916/split_pilot/train_force.npy'
OFFICIAL = PATCH / 'results/paper_classification_multiseed/paper_classification_3seed_20260909/prepared_eval'
SEEDS = (42, 43, 44)
AXES = ('action', 'weight', 'interaction', 'support_surface')
ACTION_CLASSES = ('grasp', 'hold', 'lift', 'place', 'press', 'release')
WEIGHT_CLASSES = ('heavy', 'light')
INTERACTION_CLASSES = ('accidental drop', 'slip', 'stable')
CONFIG = dict(seq_len=80, nvars=12, hidden_dim=429, depth=3, dropout=.2, out_embed_dim=768)
HYPER = dict(batch_size=512, max_epochs=400, peak_lr=.002, min_lr=1e-5,
             warmup_epochs=10, optimizer='Adam', weight_decay=0.,
             gradient_clip=1., temperature=.07, patience=75, min_delta=.0001)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def save_torch(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def now():
    return datetime.now().astimezone().isoformat()


class Encoder(nn.Module):
    """Exactly the published 960 -> 429 x 3 -> 768 MLP parameterization."""
    def __init__(self):
        super().__init__()
        layers = [nn.Flatten(start_dim=1)]
        for i in range(CONFIG['depth']):
            layers += [nn.Linear(960 if i == 0 else 429, 429), nn.GELU(), nn.Dropout(.2)]
        layers.append(nn.Linear(429, 768))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class Contrastive(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()

    def forward(self, x):
        return F.normalize(self.encoder(x), dim=-1, eps=1e-8)


class Supervised(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.encoder = Encoder()
        self.heads = nn.ModuleDict({axis: nn.Linear(768, len(classes[axis])) for axis in AXES})

    def forward(self, x):
        feature = self.encoder(x)
        return {axis: head(feature) for axis, head in self.heads.items()}


def build_model(kind, classes):
    if kind == 'contrastive':
        return Contrastive()
    if kind == 'supervised':
        return Supervised(classes)
    raise ValueError(kind)


def load_model(checkpoint, classes, device='cuda'):
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False) if isinstance(checkpoint, (str, Path)) else checkpoint
    model = build_model(ck['kind'], classes)
    model.load_state_dict(ck['model'], strict=True)
    return model.to(device).eval()


def multipositive_loss(force, text, groups):
    logits = F.normalize(force, dim=1) @ F.normalize(text, dim=1).T / HYPER['temperature']
    positives = groups[:, None].eq(groups[None, :])
    left = torch.logsumexp(logits, dim=1) - torch.logsumexp(logits.masked_fill(~positives, -torch.inf), dim=1)
    right = torch.logsumexp(logits.T, dim=1) - torch.logsumexp(logits.T.masked_fill(~positives.T, -torch.inf), dim=1)
    return .5 * (left.mean() + right.mean())


def supervised_loss(logits, labels, class_weights):
    losses = []
    for k, axis in enumerate(AXES):
        good = labels[:, k] >= 0
        if good.any():
            losses.append(F.cross_entropy(logits[axis][good], labels[good, k], weight=class_weights[axis]))
    if not losses:
        raise ValueError('Batch has no supervised targets')
    return torch.stack(losses).mean()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def lr_for_epoch(epoch):
    if epoch <= HYPER['warmup_epochs']:
        return HYPER['peak_lr'] * epoch / HYPER['warmup_epochs']
    position = (epoch - HYPER['warmup_epochs']) / (HYPER['max_epochs'] - HYPER['warmup_epochs'])
    return HYPER['min_lr'] + .5 * (HYPER['peak_lr'] - HYPER['min_lr']) * (1 + np.cos(np.pi * position))


def make_jobs():
    jobs = []
    for seed in SEEDS:
        for letter, variant in [('A', 'canonical'), ('B', 'single'), ('C', 'diverse')]:
            jobs.append(dict(id=f'lexical_{letter}_seed{seed}', stage='lexical', kind='contrastive',
                             variant=variant, subset='full', seed=seed))
    for subset in ['full', 'without_place_heavy', 'without_place_light']:
        for seed in SEEDS:
            for kind in ['contrastive', 'supervised']:
                if subset == 'full' and kind == 'contrastive':
                    continue  # Identical to lexical A; reuse its paired full-data checkpoint.
                jobs.append(dict(id=f'{kind}_{subset}_seed{seed}', stage='composition', kind=kind,
                                 variant='canonical', subset=subset, seed=seed))
    assert len(jobs) == 24
    return jobs
