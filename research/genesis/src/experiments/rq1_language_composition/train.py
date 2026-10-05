"""One resumable MLP job; evaluation is handled after the stage barrier."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import random
import time

os.environ.setdefault('WANDB_MODE', 'disabled')
import numpy as np
import pandas as pd
import torch

from common import (AXES, CONFIG, DEFAULT_RUN, HYPER, TRAIN_FORCE, build_model,
                    lr_for_epoch, multipositive_loss, now, save_torch, set_seed,
                    sha256, supervised_loss, write_json)


def population_stats(raw, indices):
    total = torch.zeros(12, device=raw.device, dtype=torch.float64)
    squared = torch.zeros_like(total)
    for start in range(0, len(indices), 1024):
        values = raw[indices[start:start + 1024]].double()
        total += values.sum((0, 1))
        squared += values.square().sum((0, 1))
    count = len(indices) * 80
    mean = total / count
    scale = (squared / count - mean.square()).clamp_min(1e-12).sqrt()
    return mean.float(), scale.float(), count


def encoder_hash(model):
    import hashlib
    digest = hashlib.sha256()
    for name, tensor in model.encoder.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def loss_for(model, job, values, semantic_ids, labels, text, class_weights, variant_ids=None):
    output = model(values)
    if job['kind'] == 'supervised':
        return supervised_loss(output, labels, class_weights)
    if text.ndim == 3:
        target = text[semantic_ids, variant_ids if variant_ids is not None else 0]
    else:
        target = text[semantic_ids]
    return multipositive_loss(output, target, semantic_ids)


def train_job(run_root, job_id, smoke_epochs=0, smoke_pause_after=0):
    run_root = Path(run_root)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    design_path = run_root / 'experiment.json'
    design = json.loads(design_path.read_text())
    design_hash = sha256(design_path)
    assert not smoke_pause_after or smoke_epochs > smoke_pause_after > 0
    for path, digest in design['source_hashes'].items():
        assert sha256(path) == digest, f'Code changed after design lock: {path}'
    job = next(j for j in design['jobs'] if j['id'] == job_id)
    out = run_root / ('smoke' if smoke_epochs else 'jobs') / job_id
    out.mkdir(parents=True, exist_ok=True)
    marker = out / 'training_complete.json'
    if marker.exists():
        saved = json.loads(marker.read_text())
        assert saved['design_sha256'] == design_hash
        print('ALREADY_COMPLETE', job_id, flush=True)
        return
    start_time = time.monotonic()
    seed = job['seed']
    set_seed(seed)
    model = build_model(job['kind'], design['classes']).cuda()
    initial = encoder_hash(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=HYPER['peak_lr'], weight_decay=0.)
    raw_numpy = np.load(TRAIN_FORCE, mmap_mode='r')
    assert raw_numpy.shape == (437244, 80, 12) and raw_numpy.dtype == np.float32
    raw = torch.tensor(raw_numpy, device='cuda')
    frame = pd.read_csv(run_root / 'data/train_rows.csv', usecols=['emb_index'])
    ids = torch.as_tensor(frame.emb_index.to_numpy(np.int64), device='cuda')
    targets = torch.as_tensor(np.load(run_root / 'data/train_labels.npy'), device='cuda', dtype=torch.long)
    splits = np.load(run_root / f'data/split_seed{seed}.npz')
    tr_np = splits[f"{job['subset']}_train"]
    va_np = splits[f"{job['subset']}_val"]
    train_ix = torch.as_tensor(tr_np, device='cuda', dtype=torch.long)
    val_ix = torch.as_tensor(va_np, device='cuda', dtype=torch.long)
    assert not np.intersect1d(tr_np, va_np).size
    mean, scale, observation_count = population_stats(raw, train_ix)
    assert torch.isfinite(mean).all() and torch.isfinite(scale).all() and (scale > 0).all()
    text = torch.load(run_root / 'text_embeddings.pt', map_location='cpu', weights_only=False)['train'][job['variant']].cuda()
    class_weights = {}
    counts = {}
    for k, axis in enumerate(AXES):
        values = targets[train_ix, k]
        count = torch.bincount(values[values >= 0], minlength=len(design['classes'][axis]))
        assert (count > 0).all(), f'{job_id} empty supervision class: {axis}, {count}'
        class_weights[axis] = count.sum() / (len(count) * count.float())
        counts[axis] = count.cpu().tolist()
    normalization = dict(mean=mean.cpu().tolist(), scale=scale.cpu().tolist(),
                         train_windows=len(tr_np), validation_windows=len(va_np),
                         observations_per_channel=observation_count)
    max_epochs = smoke_epochs or HYPER['max_epochs']
    best_loss, best_epoch, stale, last_epoch = float('inf'), 0, 0, 0
    resume = out / 'resume.pt'
    if resume.exists():
        ck = torch.load(resume, map_location='cuda', weights_only=False)
        assert ck['design_sha256'] == design_hash and ck['job_id'] == job_id
        assert ck['normalization'] == normalization and ck['initial_encoder_sha256'] == initial
        model.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['optimizer'])
        best_loss, best_epoch, stale, last_epoch = ck['best_loss'], ck['best_epoch'], ck['stale'], ck['epoch']
        random.setstate(ck['python_rng'])
        np.random.set_state(ck['numpy_rng'])
        torch.set_rng_state(ck['torch_rng'].cpu())
        torch.cuda.set_rng_state_all([x.cpu() for x in ck['cuda_rng']])
        print('RESUMED', job_id, 'epoch', last_epoch, flush=True)
    initialization = dict(job=job, normalization=normalization, initial_encoder_sha256=initial,
                          parameter_count=sum(p.numel() for p in model.parameters()),
                          encoder_parameter_count=sum(p.numel() for p in model.encoder.parameters()),
                          classes=design['classes'], train_target_counts=counts,
                          split_sha256=sha256(run_root / f'data/split_seed{seed}.npz'),
                          design_sha256=design_hash)
    write_json(out / 'initialization.json', initialization)

    def checkpoint(epoch, val_loss):
        return dict(model={k: v.detach().cpu() for k, v in model.state_dict().items()},
                    kind=job['kind'], variant=job['variant'], subset=job['subset'], seed=seed,
                    job_id=job_id, epoch=epoch, validation_loss=val_loss, normalization=normalization,
                    initial_encoder_sha256=initial, design_sha256=design_hash,
                    model_config=CONFIG)

    history_path = out / 'history.csv'
    if history_path.exists():
        history = pd.read_csv(history_path)
        # A crash between history and resume commit must not duplicate an epoch.
        history = history.loc[history.epoch <= last_epoch]
        if last_epoch and last_epoch not in history.epoch.values and 'history_row' in ck:
            history = pd.concat([history, pd.DataFrame([ck['history_row']])], ignore_index=True)
        history.to_csv(history_path, index=False)
    elif last_epoch and 'history_row' in ck:
        pd.DataFrame([ck['history_row']]).to_csv(history_path, index=False)
    if stale >= HYPER['patience']:
        max_epochs = last_epoch
    val_variants = torch.as_tensor((va_np * 17 + seed) % 32, device='cuda', dtype=torch.long)
    batch = HYPER['batch_size']
    for epoch in range(last_epoch + 1, max_epochs + 1):
        epoch_start = time.monotonic()
        lr = lr_for_epoch(epoch)
        for group in optimizer.param_groups:
            group['lr'] = lr
        # Dedicated CPU ordering/variant RNGs keep force batches identical for A/B/C.
        generator = torch.Generator().manual_seed(seed * 100000 + epoch)
        permutation = torch.randperm(len(train_ix), generator=generator).cuda()
        order = train_ix[permutation]
        variants = torch.randint(32, (len(order),), generator=torch.Generator().manual_seed(seed * 1000000 + epoch)).cuda()
        set_seed(10000 + seed * 1000 + epoch)
        model.train()
        total_loss, n_train = 0., 0
        full_steps = len(order) // batch
        if smoke_epochs:
            full_steps = min(full_steps, 4)
        for step in range(full_steps):
            lo, hi = step * batch, (step + 1) * batch
            ix = order[lo:hi]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_for(model, job, (raw[ix] - mean) / scale, ids[ix], targets[ix], text,
                            class_weights, variants[lo:hi])
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite train loss {job_id} epoch{epoch} step{step}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), HYPER['gradient_clip'], error_if_nonfinite=True)
            optimizer.step()
            total_loss += float(loss.detach()) * len(ix)
            n_train += len(ix)
        model.eval()
        total_val, n_val = 0., 0
        with torch.no_grad():
            stop = min(len(val_ix), batch * 2) if smoke_epochs else len(val_ix)
            for lo in range(0, stop, batch):
                ix = val_ix[lo:lo + batch]
                loss = loss_for(model, job, (raw[ix] - mean) / scale, ids[ix], targets[ix], text,
                                class_weights, val_variants[lo:lo + len(ix)])
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite validation loss {job_id}')
                total_val += float(loss) * len(ix)
                n_val += len(ix)
        train_loss, val_loss = total_loss / n_train, total_val / n_val
        improved = val_loss < best_loss - HYPER['min_delta']
        if improved:
            best_loss, best_epoch, stale = val_loss, epoch, 0
        else:
            stale += 1
        ck = checkpoint(epoch, val_loss)
        if improved:
            save_torch(out / 'best.pt', ck)
        save_torch(out / 'last.pt', ck)
        row = dict(epoch=epoch, train_loss=train_loss, validation_loss=val_loss, learning_rate=lr,
                   best_epoch=best_epoch, stale=stale, updates=full_steps,
                   elapsed_seconds=time.monotonic() - start_time,
                   epoch_seconds=time.monotonic() - epoch_start)
        resume_state = dict(ck, optimizer=optimizer.state_dict(), best_loss=best_loss, best_epoch=best_epoch,
                            stale=stale, python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                            lr_state=dict(epoch=epoch, lr=lr, schedule='explicit warmup/cosine'), history_row=row)
        save_torch(resume, resume_state)
        exists = history_path.exists()
        with history_path.open('a', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            if not exists:
                writer.writeheader()
            writer.writerow(row)
        write_json(out / 'status.json', dict(phase='training', job_id=job_id, pid=os.getpid(),
                   updated=now(), resumed_from_epoch=last_epoch, **row))
        print(json.dumps(dict(job_id=job_id, **row)), flush=True)
        last_epoch = epoch
        if smoke_pause_after and epoch == smoke_pause_after:
            write_json(out / 'status.json', dict(phase='preflight_paused', job_id=job_id, epoch=epoch, updated=now()))
            return
        if stale >= HYPER['patience']:
            break
    write_json(marker, dict(job=job, design_sha256=design_hash, completed=now(), last_epoch=last_epoch,
               best_epoch=best_epoch, best_loss=best_loss, stale=stale, smoke=bool(smoke_epochs),
               stop_reason='early_stopping' if stale >= HYPER['patience'] else 'max_epochs',
               elapsed_seconds=time.monotonic() - start_time,
               best_sha256=sha256(out / 'best.pt'), last_sha256=sha256(out / 'last.pt')))
    write_json(out / 'status.json', dict(phase='trained', job_id=job_id, epoch=last_epoch, updated=now()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--job', required=True)
    parser.add_argument('--smoke-epochs', type=int, default=0)
    parser.add_argument('--smoke-pause-after', type=int, default=0)
    args = parser.parse_args()
    train_job(args.run_dir, args.job, args.smoke_epochs, args.smoke_pause_after)
