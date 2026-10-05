"""Persistent two-GPU stage scheduler. Run the frozen copy inside tmux.

The scheduler has no wall-time deadline. A restart skips verified completed jobs,
and interrupted jobs resume from their committed optimizer/RNG checkpoint.
It never modifies source datasets, experiment choices, or unrelated processes.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from common import DEFAULT_RUN, TRAIN, TRAIN_FORCE, TRAIN_SHA, now, sha256, write_json

PYTHON = os.environ.get('FT_PYTHON', str(Path(os.environ.get(
    'FT_PATCHTST_ROOT', Path(__file__).resolve().parents[4] / 'PatchTST'
)) / '.venv/bin/python'))


def verify_inputs(root, design):
    assert sha256(TRAIN) == TRAIN_SHA, 'Original training CSV changed'
    manifest = json.loads((root / 'data/manifest.json').read_text())
    expected = 'c733b3474af8f37bb1d1bfedfff03dfd577163034bebaace9ea087e9b7e80b30'
    assert sha256(TRAIN_FORCE) == expected, 'Original raw training cache changed'
    for name in ('input_hashes', 'source_hashes'):
        for path, digest in design[name].items():
            assert sha256(path) == digest, f'Frozen {name} mismatch: {path}'
    return dict(verified=now(), files=len(design['input_hashes']) + len(design['source_hashes']) + 2)


def completed(root, job, phase, design_hash):
    folder = root / 'jobs' / job['id']
    marker = folder / f'{phase}_complete.json'
    if not marker.exists():
        return False
    saved = json.loads(marker.read_text())
    if phase == 'training':
        assert saved['design_sha256'] == design_hash and not saved['smoke']
        for selection in ('best', 'last'):
            assert sha256(folder / f'{selection}.pt') == saved[f'{selection}_sha256']
    else:
        assert saved['protocol_sha256'] == design_hash
        for selection, info in saved['selections'].items():
            assert sha256(folder / f'{selection}.pt') == info['checkpoint_sha256']
    return True


def environment(gpu):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               OPENBLAS_NUM_THREADS='4', WANDB_MODE='disabled', TOKENIZERS_PARALLELISM='false',
               PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
    return env


def run_phase(root, design, stage, phase, per_gpu):
    design_hash = sha256(root / 'experiment.json')
    jobs = [j for j in design['jobs'] if j['stage'] == stage]
    pending = [j for j in jobs if not completed(root, j, phase, design_hash)]
    active, failures, attempts = [], [], {}
    logs = root / 'logs'
    logs.mkdir(exist_ok=True)
    slots = [(gpu, slot) for slot in range(per_gpu if phase == 'training' else 1) for gpu in (0, 1)]
    while pending or active:
        occupied = {(p['gpu'], p['slot']) for p in active}
        for gpu, slot in slots:
            if (gpu, slot) in occupied or not pending:
                continue
            job = pending.pop(0)
            attempts[job['id']] = attempts.get(job['id'], 0) + 1
            if phase == 'training':
                cmd = [PYTHON, '-u', '-B', str(root / 'code/train.py'), '--run-dir', str(root), '--job', job['id']]
            else:
                cmd = [PYTHON, '-u', '-B', str(root / 'code/evaluate.py'), '--run-root', str(root), '--job-id', job['id'], '--device', 'cuda']
            log_path = logs / f"{job['id']}_{phase}.log"
            stream = log_path.open('a', buffering=1)
            stream.write(f'\nQUEUE START {now()} GPU={gpu} attempt={attempts[job["id"]]}\n')
            process = subprocess.Popen(cmd, stdout=stream, stderr=subprocess.STDOUT, env=environment(gpu), cwd=root)
            active.append(dict(job=job, process=process, stream=stream, gpu=gpu, slot=slot, log=str(log_path)))
            print('START', stage, phase, job['id'], 'GPU', gpu, 'PID', process.pid, flush=True)
        for item in list(active):
            code = item['process'].poll()
            if code is None:
                continue
            item['stream'].close()
            active.remove(item)
            job = item['job']
            if code == 0 and completed(root, job, phase, design_hash):
                print('DONE', phase, job['id'], flush=True)
            else:
                # Retry only signal interruptions, not code/permission failures.
                if code in (-9, -15, 137, 143) and attempts[job['id']] < 2:
                    pending.append(job)
                    print('RETRY', phase, job['id'], code, flush=True)
                else:
                    failures.append(dict(job=job['id'], returncode=code, log=item['log']))
                    print('FAILED', phase, job['id'], code, flush=True)
        write_json(root / 'queue_status.json', dict(updated=now(), pid=os.getpid(), stage=stage, phase=phase,
            active=[dict(job=p['job']['id'], pid=p['process'].pid, gpu=p['gpu'], log=p['log']) for p in active],
            pending=[j['id'] for j in pending], failed=failures,
            completed=sum(completed(root, j, phase, design_hash) for j in jobs), total=len(jobs)))
        if pending or active:
            time.sleep(10)
    if failures:
        raise RuntimeError(f'{stage}/{phase}: failed jobs; see queue_status.json: {failures}')


def main(root, per_gpu):
    root = Path(root).resolve()
    assert Path(__file__).resolve() == root / 'code/run_queue.py', 'Run the frozen run_queue.py'
    assert per_gpu in (1, 2)
    lock = (root / 'queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_file = root / 'queue_status.json'
    if status_file.exists():
        for active in json.loads(status_file.read_text()).get('active', []):
            proc = Path(f'/proc/{active["pid"]}/cmdline')
            if proc.exists() and str(root).encode() in proc.read_bytes():
                raise RuntimeError(f'Previous job still running; do not launch a duplicate: {active}')
    design = json.loads((root / 'experiment.json').read_text())
    preflight = json.loads((root / 'preflight_complete.json').read_text())
    assert preflight['design_sha256'] == sha256(root / 'experiment.json')
    try:
        write_json(root / 'input_verification_start.json', verify_inputs(root, design))
        for stage in design['stage_order']:
            run_phase(root, design, stage, 'training', per_gpu)
            verify_inputs(root, design)
            run_phase(root, design, stage, 'evaluation', per_gpu)
            cmd = [PYTHON, '-u', '-B', str(root / 'code/report.py'), '--run-root', str(root)]
            if stage != design['stage_order'][-1]:
                cmd.append('--no-bootstrap')
            with (root / 'logs' / f'{stage}_report.log').open('a') as stream:
                subprocess.run(cmd, check=True, env=environment(0), cwd=root, stdout=stream, stderr=subprocess.STDOUT)
            write_json(root / f'{stage}_complete.json', dict(completed=now(), design_sha256=sha256(root / 'experiment.json')))
        write_json(root / 'input_verification_end.json', verify_inputs(root, design))
        assert (root / 'all_complete.json').exists(), 'Final audited report did not mark completion'
        write_json(root / 'queue_status.json', dict(phase='complete', completed=now(), models=len(design['jobs'])))
        print('ALL COMPLETE', now(), root, flush=True)
    except BaseException as error:
        write_json(root / 'queue_error.json', dict(time=now(), error=repr(error), pid=os.getpid()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--per-gpu', type=int, default=2)
    args = parser.parse_args()
    main(args.run_dir, args.per_gpu)
