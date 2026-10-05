"""Test the frozen run before queuing real models; smoke results are never reported."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
import torch

from common import DEFAULT_RUN, now, sha256, write_json
from run_queue import PYTHON, environment, verify_inputs


def fixture(root, parent, name):
    result = parent / name
    result.mkdir()
    for filename in ('data', 'text_embeddings.pt'):
        (result / filename).symlink_to(root / filename, target_is_directory=filename == 'data')
    shutil.copy2(root / 'experiment.json', result / 'experiment.json')
    return result


def run_smoke(root, folder, job, gpu, pause=0):
    cmd = [PYTHON, '-u', '-B', str(root / 'code/train.py'), '--run-dir', str(folder),
           '--job', job, '--smoke-epochs', '2']
    if pause:
        cmd += ['--smoke-pause-after', str(pause)]
    with (folder / 'smoke.log').open('a') as log:
        subprocess.run(cmd, check=True, env=environment(gpu), stdout=log, stderr=subprocess.STDOUT)


def main(root):
    root = Path(root).resolve()
    assert Path(__file__).resolve() == root / 'code/preflight.py'
    design_hash = sha256(root / 'experiment.json')
    design = json.loads((root / 'experiment.json').read_text())
    verify_inputs(root, design)
    base = root / 'preflight'
    base.mkdir(exist_ok=True)
    parent = Path(tempfile.mkdtemp(prefix='attempt_', dir=base))
    with (parent / 'unit_tests.log').open('w') as log:
        subprocess.run([PYTHON, '-B', '-m', 'unittest', 'discover', '-s', str(root / 'code'), '-p', 'test*.py', '-v'],
                       check=True, env=environment(0), stdout=log, stderr=subprocess.STDOUT)
    resumed = fixture(root, parent, 'resumed')
    uninterrupted = fixture(root, parent, 'uninterrupted')
    supervised = fixture(root, parent, 'supervised')
    cl_job, sup_job = 'lexical_C_seed42', 'supervised_without_place_heavy_seed42'

    def resume_test():
        run_smoke(root, resumed, cl_job, 0, pause=1)
        assert not (resumed / 'smoke' / cl_job / 'training_complete.json').exists()
        run_smoke(root, resumed, cl_job, 0)
        run_smoke(root, uninterrupted, cl_job, 0)
        left = torch.load(resumed / 'smoke' / cl_job / 'resume.pt', weights_only=False, map_location='cpu')
        right = torch.load(uninterrupted / 'smoke' / cl_job / 'resume.pt', weights_only=False, map_location='cpu')
        assert left['epoch'] == right['epoch'] == 2
        assert left['normalization'] == right['normalization']
        assert left['validation_loss'] == right['validation_loss']
        for key in left['model']:
            assert torch.equal(left['model'][key], right['model'][key]), f'Resumed model differs: {key}'
        for index in left['optimizer']['state']:
            for key in left['optimizer']['state'][index]:
                assert torch.equal(left['optimizer']['state'][index][key], right['optimizer']['state'][index][key])
        return dict(bitwise_model_equal=True, bitwise_optimizer_equal=True, validation_loss=left['validation_loss'], epoch=2)

    with ThreadPoolExecutor(max_workers=2) as pool:
        left = pool.submit(resume_test)
        right = pool.submit(run_smoke, root, supervised, sup_job, 1)
        resume_check = left.result()
        right.result()
    # Exercise the full inference/readout/report interface on quarantined models.
    # The resulting random-model scores are deliberately not inspected or used.
    for folder, job, gpu in ((resumed, cl_job, 0), (supervised, sup_job, 1)):
        (folder / 'jobs').symlink_to(folder / 'smoke', target_is_directory=True)
        with (folder / 'evaluation.log').open('w') as log:
            subprocess.run([PYTHON, '-u', '-B', str(root / 'code/evaluate.py'), '--run-root', str(folder), '--job-id', job, '--allow-smoke'],
                           check=True, env=environment(gpu), stdout=log, stderr=subprocess.STDOUT)
        assert (folder / 'jobs' / job / 'evaluation_complete.json').exists()
    write_json(root / 'preflight_complete.json', dict(completed=now(), design_sha256=design_hash,
        artifacts=str(parent), resume=resume_check, full_inference_readouts=['contrastive', 'supervised'],
        unit_tests_passed=True, smoke_models_quarantined=True))
    print('PREFLIGHT PASSED', parent, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=DEFAULT_RUN)
    main(parser.parse_args().run_dir)
