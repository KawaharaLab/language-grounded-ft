#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

GROUP="${GROUP:-tokenfusion_byol_$(date +%Y%m%d_%H%M%S)}"
GPU="${GPU:-0}"
WANDB_MODE="${WANDB_MODE:-online}"
if [[ ! "$GROUP" =~ ^[A-Za-z0-9_-]+$ || ! "$GPU" =~ ^[0-9]+$ ]]; then
  echo "Invalid GROUP or GPU"
  exit 1
fi
PYTHON="$ROOT_DIR/.venv/bin/python"
REFERENCE_GROUP="tokenfusion_supervised_ssl_terrain0814_uniform_descent_slip5mms_20260825_015515"
REFERENCE_ROOT="$ROOT_DIR/results/tokenfusion_supervised_ssl/$REFERENCE_GROUP"
RESULT_ROOT="$ROOT_DIR/results/tokenfusion_byol/$GROUP"
SAVE_DIR="$ROOT_DIR/saved_models/tokenfusion_ssl/$GROUP/tokenfusion_ssl_byol_seed42"
RUN_NAME="tokenfusion_ssl_byol_seed42"
if [[ -e "$RESULT_ROOT" || -e "$SAVE_DIR" ]]; then
  echo "Output already exists. Choose a new GROUP; existing experiments are not overwritten."
  exit 1
fi
mkdir -p "$RESULT_ROOT/logs" "$SAVE_DIR"
exec > >(tee -a "$RESULT_ROOT/logs/pipeline.log") 2>&1
export CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}" NUMBA_NUM_THREADS="${NUMBA_NUM_THREADS:-4}"
export GROUP RESULT_ROOT SAVE_DIR ROOT_DIR REFERENCE_ROOT RUN_NAME WANDB_MODE
PHASE="preparation"
finish() {
  local code="$?"
  if (( code != 0 )); then
    printf '{"status":"failed","phase":"%s","exit_code":%s}\n' "$PHASE" "$code" > "$RESULT_ROOT/status.json"
  fi
}
trap finish EXIT
set_phase() {
  PHASE="$1"
  printf '{"status":"running","phase":"%s","gpu":%s}\n' "$PHASE" "$GPU" > "$RESULT_ROOT/status.json"
  echo "[$(date --iso-8601=seconds)] $PHASE"
}
run_logged() {
  local log_file="$1"
  shift
  printf '%q ' "$@" >> "$RESULT_ROOT/commands.sh"
  printf '\n' >> "$RESULT_ROOT/commands.sh"
  "$@" 2>&1 | tee "$log_file"
}

set_phase preparation
"$PYTHON" - <<'PY'
import hashlib
import json
import os
from pathlib import Path
import shutil
import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required; refusing to run the experiment on CPU")
root = Path(os.environ['ROOT_DIR'])
output = Path(os.environ['RESULT_ROOT'])
reference = json.loads((Path(os.environ['REFERENCE_ROOT']) / 'data_audit.json').read_text())
def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()
inputs = {reference['train_csv']: reference['train_csv_sha256']}
for entry in reference['evaluation'].values():
    inputs[entry['csv']] = entry['csv_sha256']
entry = reference['independent_evaluation']
inputs[entry['csv']] = entry['csv_sha256']
for path, expected in inputs.items():
    if digest(path) != expected:
        raise RuntimeError(f'Data differs from manuscript SSL experiment: {path}')
train_path = Path(reference['train_csv'])
prepared_train = output / 'prepared_train'
prepared_train.mkdir()
shutil.copy2(train_path, prepared_train / train_path.name)
(prepared_train / 'csv').symlink_to((train_path.parent / 'csv').resolve(), target_is_directory=True)
shutil.copytree(Path(entry['csv']).parent, output / 'prepared_eval', symlinks=True)
code_source = root / 'PatchTST_self_supervised'
code_hashes = {}
for source in code_source.rglob('*.py'):
    destination = output / 'code/PatchTST_self_supervised' / source.relative_to(code_source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    code_hashes[str(destination.relative_to(output))] = digest(destination)
shutil.copy2(root / 'scripts/run_tokenfusion_byol.sh', output / 'run_tokenfusion_byol.sh')
normalization_source = root / 'saved_models/tokenfusion_ssl' / Path(os.environ['REFERENCE_ROOT']).name / 'tokenfusion_ssl_jepa_seed42/tokenfusion_ssl_jepa_seed42_normalization.json'
shutil.copy2(normalization_source, output / 'reference_normalization.json')
manifest = {
    'group': os.environ['GROUP'], 'method': 'byol', 'seed': 42,
    'reference_results': os.environ['REFERENCE_ROOT'], 'source_csv_hashes': inputs,
    'reference_normalization_sha256': digest(normalization_source),
    'code_hashes': code_hashes, 'checkpoints': os.environ['SAVE_DIR'],
    'gpu': os.environ['CUDA_VISIBLE_DEVICES'], 'wandb_mode': os.environ['WANDB_MODE'],
    'reference_paper': 'https://arxiv.org/abs/2006.07733',
    'adaptations': ['Raw Token-Fusion encoder', 'F/T augmentations shared with VICReg',
                    'AdamW and existing SSL training/validation/linear-probe protocol'],
    'evaluation': 'best and last encoders; frozen 768-dimensional representations; single linear head per task',
}
(output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(f"CUDA: {torch.cuda.get_device_name(0)}; matched manuscript train/eval CSV hashes")
PY

CODE="$RESULT_ROOT/code/PatchTST_self_supervised"
TRAIN_DIR="$RESULT_ROOT/prepared_train"
TRAIN_CSV="train_08142026_terrain_floor_wobble_uniform_descent_contact_clean_slip5mms_balanced.csv"
EVAL_DIR="$RESULT_ROOT/prepared_eval"
EVAL_CSV="eval_08142026_terrain_independent_balanced.csv"
EVAL_PATTERN='eval_08142026_terrain_{target}_balanced.csv'
run_logged "$RESULT_ROOT/logs/audit.log" "$PYTHON" -u "$CODE/audit_classification_data.py" \
  --train_csv "$TRAIN_DIR/$TRAIN_CSV" --eval_dir "$EVAL_DIR" \
  --eval_pattern "$EVAL_PATTERN" --independent_eval_csv "$EVAL_CSV" --output_dir "$RESULT_ROOT"

set_phase pretraining
run_logged "$RESULT_ROOT/logs/train.log" "$PYTHON" -u "$CODE/token_fusion_force_ssl.py" \
  --method byol --data_dir "$TRAIN_DIR" --index_csv "$TRAIN_CSV" \
  --context_points 80 --patch_len 10 --stride 10 --n_layers 3 --n_heads 16 \
  --d_model 128 --d_ff 768 --dropout 0.2 --head_dropout 0.2 \
  --token_fusion_layers 1 --token_fusion_heads 8 --token_fusion_ff_mult 4 --out_embed_dim 768 \
  --epochs 400 --batch_size 512 --peak_lr 2e-3 --min_lr 1e-5 --warmup_epochs 10 \
  --weight_decay 1e-4 --early_stopping_patience 75 --early_stopping_min_delta 1e-5 \
  --val_from_train_pct 0.10 --mask_ratio 0.4 --zero_force_eps 1e-8 --num_workers 4 \
  --normalization_reference "$RESULT_ROOT/reference_normalization.json" \
  --byol_momentum 0.996 --byol_projection_dim 256 --byol_hidden_dim 4096 \
  --seed 42 --save_dir "$SAVE_DIR" --run_name "$RUN_NAME" \
  --wandb_mode "$WANDB_MODE" --wandb_project patchtst_tokenfusion_ssl --wandb_group "$GROUP"

for kind in best last; do
  set_phase "linear_probe_$kind"
  run_logged "$RESULT_ROOT/logs/linear_probe_$kind.log" "$PYTHON" -u "$CODE/cnn_ssl_linear_probe.py" \
    --checkpoint "$SAVE_DIR/${RUN_NAME}_${kind}.pth" \
    --train_dir "$TRAIN_DIR" --train_csv "$TRAIN_CSV" \
    --eval_dir "$EVAL_DIR" --eval_csv "$EVAL_CSV" \
    --balanced_eval --balanced_eval_pattern "$EVAL_PATTERN" \
    --targets action weight interaction --label_taxonomy terrain0814_eval --class_weight balanced \
    --epochs 200 --batch_size 512 --embedding_batch_size 512 --learning_rate 1e-2 \
    --weight_decay 0 --patience 30 --num_workers 4 --zero_force_eps 1e-8 --seed 42 \
    --output_dir "$RESULT_ROOT/linear_probes/$kind"
done

set_phase summarize
"$PYTHON" - <<'PY'
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

root = Path(os.environ['RESULT_ROOT'])
reference = Path(os.environ['REFERENCE_ROOT'])
targets = ('action', 'weight', 'interaction')
summaries = {}
artifacts = ['# BYOL evaluation artifacts', '', 'Frozen-encoder linear probes; no text prototypes.', '']
for kind in ('best', 'last'):
    directory = root / 'linear_probes' / kind
    summary = json.loads((directory / 'linear_probe_summary.json').read_text())
    assert summary['method'] == 'byol' and summary['encoder_trainable_parameters'] == 0
    for target in targets:
        predictions = pd.read_csv(directory / f'{target}_pred.csv')
        expected = pd.read_csv(root / 'prepared_eval' / f'eval_08142026_terrain_{target}_balanced.csv')
        keys = ['csv_path', 'start', 'true']
        actual_rows = predictions[keys].sort_values(keys).reset_index(drop=True)
        expected_rows = expected[['csv_path', 'start', target]].rename(columns={target: 'true'}).sort_values(keys).reset_index(drop=True)
        pd.testing.assert_frame_equal(actual_rows, expected_rows, check_dtype=False)
        metrics = summary['metrics'][target]
        assert np.isclose(metrics['accuracy'], accuracy_score(predictions['true'], predictions['pred']))
        assert np.isclose(metrics['macro_f1'], f1_score(predictions['true'], predictions['pred'], average='macro'))
        for suffix in ('umap.png', 'confusion_matrix.png', 'pred.csv'):
            path = directory / f'{target}_{suffix}'
            assert path.is_file() and path.stat().st_size > 0, path
            artifacts.append(f'- {kind} / {target}: [{path.name}]({path})')
    summaries[f'BYOL ({kind})'] = summary
for method in ('masked_reconstruction', 'vae', 'vicreg'):
    summaries[f'{method} (reference best, seed 42)'] = json.loads((reference / 'linear_probes' / method / 'linear_probe_summary.json').read_text())
lines = ['# BYOL and manuscript SSL baselines', '',
         'Single seed (42). Accuracy and Macro F1 are percentages. BYOL best/last refer to encoder checkpoints; each linear head is validation-selected.',
         'Existing JEPA is excluded from this comparison because its implementation does not match a canonical JEPA baseline. Original results are preserved.', '',
         '| Method | Metric | Action | Weight | Interaction |', '| --- | --- | ---: | ---: | ---: |']
for name, summary in summaries.items():
    for metric, label in (('accuracy', 'Accuracy'), ('macro_f1', 'Macro F1')):
        cells = [f"{100 * summary['metrics'][target][metric]:.2f}" for target in targets]
        lines.append(f'| {name} | {label} | ' + ' | '.join(cells) + ' |')
(root / 'summary.md').write_text('\n'.join(lines) + '\n')
(root / 'summary.json').write_text(json.dumps(summaries, indent=2) + '\n')
(root / 'ARTIFACTS.md').write_text('\n'.join(artifacts) + '\n')
(root / 'status.json').write_text(json.dumps({'status': 'complete', 'manuscript_updated': False}) + '\n')
print(root / 'summary.md')
print(root / 'ARTIFACTS.md')
PY
echo "Completed: $RESULT_ROOT"
