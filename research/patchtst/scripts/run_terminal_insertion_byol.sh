#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MODEL="${MODEL:-byol}"
[[ "$MODEL" == byol || "$MODEL" == supervised ]] || { echo "Unsupported MODEL: $MODEL"; exit 1; }
GROUP="${GROUP:-terminal_insertion_${MODEL}_$(date +%Y%m%d_%H%M%S)}"
[[ "$GROUP" =~ ^[A-Za-z0-9_-]+$ ]] || exit 1
PYTHON="$ROOT/.venv/bin/python"
REFERENCE="$ROOT/results/terminal_insertion_adaptation_efficiency/terminal_insertion_adaptation_efficiency_enter_minus030s_20260827_150802"
CHECKPOINT="$ROOT/saved_models/tokenfusion_ssl/tokenfusion_byol_20260912_194926/tokenfusion_ssl_byol_seed42/tokenfusion_ssl_byol_seed42_best.pth"
MODELS=(contrastive masked_reconstruction vae vicreg byol)
if [[ "$MODEL" == supervised ]]; then
  REFERENCE="$ROOT/results/terminal_insertion_byol/terminal_insertion_byol_20260913_103313"
  CHECKPOINT="$ROOT/saved_models/tokenfusion_supervised/tokenfusion_supervised_ssl_terrain0814_uniform_descent_slip5mms_20260825_015515/tokenfusion_supervised_multitask_seed42/tokenfusion_supervised_multitask_seed42_best.pth"
  MODELS+=(supervised)
fi
SOURCE_DATA="$ROOT/results/terminal_insertion/terminal_insertion_enter_minus_030s_20260827/dataset"
OUTPUT="$ROOT/results/terminal_insertion_${MODEL}/$GROUP"
WANDB_MODE="${WANDB_MODE:-online}"
[[ ! -e "$OUTPUT" ]] || { echo "Output exists; choose a new GROUP: $OUTPUT"; exit 1; }
mkdir -p "$OUTPUT/logs"
exec > >(tee -a "$OUTPUT/logs/pipeline.log") 2>&1
export ROOT REFERENCE CHECKPOINT SOURCE_DATA OUTPUT GROUP WANDB_MODE MODEL
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMBA_NUM_THREADS=4
PHASE=preparation
finish() {
  local code="$?"
  if (( code != 0 )); then
    printf '{"status":"failed","phase":"%s","exit_code":%s}\n' "$PHASE" "$code" > "$OUTPUT/status.json"
  fi
}
trap finish EXIT
set_phase() {
  PHASE="$1"
  printf '{"status":"running","phase":"%s"}\n' "$PHASE" > "$OUTPUT/status.json"
  echo "[$(date --iso-8601=seconds)] $PHASE"
}
set_phase preparation
"$PYTHON" - <<'PY'
import hashlib
import json
import os
from pathlib import Path
import shutil
import torch

root = Path(os.environ['ROOT'])
output = Path(os.environ['OUTPUT'])
reference = Path(os.environ['REFERENCE'])
checkpoint = Path(os.environ['CHECKPOINT'])
model = os.environ['MODEL']
if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
    raise RuntimeError('Two available CUDA GPUs are required')
saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
if model == 'byol':
    if saved['method'] != 'byol' or saved['epoch'] != 8:
        raise RuntimeError('Unexpected BYOL validation-best checkpoint')
else:
    rq1 = json.loads((root / 'results/paper_classification_multiseed/paper_classification_3seed_20260909/manifest.json').read_text())
    rq1_run = next(run for run in rq1['runs'] if run['model'] == 'supervised' and run['seed'] == 42)
    if checkpoint.resolve() != Path(rq1_run['checkpoints']['best']).resolve() or saved['task'] != 'multitask' or saved['model_config']['seed'] != 42:
        raise RuntimeError('Checkpoint does not match the RQ1 supervised seed 42 validation best')
shutil.copytree(Path(os.environ['SOURCE_DATA']), output / 'dataset')
for name in ('scratch', 'lora_r4'):
    (output / name).symlink_to(reference / name, target_is_directory=True)
retained = ['contrastive', 'masked_reconstruction', 'vae', 'vicreg']
if model == 'supervised':
    retained.append('byol')
for method in retained:
    destination = output / 'pretrained_frozen' / method
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(reference / 'pretrained_frozen' / method, target_is_directory=True)
    if method != 'contrastive':
        destination = output / 'pretrained_adaptation' / method
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(reference / 'pretrained_adaptation' / method, target_is_directory=True)
source_root = root / 'PatchTST_self_supervised'
for source in source_root.rglob('*.py'):
    destination = output / 'code/PatchTST_self_supervised' / source.relative_to(source_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
shutil.copy2(root / 'scripts/run_terminal_insertion_byol.sh', output / 'run_terminal_insertion_byol.sh')
shutil.copy2(root / 'scripts/finalize_terminal_insertion_byol.py', output / 'finalize_terminal_insertion_byol.py')
if model == 'supervised':
    shutil.copy2(root / 'scripts/run_terminal_insertion_supervised.sh', output / 'run_terminal_insertion_supervised.sh')
def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()
files = [checkpoint, *sorted((output / 'dataset').glob('*')), *sorted((output / 'code').rglob('*.py')), output / 'finalize_terminal_insertion_byol.py']
manifest = {
    'group': os.environ['GROUP'], 'model': model, 'checkpoint': str(checkpoint), 'pretraining_seed': 42,
    'encoder_readout': '768-dimensional representation; original task heads discarded',
    'checkpoint_epoch': saved['epoch'], 'reference_results': str(reference),
    'seeds': [42, 43, 44, 45, 46], 'episode_counts': [1, 2, 5, 10, 25, 50, 100],
    'adaptations': ['frozen_linear', 'lora'], 'epochs': 300, 'patience': 30,
    'batch_size': 256, 'weight_decay': 1e-4, 'fixed_threshold': 0.5,
    'linear_learning_rate': 2e-3, 'lora_learning_rate': 5e-4, 'lora_rank': 4, 'lora_alpha': 8,
    'input_normalization': 'fixed statistics of all 100 real training episodes',
    'frozen_feature_standardization': 'selected labeled training episodes only, as in reference',
    'sha256': {str(path): digest(path) for path in files if path.is_file()},
    'reference_reuse': 'Other pretraining methods and scratch are linked, never retrained or overwritten',
}
(output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(f'Validated {model} best epoch {saved["epoch"]}; output: {output}')
PY

CODE="$OUTPUT/code/PatchTST_self_supervised"
DATASET="$OUTPUT/dataset/terminal_insertion_windows.npz"
MANIFEST="$OUTPUT/dataset/dataset_manifest.json"
run_frozen() {
  CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u "$CODE/train_terminal_insertion_data_efficiency.py" \
    --dataset "$DATASET" --manifest "$MANIFEST" --model "$MODEL" --checkpoint "$CHECKPOINT" \
    --output-dir "$OUTPUT/pretrained_frozen/$MODEL" --episode-counts 1,2,5,10,25,50,100 \
    --seeds 42,43,44,45,46 --epochs 300 --patience 30 --min-delta 1e-5 \
    --batch-size 256 --learning-rate 2e-3 --weight-decay 1e-4 --num-workers 4 \
    --device cuda --fixed-threshold 0.5 --wandb-project terminal-insertion-adaptation-efficiency \
    --wandb-group "$GROUP" --wandb-mode "$WANDB_MODE" > "$OUTPUT/logs/frozen.log" 2>&1
}
run_lora_lane() {
  local gpu="$1" index=0 seed count output
  for seed in 42 43 44 45 46; do
    for count in 1 2 5 10 25 50 100; do
      if (( index % 2 == gpu )); then
        printf -v output '%s/pretrained_adaptation/%s/lora/seed_%s/n_%03d' "$OUTPUT" "$MODEL" "$seed" "$count"
        echo "[$(date --iso-8601=seconds)] GPU $gpu LoRA seed=$seed N=$count"
        CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u "$CODE/train_terminal_insertion_probe.py" \
          --dataset "$DATASET" --manifest "$MANIFEST" --model "$MODEL" --checkpoint "$CHECKPOINT" \
          --output-dir "$output" --pretrained-adaptation lora --lora-rank 4 --lora-alpha 8 \
          --train-episode-count "$count" --seed "$seed" --epochs 300 --patience 30 \
          --min-delta 1e-5 --batch-size 256 --learning-rate 5e-4 --weight-decay 1e-4 \
          --num-workers 0 --device cuda --fixed-threshold 0.5 \
          --wandb-project terminal-insertion-adaptation-efficiency --wandb-group "$GROUP" \
          --wandb-mode "$WANDB_MODE" > "$OUTPUT/logs/lora_seed${seed}_n${count}.log" 2>&1 || return
      fi
      index=$((index + 1))
    done
  done
}
lane_zero() {
  run_frozen || return
  run_lora_lane 0
}
set_phase training
lane_zero &
PID_ZERO=$!
run_lora_lane 1 &
PID_ONE=$!
STATUS=0
wait "$PID_ZERO" || STATUS=1
wait "$PID_ONE" || STATUS=1
(( STATUS == 0 )) || exit "$STATUS"
set_phase figures
"$PYTHON" "$CODE/summarize_terminal_insertion_pretrained_adaptation.py" \
  --run-dir "$OUTPUT" --pretraining-models "${MODELS[@]}" \
  | tee "$OUTPUT/logs/summary.log"
set_phase diagnostics
CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$OUTPUT/finalize_terminal_insertion_byol.py" --run-dir "$OUTPUT" --model "$MODEL" \
  | tee "$OUTPUT/logs/diagnostics.log"
printf '{"status":"complete","manuscript_updated":false}\n' > "$OUTPUT/status.json"
echo "[$(date --iso-8601=seconds)] Complete: $OUTPUT"
