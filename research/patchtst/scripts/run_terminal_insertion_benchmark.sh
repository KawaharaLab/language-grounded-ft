#!/usr/bin/env bash
FT_WORKSPACE_ROOT="${FT_WORKSPACE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
FT_ARTIFACT_ROOT="${FT_ARTIFACT_ROOT:-${FT_WORKSPACE_ROOT}/artifacts}"
set -euo pipefail

PATCHTST_ROOT="${FT_WORKSPACE_ROOT}/PatchTST"
RUN_ID="${1:?usage: run_terminal_insertion_benchmark.sh RUN_ID [RUN_DIR]}"
RUN_DIR="${2:-$PATCHTST_ROOT/results/terminal_insertion/$RUN_ID}"
DATASET="$RUN_DIR/dataset/terminal_insertion_windows.npz"
MANIFEST="$RUN_DIR/dataset/dataset_manifest.json"
PROJECT="terminal-insertion-linear-probe"
TRAIN="$PATCHTST_ROOT/PatchTST_self_supervised/train_terminal_insertion_probe.py"

CONTRASTIVE="$PATCHTST_ROOT/saved_models/terrain_contrastive/terrain_floor_wobble_uniform_descent_contact_clean_slip5mms_20260824_132352/terrain_floor_wobble_uniform_descent_slip5mms_tokenfusion_seed42/terrain_floor_wobble_uniform_descent_slip5mms_tokenfusion_seed42_best.pth"
SSL_ROOT="$PATCHTST_ROOT/saved_models/tokenfusion_ssl/tokenfusion_supervised_ssl_terrain0814_uniform_descent_slip5mms_20260825_015515"
MASKED="$SSL_ROOT/tokenfusion_ssl_masked_reconstruction_seed42/tokenfusion_ssl_masked_reconstruction_seed42_best.pth"
JEPA="$SSL_ROOT/tokenfusion_ssl_jepa_seed42/tokenfusion_ssl_jepa_seed42_best.pth"
VAE="$SSL_ROOT/tokenfusion_ssl_vae_seed42/tokenfusion_ssl_vae_seed42_best.pth"
VICREG="$SSL_ROOT/tokenfusion_ssl_vicreg_seed42/tokenfusion_ssl_vicreg_seed42_best.pth"

mkdir -p "$RUN_DIR/logs"
source "$PATCHTST_ROOT/.venv/bin/activate"

for required in "$DATASET" "$MANIFEST" "$CONTRASTIVE" "$MASKED" "$JEPA" "$VAE" "$VICREG"; do
  if [[ ! -f "$required" ]]; then
    echo "missing required file: $required" >&2
    exit 1
  fi
done

run_model() {
  local gpu="$1"
  local model="$2"
  local checkpoint="${3:-}"
  local output="$RUN_DIR/$model"
  local log="$RUN_DIR/logs/$model.log"
  local command=(
    python "$TRAIN"
    --dataset "$DATASET"
    --manifest "$MANIFEST"
    --model "$model"
    --output-dir "$output"
    --epochs 300
    --patience 30
    --batch-size 256
    --num-workers 4
    --seed 42
    --device cuda
    --wandb-project "$PROJECT"
    --wandb-group "$RUN_ID"
    --wandb-mode online
  )
  if [[ -n "$checkpoint" ]]; then
    command+=(--checkpoint "$checkpoint")
  fi
  echo "[$(date --iso-8601=seconds)] gpu=$gpu start model=$model" | tee -a "$RUN_DIR/logs/orchestrator.log"
  CUDA_VISIBLE_DEVICES="$gpu" "${command[@]}" >"$log" 2>&1
  echo "[$(date --iso-8601=seconds)] gpu=$gpu done model=$model" | tee -a "$RUN_DIR/logs/orchestrator.log"
}

lane_zero() {
  run_model 0 contrastive "$CONTRASTIVE"
  run_model 0 masked_reconstruction "$MASKED"
  run_model 0 vae "$VAE"
  run_model 0 scratch_mlp
}

lane_one() {
  run_model 1 jepa "$JEPA"
  run_model 1 vicreg "$VICREG"
  run_model 1 scratch_tokenfusion
}

lane_zero &
PID_ZERO=$!
lane_one &
PID_ONE=$!
STATUS=0
wait "$PID_ZERO" || STATUS=$?
wait "$PID_ONE" || STATUS=$?
if [[ "$STATUS" -ne 0 ]]; then
  echo "one or more training lanes failed; inspect $RUN_DIR/logs" >&2
  exit "$STATUS"
fi

python "$PATCHTST_ROOT/PatchTST_self_supervised/summarize_terminal_insertion_benchmark.py" \
  --run-dir "$RUN_DIR" | tee "$RUN_DIR/logs/summary.log"
echo "[$(date --iso-8601=seconds)] benchmark complete: $RUN_DIR" | tee -a "$RUN_DIR/logs/orchestrator.log"
