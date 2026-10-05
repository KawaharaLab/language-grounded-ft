#!/usr/bin/env bash
FT_WORKSPACE_ROOT="${FT_WORKSPACE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
FT_ARTIFACT_ROOT="${FT_ARTIFACT_ROOT:-${FT_WORKSPACE_ROOT}/artifacts}"
set -euo pipefail

PATCHTST_ROOT="${FT_WORKSPACE_ROOT}/PatchTST"
SOURCE_RUN="${1:-$PATCHTST_ROOT/results/terminal_insertion/terminal_insertion_enter_minus_030s_20260827}"
RUN_DIR="${2:-$PATCHTST_ROOT/results/terminal_insertion_adaptation_efficiency/terminal_insertion_adaptation_efficiency_enter_minus030s_20260827_150802}"
RUN_ID="$(basename "$RUN_DIR")"
DATASET="$SOURCE_RUN/dataset/terminal_insertion_windows.npz"
MANIFEST="$SOURCE_RUN/dataset/dataset_manifest.json"
TRAIN="$PATCHTST_ROOT/PatchTST_self_supervised/train_terminal_insertion_data_efficiency.py"
SUMMARY="$PATCHTST_ROOT/PatchTST_self_supervised/summarize_terminal_insertion_adaptation_efficiency.py"

CONTRASTIVE="$PATCHTST_ROOT/saved_models/terrain_contrastive/terrain_floor_wobble_uniform_descent_contact_clean_slip5mms_20260824_132352/terrain_floor_wobble_uniform_descent_slip5mms_tokenfusion_seed42/terrain_floor_wobble_uniform_descent_slip5mms_tokenfusion_seed42_best.pth"
SSL_ROOT="$PATCHTST_ROOT/saved_models/tokenfusion_ssl/tokenfusion_supervised_ssl_terrain0814_uniform_descent_slip5mms_20260825_015515"
MASKED="$SSL_ROOT/tokenfusion_ssl_masked_reconstruction_seed42/tokenfusion_ssl_masked_reconstruction_seed42_best.pth"
JEPA="$SSL_ROOT/tokenfusion_ssl_jepa_seed42/tokenfusion_ssl_jepa_seed42_best.pth"
VAE="$SSL_ROOT/tokenfusion_ssl_vae_seed42/tokenfusion_ssl_vae_seed42_best.pth"
VICREG="$SSL_ROOT/tokenfusion_ssl_vicreg_seed42/tokenfusion_ssl_vicreg_seed42_best.pth"

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/pretrained_frozen"
for required in "$DATASET" "$MANIFEST" "$TRAIN" "$SUMMARY" "$CONTRASTIVE" "$MASKED" "$JEPA" "$VAE" "$VICREG"; do
  if [[ ! -f "$required" ]]; then
    echo "missing required file: $required" >&2
    exit 1
  fi
done

run_model() {
  local gpu="$1"
  local model="$2"
  local checkpoint="$3"
  local output="$RUN_DIR/pretrained_frozen/$model"
  local log="$RUN_DIR/logs/pretrained_frozen_${model}.log"
  if [[ -f "$output/results.csv" ]] && [[ "$(tail -n +2 "$output/results.csv" | wc -l)" -eq 35 ]]; then
    echo "[$(date --iso-8601=seconds)] skip completed model=$model" | tee -a "$RUN_DIR/logs/pretrained_frozen_orchestrator.log"
    return
  fi
  echo "[$(date --iso-8601=seconds)] gpu=$gpu start model=$model" | tee -a "$RUN_DIR/logs/pretrained_frozen_orchestrator.log"
  CUDA_VISIBLE_DEVICES="$gpu" "$PATCHTST_ROOT/.venv/bin/python" "$TRAIN" \
    --dataset "$DATASET" \
    --manifest "$MANIFEST" \
    --model "$model" \
    --checkpoint "$checkpoint" \
    --output-dir "$output" \
    --episode-counts 1,2,5,10,25,50,100 \
    --seeds 42,43,44,45,46 \
    --epochs 300 \
    --patience 30 \
    --batch-size 256 \
    --num-workers 4 \
    --device cuda \
    --fixed-threshold 0.5 \
    --wandb-project terminal-insertion-adaptation-efficiency \
    --wandb-group "$RUN_ID" \
    --wandb-mode online >"$log" 2>&1
  echo "[$(date --iso-8601=seconds)] gpu=$gpu done model=$model" | tee -a "$RUN_DIR/logs/pretrained_frozen_orchestrator.log"
}

lane_zero() {
  run_model 0 contrastive "$CONTRASTIVE"
  run_model 0 masked_reconstruction "$MASKED"
  run_model 0 vae "$VAE"
}

lane_one() {
  run_model 1 jepa "$JEPA"
  run_model 1 vicreg "$VICREG"
}

lane_zero &
PID_ZERO=$!
lane_one &
PID_ONE=$!
STATUS=0
wait "$PID_ZERO" || STATUS=1
wait "$PID_ONE" || STATUS=1
if [[ "$STATUS" -ne 0 ]]; then
  echo "one or more pretrained frozen-probe runs failed; inspect $RUN_DIR/logs" >&2
  exit "$STATUS"
fi

"$PATCHTST_ROOT/.venv/bin/python" "$SUMMARY" --run-dir "$RUN_DIR" \
  | tee "$RUN_DIR/logs/summary_with_all_pretraining_models.log"
