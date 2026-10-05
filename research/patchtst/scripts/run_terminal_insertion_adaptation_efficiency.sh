#!/usr/bin/env bash
FT_WORKSPACE_ROOT="${FT_WORKSPACE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
FT_ARTIFACT_ROOT="${FT_ARTIFACT_ROOT:-${FT_WORKSPACE_ROOT}/artifacts}"
set -euo pipefail

PATCHTST_ROOT="${FT_WORKSPACE_ROOT}/PatchTST"
SOURCE_RUN="${1:-$PATCHTST_ROOT/results/terminal_insertion/terminal_insertion_enter_minus_030s_20260827}"
RUN_ID="${2:?usage: run_terminal_insertion_adaptation_efficiency.sh SOURCE_RUN RUN_ID [RUN_DIR]}"
RUN_DIR="${3:-$PATCHTST_ROOT/results/terminal_insertion_adaptation_efficiency/$RUN_ID}"
DATASET="$SOURCE_RUN/dataset/terminal_insertion_windows.npz"
MANIFEST="$SOURCE_RUN/dataset/dataset_manifest.json"
TRAIN="$PATCHTST_ROOT/PatchTST_self_supervised/train_terminal_insertion_probe.py"
SUMMARY="$PATCHTST_ROOT/PatchTST_self_supervised/summarize_terminal_insertion_adaptation_efficiency.py"
CHECKPOINT="$PATCHTST_ROOT/saved_models/terrain_contrastive/terrain_floor_wobble_uniform_descent_contact_clean_slip5mms_20260824_132352/terrain_floor_wobble_uniform_descent_slip5mms_tokenfusion_seed42/terrain_floor_wobble_uniform_descent_slip5mms_tokenfusion_seed42_best.pth"

EPISODE_COUNTS=(1 2 5 10 25 50 100)
mkdir -p "$RUN_DIR/logs"

for required in "$DATASET" "$MANIFEST" "$TRAIN" "$SUMMARY" "$CHECKPOINT"; do
  if [[ ! -f "$required" ]]; then
    echo "missing required file: $required" >&2
    exit 1
  fi
done

run_condition() {
  local gpu="$1"
  local condition="$2"
  local seed="$3"
  local episode_count="$4"
  local output="$RUN_DIR/$condition/seed_${seed}/n_$(printf '%03d' "$episode_count")"
  local log="$RUN_DIR/logs/${condition}_seed${seed}_n$(printf '%03d' "$episode_count").log"
  if [[ -f "$output/metrics.json" && -f "$output/history.csv" ]]; then
    echo "[$(date --iso-8601=seconds)] skip completed $condition seed=$seed n=$episode_count" | tee -a "$RUN_DIR/logs/orchestrator.log"
    return
  fi
  local patience=30
  if [[ "$episode_count" -eq 100 && "$seed" -eq 42 ]]; then
    # Keep the representative seed-42 full-data run to 300 epochs so all four
    # gradient-step curves have the same support. Other seeds use early stopping
    # and estimate the data-efficiency endpoint variance.
    patience=0
  fi
  local command=(
    "$PATCHTST_ROOT/.venv/bin/python" "$TRAIN"
    --dataset "$DATASET"
    --manifest "$MANIFEST"
    --fixed-threshold 0.5
    --train-episode-count "$episode_count"
    --output-dir "$output"
    --epochs 300
    --patience "$patience"
    --batch-size 256
    --num-workers 4
    --seed "$seed"
    --device cuda
    --wandb-project terminal-insertion-adaptation-efficiency
    --wandb-group "$RUN_ID"
    --wandb-mode disabled
  )
  case "$condition" in
    frozen_linear)
      command+=(--model contrastive --checkpoint "$CHECKPOINT" --pretrained-adaptation frozen_linear)
      ;;
    lora_r4)
      command+=(--model contrastive --checkpoint "$CHECKPOINT" --pretrained-adaptation lora --lora-rank 4 --lora-alpha 8)
      ;;
    scratch)
      command+=(--model scratch_tokenfusion)
      ;;
    full_finetune)
      command+=(--model contrastive --checkpoint "$CHECKPOINT" --pretrained-adaptation full_finetune)
      ;;
    *)
      echo "unknown condition: $condition" >&2
      return 2
      ;;
  esac
  echo "[$(date --iso-8601=seconds)] gpu=$gpu start $condition seed=$seed n=$episode_count" | tee -a "$RUN_DIR/logs/orchestrator.log"
  CUDA_VISIBLE_DEVICES="$gpu" "${command[@]}" >"$log" 2>&1
  echo "[$(date --iso-8601=seconds)] gpu=$gpu done  $condition seed=$seed n=$episode_count" | tee -a "$RUN_DIR/logs/orchestrator.log"
}

run_seed() {
  local gpu="$1"
  local condition="$2"
  local seed="$3"
  for episode_count in "${EPISODE_COUNTS[@]}"; do
    run_condition "$gpu" "$condition" "$seed" "$episode_count"
  done
}

lane_zero() {
  for seed in 42 43 44 45 46; do run_seed 0 lora_r4 "$seed"; done
  for seed in 42 43; do run_seed 0 scratch "$seed"; done
}

lane_one() {
  for seed in 42 43 44 45 46; do run_seed 1 full_finetune "$seed"; done
  for seed in 44 45 46; do run_seed 1 scratch "$seed"; done
  for seed in 42 43 44 45 46; do run_seed 1 frozen_linear "$seed"; done
}

lane_zero &
PID_ZERO=$!
lane_one &
PID_ONE=$!
STATUS=0
wait "$PID_ZERO" || STATUS=1
wait "$PID_ONE" || STATUS=1
if [[ "$STATUS" -ne 0 ]]; then
  echo "one or more efficiency runs failed; inspect $RUN_DIR/logs" >&2
  exit "$STATUS"
fi

"$PATCHTST_ROOT/.venv/bin/python" "$SUMMARY" --run-dir "$RUN_DIR" | tee "$RUN_DIR/logs/summary.log"
echo "[$(date --iso-8601=seconds)] benchmark complete: $RUN_DIR" | tee -a "$RUN_DIR/logs/orchestrator.log"
