#!/usr/bin/env bash
FT_WORKSPACE_ROOT="${FT_WORKSPACE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
FT_ARTIFACT_ROOT="${FT_ARTIFACT_ROOT:-${FT_WORKSPACE_ROOT}/artifacts}"
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

GPUS="${GPUS:-0 1}"
EVAL_DIR="${EVAL_DIR:-${FT_WORKSPACE_ROOT}/Genesis/data/eval_04272026/}"
EVAL_CSV="${EVAL_CSV:-eval_04272026_thin_15pct.csv}"
ZERO_FORCE_EPS="${ZERO_FORCE_EPS:-1e-8}"
RUN_ID="${RUN_ID:-normalized_reeval_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/normalized_reeval/$RUN_ID}"
mkdir -p "$LOG_DIR"

read -r -a GPU_VALUES <<< "$GPUS"
if (( ${#GPU_VALUES[@]} != 2 )); then
  echo 'Exactly two GPUs are required. Example: GPUS="0 1"'
  exit 1
fi

mapfile -t CHECKPOINTS < <(
  find -L \
    "$ROOT_DIR/saved_models/delta_normalized_multiseed" \
    "$ROOT_DIR/saved_models/delta_normalized_lr_sweep" \
    -type f \( -name '*_best.pth' -o -name '*_last.pth' \) \
    -print | sort
)

if (( ${#CHECKPOINTS[@]} == 0 )); then
  echo "No normalized best/last checkpoints found."
  exit 1
fi

printf '%s\n' "${CHECKPOINTS[@]}" > "$LOG_DIR/checkpoints.txt"
echo "Re-evaluating ${#CHECKPOINTS[@]} normalized checkpoints on $EVAL_DIR/$EVAL_CSV"

evaluate_one() {
  local gpu="$1"
  local checkpoint="$2"
  local model_name
  model_name="$(basename "$checkpoint" .pth)"
  local log_file="$LOG_DIR/${model_name}.gpu${gpu}.log"
  echo "===== EVAL $model_name gpu=$gpu $(date '+%F %T') =====" | tee "$log_file"
  (
    cd "$ROOT_DIR/PatchTST_self_supervised"
    CUDA_VISIBLE_DEVICES="$gpu" uv run patchtst_eval_normalized.py \
      --model_path "$checkpoint" \
      --ckpt_type contrastive \
      --model_arch patchtst \
      --arch auto \
      --token_fusion_heads 8 \
      --eval_dir "$EVAL_DIR" \
      --eval_csv "$EVAL_CSV" \
      --text_type bert \
      --zero_force_eps "$ZERO_FORCE_EPS" \
      --report_action_accuracy
  ) 2>&1 | tee -a "$log_file"
  echo "===== DONE $model_name gpu=$gpu $(date '+%F %T') =====" | tee -a "$log_file"
}

worker() {
  local worker_index="$1"
  local gpu="${GPU_VALUES[$worker_index]}"
  local index
  for ((index=worker_index; index<${#CHECKPOINTS[@]}; index+=${#GPU_VALUES[@]})); do
    evaluate_one "$gpu" "${CHECKPOINTS[$index]}"
  done
}

worker 0 &
worker0_pid=$!
worker 1 &
worker1_pid=$!

failed=0
if ! wait "$worker0_pid"; then
  failed=1
fi
if ! wait "$worker1_pid"; then
  failed=1
fi
if (( failed != 0 )); then
  echo "At least one evaluation failed. Check $LOG_DIR"
  exit 1
fi

echo "All normalized checkpoints were re-evaluated. Logs: $LOG_DIR"
