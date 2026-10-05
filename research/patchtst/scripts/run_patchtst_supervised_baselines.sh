#!/usr/bin/env bash
FT_WORKSPACE_ROOT="${FT_WORKSPACE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
FT_ARTIFACT_ROOT="${FT_ARTIFACT_ROOT:-${FT_WORKSPACE_ROOT}/artifacts}"
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SESSION="${SESSION:-patchtst_supervised_baselines}"
if [[ -z "${TMUX:-}" && "${AUTO_TMUX:-1}" == "1" ]]; then
  if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux is not installed. Run this script inside tmux, or install tmux."
    exit 1
  fi
  tmux new-session -d -s "$SESSION" "cd '$ROOT_DIR' && AUTO_TMUX=0 bash '$0'"
  echo "Started tmux session: $SESSION"
  echo "Attach with: tmux attach -t $SESSION"
  exit 0
fi

EPOCHS="${EPOCHS:-300}"
BATCH_SIZE="${BATCH_SIZE:-512}"
PEAK_LR="${PEAK_LR:-1e-3}"
MIN_LR="${MIN_LR:-1e-5}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-10}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.0}"
CLASS_WEIGHT="${CLASS_WEIGHT:-none}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-40}"
NUM_WORKERS="${NUM_WORKERS:-4}"
VAL_FROM_TRAIN_PCT="${VAL_FROM_TRAIN_PCT:-0.10}"
ZERO_FORCE_EPS="${ZERO_FORCE_EPS:-1e-8}"
SEED="${SEED:-42}"
GPU="${GPU:-0}"
GROUP="${GROUP:-patchtst_supervised_$(date +%Y%m%d_%H%M%S)}"

TRAIN_DIR="${TRAIN_DIR:-${FT_WORKSPACE_ROOT}/Genesis/data/train_04272026/}"
TRAIN_CSV="${TRAIN_CSV:-train_04272026_thin_15pct.csv}"
EVAL_DIR="${EVAL_DIR:-${FT_WORKSPACE_ROOT}/Genesis/data/eval_04272026/}"
EVAL_CSV="${EVAL_CSV:-eval_04272026_thin_15pct.csv}"

LOG_DIR="${LOG_DIR:-logs/patchtst_supervised/${GROUP}}"
SAVE_ROOT="${SAVE_ROOT:-saved_models/patchtst_supervised/${GROUP}}"
mkdir -p "$LOG_DIR" "$SAVE_ROOT"

COMMON_ARGS=(
  --train_dir "$TRAIN_DIR"
  --train_csv "$TRAIN_CSV"
  --eval_dir "$EVAL_DIR"
  --eval_csv "$EVAL_CSV"
  --epochs "$EPOCHS"
  --batch_size "$BATCH_SIZE"
  --peak_lr "$PEAK_LR"
  --min_lr "$MIN_LR"
  --warmup_epochs "$WARMUP_EPOCHS"
  --weight_decay "$WEIGHT_DECAY"
  --label_smoothing "$LABEL_SMOOTHING"
  --class_weight "$CLASS_WEIGHT"
  --early_stopping_patience "$EARLY_STOPPING_PATIENCE"
  --num_workers "$NUM_WORKERS"
  --val_from_train_pct "$VAL_FROM_TRAIN_PCT"
  --zero_force_eps "$ZERO_FORCE_EPS"
  --seed "$SEED"
  --d_model 128
  --d_ff 768
  --n_layers 3
  --n_heads 16
  --dropout 0.2
  --head_dropout 0.2
  --patch_len 10
  --stride 10
  --token_fusion_layers 1
  --token_fusion_heads 8
  --token_fusion_ff_mult 4
  --save_dir "$SAVE_ROOT"
  --wandb_project patchtst_supervised
  --wandb_group "$GROUP"
)

run_one() {
  local gpu_id="$1"
  local task="$2"
  local run_name="supervised_${task}"
  local log_file="$LOG_DIR/${run_name}.gpu${gpu_id}.log"
  local status_file="$LOG_DIR/${run_name}.status"
  local -a cmd=(
    uv run PatchTST_self_supervised/patchtst_supervised.py
    --task "$task"
    --run_name "$run_name"
    "${COMMON_ARGS[@]}"
  )

  echo "===== START task=$task gpu=$gpu_id $(date '+%F %T') =====" | tee "$log_file"
  echo "${cmd[*]}" | tee -a "$log_file"
  if CUDA_VISIBLE_DEVICES="$gpu_id" "${cmd[@]}" 2>&1 | tee -a "$log_file"; then
    echo "===== DONE task=$task gpu=$gpu_id $(date '+%F %T') =====" | tee -a "$log_file"
    echo "ok" > "$status_file"
  else
    echo "===== FAIL task=$task gpu=$gpu_id $(date '+%F %T') =====" | tee -a "$log_file"
    echo "fail" > "$status_file"
    return 1
  fi
}

echo "PatchTST supervised baselines"
echo "group=$GROUP gpu=$GPU epochs=$EPOCHS batch_size=$BATCH_SIZE"
echo "selection=validation Macro F1 class_weight=$CLASS_WEIGHT"
echo "logs=$LOG_DIR"
echo "save_root=$SAVE_ROOT"

failed=0
if ! run_one "$GPU" annotation; then
    failed=1
fi
if ! run_one "$GPU" multitask; then
    failed=1
fi

echo "annotation=$(cat "$LOG_DIR/supervised_annotation.status" 2>/dev/null || echo missing)"
echo "multitask=$(cat "$LOG_DIR/supervised_multitask.status" 2>/dev/null || echo missing)"

if (( failed != 0 )); then
  exit 1
fi
