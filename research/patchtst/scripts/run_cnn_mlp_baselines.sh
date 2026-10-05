#!/usr/bin/env bash
FT_WORKSPACE_ROOT="${FT_WORKSPACE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
FT_ARTIFACT_ROOT="${FT_ARTIFACT_ROOT:-${FT_WORKSPACE_ROOT}/artifacts}"
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SESSION="${SESSION:-cnn_mlp_baselines}"
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

EPOCHS="${EPOCHS:-2000}"
PEAK_LR="${PEAK_LR:-1e-3}"
LOSS_TYPE="${LOSS_TYPE:-infonce}"
BATCH_SIZE="${BATCH_SIZE:-512}"
NUM_WORKERS="${NUM_WORKERS:-4}"
VAL_FROM_TRAIN_PCT="${VAL_FROM_TRAIN_PCT:-0.10}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-30}"
MIN_LR="${MIN_LR:-1e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
SEED="${SEED:-42}"
ZERO_FORCE_EPS="${ZERO_FORCE_EPS:-1e-8}"
DATA_ROOT="${DATA_ROOT:-${FT_WORKSPACE_ROOT}/Genesis/data/}"
INDEX_CSV="${INDEX_CSV:-train_04272026_thin_15pct.csv}"
GROUP="${GROUP:-cnn_mlp_baselines_$(date +%Y%m%d_%H%M%S)}"
GPUS="${GPUS:-0 1}"
PRIORITY_FIRST="${PRIORITY_FIRST:-0}"
EVAL_AFTER="${EVAL_AFTER:-1}"
EVAL_CWD="${EVAL_CWD:-PatchTST_self_supervised}"
EVAL_DIR="${EVAL_DIR:-${FT_WORKSPACE_ROOT}/Genesis/data/eval_04272026/}"
EVAL_CSV="${EVAL_CSV:-eval_04272026_thin_15pct.csv}"
TEXT_TYPE="${TEXT_TYPE:-bert}"

LOG_DIR="${LOG_DIR:-logs/cnn_mlp_baselines/${GROUP}}"
SAVE_ROOT="${SAVE_ROOT:-saved_models/cnn_mlp_baselines/${GROUP}}"
mkdir -p "$LOG_DIR" "$SAVE_ROOT"

read -r -a GPU_IDS <<< "$GPUS"
if (( ${#GPU_IDS[@]} == 0 )); then
  echo "GPUS is empty. Example: GPUS=\"0 1\" bash $0"
  exit 1
fi

COMMON_ARGS=(
  --data_dir "$DATA_ROOT"
  --index_csv "$INDEX_CSV"
  --epochs "$EPOCHS"
  --peak_lr "$PEAK_LR"
  --loss_type "$LOSS_TYPE"
  --batch_size "$BATCH_SIZE"
  --num_workers "$NUM_WORKERS"
  --val_from_train_pct "$VAL_FROM_TRAIN_PCT"
  --warmup_epochs "$WARMUP_EPOCHS"
  --min_lr "$MIN_LR"
  --weight_decay "$WEIGHT_DECAY"
  --seed "$SEED"
  --zero_force_eps "$ZERO_FORCE_EPS"
)

# Format: run_name|train_script|eval_model_arch|extra_train_args
RUN_SPECS=(
  "cnn_standard_normalized|PatchTST_self_supervised/patchtst_cnn.py|cnn|--from_scratch --cnn_base_channels 0 --cnn_depth 4 --cnn_kernel_size 5 --cnn_dropout 0.2"
  "mlp_flat_normalized|PatchTST_self_supervised/patchtst_mlp.py|mlp|--from_scratch --mlp_hidden_dim 0 --mlp_depth 3 --mlp_dropout 0.2"
)

echo "CNN/MLP contrastive baseline sweep"
echo "group=$GROUP epochs=$EPOCHS peak_lr=$PEAK_LR loss_type=$LOSS_TYPE batch_size=$BATCH_SIZE gpus=${GPU_IDS[*]}"
echo "data_root=$DATA_ROOT index_csv=$INDEX_CSV zero_force_eps=$ZERO_FORCE_EPS eval_after=$EVAL_AFTER"
echo "logs=$LOG_DIR"
echo "save_root=$SAVE_ROOT"

find_best_ckpt() {
  local run_save_dir="$1"
  local run_name="$2"
  local expected="$run_save_dir/${run_name}_best.pth"
  if [[ -f "$expected" ]]; then
    echo "$expected"
    return 0
  fi
  find "$run_save_dir" -maxdepth 1 -type f -name '*_best.pth' -print | sort | tail -n 1
}

run_eval() {
  local gpu_id="$1"
  local run_name="$2"
  local model_arch="$3"
  local ckpt_path="$4"
  local log_file="$5"

  if [[ "$EVAL_AFTER" != "1" ]]; then
    return 0
  fi
  if [[ ! -f "$ckpt_path" ]]; then
    echo "[eval] checkpoint not found: $ckpt_path" | tee -a "$log_file"
    return 1
  fi

  echo "===== EVAL  $run_name arch=$model_arch gpu=$gpu_id $(date '+%F %T') =====" | tee -a "$log_file"
  (
    cd "$ROOT_DIR/$EVAL_CWD"
    CUDA_VISIBLE_DEVICES="$gpu_id" uv run patchtst_eval_normalized.py \
      --model_path "$ROOT_DIR/$ckpt_path" \
      --ckpt_type contrastive \
      --model_arch "$model_arch" \
      --eval_dir "$EVAL_DIR" \
      --eval_csv "$EVAL_CSV" \
      --text_type "$TEXT_TYPE" \
      --zero_force_eps "$ZERO_FORCE_EPS" \
      --report_action_accuracy
  ) 2>&1 | tee -a "$log_file"
}

run_one() {
  local gpu_id="$1"
  local run_name="$2"
  local train_script="$3"
  local model_arch="$4"
  local extra_args_str="$5"
  local run_save_dir="$SAVE_ROOT/$run_name"
  local log_file="$LOG_DIR/${run_name}.gpu${gpu_id}.log"
  local status_file="$LOG_DIR/${run_name}.status"
  local -a extra_args
  local -a cmd
  local ckpt_path

  read -r -a extra_args <<< "$extra_args_str"
  mkdir -p "$run_save_dir"
  cmd=(
    uv run "$train_script"
    "${COMMON_ARGS[@]}"
    --run_name "$run_name"
    --save_dir "$run_save_dir"
    "${extra_args[@]}"
  )

  echo
  echo "===== START $run_name arch=$model_arch gpu=$gpu_id $(date '+%F %T') ====="
  echo "${cmd[*]}" | tee "$log_file"
  if CUDA_VISIBLE_DEVICES="$gpu_id" WANDB_RUN_GROUP="$GROUP" WANDB_NAME="$run_name" "${cmd[@]}" 2>&1 | tee -a "$log_file"; then
    echo "===== TRAIN DONE $run_name gpu=$gpu_id $(date '+%F %T') =====" | tee -a "$log_file"
    ckpt_path="$(find_best_ckpt "$run_save_dir" "$run_name")"
    if run_eval "$gpu_id" "$run_name" "$model_arch" "$ckpt_path" "$log_file"; then
      echo "===== DONE  $run_name gpu=$gpu_id $(date '+%F %T') =====" | tee -a "$log_file"
      echo "ok" > "$status_file"
    else
      echo "===== EVAL FAIL  $run_name gpu=$gpu_id $(date '+%F %T') =====" | tee -a "$log_file"
      echo "fail" > "$status_file"
    fi
  else
    echo "===== TRAIN FAIL  $run_name gpu=$gpu_id $(date '+%F %T') =====" | tee -a "$log_file"
    echo "fail" > "$status_file"
  fi
}

run_worker() {
  local worker_id="$1"
  local gpu_id="$2"
  local start_idx="$3"
  local num_gpu_workers="${#GPU_IDS[@]}"
  local idx
  local run_name
  local train_script
  local model_arch
  local extra_args_str

  for idx in "${!RUN_SPECS[@]}"; do
    if (( idx < start_idx )); then
      continue
    fi
    if (( (idx - start_idx) % num_gpu_workers != worker_id )); then
      continue
    fi
    IFS='|' read -r run_name train_script model_arch extra_args_str <<< "${RUN_SPECS[$idx]}"
    run_one "$gpu_id" "$run_name" "$train_script" "$model_arch" "$extra_args_str"
  done
}

start_idx=0
if [[ "$PRIORITY_FIRST" == "1" ]]; then
  IFS='|' read -r first_name first_script first_arch first_args <<< "${RUN_SPECS[0]}"
  run_one "${GPU_IDS[0]}" "$first_name" "$first_script" "$first_arch" "$first_args"
  start_idx=1
fi

pids=()
for worker_id in "${!GPU_IDS[@]}"; do
  run_worker "$worker_id" "${GPU_IDS[$worker_id]}" "$start_idx" &
  pids+=("$!")
done

worker_failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    worker_failed=1
  fi
done

FAILED_RUNS=()
for idx in "${!RUN_SPECS[@]}"; do
  IFS='|' read -r run_name _ <<< "${RUN_SPECS[$idx]}"
  if [[ "$(cat "$LOG_DIR/${run_name}.status" 2>/dev/null || true)" != "ok" ]]; then
    FAILED_RUNS+=("$run_name")
  fi
done

echo
if (( ${#FAILED_RUNS[@]} > 0 )); then
  echo "Completed with failed runs: ${FAILED_RUNS[*]}"
  exit 1
fi
if (( worker_failed != 0 )); then
  echo "One or more GPU workers exited unexpectedly."
  exit 1
fi
echo "All CNN/MLP baseline runs completed."
