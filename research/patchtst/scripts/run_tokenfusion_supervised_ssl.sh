#!/usr/bin/env bash
FT_WORKSPACE_ROOT="${FT_WORKSPACE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
FT_ARTIFACT_ROOT="${FT_ARTIFACT_ROOT:-${FT_WORKSPACE_ROOT}/artifacts}"
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SESSION="${SESSION:-tokenfusion_supervised_ssl_terrain0814}"
if [[ -z "${TMUX:-}" && "${AUTO_TMUX:-1}" == "1" ]]; then
  if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux is not installed. Run this script inside tmux."
    exit 1
  fi
  tmux new-session -d -s "$SESSION" "cd '$ROOT_DIR' && AUTO_TMUX=0 bash '$0'"
  echo "Started tmux session: $SESSION"
  echo "Attach with: tmux attach -t $SESSION"
  exit 0
fi

GPUS="${GPUS:-0 1}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-400}"
BATCH_SIZE="${BATCH_SIZE:-512}"
PEAK_LR="${PEAK_LR:-2e-3}"
PATIENCE="${PATIENCE:-75}"
LINEAR_EPOCHS="${LINEAR_EPOCHS:-200}"
LINEAR_LR="${LINEAR_LR:-1e-2}"
LINEAR_PATIENCE="${LINEAR_PATIENCE:-30}"
NUM_WORKERS="${NUM_WORKERS:-4}"
WANDB_MODE="${WANDB_MODE:-disabled}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
LABEL_TAXONOMY="${LABEL_TAXONOMY:-terrain0814_eval}"
CLASS_WEIGHT="${CLASS_WEIGHT:-balanced}"

TRAIN_DIR="${TRAIN_DIR:-${FT_WORKSPACE_ROOT}/Genesis/data/train_08142026}"
TRAIN_CSV="${TRAIN_CSV:-train_08142026_terrain_floor_wobble_uniform_descent_contact_clean_slip5mms_balanced.csv}"
EVAL_DIR="${EVAL_DIR:-$ROOT_DIR/results/terrain_contrastive/terrain_floor_wobble_uniform_descent_contact_clean_slip5mms_20260824_132352/prepared_eval_templates_release_merged}"
EVAL_CSV="${EVAL_CSV:-eval_08142026_terrain_independent_balanced.csv}"
EVAL_PATTERN="${EVAL_PATTERN:-}"
if [[ -z "$EVAL_PATTERN" ]]; then
  EVAL_PATTERN='eval_08142026_terrain_{target}_balanced.csv'
fi
GROUP="${GROUP:-tokenfusion_supervised_ssl_terrain0814_uniform_descent_slip5mms_$(date +%Y%m%d_%H%M%S)}"

LOG_DIR="$ROOT_DIR/logs/tokenfusion_supervised_ssl/$GROUP"
SUPERVISED_ROOT="$ROOT_DIR/saved_models/tokenfusion_supervised/$GROUP"
SSL_ROOT="$ROOT_DIR/saved_models/tokenfusion_ssl/$GROUP"
RESULT_ROOT="$ROOT_DIR/results/tokenfusion_supervised_ssl/$GROUP"
mkdir -p "$LOG_DIR" "$SUPERVISED_ROOT" "$SSL_ROOT" "$RESULT_ROOT"

uv run PatchTST_self_supervised/audit_classification_data.py \
  --train_csv "$TRAIN_DIR/$TRAIN_CSV" \
  --eval_dir "$EVAL_DIR" \
  --eval_pattern "$EVAL_PATTERN" \
  --independent_eval_csv "$EVAL_CSV" \
  --output_dir "$RESULT_ROOT"

read -r -a GPU_IDS <<< "$GPUS"
if (( ${#GPU_IDS[@]} != 2 )); then
  echo 'Exactly two GPUs are required. Example: GPUS="0 1"'
  exit 1
fi

ARCH_ARGS=(
  --patch_len 10
  --stride 10
  --n_layers 3
  --n_heads 16
  --d_model 128
  --d_ff 768
  --dropout 0.2
  --head_dropout 0.2
  --token_fusion_layers 1
  --token_fusion_heads 8
  --token_fusion_ff_mult 4
  --out_embed_dim 768
)

run_supervised() {
  local gpu="$1"
  local task="multitask"
  local run_name="tokenfusion_supervised_${task}_seed${SEED}"
  local output_dir="$SUPERVISED_ROOT/$run_name"
  local checkpoint="$output_dir/${run_name}_best.pth"
  local log_file="$LOG_DIR/${run_name}.gpu${gpu}.log"
  if [[ "$SKIP_EXISTING" == "1" && -f "$checkpoint" && -f "$output_dir/summary.json" ]]; then
    echo "SKIP $run_name" | tee "$log_file"
  else
    local wandb_args=()
    if [[ "$WANDB_MODE" == "disabled" ]]; then
      wandb_args+=(--no_wandb)
    fi
    CUDA_VISIBLE_DEVICES="$gpu" uv run PatchTST_self_supervised/patchtst_supervised.py \
      --task "$task" \
      --train_dir "$TRAIN_DIR" \
      --train_csv "$TRAIN_CSV" \
      --eval_dir "$EVAL_DIR" \
      --eval_csv "$EVAL_CSV" \
      --epochs "$EPOCHS" \
      --batch_size "$BATCH_SIZE" \
      --peak_lr "$PEAK_LR" \
      --min_lr 1e-5 \
      --warmup_epochs 10 \
      --early_stopping_patience "$PATIENCE" \
      --num_workers "$NUM_WORKERS" \
      --val_from_train_pct 0.10 \
      --zero_force_eps 1e-8 \
      --label_taxonomy "$LABEL_TAXONOMY" \
      --class_weight "$CLASS_WEIGHT" \
      --seed "$SEED" \
      --save_dir "$SUPERVISED_ROOT" \
      --run_name "$run_name" \
      --wandb_project patchtst_tokenfusion_supervised \
      --wandb_group "$GROUP" \
      "${ARCH_ARGS[@]}" \
      "${wandb_args[@]}" 2>&1 | tee "$log_file"
  fi

  for checkpoint_name in best final; do
    local supervised_checkpoint="$output_dir/${run_name}_${checkpoint_name}.pth"
    local supervised_eval_dir="$RESULT_ROOT/supervised_eval/$checkpoint_name"
    if [[ "$SKIP_EXISTING" == "1" && -f "$supervised_eval_dir/balanced_eval_summary.json" ]]; then
      echo "SKIP supervised eval $checkpoint_name" | tee -a "$log_file"
    else
      mkdir -p "$supervised_eval_dir"
      CUDA_VISIBLE_DEVICES="$gpu" uv run PatchTST_self_supervised/evaluate_supervised_balanced.py \
        --checkpoint "$supervised_checkpoint" \
        --eval_dir "$EVAL_DIR" \
        --eval_pattern "$EVAL_PATTERN" \
        --targets action weight interaction \
        --batch_size "$BATCH_SIZE" \
        --num_workers "$NUM_WORKERS" \
        --zero_force_eps 1e-8 \
        --output_dir "$supervised_eval_dir" 2>&1 | tee -a "$log_file"
    fi
  done
}

run_ssl() {
  local gpu="$1"
  local method="$2"
  local run_name="tokenfusion_ssl_${method}_seed${SEED}"
  local save_dir="$SSL_ROOT/$run_name"
  local checkpoint="$save_dir/${run_name}_best.pth"
  local probe_dir="$RESULT_ROOT/linear_probes/$method"
  local log_file="$LOG_DIR/${run_name}.gpu${gpu}.log"
  mkdir -p "$save_dir" "$probe_dir"
  if [[ "$SKIP_EXISTING" != "1" || ! -f "$checkpoint" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" uv run PatchTST_self_supervised/token_fusion_force_ssl.py \
      --method "$method" \
      --data_dir "$TRAIN_DIR" \
      --index_csv "$TRAIN_CSV" \
      --epochs "$EPOCHS" \
      --batch_size "$BATCH_SIZE" \
      --peak_lr "$PEAK_LR" \
      --min_lr 1e-5 \
      --warmup_epochs 10 \
      --early_stopping_patience "$PATIENCE" \
      --num_workers "$NUM_WORKERS" \
      --zero_force_eps 1e-8 \
      --seed "$SEED" \
      --save_dir "$save_dir" \
      --run_name "$run_name" \
      --wandb_mode "$WANDB_MODE" \
      --wandb_project patchtst_tokenfusion_ssl \
      --wandb_group "$GROUP" \
      "${ARCH_ARGS[@]}" 2>&1 | tee "$log_file"
  else
    echo "SKIP SSL $run_name" | tee "$log_file"
  fi
  if [[ "$SKIP_EXISTING" != "1" || ! -f "$probe_dir/linear_probe_summary.json" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" uv run PatchTST_self_supervised/cnn_ssl_linear_probe.py \
      --checkpoint "$checkpoint" \
      --train_dir "$TRAIN_DIR" \
      --train_csv "$TRAIN_CSV" \
      --eval_dir "$EVAL_DIR" \
      --eval_csv "$EVAL_CSV" \
      --balanced_eval \
      --balanced_eval_pattern "$EVAL_PATTERN" \
      --targets action weight interaction \
      --label_taxonomy "$LABEL_TAXONOMY" \
      --epochs "$LINEAR_EPOCHS" \
      --batch_size "$BATCH_SIZE" \
      --embedding_batch_size "$BATCH_SIZE" \
      --learning_rate "$LINEAR_LR" \
      --class_weight "$CLASS_WEIGHT" \
      --patience "$LINEAR_PATIENCE" \
      --num_workers "$NUM_WORKERS" \
      --seed "$SEED" \
      --output_dir "$probe_dir" 2>&1 | tee -a "$log_file"
  fi
}

worker_zero() {
  run_supervised "${GPU_IDS[0]}"
  run_ssl "${GPU_IDS[0]}" masked_reconstruction
  run_ssl "${GPU_IDS[0]}" vae
}

worker_one() {
  run_ssl "${GPU_IDS[1]}" jepa
  run_ssl "${GPU_IDS[1]}" vicreg
}

echo "group=$GROUP gpus=${GPU_IDS[*]} architecture=normalized_raw_token_fusion taxonomy=$LABEL_TAXONOMY class_weight=$CLASS_WEIGHT"
worker_zero &
pid_zero=$!
worker_one &
pid_one=$!
failed=0
wait "$pid_zero" || failed=1
wait "$pid_one" || failed=1
if (( failed )); then
  echo "At least one worker failed. Check $LOG_DIR"
  exit 1
fi

uv run scripts/summarize_ssl_linear_probes.py \
  --root "$RESULT_ROOT/linear_probes" \
  --output "$RESULT_ROOT/linear_probe_comparison.md"

echo "Completed: $GROUP"
echo "Supervised: $SUPERVISED_ROOT"
echo "SSL checkpoints: $SSL_ROOT"
echo "Linear probes: $RESULT_ROOT/linear_probe_comparison.md"
