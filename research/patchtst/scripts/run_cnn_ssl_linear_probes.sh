#!/usr/bin/env bash
FT_WORKSPACE_ROOT="${FT_WORKSPACE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
FT_ARTIFACT_ROOT="${FT_ARTIFACT_ROOT:-${FT_WORKSPACE_ROOT}/artifacts}"
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SESSION="${SESSION:-cnn_ssl_linear_probes}"
if [[ -z "${TMUX:-}" && "${AUTO_TMUX:-1}" == "1" ]]; then
  tmux new-session -d -s "$SESSION" "cd '$ROOT_DIR' && AUTO_TMUX=0 bash '$0'"
  echo "Started tmux session: $SESSION"
  echo "Attach with: tmux attach -t $SESSION"
  exit 0
fi

GPU="${GPU:-0}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-512}"
LEARNING_RATE="${LEARNING_RATE:-1e-2}"
PATIENCE="${PATIENCE:-30}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SSL_GROUP="${SSL_GROUP:-cnn_ssl_patchtst_long_20260812_234821}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$ROOT_DIR/saved_models/cnn_ssl/$SSL_GROUP}"
TRAIN_DIR="${TRAIN_DIR:-${FT_WORKSPACE_ROOT}/Genesis/data/train_04272026/}"
TRAIN_CSV="${TRAIN_CSV:-train_04272026_thin_15pct.csv}"
EVAL_DIR="${EVAL_DIR:-${FT_WORKSPACE_ROOT}/Genesis/data/eval_04272026/}"
EVAL_CSV="${EVAL_CSV:-eval_04272026_thin_15pct.csv}"
GROUP="${GROUP:-cnn_ssl_linear_probes_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="$ROOT_DIR/results/cnn_ssl_linear_probes/$GROUP"
LOG_DIR="$ROOT_DIR/logs/cnn_ssl_linear_probes/$GROUP"
mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"

METHODS=(masked_unet jepa vae vicreg)
for method in "${METHODS[@]}"; do
  run_name="cnn_ssl_${method}_seed${SEED}"
  checkpoint="$CHECKPOINT_ROOT/$run_name/${run_name}_best.pth"
  output_dir="$OUTPUT_ROOT/$method"
  log_file="$LOG_DIR/${method}.gpu${GPU}.log"
  if [[ ! -f "$checkpoint" ]]; then
    echo "Checkpoint not found: $checkpoint"
    exit 1
  fi
  echo "===== LINEAR PROBE $method gpu=$GPU $(date '+%F %T') =====" | tee "$log_file"
  CUDA_VISIBLE_DEVICES="$GPU" uv run PatchTST_self_supervised/cnn_ssl_linear_probe.py \
    --checkpoint "$checkpoint" \
    --train_dir "$TRAIN_DIR" \
    --train_csv "$TRAIN_CSV" \
    --eval_dir "$EVAL_DIR" \
    --eval_csv "$EVAL_CSV" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --embedding_batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --patience "$PATIENCE" \
    --num_workers "$NUM_WORKERS" \
    --seed "$SEED" \
    --output_dir "$output_dir" 2>&1 | tee -a "$log_file"
done

uv run scripts/summarize_ssl_linear_probes.py \
  --root "$OUTPUT_ROOT" \
  --output "$OUTPUT_ROOT/linear_probe_comparison.md"

echo "Completed: $OUTPUT_ROOT/linear_probe_comparison.md"
