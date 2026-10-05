#!/usr/bin/env bash
FT_WORKSPACE_ROOT="${FT_WORKSPACE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
FT_ARTIFACT_ROOT="${FT_ARTIFACT_ROOT:-${FT_WORKSPACE_ROOT}/artifacts}"
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

GROUP="${GROUP:-paper_classification_3seed_$(date +%Y%m%d_%H%M%S)}"
SESSION="${SESSION:-$GROUP}"
GPUS="${GPUS:-0 1}"
WANDB_MODE="${WANDB_MODE:-online}"
MANUSCRIPT="${MANUSCRIPT:-${FT_WORKSPACE_ROOT}/Genesis/paper.tex}"
export GROUP SESSION GPUS WANDB_MODE MANUSCRIPT

if [[ -z "${TMUX:-}" && "${AUTO_TMUX:-1}" == 1 ]]; then
  command -v tmux >/dev/null
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Session already exists: $SESSION"
    exit 1
  fi
  printf -v launch 'cd %q && AUTO_TMUX=0 GROUP=%q SESSION=%q GPUS=%q WANDB_MODE=%q MANUSCRIPT=%q bash %q' \
    "$ROOT_DIR" "$GROUP" "$SESSION" "$GPUS" "$WANDB_MODE" "$MANUSCRIPT" "$ROOT_DIR/scripts/run_paper_classification_multiseed.sh"
  tmux new-session -d -s "$SESSION" "$launch"
  echo "Started: $SESSION"
  echo "Attach: tmux attach -t $SESSION"
  echo "Results: $ROOT_DIR/results/paper_classification_multiseed/$GROUP"
  exit 0
fi

RESULT_DIR="$ROOT_DIR/results/paper_classification_multiseed/$GROUP"
mkdir -p "$RESULT_DIR"
read -r -a GPU_IDS <<< "$GPUS"
uv run python -u scripts/paper_classification_multiseed.py run \
  --group "$GROUP" --gpus "${GPU_IDS[@]}" --manuscript "$MANUSCRIPT" \
  2>&1 | tee -a "$RESULT_DIR/runner.log"
