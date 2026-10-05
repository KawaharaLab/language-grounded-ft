#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
GROUP="${GROUP:-rq1_supervised_cnn_mlp_3seed_$(date +%Y%m%d_%H%M%S)}"
SESSION="${SESSION:-$GROUP}"
GPUS="${GPUS:-0 1}"
WANDB_MODE="${WANDB_MODE:-online}"
export GROUP SESSION GPUS WANDB_MODE
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export NUMBA_NUM_THREADS="${NUMBA_NUM_THREADS:-4}"

if [[ -z "${TMUX:-}" && "${AUTO_TMUX:-1}" == 1 ]]; then
  command -v tmux >/dev/null
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Session already exists: $SESSION"
    exit 1
  fi
  printf -v launch 'cd %q && AUTO_TMUX=0 GROUP=%q GPUS=%q WANDB_MODE=%q bash %q' \
    "$ROOT" "$GROUP" "$GPUS" "$WANDB_MODE" "$ROOT/scripts/run_paper_supervised_baselines.sh"
  tmux new-session -d -s "$SESSION" "$launch"
  echo "Started: $SESSION"
  echo "Attach: tmux attach -t $SESSION"
  echo "Results: $ROOT/results/paper_supervised_baselines/$GROUP"
  exit 0
fi

OUTPUT="$ROOT/results/paper_supervised_baselines/$GROUP"
read -r -a GPU_IDS <<< "$GPUS"
"$ROOT/.venv/bin/python" -u scripts/paper_supervised_baselines.py prepare --group "$GROUP"
"$ROOT/.venv/bin/python" -u "$OUTPUT/code/scripts/paper_supervised_baselines.py" run \
  --group "$GROUP" --gpus "${GPU_IDS[@]}" 2>&1 | tee -a "$OUTPUT/runner.log"
