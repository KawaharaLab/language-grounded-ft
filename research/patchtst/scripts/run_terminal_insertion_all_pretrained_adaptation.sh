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
TRAIN="$PATCHTST_ROOT/PatchTST_self_supervised/train_terminal_insertion_probe.py"
SUMMARY="$PATCHTST_ROOT/PatchTST_self_supervised/summarize_terminal_insertion_pretrained_adaptation.py"
EPISODE_COUNTS=(1 2 5 10 25 50 100)
SEEDS=(42 43 44 45 46)

SSL_ROOT="$PATCHTST_ROOT/saved_models/tokenfusion_ssl/tokenfusion_supervised_ssl_terrain0814_uniform_descent_slip5mms_20260825_015515"
MASKED="$SSL_ROOT/tokenfusion_ssl_masked_reconstruction_seed42/tokenfusion_ssl_masked_reconstruction_seed42_best.pth"
JEPA="$SSL_ROOT/tokenfusion_ssl_jepa_seed42/tokenfusion_ssl_jepa_seed42_best.pth"
VAE="$SSL_ROOT/tokenfusion_ssl_vae_seed42/tokenfusion_ssl_vae_seed42_best.pth"
VICREG="$SSL_ROOT/tokenfusion_ssl_vicreg_seed42/tokenfusion_ssl_vicreg_seed42_best.pth"

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/pretrained_adaptation"
for required in "$DATASET" "$MANIFEST" "$TRAIN" "$SUMMARY" "$MASKED" "$JEPA" "$VAE" "$VICREG"; do
  if [[ ! -f "$required" ]]; then
    echo "missing required file: $required" >&2
    exit 1
  fi
done

run_condition() {
  local gpu="$1"
  local model="$2"
  local adaptation="$3"
  local checkpoint="$4"
  local seed="$5"
  local episode_count="$6"
  local output="$RUN_DIR/pretrained_adaptation/$model/$adaptation/seed_${seed}/n_$(printf '%03d' "$episode_count")"
  local log="$RUN_DIR/logs/${model}_${adaptation}_seed${seed}_n$(printf '%03d' "$episode_count").log"
  if [[ -f "$output/metrics.json" && -f "$output/history.csv" ]]; then
    echo "[$(date --iso-8601=seconds)] skip $model/$adaptation seed=$seed n=$episode_count" >> "$RUN_DIR/logs/pretrained_adaptation_orchestrator.log"
    return
  fi
  local args=(
    "$PATCHTST_ROOT/.venv/bin/python" "$TRAIN"
    --dataset "$DATASET"
    --manifest "$MANIFEST"
    --model "$model"
    --checkpoint "$checkpoint"
    --pretrained-adaptation "$adaptation"
    --train-episode-count "$episode_count"
    --output-dir "$output"
    --epochs 300
    --patience 30
    --batch-size 256
    --num-workers 0
    --seed "$seed"
    --device cuda
    --fixed-threshold 0.5
    --wandb-project terminal-insertion-adaptation-efficiency
    --wandb-group "$RUN_ID"
    --wandb-mode online
  )
  if [[ "$adaptation" == "lora" ]]; then
    args+=(--lora-rank 4 --lora-alpha 8)
  fi
  echo "[$(date --iso-8601=seconds)] gpu=$gpu start $model/$adaptation seed=$seed n=$episode_count" >> "$RUN_DIR/logs/pretrained_adaptation_orchestrator.log"
  CUDA_VISIBLE_DEVICES="$gpu" "${args[@]}" >"$log" 2>&1
  echo "[$(date --iso-8601=seconds)] gpu=$gpu done  $model/$adaptation seed=$seed n=$episode_count" >> "$RUN_DIR/logs/pretrained_adaptation_orchestrator.log"
}

run_task() {
  local gpu="$1"
  local model="$2"
  local adaptation="$3"
  local checkpoint="$4"
  for seed in "${SEEDS[@]}"; do
    for episode_count in "${EPISODE_COUNTS[@]}"; do
      run_condition "$gpu" "$model" "$adaptation" "$checkpoint" "$seed" "$episode_count"
    done
  done
}

run_task 0 masked_reconstruction lora "$MASKED" & P0=$!
run_task 1 masked_reconstruction full_finetune "$MASKED" & P1=$!
run_task 1 vae lora "$VAE" & P2=$!
run_task 0 vae full_finetune "$VAE" & P3=$!
run_task 0 vicreg lora "$VICREG" & P4=$!
run_task 1 vicreg full_finetune "$VICREG" & P5=$!
run_task 1 jepa lora "$JEPA" & P6=$!
run_task 0 jepa full_finetune "$JEPA" & P7=$!

STATUS=0
for pid in "$P0" "$P1" "$P2" "$P3" "$P4" "$P5" "$P6" "$P7"; do
  wait "$pid" || STATUS=1
done
if [[ "$STATUS" -ne 0 ]]; then
  echo "one or more pretrained adaptation tasks failed; inspect $RUN_DIR/logs" >&2
  exit "$STATUS"
fi

"$PATCHTST_ROOT/.venv/bin/python" "$SUMMARY" --run-dir "$RUN_DIR" \
  | tee "$RUN_DIR/logs/summary_pretrained_lora_vs_end_to_end.log"
