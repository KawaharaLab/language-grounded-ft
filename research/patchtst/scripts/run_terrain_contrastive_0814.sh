#!/usr/bin/env bash
FT_WORKSPACE_ROOT="${FT_WORKSPACE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
FT_ARTIFACT_ROOT="${FT_ARTIFACT_ROOT:-${FT_WORKSPACE_ROOT}/artifacts}"
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SESSION="${SESSION:-terrain_contrastive_0814}"
if [[ -z "${TMUX:-}" && "${AUTO_TMUX:-1}" == "1" ]]; then
  command -v tmux >/dev/null 2>&1 || {
    echo "tmux is required."
    exit 1
  }
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session already exists: $SESSION"
    echo "Attach with: tmux attach -t $SESSION"
    exit 1
  fi
  tmux new-session -d -s "$SESSION" "cd '$ROOT_DIR' && AUTO_TMUX=0 bash '$0'"
  echo "Started tmux session: $SESSION"
  echo "Attach with: tmux attach -t $SESSION"
  exit 0
fi

SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-400}"
BATCH_SIZE="${BATCH_SIZE:-512}"
PEAK_LR="${PEAK_LR:-2e-3}"
MIN_LR="${MIN_LR:-1e-5}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-10}"
PATIENCE="${PATIENCE:-75}"
NUM_WORKERS="${NUM_WORKERS:-4}"
ZERO_FORCE_EPS="${ZERO_FORCE_EPS:-1e-8}"
WANDB_MODE="${WANDB_MODE:-online}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
FORCE_EVAL="${FORCE_EVAL:-0}"
EMBED_PYTHON="${EMBED_PYTHON:-${FT_WORKSPACE_ROOT}/Genesis/.venv/bin/python}"

TRAIN_DIR="${TRAIN_DIR:-${FT_WORKSPACE_ROOT}/Genesis/data/train_08142026}"
TRAIN_CSV="${TRAIN_CSV:-train_08142026_terrain_balanced.csv}"
TRAIN_EMB_DIR="${TRAIN_EMB_DIR:-bert_emb_terrain}"
EVAL_DIR="${EVAL_DIR:-${FT_WORKSPACE_ROOT}/Genesis/data/eval_08142026}"
EVAL_PREFIX="${EVAL_PREFIX:-eval_08142026_terrain}"
EVAL_MAP_SUFFIX="${EVAL_MAP_SUFFIX:-terrain}"
RUN_PREFIX="${RUN_PREFIX:-terrain}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-terrain0814_templates_release_merged}"
MODEL_SEQUENCE="${MODEL_SEQUENCE:-parallel}"
GPU_ID="${GPU_ID:-0}"

GROUP="${GROUP:-terrain_contrastive_$(date +%Y%m%d_%H%M%S)}"
SAVE_ROOT="$ROOT_DIR/saved_models/terrain_contrastive/$GROUP"
RESULT_ROOT="$ROOT_DIR/results/terrain_contrastive/$GROUP"
LOG_ROOT="$ROOT_DIR/logs/terrain_contrastive/$GROUP"
EVAL_RUNTIME="$RESULT_ROOT/prepared_eval_templates_release_merged"
EVAL_COMBINED="$EVAL_RUNTIME/eval_08142026_terrain_independent_balanced.csv"
TEMPLATE_PROTOTYPE_ROOT="${TEMPLATE_PROTOTYPE_ROOT:-$ROOT_DIR/results/template_prototypes_eval04272026/prototypes}"
RELEASE_PROTOTYPE_ROOT="${RELEASE_PROTOTYPE_ROOT:-$ROOT_DIR/results/synonym_prototypes_eval04272026/prototypes}"
mkdir -p "$SAVE_ROOT" "$RESULT_ROOT" "$LOG_ROOT" "$EVAL_RUNTIME"

prepare_evaluation() {
  TEMPLATE_PROTOTYPE_ROOT="$TEMPLATE_PROTOTYPE_ROOT" \
  RELEASE_PROTOTYPE_ROOT="$RELEASE_PROTOTYPE_ROOT" \
  EVAL_DIR="$EVAL_DIR" EVAL_PREFIX="$EVAL_PREFIX" \
  EVAL_MAP_SUFFIX="$EVAL_MAP_SUFFIX" EVAL_RUNTIME="$EVAL_RUNTIME" \
  EVAL_COMBINED="$EVAL_COMBINED" \
    "$EMBED_PYTHON" - <<'PY'
import json
import os
from pathlib import Path

import pandas as pd
import torch

source = Path(os.environ["EVAL_DIR"])
eval_prefix = os.environ["EVAL_PREFIX"]
map_suffix = os.environ["EVAL_MAP_SUFFIX"]
runtime = Path(os.environ["EVAL_RUNTIME"])
runtime.mkdir(parents=True, exist_ok=True)

csv_link = runtime / "csv"
if not csv_link.exists():
    csv_link.symlink_to(source / "csv", target_is_directory=True)

embedding_root = runtime / "bert_emb"
template_root = Path(os.environ["TEMPLATE_PROTOTYPE_ROOT"])
release_root = Path(os.environ["RELEASE_PROTOTYPE_ROOT"])

action_map = pd.read_csv(source / f"action_index_map_{map_suffix}.csv")
merged_action_map = action_map[action_map["action"].ne("release midair")].copy()
merged_action_map.to_csv(runtime / "action_index_map.csv", index=False)

target_maps = {"action": merged_action_map}
for target in ("weight", "interaction"):
    target_maps[target] = pd.read_csv(source / f"{target}_index_map_{map_suffix}.csv")
    target_maps[target].to_csv(runtime / f"{target}_index_map.csv", index=False)

templates = json.loads((template_root / "templates.json").read_text())
templates["action"]["release"] = "Releasing an object."
for target, target_map in target_maps.items():
    destination_root = embedding_root / target
    destination_root.mkdir(parents=True, exist_ok=True)
    standard_map = pd.read_csv(template_root / f"{target}_index_map.csv")
    standard_indices = dict(zip(standard_map[target], standard_map["emb_index"]))
    for row in target_map.itertuples(index=False):
        label = getattr(row, target)
        destination_index = int(row.emb_index)
        source_label = "drop" if target == "action" and label == "release" else label
        source_index = int(standard_indices[source_label])
        source_root = release_root if target == "action" and label == "release" else template_root
        embedding = torch.load(
            source_root / "bert_emb" / target / f"{source_index}.pt",
            map_location="cpu",
            weights_only=False,
        ).float()
        torch.save(embedding, destination_root / f"{destination_index}.pt")

(runtime / "prototype_templates.json").write_text(
    json.dumps(templates, indent=2) + "\n"
)
print(f"Prototype templates: {templates}")

frames = []
for target in ("action", "weight", "interaction"):
    frame = pd.read_csv(source / f"{eval_prefix}_{target}_balanced.csv")
    if target == "action":
        frame["action"] = frame["action"].replace({"release midair": "release"})
    frame.to_csv(runtime / f"eval_08142026_terrain_{target}_balanced.csv", index=False)
    for other in ("action", "weight", "interaction"):
        if other != target:
            frame[other] = pd.NA
    frame["balanced_target"] = target
    frames.append(frame)

combined = pd.concat(frames, ignore_index=True, sort=False)
combined.to_csv(os.environ["EVAL_COMBINED"], index=False)

action_counts = frames[0]["action"].value_counts().sort_index().to_dict()
weight_counts = frames[1]["weight"].value_counts().sort_index().to_dict()
interaction_counts = frames[2]["interaction"].value_counts().sort_index().to_dict()
expected_action = {"grasp", "hold", "lift", "place", "press", "release"}
if set(action_counts) != expected_action or len(set(action_counts.values())) != 1:
    raise RuntimeError(f"Merged action evaluation is not balanced: {action_counts}")
print(f"Prepared merged action evaluation: {action_counts}")
print(f"Prepared weight evaluation: {weight_counts}")
print(f"Prepared interaction evaluation: {interaction_counts}")
print(f"Combined evaluation rows: {len(combined)}")
PY
}

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

EVAL_ARCH_ARGS=(
  --patch_len 10
  --stride 10
  --n_layers 3
  --n_heads 16
  --d_model 128
  --d_ff 768
  --head_dropout 0.2
  --token_fusion_layers 1
  --token_fusion_heads 8
  --token_fusion_ff_mult 4
  --out_embed_dim 768
)

COMMON_ARGS=(
  --data_dir "$TRAIN_DIR"
  --index_csv "$TRAIN_CSV"
  --emb_dir "$TRAIN_EMB_DIR"
  --epochs "$EPOCHS"
  --batch_size "$BATCH_SIZE"
  --peak_lr "$PEAK_LR"
  --min_lr "$MIN_LR"
  --warmup_epochs "$WARMUP_EPOCHS"
  --early_stopping_patience "$PATIENCE"
  --early_stopping_min_delta 1e-4
  --val_from_train_pct 0.10
  --num_workers "$NUM_WORKERS"
  --zero_force_eps "$ZERO_FORCE_EPS"
  --loss_type infonce
  --seed "$SEED"
  --from_scratch
)

evaluation_complete() {
  local output_dir="$1"
  [[ -f "$output_dir/umap_metrics_summary.md" ]] || return 1
  local target
  for target in action weight interaction; do
    [[ -f "$output_dir/umap/${target}_umap.png" ]] || return 1
    [[ -f "$output_dir/heatmap/${target}_heatmap.png" ]] || return 1
  done
}

evaluate_model() {
  local gpu="$1"
  local run_name="$2"
  local model_arch="$3"
  local checkpoint_kind="$4"
  local checkpoint="$5"
  local output_name="${OUTPUT_PREFIX}_${run_name}_${checkpoint_kind}"
  local output_dir="$ROOT_DIR/data/terrain0814/$output_name"
  local log_file="$LOG_ROOT/${run_name}.log"

  if [[ "$FORCE_EVAL" != "1" && "$SKIP_EXISTING" == "1" ]] && \
    evaluation_complete "$output_dir"; then
    echo "SKIP EVAL $run_name $checkpoint_kind" | tee -a "$log_file"
    return
  fi

  CUDA_VISIBLE_DEVICES="$gpu" WANDB_MODE="$WANDB_MODE" \
    uv run PatchTST_self_supervised/patchtst_eval_normalized.py \
      --model_path "$checkpoint" \
      --ckpt_type contrastive \
      --model_arch "$model_arch" \
      --arch auto \
      --pool mean \
      "${EVAL_ARCH_ARGS[@]}" \
      --eval_dir "$EVAL_RUNTIME" \
      --eval_csv "$EVAL_COMBINED" \
      --targets action,weight,interaction \
      --text_type bert \
      --zero_force_eps "$ZERO_FORCE_EPS" \
      --output_name "$output_name" 2>&1 | tee -a "$log_file"
}

train_model() {
  local gpu="$1"
  local model_arch="$2"
  local run_name="${RUN_PREFIX}_${model_arch}_seed${SEED}"
  local save_dir="$SAVE_ROOT/$run_name"
  local best="$save_dir/${run_name}_best.pth"
  local last="$save_dir/${run_name}_last.pth"
  local log_file="$LOG_ROOT/${run_name}.log"
  mkdir -p "$save_dir"

  if [[ "$SKIP_EXISTING" != "1" || ! -f "$best" || ! -f "$last" ]]; then
    local -a command
    case "$model_arch" in
      tokenfusion)
        command=(
          uv run PatchTST_self_supervised/patchtst_bert_normalized.py
          "${COMMON_ARGS[@]}"
          "${ARCH_ARGS[@]}"
          --arch token_fusion
          --pool mean
          --input_features raw
          --run_name "$run_name"
          --save_dir "$save_dir"
        )
        ;;
      cnn)
        command=(
          uv run PatchTST_self_supervised/patchtst_cnn.py
          "${COMMON_ARGS[@]}"
          "${ARCH_ARGS[@]}"
          --cnn_base_channels 0
          --cnn_depth 4
          --cnn_kernel_size 5
          --cnn_dropout 0.2
          --run_name "$run_name"
          --save_dir "$save_dir"
        )
        ;;
      mlp)
        command=(
          uv run PatchTST_self_supervised/patchtst_mlp.py
          "${COMMON_ARGS[@]}"
          "${ARCH_ARGS[@]}"
          --mlp_hidden_dim 0
          --mlp_depth 3
          --mlp_dropout 0.2
          --run_name "$run_name"
          --save_dir "$save_dir"
        )
        ;;
      *)
        echo "Unknown model architecture: $model_arch"
        exit 1
        ;;
    esac

    echo "===== TRAIN $run_name on GPU $gpu $(date '+%F %T') =====" | tee "$log_file"
    printf '%q ' "${command[@]}" | tee -a "$log_file"
    echo | tee -a "$log_file"
    CUDA_VISIBLE_DEVICES="$gpu" WANDB_MODE="$WANDB_MODE" \
      "${command[@]}" 2>&1 | tee -a "$log_file"
  else
    echo "SKIP TRAIN $run_name" | tee "$log_file"
  fi

  local eval_arch="$model_arch"
  [[ "$model_arch" == "tokenfusion" ]] && eval_arch="patchtst"
  evaluate_model "$gpu" "$run_name" "$eval_arch" best "$best"
  evaluate_model "$gpu" "$run_name" "$eval_arch" last "$last"
}

write_summary() {
  ROOT_DIR="$ROOT_DIR" RESULT_ROOT="$RESULT_ROOT" SEED="$SEED" \
  RUN_PREFIX="$RUN_PREFIX" OUTPUT_PREFIX="$OUTPUT_PREFIX" \
    uv run python - <<'PY'
import os
from pathlib import Path

root = Path(os.environ["ROOT_DIR"])
result_root = Path(os.environ["RESULT_ROOT"])
seed = os.environ["SEED"]
run_prefix = os.environ["RUN_PREFIX"]
output_prefix = os.environ["OUTPUT_PREFIX"]
rows = []
for model in ("tokenfusion", "cnn", "mlp"):
    run_name = f"{run_prefix}_{model}_seed{seed}"
    for kind in ("best", "last"):
        summary = (
            root
            / "data"
            / "terrain0814"
            / f"{output_prefix}_{run_name}_{kind}"
            / "umap_metrics_summary.md"
        )
        metrics = {}
        for line in summary.read_text().splitlines():
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if len(cells) == 4 and cells[0] in {"Accuracy", "Macro F1"}:
                metrics[cells[0]] = cells[1:]
        rows.append((model, kind, metrics["Accuracy"], metrics["Macro F1"]))

output = result_root / "summary.md"
with output.open("w") as handle:
    handle.write("# Terrain Contrastive Evaluation\n\n")
    handle.write("All contrastive prototypes use fixed sentence templates. ")
    handle.write("`release` and `release midair` are merged, and their prototype ")
    handle.write("is `Releasing an object.`. ")
    handle.write("Each task cell is Accuracy / Macro F1.\n\n")
    handle.write("| Model | Checkpoint | Action | Weight | Interaction | Mean Accuracy |\n")
    handle.write("| --- | --- | ---: | ---: | ---: | ---: |\n")
    for model, kind, accuracy, macro_f1 in rows:
        cells = [f"{float(a):.4f} / {float(f):.4f}" for a, f in zip(accuracy, macro_f1)]
        mean_accuracy = sum(map(float, accuracy)) / len(accuracy)
        handle.write(
            f"| {model} | {kind} | {cells[0]} | {cells[1]} | {cells[2]} | "
            f"{mean_accuracy:.4f} |\n"
        )
print(f"Summary: {output}")
PY
}

prepare_evaluation

echo "group=$GROUP"
echo "train=$TRAIN_DIR/$TRAIN_CSV"
echo "embeddings=$TRAIN_DIR/$TRAIN_EMB_DIR"
echo "evaluation=$EVAL_COMBINED"
echo "model_sequence=$MODEL_SEQUENCE gpu_id=$GPU_ID"
echo "epochs=$EPOCHS batch_size=$BATCH_SIZE peak_lr=$PEAK_LR seed=$SEED"

case "$MODEL_SEQUENCE" in
  sequential)
    train_model "$GPU_ID" tokenfusion
    train_model "$GPU_ID" cnn
    train_model "$GPU_ID" mlp
    ;;
  parallel)
    train_model 0 tokenfusion &
    tokenfusion_pid=$!
    (
      train_model 1 cnn
      train_model 1 mlp
    ) &
    baseline_pid=$!
    wait "$tokenfusion_pid"
    wait "$baseline_pid"
    ;;
  *)
    echo "Unknown MODEL_SEQUENCE: $MODEL_SEQUENCE (expected parallel or sequential)"
    exit 1
    ;;
esac
write_summary

echo "Completed: $GROUP"
echo "Checkpoints: $SAVE_ROOT"
echo "Summary: $RESULT_ROOT/summary.md"
