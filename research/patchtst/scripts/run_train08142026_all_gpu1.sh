#!/usr/bin/env bash
FT_WORKSPACE_ROOT="${FT_WORKSPACE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
FT_ARTIFACT_ROOT="${FT_ARTIFACT_ROOT:-${FT_WORKSPACE_ROOT}/artifacts}"
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SESSION="${SESSION:-train0814_all_gpu1}"
if [[ -z "${TMUX:-}" && "${AUTO_TMUX:-1}" == "1" ]]; then
  command -v tmux >/dev/null 2>&1 || {
    echo "tmux is required."
    exit 1
  }
  tmux new-session -d -s "$SESSION" "cd '$ROOT_DIR' && AUTO_TMUX=0 bash '$0'"
  echo "Started tmux session: $SESSION"
  echo "Attach with: tmux attach -t $SESSION"
  exit 0
fi

GPU="${GPU:-1}"
if [[ "$GPU" != "1" ]]; then
  echo "This sweep is intentionally restricted to physical GPU 1 (GPU=$GPU)."
  exit 1
fi

SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-400}"
BATCH_SIZE="${BATCH_SIZE:-512}"
PEAK_LR="${PEAK_LR:-2e-3}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-10}"
MIN_LR="${MIN_LR:-1e-5}"
PATIENCE="${PATIENCE:-75}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LINEAR_EPOCHS="${LINEAR_EPOCHS:-200}"
LINEAR_LR="${LINEAR_LR:-1e-2}"
LINEAR_PATIENCE="${LINEAR_PATIENCE:-30}"
ZERO_FORCE_EPS="${ZERO_FORCE_EPS:-1e-8}"
WANDB_MODE="${WANDB_MODE:-online}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
EMBED_PYTHON="${EMBED_PYTHON:-${FT_WORKSPACE_ROOT}/Genesis/.venv/bin/python}"

TRAIN_DIR="${TRAIN_DIR:-${FT_WORKSPACE_ROOT}/Genesis/data/train_08142026}"
TRAIN_BALANCED_CSV="${TRAIN_BALANCED_CSV:-train_08142026_balanced.csv}"
TRAIN_FULL_CSV="${TRAIN_FULL_CSV:-train_08142026.csv}"
EVAL_DIR="${EVAL_DIR:-${FT_WORKSPACE_ROOT}/Genesis/data/eval_08142026}"
EVAL_THIN_CSV="${EVAL_THIN_CSV:-eval_08142026_thin_15pct.csv}"
EVAL_PATTERN="${EVAL_PATTERN:-}"
if [[ -z "$EVAL_PATTERN" ]]; then
  EVAL_PATTERN='eval_08142026_{target}_balanced.csv'
fi

GROUP="${GROUP:-train0814_all_$(date +%Y%m%d_%H%M%S)}"
TAXONOMY_TAG="release_merged"
LOG_DIR="$ROOT_DIR/logs/train08142026/$GROUP"
SAVE_ROOT="$ROOT_DIR/saved_models/train08142026/$GROUP"
RESULT_ROOT="$ROOT_DIR/results/train08142026/$GROUP"
CONTRASTIVE_ROOT="$SAVE_ROOT/contrastive"
SUPERVISED_ROOT="$SAVE_ROOT/supervised"
SSL_ROOT="$SAVE_ROOT/ssl"
mkdir -p "$LOG_DIR" "$CONTRASTIVE_ROOT" "$SUPERVISED_ROOT" "$SSL_ROOT" "$RESULT_ROOT"

TRAIN_INDEX="$RESULT_ROOT/train_08142026_balanced_complete.csv"
EVAL_COMBINED="$RESULT_ROOT/eval_08142026_independent_balanced.csv"
TRAIN_RUNTIME_DIR="$RESULT_ROOT/prepared_train"
EVAL_RUNTIME_DIR="$RESULT_ROOT/prepared_eval"

prepare_indices() {
  TRAIN_DIR="$TRAIN_DIR" TRAIN_BALANCED_CSV="$TRAIN_BALANCED_CSV" \
  TRAIN_FULL_CSV="$TRAIN_FULL_CSV" TRAIN_INDEX="$TRAIN_INDEX" \
  TRAIN_RUNTIME_DIR="$TRAIN_RUNTIME_DIR" EVAL_DIR="$EVAL_DIR" \
  EVAL_RUNTIME_DIR="$EVAL_RUNTIME_DIR" \
  EVAL_COMBINED="$EVAL_COMBINED" CUDA_VISIBLE_DEVICES="$GPU" \
  "$EMBED_PYTHON" - <<'PY'
import os
from pathlib import Path

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

train_dir = Path(os.environ["TRAIN_DIR"])
balanced = pd.read_csv(train_dir / os.environ["TRAIN_BALANCED_CSV"])
full = pd.read_csv(train_dir / os.environ["TRAIN_FULL_CSV"])
annotation_map = pd.read_csv(train_dir / "annotation_index_map.csv")
runtime_dir = Path(os.environ["TRAIN_RUNTIME_DIR"])
runtime_dir.mkdir(parents=True, exist_ok=True)
csv_link = runtime_dir / "csv"
if not csv_link.exists():
    csv_link.symlink_to(train_dir / "csv", target_is_directory=True)

release_rows = full[full["action"].astype(str).eq("release")]
required_annotations = set(balanced["annotation"].dropna().astype(str)) | set(
    release_rows["annotation"].dropna().astype(str)
)
known_annotations = set(annotation_map["annotation"].astype(str))
missing_embedding_texts = sorted(required_annotations - known_annotations)
next_index = int(annotation_map["emb_index"].max()) + 1
if missing_embedding_texts:
    annotation_map = pd.concat(
        [
            annotation_map,
            pd.DataFrame(
                {
                    "annotation": missing_embedding_texts,
                    "emb_index": range(next_index, next_index + len(missing_embedding_texts)),
                }
            ),
        ],
        ignore_index=True,
    )

runtime_embeddings = runtime_dir / "bert_emb"
runtime_embeddings.mkdir(parents=True, exist_ok=True)
source_index = pd.read_csv(train_dir / "annotation_index_map.csv")
for row in source_index.itertuples(index=False):
    destination = runtime_embeddings / f"{int(row.emb_index)}.pt"
    if not destination.exists():
        destination.symlink_to(train_dir / "bert_emb" / f"{int(row.emb_index)}.pt")
text_to_index = dict(zip(annotation_map["annotation"], annotation_map["emb_index"]))
texts_to_encode = [
    text
    for text in missing_embedding_texts
    if not (runtime_embeddings / f"{int(text_to_index[text])}.pt").exists()
]
if texts_to_encode:
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    embeddings = model.encode(
        texts_to_encode,
        convert_to_tensor=True,
        normalize_embeddings=True,
    ).cpu()
    for text, embedding in zip(texts_to_encode, embeddings):
        torch.save(embedding.float(), runtime_embeddings / f"{int(text_to_index[text])}.pt")
annotation_map.to_csv(runtime_dir / "annotation_index_map.csv", index=False)

# The balanced file omits the release action. Add every release sample, but do
# not insert one-off representatives for unrelated annotations absent from it.
added = release_rows.drop_duplicates(["csv_path", "start"])
added = added.drop(columns=["emb_index"], errors="ignore").merge(
    annotation_map, on="annotation", how="left", validate="many_to_one"
)
if added["emb_index"].isna().any():
    missing = sorted(added.loc[added["emb_index"].isna(), "annotation"].unique())
    raise RuntimeError(f"Missing annotation embeddings: {missing}")

train = pd.concat([balanced, added], ignore_index=True, sort=False).drop_duplicates(
    ["csv_path", "start"]
)
train = train.drop(columns=["label"], errors="ignore")
train["action"] = train["action"].replace({"release midair": "release"})
expected = {
    "action": {"grasp", "hold", "lift", "place", "press", "release"},
    "weight": {"heavy", "light"},
    "interaction": {"accidental drop", "slip", "stable"},
}
for target, required in expected.items():
    actual = set(train[target].dropna().astype(str))
    if not required.issubset(actual):
        raise RuntimeError(f"Training index lacks {target} classes: {sorted(required - actual)}")
Path(os.environ["TRAIN_INDEX"]).parent.mkdir(parents=True, exist_ok=True)
train.to_csv(os.environ["TRAIN_INDEX"], index=False)

eval_dir = Path(os.environ["EVAL_DIR"])
eval_runtime_dir = Path(os.environ["EVAL_RUNTIME_DIR"])
eval_runtime_dir.mkdir(parents=True, exist_ok=True)
eval_csv_link = eval_runtime_dir / "csv"
if not eval_csv_link.exists():
    eval_csv_link.symlink_to(eval_dir / "csv", target_is_directory=True)
eval_embedding_link = eval_runtime_dir / "bert_emb"
if not eval_embedding_link.exists():
    eval_embedding_link.symlink_to(eval_dir / "bert_emb", target_is_directory=True)

frames = []
for target in ("action", "weight", "interaction"):
    frame = pd.read_csv(eval_dir / f"eval_08142026_{target}_balanced.csv")
    if target == "action":
        frame["action"] = frame["action"].replace({"release midair": "release"})
    frame.to_csv(eval_runtime_dir / f"eval_08142026_{target}_balanced.csv", index=False)
    for other in ("action", "weight", "interaction"):
        if other != target:
            frame[other] = pd.NA
    frame["balanced_target"] = target
    frames.append(frame)
pd.concat(frames, ignore_index=True, sort=False).to_csv(
    os.environ["EVAL_COMBINED"], index=False
)
thin = pd.read_csv(eval_dir / "eval_08142026_thin_15pct.csv")
thin["action"] = thin["action"].replace({"release midair": "release"})
thin.to_csv(eval_runtime_dir / "eval_08142026_thin_15pct.csv", index=False)
action_map = pd.read_csv(eval_dir / "action_index_map.csv")
action_map = action_map[action_map["action"].ne("release midair")]
action_map.to_csv(eval_runtime_dir / "action_index_map.csv", index=False)
for target in ("weight", "interaction"):
    pd.read_csv(eval_dir / f"{target}_index_map.csv").to_csv(
        eval_runtime_dir / f"{target}_index_map.csv", index=False
    )
print(f"Prepared training rows: {len(train)} (balanced={len(balanced)}, added={len(added)})")
print(f"Prepared annotation embeddings: {len(annotation_map)} (added={len(missing_embedding_texts)})")
print(f"Prepared evaluation rows: {sum(map(len, frames))}")
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

COMMON_CONTRASTIVE_ARGS=(
  --data_dir "$TRAIN_RUNTIME_DIR"
  --index_csv "$TRAIN_INDEX"
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

contrastive_evaluation_complete() {
  local output_dir="$1"
  [[ -f "$output_dir/umap_metrics_summary.md" ]] || return 1
  local target
  for target in action weight interaction; do
    [[ -f "$output_dir/umap/${target}_umap.png" ]] || return 1
    [[ -f "$output_dir/heatmap/${target}_heatmap.png" ]] || return 1
  done
}

classifier_evaluation_complete() {
  local output_dir="$1"
  local summary_name="$2"
  [[ -f "$output_dir/$summary_name" ]] || return 1
  local target
  for target in action weight interaction; do
    [[ -f "$output_dir/${target}_umap.png" ]] || return 1
    [[ -f "$output_dir/${target}_confusion_matrix.png" ]] || return 1
  done
}

evaluate_contrastive() {
  local run_name="$1"
  local model_arch="$2"
  local checkpoint_kind="$3"
  local checkpoint="$4"
  local output_name="eval0814_${TAXONOMY_TAG}_${run_name}_${checkpoint_kind}"
  local output_dir="$ROOT_DIR/data/eval0814/$output_name"
  local log_file="$LOG_DIR/${run_name}.log"
  if [[ "$SKIP_EXISTING" == "1" ]] && contrastive_evaluation_complete "$output_dir"; then
    echo "SKIP EVAL $run_name $checkpoint_kind" | tee -a "$log_file"
    return
  fi
  CUDA_VISIBLE_DEVICES="$GPU" WANDB_MODE="$WANDB_MODE" \
    uv run PatchTST_self_supervised/patchtst_eval_normalized.py \
      --model_path "$checkpoint" \
      --ckpt_type contrastive \
      --model_arch "$model_arch" \
      --arch auto \
      --pool mean \
      --patch_len 10 \
      --stride 10 \
      --token_fusion_layers 1 \
      --token_fusion_heads 8 \
      --token_fusion_ff_mult 4 \
      --eval_dir "$EVAL_RUNTIME_DIR" \
      --eval_csv "$EVAL_COMBINED" \
      --targets action,weight,interaction \
      --text_type bert \
      --zero_force_eps "$ZERO_FORCE_EPS" \
      --output_name "$output_name" 2>&1 | tee -a "$log_file"
}

train_contrastive() {
  local model_arch="$1"
  local run_name="contrastive_${model_arch}_0814_seed${SEED}"
  local save_dir="$CONTRASTIVE_ROOT/$run_name"
  local best="$save_dir/${run_name}_best.pth"
  local last="$save_dir/${run_name}_last.pth"
  local log_file="$LOG_DIR/${run_name}.log"
  mkdir -p "$save_dir"

  if [[ "$SKIP_EXISTING" != "1" || ! -f "$best" || ! -f "$last" ]]; then
    local -a command
    case "$model_arch" in
      tokenfusion)
        command=(
          uv run PatchTST_self_supervised/patchtst_bert_normalized.py
          "${COMMON_CONTRASTIVE_ARGS[@]}"
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
          "${COMMON_CONTRASTIVE_ARGS[@]}"
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
          "${COMMON_CONTRASTIVE_ARGS[@]}"
          "${ARCH_ARGS[@]}"
          --mlp_hidden_dim 0
          --mlp_depth 3
          --mlp_dropout 0.2
          --run_name "$run_name"
          --save_dir "$save_dir"
        )
        ;;
    esac
    echo "===== TRAIN $run_name $(date '+%F %T') =====" | tee "$log_file"
    printf '%q ' "${command[@]}" | tee -a "$log_file"
    echo | tee -a "$log_file"
    CUDA_VISIBLE_DEVICES="$GPU" WANDB_MODE="$WANDB_MODE" \
      "${command[@]}" 2>&1 | tee -a "$log_file"
  else
    echo "SKIP TRAIN $run_name" | tee "$log_file"
  fi

  local eval_arch="$model_arch"
  [[ "$model_arch" == "tokenfusion" ]] && eval_arch="patchtst"
  evaluate_contrastive "$run_name" "$eval_arch" best "$best"
  evaluate_contrastive "$run_name" "$eval_arch" last "$last"
}

evaluate_supervised() {
  local run_name="$1"
  local kind="$2"
  local checkpoint="$3"
  local output_dir="$RESULT_ROOT/supervised_eval/$TAXONOMY_TAG/$kind"
  local log_file="$LOG_DIR/${run_name}.log"
  if [[ "$SKIP_EXISTING" == "1" ]] \
      && classifier_evaluation_complete "$output_dir" balanced_eval_summary.md; then
    echo "SKIP SUPERVISED EVAL $kind" | tee -a "$log_file"
    return
  fi
  CUDA_VISIBLE_DEVICES="$GPU" uv run PatchTST_self_supervised/evaluate_supervised_balanced.py \
    --checkpoint "$checkpoint" \
    --eval_dir "$EVAL_RUNTIME_DIR" \
    --eval_pattern "$EVAL_PATTERN" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --zero_force_eps "$ZERO_FORCE_EPS" \
    --output_dir "$output_dir" 2>&1 | tee -a "$log_file"
}

train_supervised() {
  local run_name="supervised_tokenfusion_multitask_0814_${TAXONOMY_TAG}_seed${SEED}"
  local output_dir="$SUPERVISED_ROOT/$run_name"
  local best="$output_dir/${run_name}_best.pth"
  local final="$output_dir/${run_name}_final.pth"
  local log_file="$LOG_DIR/${run_name}.log"
  if [[ "$SKIP_EXISTING" != "1" || ! -f "$best" || ! -f "$final" ]]; then
    echo "===== TRAIN $run_name $(date '+%F %T') =====" | tee "$log_file"
    CUDA_VISIBLE_DEVICES="$GPU" WANDB_MODE="$WANDB_MODE" \
      uv run PatchTST_self_supervised/patchtst_supervised.py \
        --task multitask \
        --train_dir "$TRAIN_RUNTIME_DIR" \
        --train_csv "$TRAIN_INDEX" \
        --eval_dir "$EVAL_RUNTIME_DIR" \
        --eval_csv "$EVAL_THIN_CSV" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --peak_lr "$PEAK_LR" \
        --min_lr "$MIN_LR" \
        --warmup_epochs "$WARMUP_EPOCHS" \
        --early_stopping_patience "$PATIENCE" \
        --num_workers "$NUM_WORKERS" \
        --val_from_train_pct 0.10 \
        --zero_force_eps "$ZERO_FORCE_EPS" \
        --seed "$SEED" \
        --save_dir "$SUPERVISED_ROOT" \
        --run_name "$run_name" \
        --wandb_project patchtst_tokenfusion_supervised_0814 \
        --wandb_group "$GROUP" \
        "${ARCH_ARGS[@]}" 2>&1 | tee -a "$log_file"
  else
    echo "SKIP TRAIN $run_name" | tee "$log_file"
  fi
  evaluate_supervised "$run_name" best "$best"
  evaluate_supervised "$run_name" last "$final"
}

evaluate_ssl() {
  local method="$1"
  local kind="$2"
  local checkpoint="$3"
  local output_dir="$RESULT_ROOT/ssl_linear_probes/$method/$TAXONOMY_TAG/$kind"
  local log_file="$LOG_DIR/ssl_${method}.log"
  if [[ "$SKIP_EXISTING" == "1" ]] \
      && classifier_evaluation_complete "$output_dir" linear_probe_summary.md; then
    echo "SKIP SSL PROBE $method $kind" | tee -a "$log_file"
    return
  fi
  CUDA_VISIBLE_DEVICES="$GPU" uv run PatchTST_self_supervised/cnn_ssl_linear_probe.py \
    --checkpoint "$checkpoint" \
    --train_dir "$TRAIN_RUNTIME_DIR" \
    --train_csv "$TRAIN_INDEX" \
    --eval_dir "$EVAL_RUNTIME_DIR" \
    --balanced_eval \
    --balanced_eval_pattern "$EVAL_PATTERN" \
    --targets action weight interaction \
    --epochs "$LINEAR_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --embedding_batch_size "$BATCH_SIZE" \
    --learning_rate "$LINEAR_LR" \
    --patience "$LINEAR_PATIENCE" \
    --num_workers "$NUM_WORKERS" \
    --zero_force_eps "$ZERO_FORCE_EPS" \
    --seed "$SEED" \
    --output_dir "$output_dir" 2>&1 | tee -a "$log_file"
}

train_ssl() {
  local method="$1"
  local run_name="ssl_tokenfusion_${method}_0814_seed${SEED}"
  local save_dir="$SSL_ROOT/$run_name"
  local best="$save_dir/${run_name}_best.pth"
  local last="$save_dir/${run_name}_last.pth"
  local log_file="$LOG_DIR/ssl_${method}.log"
  mkdir -p "$save_dir"
  if [[ "$SKIP_EXISTING" != "1" || ! -f "$best" || ! -f "$last" ]]; then
    echo "===== TRAIN $run_name $(date '+%F %T') =====" | tee "$log_file"
    CUDA_VISIBLE_DEVICES="$GPU" WANDB_MODE="$WANDB_MODE" \
      uv run PatchTST_self_supervised/token_fusion_force_ssl.py \
        --method "$method" \
        --data_dir "$TRAIN_RUNTIME_DIR" \
        --index_csv "$TRAIN_INDEX" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --peak_lr "$PEAK_LR" \
        --min_lr "$MIN_LR" \
        --warmup_epochs "$WARMUP_EPOCHS" \
        --early_stopping_patience "$PATIENCE" \
        --num_workers "$NUM_WORKERS" \
        --zero_force_eps "$ZERO_FORCE_EPS" \
        --seed "$SEED" \
        --save_dir "$save_dir" \
        --run_name "$run_name" \
        --wandb_mode "$WANDB_MODE" \
        --wandb_project patchtst_tokenfusion_ssl_0814 \
        --wandb_group "$GROUP" \
        "${ARCH_ARGS[@]}" 2>&1 | tee -a "$log_file"
  else
    echo "SKIP TRAIN $run_name" | tee "$log_file"
  fi
  evaluate_ssl "$method" best "$best"
  evaluate_ssl "$method" last "$last"
}

prepare_indices

if [[ "${PREPARE_ONLY:-0}" == "1" ]]; then
  echo "Index preparation completed: $TRAIN_INDEX"
  echo "Balanced evaluation index: $EVAL_COMBINED"
  exit 0
fi

echo "group=$GROUP"
echo "physical_gpu=$GPU (all jobs run sequentially)"
echo "train_index=$TRAIN_INDEX"
echo "eval_dir=$EVAL_DIR"
echo "epochs=$EPOCHS batch_size=$BATCH_SIZE peak_lr=$PEAK_LR seed=$SEED"

train_contrastive tokenfusion
train_contrastive cnn
train_contrastive mlp
train_supervised
for method in masked_reconstruction jepa vae vicreg; do
  train_ssl "$method"
done

echo "Completed: $GROUP"
echo "Checkpoints: $SAVE_ROOT"
echo "Supervised/SSL evaluations: $RESULT_ROOT"
echo "Contrastive plots/metrics: $ROOT_DIR/data/eval0814"
