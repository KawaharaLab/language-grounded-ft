#!/usr/bin/env bash
FT_WORKSPACE_ROOT="${FT_WORKSPACE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
FT_ARTIFACT_ROOT="${FT_ARTIFACT_ROOT:-${FT_WORKSPACE_ROOT}/artifacts}"
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SEED="${SEED:-42}"
NEW_GROUP="${NEW_GROUP:-terrain_floor_wobble_20260822_140617}"
OLD_GROUP="${OLD_GROUP:-terrain_floor_normal_20260821_011506}"
SOURCE_EVAL_DIR="${SOURCE_EVAL_DIR:-${FT_WORKSPACE_ROOT}/Genesis/data/eval_08142026}"
PREPARED_DIR="${PREPARED_DIR:-$ROOT_DIR/results/wobble_place_support_eval08142026/prepared}"
LOG_ROOT="${LOG_ROOT:-$ROOT_DIR/logs/wobble_place_support_eval08142026}"
EMBED_PYTHON="${EMBED_PYTHON:-${FT_WORKSPACE_ROOT}/Genesis/.venv/bin/python}"
mkdir -p "$PREPARED_DIR" "$LOG_ROOT"

SOURCE_EVAL_DIR="$SOURCE_EVAL_DIR" PREPARED_DIR="$PREPARED_DIR" \
  "$EMBED_PYTHON" - <<'PY'
import json
import os
from pathlib import Path

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

source = Path(os.environ["SOURCE_EVAL_DIR"])
output = Path(os.environ["PREPARED_DIR"])

frame = pd.read_csv(source / "eval_08142026_support_contrast_balanced.csv")
frame = frame[frame["action"].eq("place")].copy()
frame["support_condition"] = frame["support_condition"].replace(
    {"fixed support": "flat surface", "wobbling support": "wobbling support"}
)
expected = {"flat surface": 377, "wobbling support": 377}
counts = frame["support_condition"].value_counts().sort_index().to_dict()
if counts != expected:
    raise RuntimeError(f"Unexpected support evaluation counts: {counts}")
frame.to_csv(output / "eval_place_flat_vs_wobbling_balanced.csv", index=False)

csv_link = output / "csv"
if not csv_link.exists():
    csv_link.symlink_to(source / "csv", target_is_directory=True)

# These are the exact base descriptions used by the combined terrain-floor
# training annotations for the two observed support families.
templates = {
    "flat surface": "Placing an object on a flat surface.",
    "wobbling support": "Placing an object on a wobbling support.",
}
labels = list(templates)
encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cpu")
embeddings = encoder.encode(
    [templates[label] for label in labels],
    convert_to_tensor=True,
    normalize_embeddings=True,
).cpu()
embedding_dir = output / "bert_emb" / "support_condition"
embedding_dir.mkdir(parents=True, exist_ok=True)
for index, embedding in enumerate(embeddings):
    torch.save(embedding.float(), embedding_dir / f"{index}.pt")
pd.DataFrame(
    {"support_condition": labels, "emb_index": range(len(labels))}
).to_csv(output / "support_condition_index_map.csv", index=False)
(output / "prototype_templates.json").write_text(
    json.dumps(templates, ensure_ascii=False, indent=2) + "\n"
)

print("Support counts:", counts)
print(
    "Support x interaction:",
    frame.groupby(["support_condition", "interaction"]).size().to_dict(),
)
print(
    "Support x physical weight:",
    frame.groupby(["support_condition", "physical_weight"]).size().to_dict(),
)
print("Templates:", templates)
PY

evaluate_model() {
  local gpu="$1" group="$2" run_prefix="$3" tag="$4" model="$5"
  local model_arch="$model"
  [[ "$model" == "tokenfusion" ]] && model_arch="patchtst"
  local run_name="${run_prefix}_${model}_seed${SEED}"
  local checkpoint="$ROOT_DIR/saved_models/terrain_contrastive/$group/$run_name/${run_name}_best.pth"
  local output_name="wobbleplacesupport_${tag}_${model}_seed${SEED}_best"
  local log_file="$LOG_ROOT/${tag}_${model}_best.log"
  [[ -f "$checkpoint" ]] || { echo "Missing checkpoint: $checkpoint"; return 1; }

  CUDA_VISIBLE_DEVICES="$gpu" uv run PatchTST_self_supervised/patchtst_eval_normalized.py \
    --model_path "$checkpoint" \
    --ckpt_type contrastive \
    --model_arch "$model_arch" \
    --arch auto \
    --pool mean \
    --patch_len 10 \
    --stride 10 \
    --n_layers 3 \
    --n_heads 16 \
    --d_model 128 \
    --d_ff 768 \
    --head_dropout 0.2 \
    --token_fusion_layers 1 \
    --token_fusion_heads 8 \
    --token_fusion_ff_mult 4 \
    --out_embed_dim 768 \
    --eval_dir "$PREPARED_DIR" \
    --eval_csv eval_place_flat_vs_wobbling_balanced.csv \
    --targets support_condition \
    --text_type bert \
    --zero_force_eps 1e-8 \
    --output_name "$output_name" 2>&1 | tee "$log_file"
}

evaluate_family() {
  local group="$1" run_prefix="$2" tag="$3"
  evaluate_model 0 "$group" "$run_prefix" "$tag" tokenfusion & local pid0=$!
  evaluate_model 1 "$group" "$run_prefix" "$tag" cnn & local pid1=$!
  wait "$pid0"
  wait "$pid1"
  evaluate_model 1 "$group" "$run_prefix" "$tag" mlp
}

evaluate_family "$NEW_GROUP" terrain_floor_wobble new
evaluate_family "$OLD_GROUP" terrain_floor_normal old

echo "Completed fixed-flat versus wobbling-place evaluation."
