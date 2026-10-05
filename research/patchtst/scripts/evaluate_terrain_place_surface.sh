#!/usr/bin/env bash
FT_WORKSPACE_ROOT="${FT_WORKSPACE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
FT_ARTIFACT_ROOT="${FT_ARTIFACT_ROOT:-${FT_WORKSPACE_ROOT}/artifacts}"
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
SEED="${SEED:-42}"
MODEL_GROUP="${MODEL_GROUP:-terrain_contrastive_20260819_121436}"
MODEL_ROOT="$ROOT_DIR/saved_models/terrain_contrastive/$MODEL_GROUP"
SOURCE_EVAL_DIR="${SOURCE_EVAL_DIR:-${FT_WORKSPACE_ROOT}/Genesis/data/eval_08142026}"
PREPARED_DIR="$ROOT_DIR/results/terrain_place_surface_eval08142026/prepared"
OUTPUT_ROOT="$ROOT_DIR/data/terrainplace"
LOG_ROOT="$ROOT_DIR/logs/terrain_place_surface_eval08142026"
EMBED_PYTHON="${EMBED_PYTHON:-${FT_WORKSPACE_ROOT}/Genesis/.venv/bin/python}"
mkdir -p "$PREPARED_DIR" "$OUTPUT_ROOT" "$LOG_ROOT"

prepare_evaluation() {
  SOURCE_EVAL_DIR="$SOURCE_EVAL_DIR" PREPARED_DIR="$PREPARED_DIR" SEED="$SEED" \
    "$EMBED_PYTHON" - <<'PY'
import json
import os
from pathlib import Path

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

source = Path(os.environ["SOURCE_EVAL_DIR"])
output = Path(os.environ["PREPARED_DIR"])
seed = int(os.environ["SEED"])
output.mkdir(parents=True, exist_ok=True)

frame = pd.read_csv(source / "eval_08142026_terrain.csv")
frame = frame[frame["action"].eq("place") & frame["support_surface"].notna()].copy()

def terrain_class(value):
    if value == "a flat surface":
        return "flat surface"
    if "slope" in value:
        return "slope"
    if "step" in value:
        return "step"
    raise ValueError(f"Unsupported support surface: {value}")

frame["label"] = frame["support_surface"].map(terrain_class)
frame = frame[frame["weight"].notna() & frame["interaction"].notna()].copy()

strata = ["weight", "interaction"]
counts = frame.groupby(["label", *strata]).size().rename("count").reset_index()
common_counts = counts.groupby(strata)["count"].min().to_dict()

parts = []
for terrain in ("flat surface", "slope", "step"):
    terrain_rows = frame[frame["label"].eq(terrain)]
    for stratum, sample_count in common_counts.items():
        weight, interaction = stratum
        candidates = terrain_rows[
            terrain_rows["weight"].eq(weight)
            & terrain_rows["interaction"].eq(interaction)
        ]
        parts.append(candidates.sample(n=int(sample_count), random_state=seed))

balanced = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=seed)
balanced.to_csv(output / "eval_terrain_place_surface_balanced.csv", index=False)

csv_link = output / "csv"
if not csv_link.exists():
    csv_link.symlink_to(source / "csv", target_is_directory=True)

templates = {
    "flat surface": "Placing an object on a flat surface.",
    "slope": "Placing an object on a slope.",
    "step": "Placing an object on a step.",
}
labels = list(templates)
encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cpu")
embeddings = encoder.encode(
    [templates[label] for label in labels],
    convert_to_tensor=True,
    normalize_embeddings=True,
).cpu()
embedding_dir = output / "bert_emb" / "label"
embedding_dir.mkdir(parents=True, exist_ok=True)
for index, embedding in enumerate(embeddings):
    torch.save(embedding.float(), embedding_dir / f"{index}.pt")
pd.DataFrame({"label": labels, "emb_index": range(len(labels))}).to_csv(
    output / "label_index_map.csv", index=False
)
(output / "templates.json").write_text(
    json.dumps(templates, ensure_ascii=False, indent=2) + "\n"
)

print("Original terrain counts:", frame["label"].value_counts().sort_index().to_dict())
print("Balanced terrain counts:", balanced["label"].value_counts().sort_index().to_dict())
print("Balanced strata:")
print(pd.crosstab(balanced["label"], [balanced["weight"], balanced["interaction"]]))
print("Templates:", templates)
PY
}

evaluate_model() {
  local gpu="$1"
  local model="$2"
  local model_arch="$model"
  [[ "$model" == "tokenfusion" ]] && model_arch="patchtst"
  local checkpoint="$MODEL_ROOT/terrain_${model}_seed${SEED}/terrain_${model}_seed${SEED}_best.pth"
  local output_name="terrainplace_surface_${model}_seed${SEED}_best"
  local log_file="$LOG_ROOT/${model}_best.log"

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
    --eval_csv eval_terrain_place_surface_balanced.csv \
    --targets label \
    --text_type bert \
    --zero_force_eps 1e-8 \
    --output_name "$output_name" 2>&1 | tee "$log_file"
}

prepare_evaluation

evaluate_model "$GPU0" tokenfusion &
pid0=$!
evaluate_model "$GPU1" cnn &
pid1=$!
wait "$pid0"
wait "$pid1"
evaluate_model "$GPU1" mlp

echo "Completed terrain place-surface evaluation."
echo "Prepared data: $PREPARED_DIR"
echo "Results: $OUTPUT_ROOT"
