# ...existing code...
import os
from pathlib import Path
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

data_type = os.environ.get("DATA_TYPE", "train_08142026")
genesis_root = Path(
    os.environ.get("FT_GENESIS_ROOT", Path(__file__).resolve().parents[3])
).expanduser().resolve()
data_root = Path(os.environ.get("FT_GENESIS_DATA_ROOT", genesis_root / "data"))
data_dir = str(data_root / data_type)
save_dir = os.path.join(data_dir, os.environ.get("EMB_DIR", "bert_emb"))
os.makedirs(save_dir, exist_ok=True)

model_name = os.environ.get("FT_TEXT_MODEL", "sentence-transformers/all-mpnet-base-v2")
model_revision = os.environ.get(
    "FT_TEXT_MODEL_REVISION", "e8c3b32edf5434bc2275fc9bab85f82640a19130"
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentenceTransformer(
    model_name,
    device=device,
    revision=None if Path(model_name).expanduser().exists() else model_revision,
)
model.max_seq_length = 512

csv_name = os.environ.get("TRAIN_CSV", f"{data_type}_balanced.csv")
csv_path = os.path.join(data_dir, csv_name)
df = pd.read_csv(csv_path)

# Collect unique annotations (ignore NaN / empty)
series = df.get("annotation", pd.Series([], dtype=str)).fillna("").astype(str).str.strip()
series = series[series != ""]
unique_ann = sorted(series.unique())
print(f"unique annotations: {len(unique_ann)} / rows: {len(df)}")

batch_size = 64
embeddings_list: list[torch.Tensor] = []

def encode_texts(texts: list[str]) -> list[torch.Tensor]:
    out: list[torch.Tensor] = []
    for start in tqdm(range(0, len(texts), batch_size), desc="encode(unique)", leave=False):
        batch = texts[start:start + batch_size]
        feats = model.encode(
            batch,
            batch_size=len(batch),
            convert_to_tensor=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        out.extend(list(feats.detach().cpu()))
    return out

if len(unique_ann) == 0:
    print("No annotations found. Skipping embedding generation.")
    df["emb_index"] = pd.NA
else:
    embeddings_list = encode_texts(unique_ann)
    if len(embeddings_list) != len(unique_ann):
        raise RuntimeError("embedding count mismatch")

    # Save each unique embedding once; file name = emb_index.pt (0..U-1)
    for idx, vec in enumerate(embeddings_list):
        torch.save(vec, os.path.join(save_dir, f"{idx}.pt"))

    # Map annotation -> index
    ann_to_index = {ann: i for i, ann in enumerate(unique_ann)}

    def map_index(v):
        if pd.isna(v):
            return pd.NA
        v_str = str(v).strip()
        if v_str == "":
            return pd.NA
        return ann_to_index.get(v_str, pd.NA)

    df["emb_index"] = df.get("annotation", pd.Series([], dtype=str)).map(map_index)

    # Write mapping file (annotation -> emb_index)
    mapping_name = os.environ.get("MAPPING_NAME", "annotation_index_map.csv")
    mapping_path = os.path.join(os.path.dirname(csv_path), mapping_name)
    pd.DataFrame({"annotation": unique_ann, "emb_index": [ann_to_index[a] for a in unique_ann]}).to_csv(mapping_path, index=False)
    print(f"wrote mapping -> {mapping_path}")

df.to_csv(csv_path, index=False)
print(f"[DONE] updated {csv_path} with emb_index (unique annotation embeddings in {save_dir})")
# ...existing
