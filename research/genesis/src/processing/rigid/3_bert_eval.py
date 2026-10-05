import os
from pathlib import Path

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

data_type = os.environ.get("DATA_TYPE", "eval_08142026")
thin_pct = int(os.environ.get("THIN_PCT", "15"))
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

csv_name = os.environ.get("EVAL_CSV", f"{data_type}_thin_{thin_pct}pct.csv")
csv_path = os.path.join(data_dir, csv_name)
df = pd.read_csv(csv_path)

target_cols = [
    value.strip()
    for value in os.environ.get(
        "TARGET_COLS", "label,action,weight,interaction,placement_outcome,annotation"
    ).split(",")
    if value.strip()
]
batch_size = 64

def collect_unique(series: pd.Series) -> list[str]:
    s = series.dropna().astype(str).str.strip()
    s = s[s != ""]
    return sorted(s.unique())

def encode_texts(texts: list[str]) -> list[torch.Tensor]:
    out: list[torch.Tensor] = []
    for start in tqdm(range(0, len(texts), batch_size), desc="encode", leave=False):
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

for col in target_cols:
    if col not in df.columns:
        print(f"[WARN] column '{col}' not found. skip.")
        continue

    values = df[col]
    eligible_rows = None
    if col == "action":
        # Action accuracy excludes the entire accidental_drop_* episode, not
        # only the window in which the accidental-drop event was annotated.
        interaction = (
            df.get("interaction", pd.Series("", index=df.index))
            .fillna("")
            .astype(str)
            .str.lower()
        )
        legacy_action = df["action"].fillna("").astype(str).str.lower()
        csv_paths = (
            df.get("csv_path", pd.Series("", index=df.index))
            .fillna("")
            .astype(str)
        )
        accidental_episode = csv_paths.map(
            lambda value: Path(value).parent.name.lower().startswith("accidental_drop_")
        )
        valid_action = (
            ~accidental_episode
            & ~interaction.eq("accidental drop")
            & ~legacy_action.str.contains("accidental drop", regex=False)
        )
        valid_action &= ~legacy_action.isin(["", "unknown"])
        valid_action &= ~legacy_action.str.contains(" then ", regex=False)
        eligible_rows = valid_action
        values = values[valid_action]
    uniq = collect_unique(values)
    print(f"[INFO] {col}: unique valid entries = {len(uniq)}")
    if len(uniq) == 0:
        df[f"{col}_emb_index"] = pd.NA
        continue

    col_dir = os.path.join(save_dir, col)
    os.makedirs(col_dir, exist_ok=True)

    embeddings = encode_texts(uniq)
    if len(embeddings) != len(uniq):
        raise RuntimeError(f"embedding count mismatch for {col}")

    text_to_index = {}
    for idx, vec in enumerate(embeddings):
        torch.save(vec, os.path.join(col_dir, f"{idx}.pt"))
        text_to_index[uniq[idx]] = idx

    def map_index(v):
        if pd.isna(v):
            return pd.NA
        v_str = str(v).strip()
        if v_str == "":
            return pd.NA
        return text_to_index.get(v_str, pd.NA)

    df[f"{col}_emb_index"] = df[col].map(map_index)
    if eligible_rows is not None:
        df.loc[~eligible_rows, f"{col}_emb_index"] = pd.NA

    mapping_suffix = os.environ.get("MAPPING_SUFFIX", "").strip()
    mapping_name = f"{col}_index_map{f'_{mapping_suffix}' if mapping_suffix else ''}.csv"
    mapping_path = os.path.join(data_dir, mapping_name)
    pd.DataFrame({col: uniq, "emb_index": [text_to_index[t] for t in uniq]}).to_csv(mapping_path, index=False)
    print(f"[INFO] {col}: saved {len(embeddings)} embeddings -> {col_dir}")

df.to_csv(csv_path, index=False)
print(f"[DONE] updated {csv_path} with *_emb_index columns")
