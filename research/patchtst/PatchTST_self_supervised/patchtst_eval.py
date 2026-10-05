from workspace_paths import resolve_recorded_path, workspace_path
import os
import argparse
import math
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import umap
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

from src.models.patchTST import PatchTST
from src.callback.patch_mask import create_patch   # patch creation function
from src.learner import load_model   # for loading state_dict
from baseline_models import (
    FlatMLPContrastive,
    LSTMContrastive,
    ResidualTCNContrastive,
    StandardCNNContrastive,
)

PATCHTST_BERT_REF = {
    "context_points": 80,
    "patch_len": 10,
    "stride": 10,
    "n_layers": 3,
    "n_heads": 16,
    "d_model": 128,
    "d_ff": 768,
    "dropout": 0.2,
    "head_dropout": 0.2,
    "out_embed_dim": 768,
}
MLP_WIDTH_CANDIDATES = [64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 448, 512]


class SingleStageAttentionPooling(torch.nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.score = torch.nn.Linear(d_model, 1)

    def forward(self, z: torch.Tensor):
        tok = z.permute(0, 1, 3, 2).contiguous().view(z.size(0), -1, z.size(2))  # [B, V*P, D]
        w = torch.softmax(self.score(tok).squeeze(-1), dim=-1)
        return torch.sum(tok * w.unsqueeze(-1), dim=1)


class TwoStageAttentionPooling(torch.nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.patch_score = torch.nn.Linear(d_model, 1)
        self.channel_score = torch.nn.Linear(d_model, 1)

    def forward(self, z: torch.Tensor):
        tok = z.permute(0, 1, 3, 2).contiguous()  # [B, V, P, D]
        w_patch = torch.softmax(self.patch_score(tok).squeeze(-1), dim=2)  # [B, V, P]
        ch_repr = torch.sum(tok * w_patch.unsqueeze(-1), dim=2)  # [B, V, D]
        w_ch = torch.softmax(self.channel_score(ch_repr).squeeze(-1), dim=1)  # [B, V]
        return torch.sum(ch_repr * w_ch.unsqueeze(-1), dim=1)  # [B, D]


class ChannelCLSPatchAttentionPooling(torch.nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.cls = torch.nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    def forward(self, z: torch.Tensor):
        tok = z.permute(0, 1, 3, 2).contiguous()  # [B, V, P, D]
        cls = self.cls.expand(tok.size(0), tok.size(1), -1)  # [B, V, D]
        scores = torch.einsum("bvpd,bvd->bvp", tok, cls) / math.sqrt(tok.size(-1))
        w = torch.softmax(scores, dim=2)
        ch_repr = torch.sum(tok * w.unsqueeze(-1), dim=2)
        return ch_repr.mean(dim=1)


class TemporalConvStem(torch.nn.Module):
    """Local temporal filter before patching while preserving channel count."""
    def __init__(self, nvars: int, kernel_size: int, dropout: float):
        super().__init__()
        padding = kernel_size // 2
        self.net = torch.nn.Sequential(
            torch.nn.Conv1d(nvars, nvars, kernel_size, padding=padding, groups=nvars),
            torch.nn.GELU(),
            torch.nn.Conv1d(nvars, nvars, 1),
            torch.nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor):
        y = x.transpose(1, 2)
        y = y + self.net(y)
        return y.transpose(1, 2)


class TokenFusionPooling(torch.nn.Module):
    """Transformer pooling over channel-patch tokens for cross-axis interaction."""
    def __init__(self, nvars: int, num_patch: int, d_model: int,
                 n_heads: int, n_layers: int, ff_mult: int, dropout: float):
        super().__init__()
        self.channel_embed = torch.nn.Parameter(torch.randn(1, nvars, 1, d_model) * 0.02)
        self.patch_embed = torch.nn.Parameter(torch.randn(1, 1, num_patch, d_model) * 0.02)
        self.cls = torch.nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = torch.nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, z: torch.Tensor):
        tok = z.permute(0, 1, 3, 2).contiguous()
        tok = tok + self.channel_embed + self.patch_embed
        tok = tok.reshape(tok.size(0), -1, tok.size(-1))
        cls = self.cls.expand(tok.size(0), -1, -1)
        tok = torch.cat([cls, tok], dim=1)
        tok = self.encoder(tok)
        return tok[:, 0]


class PatchTSTContrastive(torch.nn.Module):
    def __init__(self, nvars: int, context_points: int, patch_len: int, stride: int,
                 n_layers: int, n_heads: int, d_model: int, d_ff: int, dropout: float,
                 head_dropout: float, out_embed_dim: int, pool: str = 'mean',
                 arch: str = 'patchtst', input_features: str = 'raw',
                 conv_stem_kernel: int = 5, token_fusion_layers: int = 1,
                 token_fusion_heads: int = 4, token_fusion_ff_mult: int = 4,
                 use_channel_embedding: bool = False):
        super().__init__()
        self.nvars = nvars
        self.patch_len = patch_len
        self.stride = stride
        self.pool = pool
        self.arch = arch
        self.input_features = 'raw_delta' if arch == 'delta_token_fusion' else input_features
        if self.input_features not in {'raw', 'raw_delta'}:
            raise ValueError(f"Unsupported input_features: {self.input_features}")
        self.use_conv_stem = arch in {'conv_stem', 'conv_token_fusion'}
        self.use_token_fusion = arch in {'token_fusion', 'conv_token_fusion', 'delta_token_fusion'}
        self.use_channel_embedding = use_channel_embedding or arch == 'axis_embed'
        self.input_nvars = nvars * (2 if self.input_features == 'raw_delta' else 1)
        self.conv_stem = TemporalConvStem(nvars, conv_stem_kernel, dropout) if self.use_conv_stem else None

        num_patch = (max(context_points, patch_len) - patch_len) // stride + 1
        self.backbone = PatchTST(
            c_in=self.input_nvars,
            target_dim=0,
            patch_len=patch_len,
            stride=stride,
            num_patch=num_patch,
            n_layers=n_layers,
            n_heads=n_heads,
            d_model=d_model,
            shared_embedding=True,
            d_ff=d_ff,
            dropout=dropout,
            head_dropout=head_dropout,
            act='gelu',
            head_type='pretrain',
            res_attention=False,
        ).backbone
        if self.use_token_fusion:
            self.pooler = TokenFusionPooling(
                self.input_nvars,
                num_patch,
                d_model,
                token_fusion_heads,
                token_fusion_layers,
                token_fusion_ff_mult,
                dropout,
            )
        elif self.pool == 'single_attn':
            self.pooler = SingleStageAttentionPooling(d_model)
        elif self.pool == 'two_stage_attn':
            self.pooler = TwoStageAttentionPooling(d_model)
        elif self.pool == 'channel_cls_patch_attn':
            self.pooler = ChannelCLSPatchAttentionPooling(d_model)
        else:
            self.pooler = None
        if self.use_channel_embedding and not self.use_token_fusion:
            self.channel_embed = torch.nn.Parameter(torch.randn(1, self.input_nvars, d_model, 1) * 0.02)
        else:
            self.channel_embed = None

        if self.pool == 'flatten':
            pooled_dim = self.input_nvars * num_patch * d_model
        else:
            pooled_dim = d_model
        proj_layers = [torch.nn.Linear(pooled_dim, out_embed_dim)] if pooled_dim == out_embed_dim else [
            torch.nn.Linear(pooled_dim, d_model), torch.nn.GELU(), torch.nn.Dropout(head_dropout), torch.nn.Linear(d_model, out_embed_dim)
        ]
        self.proj = torch.nn.Sequential(*proj_layers)

    def _build_input_features(self, x: torch.Tensor):
        if self.conv_stem is not None:
            x = self.conv_stem(x)
        if self.input_features == 'raw_delta':
            delta = torch.zeros_like(x)
            delta[:, 1:] = x[:, 1:] - x[:, :-1]
            x = torch.cat([x, delta], dim=-1)
        return x

    def forward(self, x: torch.Tensor):
        x = self._build_input_features(x)
        xb_patch, _ = create_patch(x, self.patch_len, self.stride)
        z = self.backbone(xb_patch)
        if self.channel_embed is not None:
            z = z + self.channel_embed
        if self.use_token_fusion:
            z = self.pooler(z)
        elif self.pool == 'mean':
            z = z.mean(dim=(1, 3))
        elif self.pool == 'last':
            z = z[:, :, :, -1].mean(dim=1)
        elif self.pool == 'flatten':
            z = z.permute(0,1,3,2).contiguous().view(z.size(0), -1)
        elif self.pooler is not None:
            z = self.pooler(z)
        else:
            raise ValueError(f"Unsupported pool type: {self.pool}")
        emb = self.proj(z)
        emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-8)
        return emb


class ForceAxisMetadataEmbedding(torch.nn.Module):
    def __init__(self, nvars: int, d_model: int):
        super().__init__()
        self.base_embed = torch.nn.Parameter(torch.randn(1, nvars, d_model) * 0.02)
        self.sensor_embed = torch.nn.Embedding(2, d_model)
        self.kind_embed = torch.nn.Embedding(2, d_model)
        self.axis_embed = torch.nn.Embedding(3, d_model)
        self.feature_embed = torch.nn.Embedding(2, d_model)
        sensor_ids, kind_ids, axis_ids, feature_ids = self._build_ids(nvars)
        self.register_buffer("sensor_ids", sensor_ids, persistent=False)
        self.register_buffer("kind_ids", kind_ids, persistent=False)
        self.register_buffer("axis_ids", axis_ids, persistent=False)
        self.register_buffer("feature_ids", feature_ids, persistent=False)

    @staticmethod
    def _build_ids(nvars: int):
        if nvars % 12 != 0:
            sensor_ids = torch.zeros(nvars, dtype=torch.long)
            kind_ids = torch.zeros(nvars, dtype=torch.long)
            axis_ids = torch.arange(nvars, dtype=torch.long) % 3
            feature_ids = torch.zeros(nvars, dtype=torch.long)
            return sensor_ids, kind_ids, axis_ids, feature_ids
        base_sensor = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=torch.long)
        base_kind = torch.tensor([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1], dtype=torch.long)
        base_axis = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=torch.long)
        repeats = nvars // 12
        return (
            base_sensor.repeat(repeats),
            base_kind.repeat(repeats),
            base_axis.repeat(repeats),
            torch.arange(repeats, dtype=torch.long).repeat_interleave(12).clamp(max=1),
        )

    def forward(self, batch_size: int):
        emb = (
            self.base_embed
            + self.sensor_embed(self.sensor_ids).unsqueeze(0)
            + self.kind_embed(self.kind_ids).unsqueeze(0)
            + self.axis_embed(self.axis_ids).unsqueeze(0)
            + self.feature_embed(self.feature_ids).unsqueeze(0)
        )
        return emb.expand(batch_size, -1, -1)


class InvertedTemporalBackbone(torch.nn.Module):
    def __init__(self, nvars: int, context_points: int, n_layers: int, n_heads: int,
                 d_model: int, d_ff: int, dropout: float, head_dropout: float,
                 arch: str, input_features: str, conv_stem_kernel: int,
                 pool: str, use_metadata: bool):
        super().__init__()
        self.nvars = nvars
        self.context_points = context_points
        self.arch = arch
        self.input_features = input_features
        self.pool = pool
        self.input_nvars = nvars * (2 if input_features == "raw_delta" else 1)
        self.conv_stem = TemporalConvStem(nvars, conv_stem_kernel, dropout) if "conv" in arch else None
        self.value_proj = torch.nn.Sequential(
            torch.nn.LayerNorm(context_points),
            torch.nn.Linear(context_points, d_model),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
        )
        self.token_embed = ForceAxisMetadataEmbedding(self.input_nvars, d_model) if use_metadata else torch.nn.Parameter(
            torch.randn(1, self.input_nvars, d_model) * 0.02
        )
        self.cls = torch.nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = torch.nn.TransformerEncoder(layer, num_layers=n_layers)
        self.attn_score = torch.nn.Linear(d_model, 1) if pool == "attn" else None
        self.out_norm = torch.nn.LayerNorm(d_model)
        self.out_drop = torch.nn.Dropout(head_dropout)

    def _build_input_features(self, x: torch.Tensor):
        if self.conv_stem is not None:
            x = self.conv_stem(x)
        if self.input_features == "raw_delta":
            delta = torch.zeros_like(x)
            delta[:, 1:] = x[:, 1:] - x[:, :-1]
            x = torch.cat([x, delta], dim=-1)
        return x

    def _token_embedding(self, batch_size: int):
        if isinstance(self.token_embed, ForceAxisMetadataEmbedding):
            return self.token_embed(batch_size)
        return self.token_embed.expand(batch_size, -1, -1)

    def forward(self, x: torch.Tensor):
        x = self._build_input_features(x)
        tok = self.value_proj(x.transpose(1, 2)) + self._token_embedding(x.size(0))
        tok = self.encoder(torch.cat([self.cls.expand(tok.size(0), -1, -1), tok], dim=1))
        if self.pool == "cls":
            z = tok[:, 0]
        elif self.pool == "mean":
            z = tok[:, 1:].mean(dim=1)
        elif self.pool == "attn":
            body = tok[:, 1:]
            w = torch.softmax(self.attn_score(body).squeeze(-1), dim=1)
            z = torch.sum(body * w.unsqueeze(-1), dim=1)
        else:
            raise ValueError(f"Unsupported iTransformer pool type: {self.pool}")
        return self.out_drop(self.out_norm(z))


class ITransformerContrastive(torch.nn.Module):
    def __init__(self, nvars: int, context_points: int, n_layers: int, n_heads: int,
                 d_model: int, d_ff: int, dropout: float, head_dropout: float,
                 out_embed_dim: int, arch: str = "itransformer",
                 input_features: str = "raw", conv_stem_kernel: int = 5,
                 pool: str = "cls", use_metadata: bool = False):
        super().__init__()
        if arch in {"itransformer_delta", "itransformer_delta_metadata"}:
            input_features = "raw_delta"
        if arch in {"itransformer_metadata", "itransformer_delta_metadata", "itransformer_conv_metadata"}:
            use_metadata = True
        if arch == "itransformer_conv_metadata":
            input_features = "raw_delta"
        self.arch = arch
        self.input_features = input_features
        self.backbone = InvertedTemporalBackbone(
            nvars=nvars,
            context_points=context_points,
            n_layers=n_layers,
            n_heads=n_heads,
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            head_dropout=head_dropout,
            arch=arch,
            input_features=input_features,
            conv_stem_kernel=conv_stem_kernel,
            pool=pool,
            use_metadata=use_metadata,
        )
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model),
            torch.nn.GELU(),
            torch.nn.Dropout(head_dropout),
            torch.nn.Linear(d_model, out_embed_dim),
        )

    def forward(self, x: torch.Tensor):
        z = self.backbone(x)
        emb = self.proj(z)
        emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-8)
        return emb

# CNN baseline model (same structure as patchtst_cnn.py)
class ResidualTemporalBlock(torch.nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv1 = torch.nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.bn1 = torch.nn.BatchNorm1d(channels)
        self.conv2 = torch.nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.bn2 = torch.nn.BatchNorm1d(channels)
        self.act = torch.nn.GELU()
        self.drop = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.act(h)
        h = self.drop(h)
        h = self.conv2(h)
        h = self.bn2(h)
        return self.act(x + self.drop(h))


class CNNContrastive(torch.nn.Module):
    def __init__(self, nvars: int, base_channels: int, depth: int, kernel_size: int,
                 dropout: float, out_embed_dim: int, head_dropout: float):
        super().__init__()
        self.stem = torch.nn.Sequential(
            torch.nn.Conv1d(nvars, base_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            torch.nn.BatchNorm1d(base_channels),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
        )
        blocks = []
        for i in range(depth):
            blocks.append(ResidualTemporalBlock(base_channels, kernel_size, 2 ** (i % 4), dropout))
        self.encoder = torch.nn.Sequential(*blocks)
        self.pool = torch.nn.AdaptiveAvgPool1d(1)
        if base_channels == out_embed_dim:
            self.proj = torch.nn.Linear(base_channels, out_embed_dim)
        else:
            self.proj = torch.nn.Sequential(
                torch.nn.Linear(base_channels, base_channels),
                torch.nn.GELU(),
                torch.nn.Dropout(head_dropout),
                torch.nn.Linear(base_channels, out_embed_dim),
            )

    def forward(self, x: torch.Tensor):
        x = x.transpose(1, 2)
        z = self.stem(x)
        z = self.encoder(z)
        z = self.pool(z).squeeze(-1)
        emb = self.proj(z)
        emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-8)
        return emb


class ResidualMLPBlock(torch.nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim * 4),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim * 4, hidden_dim),
            torch.nn.Dropout(dropout),
        )
        self.norm = torch.nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor):
        return self.norm(x + self.net(x))


class MLPContrastive(torch.nn.Module):
    def __init__(self, nvars: int, seq_len: int, hidden_dim: int, depth: int,
                 dropout: float, out_embed_dim: int, head_dropout: float):
        super().__init__()
        self.nvars = nvars
        self.seq_len = seq_len
        self.stem = torch.nn.Sequential(
            torch.nn.Linear(seq_len, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
        )
        self.encoder = torch.nn.Sequential(*[ResidualMLPBlock(hidden_dim, dropout) for _ in range(depth)])
        if hidden_dim == out_embed_dim:
            self.proj = torch.nn.Linear(hidden_dim, out_embed_dim)
        else:
            self.proj = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Dropout(head_dropout),
                torch.nn.Linear(hidden_dim, out_embed_dim),
            )

    def forward(self, x: torch.Tensor):
        # x: [B, T, V]
        b, t, v = x.shape
        if t != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {t}")
        if v != self.nvars:
            raise ValueError(f"Expected nvars={self.nvars}, got {v}")
        x = x.transpose(1, 2).contiguous().view(b * v, t)
        z = self.stem(x)
        z = self.encoder(z)
        z = z.view(b, v, -1).mean(dim=1)
        emb = self.proj(z)
        emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-8)
        return emb

# Load force evaluation dataset (validation directory)
def load_force_segments(eval_dir, index_csv="eval_thin_17pct.csv",
                        cols=None, seq_len=80, target_col="label", drop_nonfinite=True, debug=False,
                        include_action=False, drop_zero_force=True, zero_force_eps=1e-8):
    if cols is None:
        cols = [
            "left_fx","left_fy","left_fz","left_tx","left_ty","left_tz",
            "right_fx","right_fy","right_fz","right_tx","right_ty","right_tz"
        ]
    csv_path = os.path.join(eval_dir, index_csv)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)
    required = {"csv_path","start",f"{target_col}_emb_index"}
    miss = required - set(df.columns)
    if miss:
        raise ValueError(f"missing columns: {miss}")
    # Drop rows with NaN in target column (for action/weight/interaction etc.)
    if target_col in df.columns:
        before = len(df)
        df = df.dropna(subset=[target_col])
        dropped = before - len(df)
        if dropped > 0 and debug:
            print(f"[debug] dropped {dropped} rows with NaN in target '{target_col}'")

    segments = []
    labels = []
    actions = []
    meta_csv_paths = []
    meta_starts = []
    skipped_nonfinite = 0
    skipped_short = 0
    skipped_zero_force = 0
    for _, row in df.iterrows():
        recorded = str(resolve_recorded_path(str(row["csv_path"])))
        fcsv = recorded if os.path.isabs(recorded) else os.path.join(eval_dir, "csv", recorded)
        if not os.path.exists(fcsv):
            continue
        raw = pd.read_csv(fcsv)
        if any(c not in raw.columns for c in cols):
            continue
        start = int(row["start"])
        arr = raw[cols].values.astype("float32")[start:start+seq_len, :]
        if arr.shape[0] != seq_len:
            skipped_short += 1
            continue
        if drop_nonfinite and not np.isfinite(arr).all():
            skipped_nonfinite += 1
            continue
        if drop_zero_force and np.max(np.abs(arr)) <= zero_force_eps:
            skipped_zero_force += 1
            continue
        segments.append(arr)  # [seq_len, nvars]
        meta_csv_paths.append(row["csv_path"])
        meta_starts.append(start)
        # Prefer explicit target column if present; fallback to target_col_emb_index
        if target_col in row and not pd.isna(row[target_col]):
            labels.append(str(row[target_col]))
        else:
            labels.append(str(row.get(target_col, row.get(f"{target_col}_emb_index"))))
        if include_action:
            actions.append(str(row["action"]) if "action" in row and not pd.isna(row["action"]) else "")
    if debug:
        print(
            f"[debug] load_force_segments: total_rows={len(df)} usable={len(segments)} "
            f"short_skipped={skipped_short} nonfinite_skipped={skipped_nonfinite} "
            f"zero_force_skipped={skipped_zero_force} zero_force_eps={zero_force_eps}"
        )
    if len(segments)==0:
        raise RuntimeError("No usable segments after filtering.")
    return np.stack(segments), labels, meta_csv_paths, meta_starts, (actions if include_action else None)   # (N, seq_len, nvars), labels, metadata


def load_force_segments_all_rows(eval_dir, index_csv="eval_thin_17pct.csv",
                                 cols=None, seq_len=80, drop_nonfinite=True, debug=False,
                                 drop_zero_force=True, zero_force_eps=1e-8):
    if cols is None:
        cols = [
            "left_fx","left_fy","left_fz","left_tx","left_ty","left_tz",
            "right_fx","right_fy","right_fz","right_tx","right_ty","right_tz"
        ]
    csv_path = os.path.join(eval_dir, index_csv)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)
    required = {"csv_path", "start"}
    miss = required - set(df.columns)
    if miss:
        raise ValueError(f"missing columns: {miss}")

    segments = []
    valid_row_idx = []
    csv_paths = []
    starts = []
    skipped_nonfinite = 0
    skipped_short = 0
    skipped_missing = 0
    skipped_zero_force = 0
    for i, row in df.iterrows():
        recorded = str(resolve_recorded_path(str(row["csv_path"])))
        fcsv = recorded if os.path.isabs(recorded) else os.path.join(eval_dir, "csv", recorded)
        if not os.path.exists(fcsv):
            skipped_missing += 1
            continue
        raw = pd.read_csv(fcsv)
        if any(c not in raw.columns for c in cols):
            skipped_missing += 1
            continue
        start = int(row["start"])
        arr = raw[cols].values.astype("float32")[start:start + seq_len, :]
        if arr.shape[0] != seq_len:
            skipped_short += 1
            continue
        if drop_nonfinite and not np.isfinite(arr).all():
            skipped_nonfinite += 1
            continue
        if drop_zero_force and np.max(np.abs(arr)) <= zero_force_eps:
            skipped_zero_force += 1
            continue
        segments.append(arr)
        valid_row_idx.append(int(i))
        csv_paths.append(str(row["csv_path"]))
        starts.append(start)

    if debug:
        print(
            f"[debug] load_force_segments_all_rows: total_rows={len(df)} usable={len(segments)} "
            f"short_skipped={skipped_short} nonfinite_skipped={skipped_nonfinite} "
            f"zero_force_skipped={skipped_zero_force} zero_force_eps={zero_force_eps} "
            f"missing_skipped={skipped_missing}"
        )
    if len(segments) == 0:
        raise RuntimeError("No usable segments after filtering.")
    return np.stack(segments), df, valid_row_idx, csv_paths, starts


def compute_action_accuracy(y_true, preds, actions):
    if actions is None or len(actions) == 0:
        return None
    if not (len(y_true) == len(preds) == len(actions)):
        return None
    df = pd.DataFrame({"true": y_true, "pred": preds, "action": actions})
    if len(df) == 0:
        return None
    grp = df.groupby("action", dropna=False)
    out = grp.apply(lambda g: pd.Series({
        "total": int(len(g)),
        "correct": int((g["true"] == g["pred"]).sum()),
    }), include_groups=False).reset_index()
    out["accuracy"] = np.where(out["total"] > 0, out["correct"] / out["total"], 0.0)
    out = out.sort_values(["accuracy", "total"], ascending=[False, False]).reset_index(drop=True)
    return out

def build_patchtst_model(nvars, args):
    num_patch = (max(args.context_points, args.patch_len)-args.patch_len)//args.stride + 1
    model = PatchTST(
        c_in=nvars,
        target_dim=args.target_points,
        patch_len=args.patch_len,
        stride=args.stride,
        num_patch=num_patch,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_model=args.d_model,
        shared_embedding=True,
        d_ff=args.d_ff,
        dropout=0.0,
        head_dropout=0.0,
        act="gelu",
        head_type="pretrain",
        res_attention=False
    )
    return model


def infer_cnn_config_from_state(state, args):
    # Prefer checkpoint metadata if available; fallback to state_dict shape inference.
    base_channels = state.get("cnn_base_channels", None)
    depth = state.get("cnn_depth", None)
    if base_channels is None and "stem.0.weight" in state:
        base_channels = int(state["stem.0.weight"].shape[0])
    if depth is None:
        enc_ids = []
        for k in state.keys():
            if k.startswith("encoder.") and ".conv1.weight" in k:
                try:
                    enc_ids.append(int(k.split(".")[1]))
                except Exception:
                    pass
        depth = (max(enc_ids) + 1) if len(enc_ids) > 0 else args.cnn_depth

    out_embed_dim = args.out_embed_dim
    if "proj.weight" in state:
        out_embed_dim = int(state["proj.weight"].shape[0])
    elif "proj.3.weight" in state:
        out_embed_dim = int(state["proj.3.weight"].shape[0])
    elif "proj.0.weight" in state and "proj.3.weight" not in state:
        out_embed_dim = int(state["proj.0.weight"].shape[0])

    if base_channels is None:
        base_channels = args.cnn_base_channels
    return int(base_channels), int(depth), int(out_embed_dim)


def build_cnn_model(nvars, args, state=None, checkpoint=None):
    if checkpoint and checkpoint.get("architecture_version") == "standard_v1":
        config = checkpoint["model_config"]
        model = StandardCNNContrastive(
            nvars=nvars,
            base_channels=int(config["base_channels"]),
            depth=int(config["depth"]),
            kernel_size=int(config["kernel_size"]),
            dropout=float(config["dropout"]),
            out_embed_dim=int(config["out_embed_dim"]),
        )
        return (
            model,
            int(config["base_channels"]),
            int(config["depth"]),
            int(config["out_embed_dim"]),
        )
    if state is not None:
        base_channels, depth, out_embed_dim = infer_cnn_config_from_state(state, args)
    else:
        base_channels, depth, out_embed_dim = args.cnn_base_channels, args.cnn_depth, args.out_embed_dim
    model = CNNContrastive(
        nvars=nvars,
        base_channels=base_channels,
        depth=depth,
        kernel_size=args.cnn_kernel_size,
        dropout=args.cnn_dropout,
        out_embed_dim=out_embed_dim,
        head_dropout=args.head_dropout,
    )
    return model, base_channels, depth, out_embed_dim


def infer_mlp_config_from_state(state, args):
    hidden_dim = state.get("mlp_hidden_dim", None)
    depth = state.get("mlp_depth", None)
    if hidden_dim is None and "stem.0.weight" in state:
        hidden_dim = int(state["stem.0.weight"].shape[0])
    if depth is None:
        enc_ids = []
        for k in state.keys():
            if k.startswith("encoder.") and ".net.0.weight" in k:
                try:
                    enc_ids.append(int(k.split(".")[1]))
                except Exception:
                    pass
        depth = (max(enc_ids) + 1) if len(enc_ids) > 0 else args.mlp_depth

    out_embed_dim = args.out_embed_dim
    if "proj.weight" in state:
        out_embed_dim = int(state["proj.weight"].shape[0])
    elif "proj.3.weight" in state:
        out_embed_dim = int(state["proj.3.weight"].shape[0])
    elif "proj.0.weight" in state and "proj.3.weight" not in state:
        out_embed_dim = int(state["proj.0.weight"].shape[0])

    if hidden_dim is None:
        hidden_dim = args.mlp_hidden_dim
    return int(hidden_dim), int(depth), int(out_embed_dim)


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def target_patchtst_bert_params(nvars, out_embed_dim):
    cfg = PATCHTST_BERT_REF
    num_patch = (max(cfg["context_points"], cfg["patch_len"]) - cfg["patch_len"]) // cfg["stride"] + 1
    ref = PatchTST(
        c_in=nvars,
        target_dim=0,
        patch_len=cfg["patch_len"],
        stride=cfg["stride"],
        num_patch=num_patch,
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        d_model=cfg["d_model"],
        shared_embedding=True,
        d_ff=cfg["d_ff"],
        dropout=cfg["dropout"],
        head_dropout=cfg["head_dropout"],
        act="gelu",
        head_type="pretrain",
        res_attention=False,
    ).backbone
    if cfg["d_model"] == out_embed_dim:
        ref_proj = torch.nn.Sequential(torch.nn.Linear(cfg["d_model"], out_embed_dim))
    else:
        ref_proj = torch.nn.Sequential(
            torch.nn.Linear(cfg["d_model"], cfg["d_model"]),
            torch.nn.GELU(),
            torch.nn.Dropout(cfg["head_dropout"]),
            torch.nn.Linear(cfg["d_model"], out_embed_dim),
        )
    return count_trainable_params(ref) + count_trainable_params(ref_proj)


def choose_mlp_width_for_eval(nvars, args, out_embed_dim, depth):
    target_params = target_patchtst_bert_params(nvars, out_embed_dim)
    best_width = MLP_WIDTH_CANDIDATES[0]
    best_params = -1
    best_gap = float("inf")
    for width in MLP_WIDTH_CANDIDATES:
        model = MLPContrastive(
            nvars=nvars,
            seq_len=args.context_points,
            hidden_dim=width,
            depth=depth,
            dropout=args.mlp_dropout,
            out_embed_dim=out_embed_dim,
            head_dropout=args.head_dropout,
        )
        pcount = count_trainable_params(model)
        gap = abs(pcount - target_params)
        if gap < best_gap:
            best_gap = gap
            best_width = width
            best_params = pcount
    return best_width, best_params, target_params


def build_mlp_model(nvars, args, state=None, checkpoint=None):
    if checkpoint and checkpoint.get("architecture_version") == "standard_v1":
        config = checkpoint["model_config"]
        model = FlatMLPContrastive(
            nvars=nvars,
            seq_len=int(config["seq_len"]),
            hidden_dim=int(config["hidden_dim"]),
            depth=int(config["depth"]),
            dropout=float(config["dropout"]),
            out_embed_dim=int(config["out_embed_dim"]),
        )
        return (
            model,
            int(config["hidden_dim"]),
            int(config["depth"]),
            int(config["out_embed_dim"]),
        )
    if state is not None:
        hidden_dim, depth, out_embed_dim = infer_mlp_config_from_state(state, args)
    else:
        hidden_dim, depth, out_embed_dim = args.mlp_hidden_dim, args.mlp_depth, args.out_embed_dim
    if hidden_dim <= 0:
        hidden_dim, mlp_params, tgt_params = choose_mlp_width_for_eval(nvars, args, out_embed_dim, depth)
        print(
            f"[info] auto-selected mlp_hidden_dim={hidden_dim} for eval "
            f"(mlp_params={mlp_params}, target_patchtst_params={tgt_params})"
        )
    model = MLPContrastive(
        nvars=nvars,
        seq_len=args.context_points,
        hidden_dim=hidden_dim,
        depth=depth,
        dropout=args.mlp_dropout,
        out_embed_dim=out_embed_dim,
        head_dropout=args.head_dropout,
    )
    return model, hidden_dim, depth, out_embed_dim


def build_sequence_baseline_model(nvars, architecture, checkpoint):
    if not checkpoint or checkpoint.get("architecture_version") != "standard_v1":
        raise ValueError(
            f"{architecture} evaluation requires a standard_v1 contrastive checkpoint"
        )
    if checkpoint.get("model_arch") != architecture:
        raise ValueError(
            f"Checkpoint model_arch={checkpoint.get('model_arch')!r}, expected {architecture!r}"
        )
    config = checkpoint["model_config"]
    if architecture == "tcn":
        return ResidualTCNContrastive(nvars=nvars, **config), config
    if architecture == "lstm":
        return LSTMContrastive(nvars=nvars, **config), config
    raise ValueError(f"Unsupported sequence baseline: {architecture}")


def infer_proj_out_dim_from_state(state, default_out_dim):
    if "proj.weight" in state:
        return int(state["proj.weight"].shape[0])
    if "proj.3.weight" in state:
        return int(state["proj.3.weight"].shape[0])
    if "proj.0.weight" in state:
        return int(state["proj.0.weight"].shape[0])
    return int(default_out_dim)


def infer_patchtst_variant_from_state(state, args, raw_nvars):
    arch = args.arch
    pool = args.pool
    itransformer_pool = getattr(args, "itransformer_pool", "cls")
    input_features = args.input_features
    use_channel_embedding = args.use_channel_embedding

    has_conv_stem = any(k.startswith("conv_stem.") for k in state.keys())
    has_itransformer_conv_stem = any(k.startswith("backbone.conv_stem.") for k in state.keys())
    has_itransformer = any(
        k.startswith(("backbone.value_proj.", "backbone.cls", "backbone.out_norm."))
        for k in state.keys()
    )
    has_itransformer_metadata = any(k.startswith("backbone.token_embed.base_embed") for k in state.keys())
    if "backbone.token_embed.base_embed" in state:
        token_nvars = int(state["backbone.token_embed.base_embed"].shape[1])
    elif "backbone.token_embed" in state:
        token_nvars = int(state["backbone.token_embed"].shape[1])
    else:
        token_nvars = None
    has_token_fusion = any(k.startswith("pooler.encoder.") for k in state.keys())
    has_channel_embedding = "channel_embed" in state
    if "pooler.channel_embed" in state:
        token_nvars = int(state["pooler.channel_embed"].shape[1])
    elif "channel_embed" in state:
        token_nvars = int(state["channel_embed"].shape[1])

    if arch == "auto":
        if has_itransformer:
            if has_itransformer_conv_stem:
                arch = "itransformer_conv_metadata"
            elif has_itransformer_metadata and token_nvars == raw_nvars * 2:
                arch = "itransformer_delta_metadata"
            elif has_itransformer_metadata:
                arch = "itransformer_metadata"
            elif token_nvars == raw_nvars * 2:
                arch = "itransformer_delta"
            else:
                arch = "itransformer"
        elif has_token_fusion and has_conv_stem:
            arch = "conv_token_fusion"
        elif has_token_fusion and token_nvars == raw_nvars * 2:
            arch = "delta_token_fusion"
        elif has_token_fusion:
            arch = "token_fusion"
        elif has_conv_stem:
            arch = "conv_stem"
        elif has_channel_embedding:
            arch = "axis_embed"
        else:
            arch = "patchtst"

    if arch in {"delta_token_fusion", "itransformer_delta", "itransformer_delta_metadata", "itransformer_conv_metadata"}:
        input_features = "raw_delta"
    elif token_nvars == raw_nvars * 2:
        input_features = "raw_delta"

    if arch == "axis_embed" or has_channel_embedding or has_itransformer_metadata:
        use_channel_embedding = True

    if "backbone.attn_score.weight" in state:
        itransformer_pool = "attn"

    if not has_token_fusion and pool == "mean":
        if "pooler.patch_score.weight" in state and "pooler.channel_score.weight" in state:
            pool = "two_stage_attn"
        elif "pooler.score.weight" in state:
            pool = "single_attn"
        elif "pooler.cls" in state:
            pool = "channel_cls_patch_attn"

    token_fusion_layers = args.token_fusion_layers
    if has_token_fusion:
        layer_ids = []
        for k in state.keys():
            if k.startswith("pooler.encoder.layers."):
                try:
                    layer_ids.append(int(k.split(".")[3]))
                except Exception:
                    pass
        if layer_ids:
            token_fusion_layers = max(layer_ids) + 1

    conv_stem_kernel = args.conv_stem_kernel
    if "conv_stem.net.0.weight" in state:
        conv_stem_kernel = int(state["conv_stem.net.0.weight"].shape[-1])
    elif "backbone.conv_stem.net.0.weight" in state:
        conv_stem_kernel = int(state["backbone.conv_stem.net.0.weight"].shape[-1])

    return arch, pool, itransformer_pool, input_features, use_channel_embedding, token_fusion_layers, conv_stem_kernel


def extract_embeddings_patchtst(model, force_array, patch_len, stride, pool="mean", proj=None):
    """
    force_array: (N, seq_len, nvars)
    output: (N, d_model)
    """
    device = next(model.parameters()).device
    model.eval()
    embs = []
    with torch.no_grad():
        for batch in torch.split(torch.from_numpy(force_array), 64):  # batch size 64
            # batch: [B, seq_len, nvars] -> [B, num_patch, nvars, patch_len]
            xb_patch, _ = create_patch(batch.to(device), patch_len, stride)
            # backbone 出力: [B, nvars, d_model, num_patch]
            z = model.backbone(xb_patch)
            if pool == "mean":
                # nvars & num_patch 平均 -> [B, d_model]
                z_pool = z.mean(dim=(1,3))
            elif pool == "patchcat_mean":
                # パッチ方向に連結 -> [B, nvars, num_patch*d_model]、その後チャンネル平均 -> [B, num_patch*d_model]
                z_perm = z.permute(0, 1, 3, 2).contiguous()  # [B, nvars, num_patch, d_model]
                z_pool = z_perm.view(z_perm.size(0), z_perm.size(1), -1).mean(dim=1)
            elif pool == "flatten":
                z_pool = z.permute(0,1,3,2).contiguous().view(z.size(0), -1)
            else:
                # 末尾パッチ平均のみ
                z_pool = z[:,:,:, -1].mean(dim=1)  # [B, d_model]
            if proj is not None:
                z_pool = proj(z_pool)
            # L2 正規化（contrastive 空間に揃える）
            z_pool = z_pool / (z_pool.norm(dim=-1, keepdim=True) + 1e-8)
            embs.append(z_pool.cpu())
    return torch.cat(embs, dim=0).numpy()


def extract_embeddings_cnn(model, force_array):
    device = next(model.parameters()).device
    model.eval()
    embs = []
    with torch.no_grad():
        for batch in torch.split(torch.from_numpy(force_array), 64):
            z = model(batch.to(device).float())
            z = z / (z.norm(dim=-1, keepdim=True) + 1e-8)
            embs.append(z.cpu())
    return torch.cat(embs, dim=0).numpy()


def extract_embeddings_mlp(model, force_array):
    device = next(model.parameters()).device
    model.eval()
    embs = []
    with torch.no_grad():
        for batch in torch.split(torch.from_numpy(force_array), 64):
            z = model(batch.to(device).float())
            z = z / (z.norm(dim=-1, keepdim=True) + 1e-8)
            embs.append(z.cpu())
    return torch.cat(embs, dim=0).numpy()


def extract_embeddings_generic(model, force_array):
    device = next(model.parameters()).device
    model.eval()
    embs = []
    with torch.no_grad():
        for batch in torch.split(torch.from_numpy(force_array), 64):
            z = model(batch.to(device).float())
            z = z / (z.norm(dim=-1, keepdim=True) + 1e-8)
            embs.append(z.cpu())
    return torch.cat(embs, dim=0).numpy()


def plot_umap_dual(force_emb, force_labels, out_path, text_emb=None, text_labels=None, ckpt_type="pretrained", debug=False):
    # Merge labels for unified mapping
    all_labels = list(force_labels)
    if text_labels is not None:
        all_labels += list(text_labels)
    unique = sorted(set(all_labels))
    mapping = {l:i for i,l in enumerate(unique)}
    t2d = None
    reducer = umap.UMAP(random_state=42)

    """Fit UMAP on force embeddings only, and then map text embeddings over it: looks good"""
    # f2d = reducer.fit_transform(force_emb)  # (N, 2)
    # if text_emb is not None:
    #     t2d = reducer.transform(text_emb)    # (N, 2)
    """textbase: doesn't work except for annotations"""
    # t2d = reducer.fit_transform(text_emb)
    # f2d = reducer.transform(force_emb)  # (N, 2)
    """create a joint embedding space by fitting on both force and text: looks awful"""
    if ckpt_type == "contrastive" and text_emb is not None:
        combined = np.concatenate([force_emb, text_emb], axis=0)
        combined_2d = reducer.fit_transform(combined)
        f2d = combined_2d[:len(force_emb)]
        t2d = combined_2d[len(force_emb):]
    else:
        f2d = reducer.fit_transform(force_emb)
        t2d = None
        if ckpt_type == "contrastive" and text_emb is not None:
            # Fallback transform (should rarely happen now) – but keeps previous behavior if condition mismatched
            t2d = reducer.transform(text_emb)

    plt.figure(figsize=(5,5))
    # Force points
    f_colors = [mapping[l] for l in force_labels]
    sc = plt.scatter(f2d[:,0], f2d[:,1], c=f_colors, cmap="tab10", s=5, alpha=0.8, label="force")
    # Text points (X marker)
    if t2d is not None:
        t_colors = [mapping[l] for l in text_labels]
    # Outer black outline (slightly larger)
        plt.scatter(t2d[:,0], t2d[:,1], c='black', s=100, marker='X', linewidths=5.0, alpha=0.95, zorder=3)
    # Inner colored marker (overlay slightly smaller)
        plt.scatter(t2d[:,0], t2d[:,1], c=t_colors, cmap="tab10", s=80, marker='X', linewidths=1.2, alpha=0.95, zorder=4, label="text")
    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(color=sc.cmap(sc.norm(mapping[lbl])), label=lbl) for lbl in unique]
    plt.legend(handles=handles, bbox_to_anchor=(1.04,1), loc="upper left", fontsize=12)
    plt.xticks([]); plt.yticks([])
    for spine in ['top','right','left','bottom']:
        plt.gca().spines[spine].set_visible(False)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    if debug:
        print(f"[debug] UMAP(force fit on {'force+text' if (ckpt_type=='contrastive' and text_emb is not None) else 'force only'}) force_emb shape={force_emb.shape} text_emb shape={None if text_emb is None else text_emb.shape}")
    print("Saved:", out_path)

def load_label_text_embeddings(eval_dir, emb_dir="text_emb", debug=False, target_col="label"):
    csv_name=f"{target_col}_index_map.csv"
    emb_dir=f"{emb_dir}/{target_col}"
    path = os.path.join(eval_dir, csv_name)
    if not os.path.exists(path):
        return None, None
    df = pd.read_csv(path)
    if not {target_col,"emb_index"}.issubset(df.columns):
        return None, None
    labels = []
    vecs = []
    skipped_nonfinite = 0
    for _, row in df.iterrows():
        raw_label = str(row[target_col]).strip()
        if target_col == "action":
            normalized_action = raw_label.lower()
            if "accidental drop" in normalized_action or " then " in normalized_action:
                continue
        p = os.path.join(eval_dir, emb_dir, str(row["emb_index"]) + ".pt")
        try:
            t = torch.load(p, map_location='cpu')
        except Exception:
            continue
        if isinstance(t, torch.Tensor):
            if t.ndim == 2 and t.shape[0] == 1:
                t = t.squeeze(0)
            t = t.float()
            if not torch.isfinite(t).all():
                skipped_nonfinite += 1
                continue
            if target_col == "interaction" and raw_label.lower() in {"slow slip", "fast slip"}:
                raise ValueError(
                    "Legacy slow/fast slip prototypes cannot be averaged implicitly. "
                    "Provide one explicit sentence prototype for the slip class."
                )
            vecs.append(t.numpy())
            labels.append(raw_label)
    if skipped_nonfinite > 0 and debug:
        print(f"[debug] skipped {skipped_nonfinite} non-finite text embeddings in {emb_dir}")
    if not vecs:
        return None, None
    return np.stack(vecs), labels

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True,
                    help="pretrained PatchTST .pth (pretrain head)")
    ap.add_argument("--ckpt_type", type=str, required=True, choices=["pretrained","contrastive"],
                    help="checkpoint type: 'pretrained' for original, 'contrastive' to load backbone only")
    ap.add_argument("--model_arch", type=str, default="patchtst", choices=["patchtst", "cnn", "mlp", "tcn", "lstm"],
                    help="model architecture to evaluate")

    ap.add_argument("--eval_dir", type=str, default=str(workspace_path("Genesis/data/eval_04272026/")))
    ap.add_argument("--eval_csv", type=str, default="eval_04272026_thin_15pct.csv")
    ap.add_argument("--context_points", type=int, default=80)
    ap.add_argument("--target_points", type=int, default=96)
    ap.add_argument("--zero_force_eps", type=float, default=1e-8,
                    help="Drop windows whose max absolute force/torque value is <= this threshold.")
    ap.add_argument("--keep_zero_force", action="store_true",
                    help="Keep all-zero force/torque windows in evaluation.")
    ap.add_argument("--patch_len", type=int, default=10)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--n_layers", type=int, default=3)
    ap.add_argument("--n_heads", type=int, default=16)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--d_ff", type=int, default=768)
    ap.add_argument("--pool", type=str, default="mean",
                    choices=["mean","flatten","last","patchcat_mean","single_attn","two_stage_attn","channel_cls_patch_attn"])
    ap.add_argument("--arch", type=str, default="auto",
                    choices=[
                        "auto","patchtst","axis_embed","conv_stem","token_fusion","conv_token_fusion","delta_token_fusion",
                        "itransformer","itransformer_metadata","itransformer_delta",
                        "itransformer_delta_metadata","itransformer_conv_metadata"
                    ],
                    help="PatchTST contrastive architecture variant. auto infers from checkpoint keys.")
    ap.add_argument("--itransformer_pool", type=str, default="cls", choices=["cls","mean","attn"],
                    help="Pooling for iTransformer contrastive checkpoints.")
    ap.add_argument("--input_features", type=str, default="raw", choices=["raw","raw_delta"],
                    help="raw_delta concatenates temporal differences; inferred for delta checkpoints when possible.")
    ap.add_argument("--conv_stem_kernel", type=int, default=5)
    ap.add_argument("--token_fusion_layers", type=int, default=1)
    ap.add_argument("--token_fusion_heads", type=int, default=4)
    ap.add_argument("--token_fusion_ff_mult", type=int, default=4)
    ap.add_argument("--use_channel_embedding", action="store_true")
    ap.add_argument("--head_dropout", type=float, default=0.2)
    # CNN args (default to patchtst_cnn.py defaults)
    ap.add_argument("--cnn_base_channels", type=int, default=128)
    ap.add_argument("--cnn_depth", type=int, default=6)
    ap.add_argument("--cnn_kernel_size", type=int, default=5)
    ap.add_argument("--cnn_dropout", type=float, default=0.2)
    # MLP args (default to patchtst_mlp.py defaults)
    ap.add_argument("--mlp_hidden_dim", type=int, default=0,
                    help="<=0: auto-match PatchTST parameter scale")
    ap.add_argument("--mlp_depth", type=int, default=6)
    ap.add_argument("--mlp_dropout", type=float, default=0.2)
    ap.add_argument("--out_embed_dim", type=int, default=768)
    ap.add_argument("--out", type=str, default="patchtst_umap.png")
    ap.add_argument(
        "--output_name",
        default="",
        help="Optional result directory name; defaults to the checkpoint filename.",
    )
    ap.add_argument("--label_index_map", type=str, default="label_index_map.csv")
    ap.add_argument("--text_type", type=str, default="bert", choices=["siglip","bert","clip"], help="text embedding type")
    ap.add_argument("--report_action_accuracy", action="store_true",
                    help="report per-action accuracy for interaction/weight targets")
    ap.add_argument(
        "--targets",
        default="annotation,label,action,weight,interaction",
        help="Comma-separated evaluation targets; useful with independent target-balanced CSVs.",
    )
    ap.add_argument("--debug", action="store_true")
    ap.add_argument(
        "--skip_plots",
        action="store_true",
        help="Compute classification metrics and predictions without UMAP/confusion-matrix images.",
    )
    args = ap.parse_args()
    allowed_targets = {
        "annotation",
        "label",
        "action",
        "weight",
        "interaction",
        "support_condition",
    }
    targets = [value.strip() for value in args.targets.split(",") if value.strip()]
    invalid_targets = sorted(set(targets) - allowed_targets)
    if not targets or invalid_targets:
        raise ValueError(f"Invalid --targets selection: {invalid_targets or targets}")
    # targets = ["annotation"]
    model_name = args.output_name or os.path.basename(args.model_path).replace('.pth','')
    model_id = model_name.split('_')[0]
    os.makedirs(f"data/{model_id}/{model_name}/umap", exist_ok=True)
    if args.ckpt_type == 'contrastive':
        os.makedirs(f"data/{model_id}/{model_name}/heatmap", exist_ok=True)
        os.makedirs(f"data/{model_id}/{model_name}/pred", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir_map = {"bert": "bert_emb", "clip": "clip_emb", "siglip": "text_emb"}
    emb_base = base_dir_map[args.text_type]
    # ターゲット別メトリクス格納
    metrics_by_target = {}
    action_accuracy_tables = {}
    model = None
    proj = None
    contrastive_state = None
    contrastive_checkpoint = None

    if args.ckpt_type == "contrastive":
        ckpt_all = torch.load(args.model_path, map_location=device)
        contrastive_checkpoint = ckpt_all if isinstance(ckpt_all, dict) else None
        contrastive_state = ckpt_all.get('model', ckpt_all)

    force_segments, eval_df, valid_row_idx, _, _ = load_force_segments_all_rows(
        args.eval_dir, args.eval_csv,
        seq_len=args.context_points,
        drop_nonfinite=True,
        drop_zero_force=not args.keep_zero_force,
        zero_force_eps=args.zero_force_eps,
        debug=args.debug,
    )
    valid_row_idx_set = set(valid_row_idx)
    rowid_to_embed = {rid: i for i, rid in enumerate(valid_row_idx)}
    nvars = force_segments.shape[2]

    if args.model_arch == "patchtst":
        if args.ckpt_type == "contrastive":
            if args.pool == "patchcat_mean":
                raise ValueError("pool='patchcat_mean' is only supported in pretrained eval path, not contrastive wrapper.")
            out_embed_dim = infer_proj_out_dim_from_state(contrastive_state, args.out_embed_dim)
            arch, pool, itransformer_pool, input_features, use_channel_embedding, token_fusion_layers, conv_stem_kernel = infer_patchtst_variant_from_state(
                contrastive_state,
                args,
                nvars,
            )
            args.arch = arch
            args.pool = pool
            args.itransformer_pool = itransformer_pool
            args.input_features = input_features
            args.use_channel_embedding = use_channel_embedding
            args.token_fusion_layers = token_fusion_layers
            args.conv_stem_kernel = conv_stem_kernel
            print(
                f"[info] patchtst contrastive variant: arch={arch} pool={pool} itransformer_pool={itransformer_pool} input_features={input_features} "
                f"use_channel_embedding={use_channel_embedding} token_fusion_layers={token_fusion_layers} "
                f"conv_stem_kernel={conv_stem_kernel}"
            )
            if arch.startswith("itransformer"):
                model = ITransformerContrastive(
                    nvars=nvars,
                    context_points=args.context_points,
                    n_layers=args.n_layers,
                    n_heads=args.n_heads,
                    d_model=args.d_model,
                    d_ff=args.d_ff,
                    dropout=0.0,
                    head_dropout=args.head_dropout,
                    out_embed_dim=out_embed_dim,
                    arch=arch,
                    input_features=input_features,
                    conv_stem_kernel=conv_stem_kernel,
                    pool=itransformer_pool,
                    use_metadata=use_channel_embedding,
                ).to(device)
            else:
                model = PatchTSTContrastive(
                    nvars=nvars,
                    context_points=args.context_points,
                    patch_len=args.patch_len,
                    stride=args.stride,
                    n_layers=args.n_layers,
                    n_heads=args.n_heads,
                    d_model=args.d_model,
                    d_ff=args.d_ff,
                    dropout=0.0,
                    head_dropout=args.head_dropout,
                    out_embed_dim=out_embed_dim,
                    pool=pool,
                    arch=arch,
                    input_features=input_features,
                    conv_stem_kernel=conv_stem_kernel,
                    token_fusion_layers=token_fusion_layers,
                    token_fusion_heads=args.token_fusion_heads,
                    token_fusion_ff_mult=args.token_fusion_ff_mult,
                    use_channel_embedding=use_channel_embedding,
                ).to(device)
        else:
            model = build_patchtst_model(nvars, args).to(device)
    else:
        model = None

    if args.ckpt_type == "pretrained":
        if args.model_arch == "patchtst":
            if args.pool in {"single_attn", "two_stage_attn", "channel_cls_patch_attn"}:
                raise ValueError(f"pool='{args.pool}' requires contrastive checkpoint trained with the same pool head.")
            load_model(args.model_path, model, device=device, with_opt=False)
            print("[info] loaded pretrained checkpoint fully")
        else:
            print("[warn] ckpt_type=pretrained with model_arch in {cnn,mlp} is not supported for embedding eval; use contrastive ckpt.")
            return
    else:  # contrastive
        state = contrastive_state
        if args.model_arch == "patchtst":
            msg = model.load_state_dict(state, strict=False)
            print(
                f"[info] loaded patchtst contrastive ckpt "
                f"(pool={args.pool}) missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}"
            )
        elif args.model_arch == "cnn":
            model, bw, dp, od = build_cnn_model(
                nvars,
                args,
                state=state,
                checkpoint=contrastive_checkpoint,
            )
            model = model.to(device)
            msg = model.load_state_dict(state, strict=False)
            print(
                f"[info] loaded cnn contrastive ckpt "
                f"(base_channels={bw}, depth={dp}, out_embed_dim={od}) "
                f"missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}"
            )
        elif args.model_arch == "mlp":
            model, hw, dp, od = build_mlp_model(
                nvars,
                args,
                state=state,
                checkpoint=contrastive_checkpoint,
            )
            model = model.to(device)
            msg = model.load_state_dict(state, strict=False)
            print(
                f"[info] loaded mlp contrastive ckpt "
                f"(hidden_dim={hw}, depth={dp}, out_embed_dim={od}) "
                f"missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}"
            )
        else:
            model, config = build_sequence_baseline_model(
                nvars,
                args.model_arch,
                contrastive_checkpoint,
            )
            model = model.to(device)
            msg = model.load_state_dict(state, strict=False)
            print(
                f"[info] loaded {args.model_arch} contrastive ckpt "
                f"(config={config}) missing={len(msg.missing_keys)} "
                f"unexpected={len(msg.unexpected_keys)}"
            )
    model.eval()
    # Extract embeddings once and reuse for all targets.
    if args.model_arch == "patchtst":
        if args.ckpt_type == "contrastive":
            embeddings = extract_embeddings_generic(model, force_segments)
        else:
            embeddings = extract_embeddings_patchtst(model, force_segments, args.patch_len,
                                                     args.stride, pool=args.pool, proj=proj)
    elif args.model_arch == "cnn":
        embeddings = extract_embeddings_cnn(model, force_segments)
    elif args.model_arch == "mlp":
        embeddings = extract_embeddings_mlp(model, force_segments)
    else:
        embeddings = extract_embeddings_generic(model, force_segments)

    for target in targets:
        umap_out = f"data/{model_id}/{model_name}/umap/{target}_umap.png"
        heatmap_out = f"data/{model_id}/{model_name}/heatmap/{target}_heatmap.png"
        pred_out = f"data/{model_id}/{model_name}/pred/{target}_pred.csv"
        if target not in eval_df.columns:
            raise ValueError(f"missing columns: {{{target}}}")

        use_action = args.report_action_accuracy and target in {"interaction", "weight"}
        target_df = eval_df.copy()
        interaction_values = target_df.get(
            "interaction", pd.Series("", index=target_df.index)
        ).fillna("").astype(str).str.strip().str.lower()
        legacy_action_values = target_df.get(
            "action", pd.Series("", index=target_df.index)
        ).fillna("").astype(str).str.strip().str.lower()
        accidental_drop_mask = interaction_values.eq("accidental drop") | legacy_action_values.str.contains(
            "accidental drop", regex=False
        )
        if target == "action":
            # The intended action is evaluated only on windows without an
            # accidental-drop interaction.
            target_df = target_df[~accidental_drop_mask].copy()
            valid_action = target_df["action"].fillna("").astype(str).str.strip().str.lower()
            target_df = target_df[
                ~valid_action.isin(["", "unknown"])
                & ~valid_action.str.contains(" then ", regex=False)
            ].copy()
        elif target == "interaction":
            # Accept legacy CSVs where accidental drop lived in action and
            # collapse slow/fast slip into a single occurrence label.
            target_df.loc[accidental_drop_mask, "interaction"] = "accidental drop"
            target_df["interaction"] = target_df["interaction"].replace(
                {"slow slip": "slip", "fast slip": "slip"}
            )
        target_df = target_df.dropna(subset=[target]).copy()
        target_df["__rowid__"] = target_df.index.astype(int)
        target_df = target_df[target_df["__rowid__"].isin(valid_row_idx_set)].copy()
        if len(target_df) == 0:
            raise RuntimeError(f"No usable rows for target={target} after filtering.")

        sel_embed_idx = [rowid_to_embed[int(rid)] for rid in target_df["__rowid__"].tolist()]
        emb_target = embeddings[sel_embed_idx]
        labels = [str(v) for v in target_df[target].tolist()]
        pred_csv_paths = [str(v) for v in target_df["csv_path"].tolist()]
        pred_starts = [int(v) for v in target_df["start"].tolist()]
        actions = None
        if use_action:
            if "action" not in target_df.columns:
                raise ValueError("missing columns: {'action'}")
            actions = [str(v) if pd.notna(v) else "" for v in target_df["action"].tolist()]

        # Load text embeddings with labels when contrastive
        text_emb = text_labels = None
        if args.ckpt_type == 'contrastive':
            print("contrastive ckpt; loading text embeddings")
            text_emb, text_labels = load_label_text_embeddings(args.eval_dir, emb_dir=emb_base, target_col=target, debug=args.debug)
            if text_emb is not None and args.debug:
                print(f"[debug] loaded text_emb shape={text_emb.shape} labels={len(text_labels)} from {emb_base}")

        # UMAP plot 1: all (text embeddings with X)
        if not args.skip_plots:
            plot_umap_dual(emb_target, labels, umap_out, text_emb, text_labels, ckpt_type=args.ckpt_type, debug=args.debug)
        if args.ckpt_type == 'pretrained':
            continue
        # Minimal confusion matrix (contrastive + prototypes available)
        if args.ckpt_type == 'contrastive' and text_emb is not None and len(text_labels) > 0:

            # embeddings already L2-normalized in extract_embeddings
            emb_t = torch.from_numpy(emb_target).float()
            prot = torch.from_numpy(text_emb).float()
            prot = prot / (prot.norm(dim=-1, keepdim=True) + 1e-8)
            sims = emb_t @ prot.t()
            pred_idx = sims.argmax(dim=1).cpu().numpy()
            preds = [text_labels[i] for i in pred_idx]
            y_true = labels
            # Order: sorted unique labels from data (deterministic)
            sel_order = sorted(set(y_true))
            cm = confusion_matrix(y_true, preds, labels=sel_order)
            if not args.skip_plots:
                plt.figure(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=sel_order, yticklabels=sel_order, vmax=max(cm.max(), 1))
                plt.ylabel('True label')
                plt.xlabel('Predicted label')
                plt.tight_layout()
                plt.savefig(heatmap_out)
                plt.close()
            acc = (cm.trace() / cm.sum()) if cm.sum() > 0 else 0.0
            # 追加メトリクス
            micro_precision = precision_score(y_true, preds, average='micro', zero_division=0)
            micro_recall    = recall_score(y_true, preds, average='micro', zero_division=0)
            micro_f1        = f1_score(y_true, preds, average='micro', zero_division=0)
            macro_precision = precision_score(y_true, preds, average='macro', zero_division=0)
            macro_recall    = recall_score(y_true, preds, average='macro', zero_division=0)
            macro_f1        = f1_score(y_true, preds, average='macro', zero_division=0)
            metrics_by_target[target] = {
                "Accuracy": acc,
                "Micro Precision": micro_precision,
                "Micro Recall": micro_recall,
                "Micro F1": micro_f1,
                "Macro Precision": macro_precision,
                "Macro Recall": macro_recall,
                "Macro F1": macro_f1
            }
            if args.skip_plots:
                print(f"[info] classification metrics computed (acc={acc:.4f}, samples={len(y_true)})")
            else:
                print(f"[info] UMAP confusion matrix saved to {heatmap_out} (acc={acc:.4f}, samples={len(y_true)})")
            # Export per-sample predictions with cosine similarities
            export_len = min(len(y_true), len(pred_csv_paths))
            sims_np = sims.cpu().numpy()
            # Prepare label order file (order corresponds to columns of sims/prototypes)
            order_txt = os.path.join(os.path.dirname(pred_out), f"{target}_pred_label_order.txt")
            with open(order_txt, 'w', encoding='utf-8') as f:
                for j, lbl in enumerate(text_labels):
                    f.write(f"{j}\t{lbl}\n")
            # Build rows
            rows = []
            for i in range(export_len):
                row = {
                    "csv_path": pred_csv_paths[i],
                    "start": int(pred_starts[i]),
                    "true": y_true[i],
                    "pred": preds[i]
                }
                # add cosine similarities per label as sim_0, sim_1, ... aligned with text_labels order
                for j in range(len(text_labels)):
                    row[f"sim_{j}"] = float(sims_np[i, j])
                rows.append(row)
            pd.DataFrame(rows).to_csv(pred_out, index=False)
            print(f"[info] Saved per-sample predictions to {pred_out} ({export_len} rows)")
            print(f"[info] Saved label order mapping to {order_txt}")

            if args.report_action_accuracy and target in {"interaction", "weight"}:
                action_acc_df = compute_action_accuracy(y_true, preds, actions)
                if action_acc_df is not None and len(action_acc_df) > 0:
                    action_out = os.path.join(os.path.dirname(pred_out), f"{target}_action_accuracy.csv")
                    action_acc_df.to_csv(action_out, index=False)
                    action_accuracy_tables[target] = action_acc_df
                    best = action_acc_df.iloc[0]
                    worst = action_acc_df.iloc[-1]
                    print(
                        f"[info] action accuracy ({target}) saved to {action_out} | "
                        f"best={best['action']}:{best['accuracy']:.4f} ({int(best['correct'])}/{int(best['total'])}) "
                        f"worst={worst['action']}:{worst['accuracy']:.4f} ({int(worst['correct'])}/{int(worst['total'])})"
                    )
                else:
                    print(f"[warn] action accuracy ({target}) skipped due to missing/invalid action metadata")
        else:
            print("[info] Confusion matrix skipped (need contrastive + text embeddings).")

    
    if args.ckpt_type == "contrastive" and len(metrics_by_target) > 0:
        all_metrics = [
            "Accuracy",
            "Micro Precision","Micro Recall","Micro F1",
            "Macro Precision","Macro Recall","Macro F1"
        ]
        header = ["Metric"] + targets
        lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
        for m in all_metrics:
            row = [m]
            for t in targets:
                if t in metrics_by_target:
                    row.append(f"{metrics_by_target[t][m]:.4f}")
                else:
                    row.append("-")
            lines.append("| " + " | ".join(row) + " |")
        out_md = f"data/{model_id}/{model_name}/umap_metrics_summary.md"
        with open(out_md, 'w', encoding='utf-8') as f:
            f.write("# UMAP Prototype Classification Metrics Summary\n\n")
            f.write("\n".join(lines) + "\n")
            if args.report_action_accuracy:
                for t in ["interaction", "weight"]:
                    if t not in action_accuracy_tables:
                        continue
                    f.write(f"\n## Per-action Accuracy ({t})\n\n")
                    f.write("| action | accuracy | correct | total |\n")
                    f.write("| --- | ---: | ---: | ---: |\n")
                    for _, r in action_accuracy_tables[t].iterrows():
                        f.write(
                            f"| {r['action']} | {float(r['accuracy']):.4f} | {int(r['correct'])} | {int(r['total'])} |\n"
                        )
        print(f"Saved aggregated UMAP metrics markdown -> {out_md}")

if __name__ == "__main__":
    main()
