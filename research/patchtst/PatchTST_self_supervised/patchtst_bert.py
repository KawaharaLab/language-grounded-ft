from workspace_paths import resolve_recorded_path, workspace_path
import os
import math
import argparse
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src.models.patchTST import PatchTST
from src.callback.patch_mask import create_patch
from src.learner import transfer_weights, load_model

import wandb



################################################################################
# Dataset
################################################################################

DEFAULT_FORCE_COLS = [
    "left_fx","left_fy","left_fz","left_tx","left_ty","left_tz",
    "right_fx","right_fy","right_fz","right_tx","right_ty","right_tz"
]


class ForceContrastiveDataset(Dataset):
    """
    Returns (force_seq, bert_embedding)
    force_seq: [seq_len, nvars]
    bert_embedding: tensor [dim]
    """
    def __init__(self, data_dir: str, index_csv: str,
                 seq_len: int = 80, use_cols: List[str] = None, mode: str = 'train',
                 drop_zero_force: bool = True, zero_force_eps: float = 1e-8,
                 return_group_ids: bool = False, emb_dir: str = 'bert_emb'):
        super().__init__()
        self.data_dir = data_dir
        self.index_csv = os.path.join(data_dir, index_csv)
        self.seq_len = seq_len
        self.use_cols = use_cols or DEFAULT_FORCE_COLS
        self.index_name = "emb_index"
        self.drop_zero_force = drop_zero_force
        self.zero_force_eps = zero_force_eps
        self.return_group_ids = return_group_ids
        self.emb_dir = emb_dir

        if not os.path.exists(self.index_csv):
            raise FileNotFoundError(self.index_csv)
        df = pd.read_csv(self.index_csv)
        required = {"csv_path", "start", self.index_name}
        if self.return_group_ids:
            required.add("annotation")
        miss = required - set(df.columns)
        if miss:
            raise ValueError(f"Missing columns {miss} in {self.index_csv}")

        self.force_segments = []
        self.bert_embeddings = []
        self.group_ids = []
        group_map = {}

        # cache each csv
        csv_cache = {}
        embedding_cache = {}
        unique_paths = df['csv_path'].unique()
        for p in unique_paths:
            recorded = resolve_recorded_path(str(p))
            full_p = str(recorded if recorded.is_absolute() else workspace_path(str(recorded)))
            if not os.path.exists(full_p):
                full_p = os.path.join(data_dir, 'csv', str(p))
            if os.path.exists(full_p):
                csv_cache[p] = pd.read_csv(full_p, usecols=self.use_cols).values.astype('float32')

        debug_limit = 100
        debug_prints = 0
        skipped_zero_force = 0
        for _, row in df.iterrows():
            p = row['csv_path']
            if p not in csv_cache:
                continue
            start = int(row['start'])
            seg = csv_cache[p][start:start+self.seq_len, :]
            if seg.shape[0] != self.seq_len:  # skip short
                continue

            # Check finiteness for force segment
            if not np.isfinite(seg).all():
                nonfinite = ~np.isfinite(seg)
                nf_total = int(nonfinite.sum())
                per_col = nonfinite.sum(axis=0).tolist()
                if debug_prints < debug_limit:
                    print(f"[ForceContrastiveDataset] non-finite force segment csv={p} start={start} total={nf_total} per_col(head)={per_col[:min(12,len(per_col))]}")
                    debug_prints += 1
                continue  # drop this pair
            if self.drop_zero_force and np.max(np.abs(seg)) <= self.zero_force_eps:
                skipped_zero_force += 1
                if debug_prints < debug_limit:
                    print(f"[ForceContrastiveDataset] zero-force segment csv={p} start={start} eps={self.zero_force_eps}")
                    debug_prints += 1
                continue
            emb_file = os.path.join(data_dir, self.emb_dir, f"{row[self.index_name]}.pt")
            if not os.path.exists(emb_file):
                continue
            t_emb = embedding_cache.get(emb_file)
            if t_emb is None:
                try:
                    t_emb = torch.load(emb_file, map_location='cpu')
                except Exception as e:
                    if debug_prints < debug_limit:
                        print(f"[ForceContrastiveDataset] failed to load text emb: {emb_file} err={e}")
                        debug_prints += 1
                    continue
                if isinstance(t_emb, torch.Tensor):
                    t_emb = t_emb.detach().cpu()
                else:
                    try:
                        t_emb = torch.tensor(t_emb)
                    except Exception:
                        if debug_prints < debug_limit:
                            print(f"[ForceContrastiveDataset] text emb not tensor-like: {emb_file}")
                            debug_prints += 1
                        continue
                # squeeze optional leading dim
                if t_emb.ndim == 2 and t_emb.shape[0] == 1:
                    t_emb = t_emb.squeeze(0)
                # Check finiteness once for each unique text embedding.
                if not torch.isfinite(t_emb).all():
                    nf_total = int((~torch.isfinite(t_emb)).sum().item())
                    if debug_prints < debug_limit:
                        print(f"[ForceContrastiveDataset] non-finite text emb emb_index={row[self.index_name]} file={emb_file} total={nf_total}")
                        debug_prints += 1
                    continue  # drop this pair
                t_emb = t_emb.float()
                embedding_cache[emb_file] = t_emb

            # Valid pair -> append
            self.force_segments.append(seg)
            self.bert_embeddings.append(t_emb)
            if self.return_group_ids:
                annotation = str(row['annotation'])
                if annotation not in group_map:
                    group_map[annotation] = len(group_map)
                self.group_ids.append(group_map[annotation])

        if len(self.force_segments) == 0:
            raise RuntimeError(f"No usable pairs in {data_dir}")
        if self.drop_zero_force:
            print(f"[ForceContrastiveDataset] skipped_zero_force={skipped_zero_force} eps={self.zero_force_eps}")

    def __len__(self):
        return len(self.force_segments)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.force_segments[idx])          # [seq_len, nvars]
        y = self.bert_embeddings[idx]                          # [dim]
        if self.return_group_ids:
            return x, y, self.group_ids[idx]
        return x, y


################################################################################
# Sigmoid (SigLIP-style) Contrastive Loss
################################################################################

class SigmoidLoss(nn.Module):
    """SigLIP-style symmetric sigmoid loss with learnable temperature & bias."""
    def __init__(self, initial_t_prime: float = 0.0, initial_b: float = 0.0):
        super().__init__()
        initial_t_prime = math.log(1/0.07)
        self.t_prime = nn.Parameter(torch.tensor(initial_t_prime))
        self.b = nn.Parameter(torch.tensor(initial_b))

    def forward(self, img_emb: torch.Tensor, txt_emb: torch.Tensor):
        # img_emb, txt_emb: [N, D]
        n = img_emb.size(0)
        device = img_emb.device
        t = torch.exp(self.t_prime)  # temperature > 0
        logits = (img_emb @ txt_emb.T) * t + self.b
        labels = 2 * torch.eye(n, device=device) - torch.ones(n, n, device=device)
        loss = -torch.sum(F.logsigmoid(labels * logits)) / n
        return loss


class InfoNCELoss(nn.Module):
    """InfoNCE loss with optional force-to-text-only optimization.

    logit_scale = exp(logit_scale_param) where initial scale ~ 1/0.07 (CLIP default temp=0.07).
    """
    def __init__(self, initial_logit_scale: float = math.log(1/0.07), learnable: bool = True,
                 max_scale: float = 100.0, symmetric: bool = True,
                 mask_same_annotation: bool = False,
                 multi_positive_numerator: bool = False):
        super().__init__()
        if learnable:
            self.logit_scale_param = nn.Parameter(torch.tensor(initial_logit_scale))
        else:
            self.register_buffer('logit_scale_param', torch.tensor(initial_logit_scale))
        self.max_scale = max_scale
        self.symmetric = symmetric
        self.mask_same_annotation = mask_same_annotation
        self.multi_positive_numerator = multi_positive_numerator

    def forward(self, img_emb: torch.Tensor, txt_emb: torch.Tensor,
                group_ids: torch.Tensor = None):
        # img_emb, txt_emb: [N, D]
        n = img_emb.size(0)
        device = img_emb.device
        # Normalize just in case (upstream may already do it)
        img_emb = img_emb / (img_emb.norm(dim=-1, keepdim=True) + 1e-8)
        txt_emb = txt_emb / (txt_emb.norm(dim=-1, keepdim=True) + 1e-8)
        scale = self.logit_scale_param.exp().clamp(max=self.max_scale)
        logits_per_image = (img_emb @ txt_emb.T) * scale
        same_annotation = None
        if self.mask_same_annotation or self.multi_positive_numerator:
            if group_ids is None:
                raise ValueError("group_ids are required for annotation-aware InfoNCE")
            group_ids = group_ids.to(device=device)
            same_annotation = group_ids[:, None].eq(group_ids[None, :])
        if self.multi_positive_numerator:
            def multi_positive_loss(logits, positive_mask):
                positive_logits = logits.masked_fill(~positive_mask, -torch.inf)
                return (torch.logsumexp(logits, dim=1)
                        - torch.logsumexp(positive_logits, dim=1)).mean()

            loss_i = multi_positive_loss(logits_per_image, same_annotation)
            if not self.symmetric:
                return loss_i
            loss_t = multi_positive_loss(logits_per_image.T, same_annotation.T)
            return (loss_i + loss_t) / 2.0
        if self.mask_same_annotation:
            diagonal = torch.eye(n, dtype=torch.bool, device=device)
            false_negative_mask = same_annotation & ~diagonal
            logits_per_image = logits_per_image.masked_fill(false_negative_mask, -torch.inf)
        labels = torch.arange(n, device=device)
        loss_i = F.cross_entropy(logits_per_image, labels)
        if not self.symmetric:
            return loss_i
        logits_per_text = logits_per_image.T
        loss_t = F.cross_entropy(logits_per_text, labels)
        return (loss_i + loss_t) / 2.0


################################################################################
# PatchTST Contrastive Wrapper
################################################################################

class SingleStageAttentionPooling(nn.Module):
    """Attention pooling over all (channel, patch) tokens."""
    def __init__(self, d_model: int):
        super().__init__()
        self.score = nn.Linear(d_model, 1)

    def forward(self, z: torch.Tensor):
        # z: [B, V, D, P]
        tok = z.permute(0, 1, 3, 2).contiguous().view(z.size(0), -1, z.size(2))  # [B, V*P, D]
        w = torch.softmax(self.score(tok).squeeze(-1), dim=-1)  # [B, V*P]
        return torch.sum(tok * w.unsqueeze(-1), dim=1)  # [B, D]


class TwoStageAttentionPooling(nn.Module):
    """Patch attention within each channel, then channel attention."""
    def __init__(self, d_model: int):
        super().__init__()
        self.patch_score = nn.Linear(d_model, 1)
        self.channel_score = nn.Linear(d_model, 1)

    def forward(self, z: torch.Tensor):
        # z: [B, V, D, P]
        tok = z.permute(0, 1, 3, 2).contiguous()  # [B, V, P, D]
        w_patch = torch.softmax(self.patch_score(tok).squeeze(-1), dim=2)  # [B, V, P]
        ch_repr = torch.sum(tok * w_patch.unsqueeze(-1), dim=2)  # [B, V, D]
        w_ch = torch.softmax(self.channel_score(ch_repr).squeeze(-1), dim=1)  # [B, V]
        return torch.sum(ch_repr * w_ch.unsqueeze(-1), dim=1)  # [B, D]


class ChannelCLSPatchAttentionPooling(nn.Module):
    """Per-channel CLS-query patch attention, then mean over channels."""
    def __init__(self, d_model: int):
        super().__init__()
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    def forward(self, z: torch.Tensor):
        # z: [B, V, D, P]
        tok = z.permute(0, 1, 3, 2).contiguous()  # [B, V, P, D]
        cls = self.cls.expand(tok.size(0), tok.size(1), -1)  # [B, V, D]
        scores = torch.einsum("bvpd,bvd->bvp", tok, cls) / math.sqrt(tok.size(-1))
        w = torch.softmax(scores, dim=2)  # [B, V, P]
        ch_repr = torch.sum(tok * w.unsqueeze(-1), dim=2)  # [B, V, D]
        return ch_repr.mean(dim=1)  # [B, D]


class TemporalConvStem(nn.Module):
    """Local temporal filter before patching while preserving channel count."""
    def __init__(self, nvars: int, kernel_size: int, dropout: float):
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(nvars, nvars, kernel_size, padding=padding, groups=nvars),
            nn.GELU(),
            nn.Conv1d(nvars, nvars, 1),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor):
        # x: [B, T, V]
        y = x.transpose(1, 2)
        y = y + self.net(y)
        return y.transpose(1, 2)


class TokenFusionPooling(nn.Module):
    """Transformer pooling over channel-patch tokens for cross-axis interaction."""
    def __init__(self, nvars: int, num_patch: int, d_model: int,
                 n_heads: int, n_layers: int, ff_mult: int, dropout: float):
        super().__init__()
        self.channel_embed = nn.Parameter(torch.randn(1, nvars, 1, d_model) * 0.02)
        self.patch_embed = nn.Parameter(torch.randn(1, 1, num_patch, d_model) * 0.02)
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, z: torch.Tensor):
        # z: [B, V, D, P]
        tok = z.permute(0, 1, 3, 2).contiguous()  # [B, V, P, D]
        tok = tok + self.channel_embed + self.patch_embed
        tok = tok.reshape(tok.size(0), -1, tok.size(-1))  # [B, V*P, D]
        cls = self.cls.expand(tok.size(0), -1, -1)
        tok = torch.cat([cls, tok], dim=1)
        tok = self.encoder(tok)
        return tok[:, 0]


class PatchTSTContrastive(nn.Module):
    def __init__(self, nvars: int, context_points: int, patch_len: int, stride: int,
                 n_layers: int, n_heads: int, d_model: int, d_ff: int, dropout: float,
                 head_dropout: float, out_embed_dim: int, pool: str = 'mean',
                 arch: str = 'patchtst', input_features: str = 'raw',
                 conv_stem_kernel: int = 5, token_fusion_layers: int = 1,
                 token_fusion_heads: int = 4, token_fusion_ff_mult: int = 4,
                 use_channel_embedding: bool = False):
        super().__init__()
        self.context_points = context_points
        self.patch_len = patch_len
        self.stride = stride
        self.pool = pool
        self.nvars = nvars
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
        self.num_patch = num_patch
        # Use head_type='pretrain' to reuse backbone weights; we'll add our projection head.
        self.backbone = PatchTST(
            c_in=self.input_nvars,
            target_dim=0,             # unused
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
        ).backbone  # only keep encoder part

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
            self.channel_embed = nn.Parameter(torch.randn(1, self.input_nvars, d_model, 1) * 0.02)
        else:
            self.channel_embed = None

        if self.pool == 'flatten':
            pooled_dim = self.input_nvars * num_patch * d_model
        else:
            pooled_dim = d_model
        proj_layers = [nn.Linear(pooled_dim, out_embed_dim)] if pooled_dim == out_embed_dim else [
            nn.Linear(pooled_dim, d_model), nn.GELU(), nn.Dropout(head_dropout), nn.Linear(d_model, out_embed_dim)
        ]
        self.proj = nn.Sequential(*proj_layers)

    def _build_input_features(self, x: torch.Tensor):
        if self.conv_stem is not None:
            x = self.conv_stem(x)
        if self.input_features == 'raw_delta':
            delta = torch.zeros_like(x)
            delta[:, 1:] = x[:, 1:] - x[:, :-1]
            x = torch.cat([x, delta], dim=-1)
        return x

    def forward(self, x: torch.Tensor):
        """
        x: [B, seq_len, nvars]
        Returns: normalized embedding [B, out_embed_dim]
        """
        x = self._build_input_features(x)
        # patching expects [B, seq_len, nvars]; create_patch yields [B, num_patch, nvars, patch_len]
        xb_patch, _ = create_patch(x, self.patch_len, self.stride)
        # backbone expects [B, num_patch, nvars, patch_len]
        z = self.backbone(xb_patch)  # [B, nvars, d_model, num_patch]
        if self.channel_embed is not None:
            z = z + self.channel_embed
        if self.use_token_fusion:
            z = self.pooler(z)
        elif self.pool == 'mean':
            z = z.mean(dim=(1, 3))          # [B, d_model]
        elif self.pool == 'last':
            z = z[:, :, :, -1].mean(dim=1)  # [B, d_model]
        elif self.pool == 'flatten':
            z = z.permute(0,1,3,2).contiguous().view(z.size(0), -1) # [B, nvars*num_patch*d_model]
        elif self.pooler is not None:
            z = self.pooler(z)
        else:
            raise ValueError(f"Unsupported pool type: {self.pool}")
        emb = self.proj(z)
        emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-8)
        return emb


class ForceAxisMetadataEmbedding(nn.Module):
    """Domain metadata for inverted force/torque tokens."""
    def __init__(self, nvars: int, d_model: int):
        super().__init__()
        self.base_embed = nn.Parameter(torch.randn(1, nvars, d_model) * 0.02)
        self.sensor_embed = nn.Embedding(2, d_model)
        self.kind_embed = nn.Embedding(2, d_model)
        self.axis_embed = nn.Embedding(3, d_model)
        self.feature_embed = nn.Embedding(2, d_model)
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
        sensor_ids = base_sensor.repeat(repeats)
        kind_ids = base_kind.repeat(repeats)
        axis_ids = base_axis.repeat(repeats)
        feature_ids = torch.arange(repeats, dtype=torch.long).repeat_interleave(12).clamp(max=1)
        return sensor_ids, kind_ids, axis_ids, feature_ids

    def forward(self, batch_size: int):
        emb = (
            self.base_embed
            + self.sensor_embed(self.sensor_ids).unsqueeze(0)
            + self.kind_embed(self.kind_ids).unsqueeze(0)
            + self.axis_embed(self.axis_ids).unsqueeze(0)
            + self.feature_embed(self.feature_ids).unsqueeze(0)
        )
        return emb.expand(batch_size, -1, -1)


class InvertedTemporalBackbone(nn.Module):
    """iTransformer-style backbone: variables are tokens, time is token content."""
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
        self.value_proj = nn.Sequential(
            nn.LayerNorm(context_points),
            nn.Linear(context_points, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.token_embed = ForceAxisMetadataEmbedding(self.input_nvars, d_model) if use_metadata else nn.Parameter(
            torch.randn(1, self.input_nvars, d_model) * 0.02
        )
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.attn_score = nn.Linear(d_model, 1) if pool == "attn" else None
        self.out_norm = nn.LayerNorm(d_model)
        self.out_drop = nn.Dropout(head_dropout)

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
        cls = self.cls.expand(tok.size(0), -1, -1)
        tok = self.encoder(torch.cat([cls, tok], dim=1))
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


class ITransformerContrastive(nn.Module):
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
        if input_features not in {"raw", "raw_delta"}:
            raise ValueError(f"Unsupported input_features: {input_features}")
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
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(d_model, out_embed_dim),
        )

    def forward(self, x: torch.Tensor):
        z = self.backbone(x)
        emb = self.proj(z)
        emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-8)
        return emb


################################################################################
# Scheduler (warmup + cosine)
################################################################################

class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-5, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [base_lr * (self.last_epoch + 1) / self.warmup_epochs for base_lr in self.base_lrs]
        progress = (self.last_epoch - self.warmup_epochs) / max(1, (self.total_epochs - self.warmup_epochs))
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return [self.min_lr + (base_lr - self.min_lr) * cosine for base_lr in self.base_lrs]


################################################################################
# Training Loop
################################################################################

def parse_args():
    ap = argparse.ArgumentParser()
    # Data
    ap.add_argument('--data_dir', type=str, default=str(workspace_path("Genesis/data/train_04272026/")))
    ap.add_argument('--index_csv', type=str, default='train_04272026_thin_15pct.csv')
    ap.add_argument('--emb_dir', type=str, default='bert_emb',
                    help='Embedding directory under data_dir.')
    ap.add_argument('--context_points', type=int, default=80)
    ap.add_argument('--force_cols', type=str, default=','.join(DEFAULT_FORCE_COLS))
    ap.add_argument('--zero_force_eps', type=float, default=1e-8,
                    help='Drop samples whose max absolute force/torque value is <= this threshold.')
    ap.add_argument('--keep_zero_force', action='store_true',
                    help='Keep all-zero force/torque windows instead of dropping them.')
    # Validation from train split
    ap.add_argument('--val_from_train_pct', type=float, default=0.10, help='Use this percentage of training data as validation during training.')
    # Model / backbone
    ap.add_argument('--pretrained_path', type=str, default='', help='Path to SSL pre-trained PatchTST weights (.pth). Leave empty to train from scratch.')
    ap.add_argument('--from_scratch', action='store_true', help='Ignore pretrained and start from random initialization.')
    ap.add_argument('--patch_len', type=int, default=10)
    ap.add_argument('--stride', type=int, default=10)
    ap.add_argument('--n_layers', type=int, default=3)
    ap.add_argument('--n_heads', type=int, default=16)
    ap.add_argument('--d_model', type=int, default=128)
    ap.add_argument('--d_ff', type=int, default=768)
    ap.add_argument('--dropout', type=float, default=0.2)
    ap.add_argument('--head_dropout', type=float, default=0.2)
    ap.add_argument('--out_embed_dim', type=int, default=768)
    ap.add_argument('--pool', type=str, default='mean',
                    choices=['mean','last','flatten','single_attn','two_stage_attn','channel_cls_patch_attn'])
    ap.add_argument('--arch', type=str, default='patchtst',
                    choices=[
                        'patchtst','axis_embed','conv_stem','token_fusion','conv_token_fusion','delta_token_fusion',
                        'itransformer','itransformer_metadata','itransformer_delta',
                        'itransformer_delta_metadata','itransformer_conv_metadata'
                    ],
                    help='Architecture variant to evaluate.')
    ap.add_argument('--itransformer_pool', type=str, default='cls', choices=['cls','mean','attn'],
                    help='Pooling for iTransformer variants.')
    ap.add_argument('--input_features', type=str, default='raw', choices=['raw','raw_delta'],
                    help='raw_delta concatenates temporal differences to preserve transition cues.')
    ap.add_argument('--conv_stem_kernel', type=int, default=5,
                    help='Temporal convolution kernel for conv_stem variants.')
    ap.add_argument('--token_fusion_layers', type=int, default=1,
                    help='Cross-axis token fusion Transformer layers.')
    ap.add_argument('--token_fusion_heads', type=int, default=4,
                    help='Attention heads for token fusion. Must divide d_model.')
    ap.add_argument('--token_fusion_ff_mult', type=int, default=4,
                    help='Feed-forward width multiplier for token fusion.')
    ap.add_argument('--use_channel_embedding', action='store_true',
                    help='Add learned axis/channel identity embeddings before pooling.')
    # Optim
    ap.add_argument('--epochs', type=int, default=5000)
    ap.add_argument('--warmup_epochs', type=int, default=30)
    ap.add_argument('--peak_lr', type=float, default=5e-4)
    ap.add_argument('--weight_decay', type=float, default=0.0)
    ap.add_argument('--batch_size', type=int, default=512)
    ap.add_argument('--gradient_clipping', type=float, default=1.0)
    ap.add_argument('--min_lr', type=float, default=1e-5)
    ap.add_argument('--freeze_backbone_epochs', type=int, default=0, help='Freeze backbone for initial epochs')
    # Loss
    ap.add_argument('--loss_type', type=str, default='infonce',
                    choices=['sigmoid','infonce','infonce_force_to_text','infonce_multi_positive',
                             'infonce_multi_positive_numerator'],
                    help='Contrastive loss type. multi_positive masks false negatives; '
                         'multi_positive_numerator puts all same-annotation samples in the numerator.')
    ap.add_argument('--learnable_temp', action='store_true', help='Make temperature (logit scale) learnable for InfoNCE')
    # Misc
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--wandb_api_key', type=str, default=os.environ.get("WANDB_API_KEY", ""))  # user will replace
    ap.add_argument('--run_name', type=str, default='', help='Explicit W&B run and checkpoint name.')
    ap.add_argument('--save_dir', type=str, default='saved_models/bert/')
    return ap.parse_args()


def set_seed(seed: int):
    if seed is None:
        return
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_backbone_weights(model: PatchTSTContrastive, pretrained_path: str, device: str):
    """Transfer weights from a pre-trained PatchTST (pretrain head)."""
    state = torch.load(pretrained_path, map_location=device)
    if 'model' in state:  # saved with optim
        state = state['model']
    # Filter keys belonging to backbone.*
    model_dict = model.backbone.state_dict()
    filtered = {k.replace('backbone.', ''): v for k,v in state.items() if k.startswith('backbone.')}
    # In some saved files keys might already start directly with encoder layers
    missing = []
    for k,v in filtered.items():
        if k in model_dict and model_dict[k].shape == v.shape:
            model_dict[k] = v
        else:
            missing.append(k)
    model.backbone.load_state_dict(model_dict, strict=False)
    if missing:
        print(f"[load_backbone_weights] Unmatched keys (ignored): {missing[:5]} ... total {len(missing)}")
    else:
        print("Backbone weights loaded.")


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    # W&B init
    if args.from_scratch:
        project_name = 'patchtst_scratch_bert'
    else:
        project_name = 'patchtst_bert'
    wandb.login(key=args.wandb_api_key)
    wandb.init(project=project_name, name=args.run_name or None, config=vars(args))
    run_name = wandb.run.name

    os.makedirs(args.save_dir, exist_ok=True)

    cols = args.force_cols.split(',') if args.force_cols else DEFAULT_FORCE_COLS
    use_annotation_groups = args.loss_type in {
        'infonce_multi_positive', 'infonce_multi_positive_numerator'
    }
    dataset_full = ForceContrastiveDataset(
        args.data_dir,
        args.index_csv,
        args.context_points,
        cols,
        mode='train',
        drop_zero_force=not args.keep_zero_force,
        zero_force_eps=args.zero_force_eps,
        return_group_ids=use_annotation_groups,
        emb_dir=args.emb_dir,
    )
    nvars = dataset_full[0][0].shape[1]

    # Build train/val loaders
    if args.val_from_train_pct and args.val_from_train_pct > 0:
        total_len = len(dataset_full)
        val_len = max(1, int(round(total_len * args.val_from_train_pct)))
        train_len = max(1, total_len - val_len)
        if train_len + val_len > total_len:
            val_len = total_len - train_len
        g = torch.Generator()
        g.manual_seed(args.seed if args.seed is not None else 42)
        train_subset, val_subset = torch.utils.data.random_split(dataset_full, [train_len, val_len], generator=g)
        loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
        print(f"Train/Val split from train: train={train_len} val={val_len} (pct={args.val_from_train_pct:.2f})")
    else:
        raise ValueError("validation set must be derived from training set")
    is_itransformer = args.arch.startswith("itransformer")
    if is_itransformer:
        model = ITransformerContrastive(
            nvars=nvars,
            context_points=args.context_points,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            d_model=args.d_model,
            d_ff=args.d_ff,
            dropout=args.dropout,
            head_dropout=args.head_dropout,
            out_embed_dim=args.out_embed_dim,
            arch=args.arch,
            input_features=args.input_features,
            conv_stem_kernel=args.conv_stem_kernel,
            pool=args.itransformer_pool,
            use_metadata=args.use_channel_embedding,
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
            dropout=args.dropout,
            head_dropout=args.head_dropout,
            out_embed_dim=args.out_embed_dim,
            pool=args.pool,
            arch=args.arch,
            input_features=args.input_features,
            conv_stem_kernel=args.conv_stem_kernel,
            token_fusion_layers=args.token_fusion_layers,
            token_fusion_heads=args.token_fusion_heads,
            token_fusion_ff_mult=args.token_fusion_ff_mult,
            use_channel_embedding=args.use_channel_embedding,
        ).to(device)

    if (not args.from_scratch) and args.pretrained_path:
        if is_itransformer:
            print("[info] iTransformer variants are not compatible with PatchTST SSL weights; training from scratch.")
        else:
            load_backbone_weights(model, args.pretrained_path, device)
    else:
        print("[info] Training from scratch (no pretrained backbone loaded).")

    if args.loss_type == 'sigmoid':
        criterion = SigmoidLoss().to(device)
        crit_params = list(criterion.parameters())
    else:  # InfoNCE
        criterion = InfoNCELoss(
            learnable=args.learnable_temp,
            symmetric=args.loss_type != 'infonce_force_to_text',
            mask_same_annotation=args.loss_type == 'infonce_multi_positive',
            multi_positive_numerator=args.loss_type == 'infonce_multi_positive_numerator',
        ).to(device)
        # If not learnable, criterion has no grad params
        crit_params = [p for p in criterion.parameters() if p.requires_grad]
    # Combine parameters (optionally freeze backbone initially)
    backbone_params = list(model.backbone.parameters())
    backbone_param_ids = {id(p) for p in backbone_params}
    head_params = [p for p in model.parameters() if id(p) not in backbone_param_ids]
    optimizer = torch.optim.Adam(
        [{'params': backbone_params, 'lr': args.peak_lr},
         {'params': head_params, 'lr': args.peak_lr},
         {'params': crit_params, 'lr': args.peak_lr}],
        lr=args.peak_lr, weight_decay=args.weight_decay
    )

    scheduler = WarmupCosineScheduler(optimizer, args.warmup_epochs, args.epochs, min_lr=args.min_lr)

    # Optionally freeze backbone
    if args.freeze_backbone_epochs > 0:
        for p in backbone_params:
            p.requires_grad = False
        print(f"Backbone frozen for first {args.freeze_backbone_epochs} epochs.")


    best_val = float('inf')
    best_path = os.path.join(args.save_dir, f"{run_name}_best.pth")

    for epoch in range(args.epochs):
        model.train()
        if args.freeze_backbone_epochs and epoch == args.freeze_backbone_epochs:
            for p in backbone_params:
                p.requires_grad = True
            print('Backbone unfrozen.')

        scheduler.step()
        total_loss = 0.0
        for step, batch in enumerate(loader):
            if use_annotation_groups:
                force_seq, txt_emb, group_ids = batch
                group_ids = group_ids.to(device)
            else:
                force_seq, txt_emb = batch
                group_ids = None
            force_seq = force_seq.to(device)                  # [B, seq_len, nvars]
            txt_emb = txt_emb.to(device).float()
            # Normalize text embeddings
            txt_emb = txt_emb / (txt_emb.norm(dim=-1, keepdim=True) + 1e-8)

            optimizer.zero_grad(set_to_none=True)
            force_emb = model(force_seq)                      # [B, out_embed_dim]
            if isinstance(criterion, InfoNCELoss):
                loss = criterion(force_emb, txt_emb, group_ids=group_ids)
            else:
                loss = criterion(force_emb, txt_emb)
            if torch.isnan(loss):
                print(f"NaN loss at epoch {epoch} step {step}")
                return
            loss.backward()
            if args.gradient_clipping > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clipping)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / (step + 1)
        val_loss = None
        if val_loader is not None:
            model.eval()
            v_total = 0.0
            v_steps = 0
            with torch.no_grad():
                for val_batch in val_loader:
                    if use_annotation_groups:
                        v_force, v_txt, v_group_ids = val_batch
                        v_group_ids = v_group_ids.to(device)
                    else:
                        v_force, v_txt = val_batch
                        v_group_ids = None
                    v_force = v_force.to(device)
                    v_txt = v_txt.to(device).float()
                    v_txt = v_txt / (v_txt.norm(dim=-1, keepdim=True) + 1e-8)
                    v_force_emb = model(v_force)
                    if isinstance(criterion, InfoNCELoss):
                        v_loss = criterion(v_force_emb, v_txt, group_ids=v_group_ids)
                    else:
                        v_loss = criterion(v_force_emb, v_txt)
                    v_total += v_loss.item()
                    v_steps += 1
            val_loss = v_total / max(1, v_steps)
            if val_loss < best_val:
                best_val = val_loss
                torch.save({'model': model.state_dict()}, best_path)
                print(f"  * New best val_loss {best_val:.4f} -> saved {best_path}")

        log_data = {'epoch': epoch+1, 'train_loss': avg_loss, 'lr': scheduler.get_last_lr()[0]}
        # Add loss-specific parameters
        if isinstance(criterion, SigmoidLoss):
            log_data['t_prime'] = criterion.t_prime.item()
            log_data['bias'] = criterion.b.item()
        elif isinstance(criterion, InfoNCELoss):
            log_data['logit_scale'] = criterion.logit_scale_param.exp().item()
        if val_loss is not None:
            log_data['valid_loss'] = val_loss
        msg = f"Epoch {epoch+1}/{args.epochs} - train: {avg_loss:.4f}"
        if val_loss is not None:
            msg += f" val: {val_loss:.4f}"
        msg += f" lr: {log_data['lr']:.6f}"
        print(msg)
        wandb.log(log_data)

        # Save periodic
        if (epoch + 1) % max(10, args.epochs//10) == 0 or (epoch + 1) == args.epochs:
            ckpt_path = os.path.join(args.save_dir, f"{run_name}_epoch{epoch+1}.pth")
            torch.save({'model': model.state_dict()}, ckpt_path)
            print('Saved', ckpt_path)

    wandb.finish()


if __name__ == '__main__':
    main()
