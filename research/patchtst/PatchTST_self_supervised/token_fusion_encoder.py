import torch
import torch.nn as nn

from patchtst_bert import TokenFusionPooling
from src.callback.patch_mask import create_patch
from src.models.patchTST import PatchTST


class RawTokenFusionEncoder(nn.Module):
    def __init__(
        self,
        nvars=12,
        context_points=80,
        patch_len=10,
        stride=10,
        n_layers=3,
        n_heads=16,
        d_model=128,
        d_ff=768,
        dropout=0.2,
        head_dropout=0.2,
        token_fusion_layers=1,
        token_fusion_heads=8,
        token_fusion_ff_mult=4,
        out_embed_dim=768,
        normalize_output=False,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.normalize_output = normalize_output
        self.output_dim = out_embed_dim
        self.num_patch = (max(context_points, patch_len) - patch_len) // stride + 1
        self.backbone = PatchTST(
            c_in=nvars,
            target_dim=0,
            patch_len=patch_len,
            stride=stride,
            num_patch=self.num_patch,
            n_layers=n_layers,
            n_heads=n_heads,
            d_model=d_model,
            shared_embedding=True,
            d_ff=d_ff,
            dropout=dropout,
            head_dropout=head_dropout,
            act="gelu",
            head_type="pretrain",
            res_attention=False,
        ).backbone
        self.pooler = TokenFusionPooling(
            nvars=nvars,
            num_patch=self.num_patch,
            d_model=d_model,
            n_heads=token_fusion_heads,
            n_layers=token_fusion_layers,
            ff_mult=token_fusion_ff_mult,
            dropout=dropout,
        )
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(d_model, out_embed_dim),
        )

    @property
    def output(self):
        return self.proj[-1]

    def forward_features(self, force_sequence):
        patches, _ = create_patch(force_sequence, self.patch_len, self.stride)
        encoded = self.backbone(patches)
        return self.pooler(encoded)

    def forward(self, force_sequence):
        embedding = self.proj(self.forward_features(force_sequence))
        if self.normalize_output:
            embedding = torch.nn.functional.normalize(embedding, dim=-1)
        return embedding


def encoder_config_from_args(args, nvars):
    return {
        "encoder_type": "patchtst_raw_token_fusion",
        "nvars": int(nvars),
        "context_points": int(args.context_points),
        "patch_len": int(args.patch_len),
        "stride": int(args.stride),
        "n_layers": int(args.n_layers),
        "n_heads": int(args.n_heads),
        "d_model": int(args.d_model),
        "d_ff": int(args.d_ff),
        "dropout": float(args.dropout),
        "head_dropout": float(args.head_dropout),
        "token_fusion_layers": int(args.token_fusion_layers),
        "token_fusion_heads": int(args.token_fusion_heads),
        "token_fusion_ff_mult": int(args.token_fusion_ff_mult),
        "out_embed_dim": int(args.out_embed_dim),
        "normalize_output": False,
    }


def build_token_fusion_encoder(config):
    allowed = {
        "nvars",
        "context_points",
        "patch_len",
        "stride",
        "n_layers",
        "n_heads",
        "d_model",
        "d_ff",
        "dropout",
        "head_dropout",
        "token_fusion_layers",
        "token_fusion_heads",
        "token_fusion_ff_mult",
        "out_embed_dim",
        "normalize_output",
    }
    return RawTokenFusionEncoder(**{key: value for key, value in config.items() if key in allowed})
