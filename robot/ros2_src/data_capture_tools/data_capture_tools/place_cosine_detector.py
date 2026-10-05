"""PatchTST cosine detector used by the real-robot place evaluation."""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn


FORCE_COLUMNS = [
    "left_fx",
    "left_fy",
    "left_fz",
    "left_tx",
    "left_ty",
    "left_tz",
    "right_fx",
    "right_fy",
    "right_fz",
    "right_tx",
    "right_ty",
    "right_tz",
]


class TokenFusionPooling(nn.Module):
    """Pool the channel-patch tokens exactly as patchtst_bert.py does."""

    def __init__(
        self,
        nvars: int,
        num_patch: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        ff_mult: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.channel_embed = nn.Parameter(
            torch.randn(1, nvars, 1, d_model) * 0.02
        )
        self.patch_embed = nn.Parameter(
            torch.randn(1, 1, num_patch, d_model) * 0.02
        )
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        tokens = value.permute(0, 1, 3, 2).contiguous()
        tokens = tokens + self.channel_embed + self.patch_embed
        tokens = tokens.reshape(tokens.size(0), -1, tokens.size(-1))
        cls = self.cls.expand(tokens.size(0), -1, -1)
        return self.encoder(torch.cat([cls, tokens], dim=1))[:, 0]


class TokenFusionContrastive(nn.Module):
    """Inference-only, load-compatible token-fusion architecture."""

    def __init__(self, train_args: dict, patchtst_code: Path) -> None:
        super().__init__()
        if str(train_args.get("arch")) != "token_fusion":
            raise ValueError(
                "This real-robot evaluator currently requires arch=token_fusion"
            )
        if str(train_args.get("input_features", "raw")) != "raw":
            raise ValueError("This evaluator currently requires raw input features")

        if str(patchtst_code) not in sys.path:
            sys.path.insert(0, str(patchtst_code))
        from src.models.patchTST import PatchTST  # pylint: disable=import-outside-toplevel

        nvars = len(FORCE_COLUMNS)
        context_points = int(train_args["context_points"])
        patch_len = int(train_args["patch_len"])
        stride = int(train_args["stride"])
        d_model = int(train_args["d_model"])
        dropout = float(train_args["dropout"])
        head_dropout = float(train_args["head_dropout"])
        num_patch = (max(context_points, patch_len) - patch_len) // stride + 1

        self.context_points = context_points
        self.patch_len = patch_len
        self.stride = stride
        self.backbone = PatchTST(
            c_in=nvars,
            target_dim=0,
            patch_len=patch_len,
            stride=stride,
            num_patch=num_patch,
            n_layers=int(train_args["n_layers"]),
            n_heads=int(train_args["n_heads"]),
            d_model=d_model,
            shared_embedding=True,
            d_ff=int(train_args["d_ff"]),
            dropout=dropout,
            head_dropout=head_dropout,
            act="gelu",
            head_type="pretrain",
            res_attention=False,
        ).backbone
        self.pooler = TokenFusionPooling(
            nvars=nvars,
            num_patch=num_patch,
            d_model=d_model,
            n_heads=int(train_args.get("token_fusion_heads", 4)),
            n_layers=int(train_args.get("token_fusion_layers", 1)),
            ff_mult=int(train_args.get("token_fusion_ff_mult", 4)),
            dropout=dropout,
        )
        out_dim = int(train_args["out_embed_dim"])
        if d_model == out_dim:
            self.proj = nn.Sequential(nn.Linear(d_model, out_dim))
        else:
            self.proj = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(head_dropout),
                nn.Linear(d_model, out_dim),
            )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        seq_len = value.shape[1]
        num_patch = (max(seq_len, self.patch_len) - self.patch_len) // self.stride + 1
        target_len = self.patch_len + self.stride * (num_patch - 1)
        patches = value[:, seq_len - target_len:, :].unfold(
            dimension=1,
            size=self.patch_len,
            step=self.stride,
        )
        features = self.backbone(patches)
        embedding = self.proj(self.pooler(features))
        return embedding / (embedding.norm(dim=-1, keepdim=True) + 1.0e-8)


def _checkpoint_model_arch(checkpoint: dict, train_args: dict) -> str:
    """Return one canonical architecture name for all supported checkpoints."""
    model_arch = str(checkpoint.get("model_arch", "")).strip().lower()
    if model_arch in {"cnn", "mlp"}:
        return model_arch
    patchtst_arch = str(train_args.get("arch", "")).strip().lower()
    if patchtst_arch == "token_fusion":
        return "tokenfusion"
    raise ValueError(
        "Unsupported force encoder checkpoint: "
        f"model_arch={model_arch!r}, arch={patchtst_arch!r}"
    )


def _build_force_encoder(
    checkpoint: dict,
    train_args: dict,
    patchtst_code: Path,
) -> tuple[nn.Module, str]:
    """Build the architecture recorded in a contrastive checkpoint."""
    model_arch = _checkpoint_model_arch(checkpoint, train_args)
    if model_arch == "tokenfusion":
        return TokenFusionContrastive(train_args, patchtst_code), model_arch

    input_features = str(train_args.get("input_features", "raw"))
    if input_features != "raw":
        raise ValueError(
            f"The real-robot evaluator requires raw input, got {input_features!r}"
        )
    if str(patchtst_code) not in sys.path:
        sys.path.insert(0, str(patchtst_code))
    from baseline_models import (  # pylint: disable=import-outside-toplevel
        FlatMLPContrastive,
        StandardCNNContrastive,
    )

    config = checkpoint.get("model_config")
    if not isinstance(config, dict):
        raise ValueError(
            f"{model_arch} checkpoint does not contain model_config"
        )
    out_embed_dim = int(
        config["out_embed_dim"]
        if "out_embed_dim" in config
        else train_args["out_embed_dim"]
    )
    if model_arch == "cnn":
        model = StandardCNNContrastive(
            nvars=len(FORCE_COLUMNS),
            base_channels=int(config["base_channels"]),
            depth=int(config["depth"]),
            kernel_size=int(config["kernel_size"]),
            dropout=float(config["dropout"]),
            out_embed_dim=out_embed_dim,
        )
    else:
        context_points = int(train_args["context_points"])
        seq_len = int(config.get("seq_len", context_points))
        if seq_len != context_points:
            raise ValueError(
                "MLP checkpoint sequence length does not match context_points: "
                f"{seq_len} != {context_points}"
            )
        model = FlatMLPContrastive(
            nvars=len(FORCE_COLUMNS),
            seq_len=seq_len,
            hidden_dim=int(config["hidden_dim"]),
            depth=int(config["depth"]),
            dropout=float(config["dropout"]),
            out_embed_dim=out_embed_dim,
        )
    return model, model_arch


class CosinePlaceDetector:
    """Load a force encoder and score hold/place fixed-text prototypes."""

    def __init__(
        self,
        checkpoint_path: Path,
        normalization_path: Path,
        hold_prototype_path: Path,
        place_prototype_path: Path,
        patchtst_code: Path,
        device_name: str = "auto",
        expected_model_arch: str | None = None,
    ) -> None:
        if device_name == "auto":
            device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_name)

        checkpoint_path = checkpoint_path.expanduser().resolve()
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )
        if not isinstance(checkpoint, dict) or "train_args" not in checkpoint:
            raise ValueError(f"Checkpoint has no train_args: {checkpoint_path}")
        train_args = dict(checkpoint["train_args"])
        force_columns = [
            item for item in str(train_args.get("force_cols", "")).split(",") if item
        ]
        if force_columns != FORCE_COLUMNS:
            raise ValueError(
                f"Checkpoint force-axis order mismatch: {force_columns}"
            )
        if int(train_args.get("context_points", -1)) != 80:
            raise ValueError("The requested evaluation requires context_points=80")

        self.model, self.model_arch = _build_force_encoder(
            checkpoint, train_args, patchtst_code
        )
        if (
            expected_model_arch is not None
            and self.model_arch != expected_model_arch
        ):
            raise ValueError(
                "Selected model architecture does not match checkpoint: "
                f"expected {expected_model_arch!r}, got {self.model_arch!r}"
            )
        self.model = self.model.to(self.device)
        self.model.load_state_dict(checkpoint.get("model", checkpoint), strict=True)
        self.model.eval()
        self.context_points = int(train_args["context_points"])

        # Keep the exact bytes loaded at startup for provenance, even if the
        # source file is later replaced while this process is still running.
        self.normalization_snapshot_bytes = (
            normalization_path.expanduser().resolve().read_bytes()
        )
        self.normalization_sha256 = hashlib.sha256(
            self.normalization_snapshot_bytes
        ).hexdigest()
        normalization = json.loads(self.normalization_snapshot_bytes)
        self.normalization_id = normalization.get("normalization_id")
        if self.normalization_id is not None and (
            not isinstance(self.normalization_id, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", self.normalization_id)
            is None
        ):
            raise ValueError("normalization_id must be a path-safe label of 1-64 characters")
        if list(normalization.get("force_cols", [])) != FORCE_COLUMNS:
            raise ValueError("Real normalization force-axis order is invalid")
        mean = torch.as_tensor(
            normalization["mean"], dtype=torch.float32, device=self.device
        )
        scale = torch.as_tensor(
            normalization.get("scale", normalization["std"]),
            dtype=torch.float32,
            device=self.device,
        )
        if (
            mean.numel() != 12
            or scale.numel() != 12
            or not torch.isfinite(mean).all()
            or not torch.isfinite(scale).all()
            or torch.any(scale <= 0)
        ):
            raise ValueError("Real normalization requires 12 finite axes and positive scales")
        self.mean = mean.view(1, 1, 12)
        self.scale = scale.view(1, 1, 12)
        self.normalization_metadata = normalization

        self.hold = self._load_prototype(hold_prototype_path)
        self.place = self._load_prototype(place_prototype_path)
        self.fixed_surface: torch.Tensor | None = None
        self.wobbling_surface: torch.Tensor | None = None
        self.fixed_surface_prototype_path: Path | None = None
        self.wobbling_surface_prototype_path: Path | None = None
        if self.hold.numel() != int(train_args["out_embed_dim"]):
            raise ValueError("Text and model embedding dimensions do not match")
        self.checkpoint_path = checkpoint_path
        self.normalization_path = normalization_path.expanduser().resolve()
        self.hold_prototype_path = hold_prototype_path.expanduser().resolve()
        self.place_prototype_path = place_prototype_path.expanduser().resolve()

        # Materialize CUDA kernels before the robot starts moving.
        self.score(np.zeros((self.context_points, 12), dtype=np.float32))

    def configure_surface_prototypes(
        self,
        fixed_surface_path: Path,
        wobbling_surface_path: Path,
    ) -> None:
        """Enable fixed-versus-wobbling support scores for every window."""
        fixed_surface_path = fixed_surface_path.expanduser().resolve()
        wobbling_surface_path = wobbling_surface_path.expanduser().resolve()
        fixed_surface = self._load_prototype(fixed_surface_path)
        wobbling_surface = self._load_prototype(wobbling_surface_path)
        expected = self.hold.numel()
        if fixed_surface.numel() != expected or wobbling_surface.numel() != expected:
            raise ValueError(
                "Support prototypes and model embeddings do not have the same size"
            )
        self.fixed_surface = fixed_surface
        self.wobbling_surface = wobbling_surface
        self.fixed_surface_prototype_path = fixed_surface_path
        self.wobbling_surface_prototype_path = wobbling_surface_path

    def _load_prototype(self, path: Path) -> torch.Tensor:
        value = torch.load(
            path.expanduser().resolve(), map_location=self.device, weights_only=True
        )
        value = torch.as_tensor(value, dtype=torch.float32, device=self.device).reshape(-1)
        return value / (value.norm() + 1.0e-8)

    def score(self, window: Sequence[Sequence[float]]) -> dict[str, float]:
        array = np.asarray(window, dtype=np.float32)
        expected = (self.context_points, 12)
        if array.shape != expected or not np.isfinite(array).all():
            raise ValueError(f"Expected a finite force window {expected}, got {array.shape}")
        started_ns = time.perf_counter_ns()
        value = torch.from_numpy(array).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            embedding = self.model((value - self.mean) / self.scale)[0]
            hold_score = torch.dot(embedding, self.hold)
            place_score = torch.dot(embedding, self.place)
            fixed_surface_score = (
                torch.dot(embedding, self.fixed_surface)
                if self.fixed_surface is not None
                else None
            )
            wobbling_surface_score = (
                torch.dot(embedding, self.wobbling_surface)
                if self.wobbling_surface is not None
                else None
            )
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
        elapsed_ms = (time.perf_counter_ns() - started_ns) / 1.0e6
        scores = {
            "hold_score": float(hold_score.detach().cpu()),
            "place_score": float(place_score.detach().cpu()),
            "margin": float((place_score - hold_score).detach().cpu()),
            "inference_ms": elapsed_ms,
        }
        if fixed_surface_score is not None and wobbling_surface_score is not None:
            fixed_value = float(fixed_surface_score.detach().cpu())
            wobbling_value = float(wobbling_surface_score.detach().cpu())
            scores.update(
                {
                    "fixed_surface_score": fixed_value,
                    "wobbling_surface_score": wobbling_value,
                    "fixed_minus_wobbling": fixed_value - wobbling_value,
                }
            )
        return scores

    def write_normalization_snapshot(self, path: Path) -> None:
        """Save precisely the normalization loaded by this detector."""
        path.write_bytes(self.normalization_snapshot_bytes)

    def metadata(self) -> dict:
        metadata = {
            "model_arch": self.model_arch,
            "checkpoint_path": str(self.checkpoint_path),
            "normalization_path": str(self.normalization_path),
            "normalization_id": self.normalization_id,
            "normalization_sha256": self.normalization_sha256,
            "normalization_coordinate_transform": self.normalization_metadata.get(
                "coordinate_transform"
            ),
            "normalization_episode_count": self.normalization_metadata.get("episodes"),
            "hold_prototype_path": str(self.hold_prototype_path),
            "hold_text": "Holding an object.",
            "place_prototype_path": str(self.place_prototype_path),
            "place_text": "Placing an object.",
            "context_points": self.context_points,
            "device": str(self.device),
        }
        if (
            self.fixed_surface_prototype_path is not None
            and self.wobbling_surface_prototype_path is not None
        ):
            metadata.update(
                {
                    "fixed_surface_prototype_path": str(
                        self.fixed_surface_prototype_path
                    ),
                    "fixed_surface_text": "Placing an object on a flat surface.",
                    "wobbling_surface_prototype_path": str(
                        self.wobbling_surface_prototype_path
                    ),
                    "wobbling_surface_text": (
                        "Placing an object on a wobbling support."
                    ),
                }
            )
        return metadata
