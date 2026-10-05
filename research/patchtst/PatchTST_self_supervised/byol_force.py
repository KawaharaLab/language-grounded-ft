import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_ema_momentum(base_momentum, step, total_steps):
    if not 0.0 <= base_momentum <= 1.0:
        raise ValueError("BYOL base momentum must be between zero and one")
    if total_steps < 1 or step < 0:
        raise ValueError("BYOL requires a positive total_steps and nonnegative step")
    progress = min(step / max(1, total_steps - 1), 1.0)
    return 1.0 - (1.0 - base_momentum) * (1.0 + math.cos(math.pi * progress)) / 2.0


def prediction_mlp(input_dim, hidden_dim, output_dim):
    if min(input_dim, hidden_dim, output_dim) < 1:
        raise ValueError("BYOL MLP dimensions must be positive")
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim, bias=False),
    )


class BYOLState(nn.Module):
    def __init__(self, encoder, projection_dim=256, hidden_dim=4096):
        super().__init__()
        self.encoder = encoder
        self.projector = prediction_mlp(encoder.output.out_features, hidden_dim, projection_dim)
        self.predictor = prediction_mlp(projection_dim, hidden_dim, projection_dim)
        self.target_encoder = copy.deepcopy(self.encoder).requires_grad_(False)
        self.target_projector = copy.deepcopy(self.projector).requires_grad_(False)
        self.metrics = {}

    def forward(self, first_view, second_view):
        first_embedding = self.encoder(first_view)
        second_embedding = self.encoder(second_view)
        first_projection = self.projector(first_embedding)
        second_projection = self.projector(second_embedding)
        first_prediction = F.normalize(self.predictor(first_projection), dim=-1)
        second_prediction = F.normalize(self.predictor(second_projection), dim=-1)
        with torch.no_grad():
            first_target = F.normalize(self.target_projector(self.target_encoder(first_view)), dim=-1)
            second_target = F.normalize(self.target_projector(self.target_encoder(second_view)), dim=-1)
        first_distance = (first_prediction - second_target).square().sum(dim=-1)
        second_distance = (second_prediction - first_target).square().sum(dim=-1)
        with torch.no_grad():
            normalized_embedding = F.normalize(first_embedding, dim=-1)
            normalized_projection = F.normalize(first_projection, dim=-1)
            self.metrics = {
                "encoder_std": first_embedding.std(dim=0, unbiased=False).mean(),
                "encoder_normalized_std": normalized_embedding.std(dim=0, unbiased=False).mean(),
                "encoder_norm": first_embedding.norm(dim=-1).mean(),
                "projector_normalized_std": normalized_projection.std(dim=0, unbiased=False).mean(),
                "target_normalized_std": first_target.std(dim=0, unbiased=False).mean(),
                "prediction_target_cosine": 1.0 - (first_distance + second_distance).mean() / 4.0,
            }
        return (first_distance + second_distance).mean()

    @torch.no_grad()
    def update_target(self, momentum):
        if not 0.0 <= momentum <= 1.0:
            raise ValueError("BYOL momentum must be between zero and one")
        for online_module, target_module in (
            (self.encoder, self.target_encoder),
            (self.projector, self.target_projector),
        ):
            online_parameters = dict(online_module.named_parameters())
            for name, target_parameter in target_module.named_parameters():
                target_parameter.lerp_(online_parameters[name], 1.0 - momentum)
