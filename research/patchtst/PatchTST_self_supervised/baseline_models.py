import torch
import torch.nn as nn


class StandardCNNContrastive(nn.Module):
    """Conventional 1D CNN over all force/torque channels."""

    def __init__(
        self,
        nvars: int,
        base_channels: int,
        depth: int,
        kernel_size: int,
        dropout: float,
        out_embed_dim: int,
        normalize_output: bool = True,
    ):
        super().__init__()
        self.normalize_output = normalize_output
        if depth < 1:
            raise ValueError("CNN depth must be positive")
        if kernel_size % 2 == 0:
            raise ValueError("CNN kernel size must be odd")

        blocks = []
        in_channels = nvars
        for stage in range(depth):
            out_channels = base_channels * (2 ** min(stage, 2))
            blocks.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel_size,
                        padding=kernel_size // 2,
                        bias=False,
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.GELU(),
                    nn.MaxPool1d(kernel_size=2, stride=2),
                    nn.Dropout(dropout),
                )
            )
            in_channels = out_channels

        self.features = nn.Sequential(*blocks)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.output = nn.Linear(in_channels, out_embed_dim)

    def forward(self, force: torch.Tensor):
        features = self.features(force.transpose(1, 2))
        embedding = self.output(self.global_pool(features).squeeze(-1))
        if not self.normalize_output:
            return embedding
        return embedding / (embedding.norm(dim=-1, keepdim=True) + 1e-8)


class FlatMLPContrastive(nn.Module):
    """Single MLP over the flattened time-by-axis input without pooling."""

    def __init__(
        self,
        nvars: int,
        seq_len: int,
        hidden_dim: int,
        depth: int,
        dropout: float,
        out_embed_dim: int,
        normalize_output: bool = True,
    ):
        super().__init__()
        self.normalize_output = normalize_output
        if depth < 1:
            raise ValueError("MLP depth must be positive")

        input_dim = nvars * seq_len
        layers = [nn.Flatten(start_dim=1)]
        for layer_index in range(depth):
            layers.extend(
                [
                    nn.Linear(input_dim if layer_index == 0 else hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
        layers.append(nn.Linear(hidden_dim, out_embed_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, force: torch.Tensor):
        embedding = self.network(force)
        if not self.normalize_output:
            return embedding
        return embedding / (embedding.norm(dim=-1, keepdim=True) + 1e-8)


class CausalConv1d(nn.Conv1d):
    """Length-preserving causal convolution."""

    def forward(self, inputs: torch.Tensor):
        outputs = super().forward(inputs)
        return outputs[..., : inputs.size(-1)]


class ResidualTCNBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.network = nn.Sequential(
            CausalConv1d(
                channels,
                channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            CausalConv1d(
                channels,
                channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: torch.Tensor):
        return inputs + self.network(inputs)


class ResidualTCNContrastive(nn.Module):
    """Dilated residual TCN without temporal downsampling.

    The final time step sees the complete fixed-length history when the
    receptive field is at least the input length.  This keeps the model simple
    and avoids the repeated max pooling used by the conventional CNN baseline.
    """

    def __init__(
        self,
        nvars: int,
        channels: int,
        depth: int,
        kernel_size: int,
        dropout: float,
        out_embed_dim: int,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError("TCN depth must be positive")
        if kernel_size < 2:
            raise ValueError("TCN kernel size must be at least 2")

        self.input_projection = nn.Conv1d(nvars, channels, kernel_size=1)
        self.blocks = nn.Sequential(
            *[
                ResidualTCNBlock(
                    channels=channels,
                    kernel_size=kernel_size,
                    dilation=2 ** block_index,
                    dropout=dropout,
                )
                for block_index in range(depth)
            ]
        )
        self.output_norm = nn.LayerNorm(channels)
        self.output = nn.Linear(channels, out_embed_dim)

    def forward(self, force: torch.Tensor):
        features = self.input_projection(force.transpose(1, 2))
        features = self.blocks(features)
        embedding = self.output(self.output_norm(features[..., -1]))
        return embedding / (embedding.norm(dim=-1, keepdim=True) + 1e-8)


class LSTMContrastive(nn.Module):
    """Unidirectional stacked LSTM using the final hidden state."""

    def __init__(
        self,
        nvars: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        out_embed_dim: int,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("LSTM layer count must be positive")
        self.encoder = nn.LSTM(
            input_size=nvars,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, out_embed_dim)

    def forward(self, force: torch.Tensor):
        _, (hidden, _) = self.encoder(force)
        embedding = self.output(
            self.output_dropout(self.output_norm(hidden[-1]))
        )
        return embedding / (embedding.norm(dim=-1, keepdim=True) + 1e-8)


def count_trainable_params(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def choose_cnn_width(
    nvars: int,
    depth: int,
    kernel_size: int,
    dropout: float,
    out_embed_dim: int,
    target_params: int,
):
    best = None
    for width in range(8, 257):
        params = 0
        in_channels = nvars
        for stage in range(depth):
            out_channels = width * (2 ** min(stage, 2))
            params += out_channels * in_channels * kernel_size
            params += 2 * out_channels
            in_channels = out_channels
        params += out_embed_dim * in_channels + out_embed_dim
        candidate = (abs(params - target_params), width, params)
        if best is None or candidate < best:
            best = candidate
    _, width, params = best
    return width, params


def choose_mlp_width(
    nvars: int,
    seq_len: int,
    depth: int,
    dropout: float,
    out_embed_dim: int,
    target_params: int,
):
    best = None
    for width in range(32, 2049):
        input_dim = nvars * seq_len
        params = input_dim * width + width
        params += (depth - 1) * (width * width + width)
        params += width * out_embed_dim + out_embed_dim
        candidate = (abs(params - target_params), width, params)
        if best is None or candidate < best:
            best = candidate
    _, width, params = best
    return width, params


def choose_tcn_width(
    nvars: int,
    depth: int,
    kernel_size: int,
    dropout: float,
    out_embed_dim: int,
    target_params: int,
):
    best = None
    for width in range(8, 513):
        # Input projection, two convolutions and two BatchNorms per block,
        # final LayerNorm, and the output projection.
        params = nvars * width + width
        params += depth * (2 * width * width * kernel_size + 6 * width)
        params += 2 * width
        params += out_embed_dim * width + out_embed_dim
        candidate = (abs(params - target_params), width, params)
        if best is None or candidate < best:
            best = candidate
    _, width, params = best
    return width, params


def choose_lstm_width(
    nvars: int,
    num_layers: int,
    dropout: float,
    out_embed_dim: int,
    target_params: int,
):
    best = None
    for width in range(16, 1025):
        # PyTorch LSTM has input/hidden matrices and two bias vectors for each
        # of its four gates. Subsequent layers receive width-dimensional input.
        params = 4 * width * nvars + 4 * width * width + 8 * width
        params += (num_layers - 1) * (8 * width * width + 8 * width)
        params += 2 * width
        params += out_embed_dim * width + out_embed_dim
        candidate = (abs(params - target_params), width, params)
        if best is None or candidate < best:
            best = candidate
    _, width, params = best
    return width, params
