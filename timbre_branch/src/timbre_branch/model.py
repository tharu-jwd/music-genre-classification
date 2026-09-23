"""Neural timbre branch implementing the proposal's strict concept bottleneck."""

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class TimbreBranchConfig:
    input_dim: int = 128
    hidden_dim: int = 128
    bottleneck_hidden_dim: int = 64
    output_dim: int = 35
    dropout: float = 0.20

    def __post_init__(self) -> None:
        for name in ("input_dim", "hidden_dim", "bottleneck_hidden_dim", "output_dim"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.input_dim != 128:
            raise ValueError("The shared encoder contract requires input_dim=128")
        if self.output_dim != 35:
            raise ValueError("The timbre concept contract requires output_dim=35")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "TimbreBranchConfig":
        return cls(**dict(values))


class TimbreBranch(nn.Module):
    """Predict 35 standardized, named timbre concepts from a 128-D embedding.

    The returned tensor is both ``d_hat`` and ``z_timbre``. There is deliberately
    no path that passes the unrestricted shared embedding around this output.
    """

    def __init__(self, config: TimbreBranchConfig | None = None) -> None:
        super().__init__()
        self.config = config or TimbreBranchConfig()
        self.network = nn.Sequential(
            nn.Linear(self.config.input_dim, self.config.hidden_dim),
            nn.LayerNorm(self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.bottleneck_hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.bottleneck_hidden_dim, self.config.output_dim),
        )

    def forward(self, h_audio: Tensor) -> Tensor:
        if h_audio.ndim != 2:
            raise ValueError(
                f"h_audio must have shape [batch, 128], received {tuple(h_audio.shape)}"
            )
        if h_audio.shape[-1] != self.config.input_dim:
            raise ValueError(
                f"h_audio last dimension must be {self.config.input_dim}, "
                f"received {h_audio.shape[-1]}"
            )
        return self.network(h_audio)
