"""Branch-private residual adapters for integration experiment I1."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class I1AdapterConfig:
    input_dim: int = 128
    hidden_dim: int = 32
    dropout: float = 0.10

    def __post_init__(self) -> None:
        if self.input_dim != 128:
            raise ValueError("I1 must preserve the shared encoder width of 128")
        if self.hidden_dim != 32:
            raise ValueError("I1 requires the documented 128 -> 32 -> 128 adapter")
        if self.dropout != 0.10:
            raise ValueError("I1 requires adapter dropout=0.10")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


class ZeroInitializedResidualAdapter(nn.Module):
    """Shape-preserving I1 adapter with an initially exact identity mapping."""

    def __init__(self, config: I1AdapterConfig | None = None) -> None:
        super().__init__()
        self.config = config or I1AdapterConfig()
        self.norm = nn.LayerNorm(self.config.input_dim)
        self.down = nn.Linear(self.config.input_dim, self.config.hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(self.config.dropout)
        self.up = nn.Linear(self.config.hidden_dim, self.config.input_dim)
        self.residual_scale = nn.Parameter(torch.zeros(()))

    def forward(self, values: Tensor, mask: Tensor | None = None) -> Tensor:
        if values.ndim not in (2, 3) or values.shape[-1] != self.config.input_dim:
            raise ValueError(
                "I1 adapter input must be [B,128] or [B,T,128], received "
                f"{tuple(values.shape)}"
            )
        residual = self.up(self.dropout(self.activation(self.down(self.norm(values)))))
        adapted = values + self.residual_scale * residual
        if mask is not None:
            if values.ndim != 3 or mask.shape != values.shape[:2]:
                raise ValueError("temporal adapter mask must have shape [B,T]")
            adapted = adapted * mask.unsqueeze(-1).to(adapted.dtype)
        return adapted


class I1BranchAdapters(nn.Module):
    """Four independent adapters without changing the established routing."""

    def __init__(self, config: I1AdapterConfig | None = None) -> None:
        super().__init__()
        config = config or I1AdapterConfig()
        self.config = config
        self.instrument = ZeroInitializedResidualAdapter(config)
        self.timbre = ZeroInitializedResidualAdapter(config)
        self.rhythm = ZeroInitializedResidualAdapter(config)
        self.harmony = ZeroInitializedResidualAdapter(config)

    def pooled(self, branch: str, pooled_song: Tensor) -> Tensor:
        if branch not in {"instrument", "timbre"}:
            raise ValueError(f"{branch} is not a pooled I1 branch")
        return getattr(self, branch)(pooled_song)

    def temporal(self, branch: str, sequence: Tensor, mask: Tensor) -> Tensor:
        if branch not in {"rhythm", "harmony"}:
            raise ValueError(f"{branch} is not a temporal I1 branch")
        return getattr(self, branch)(sequence, mask)

