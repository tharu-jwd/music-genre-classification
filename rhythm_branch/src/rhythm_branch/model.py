"""Temporal rhythm branch over mel-derived shared-encoder features."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from .constants import N_RHYTHM_FEATURES, RHYTHM_EMBEDDING_DIM, RHYTHM_INPUT_DIM


@dataclass(frozen=True)
class RhythmBranchConfig:
    input_dim: int = RHYTHM_INPUT_DIM
    hidden_dim: int = 64
    embedding_dim: int = RHYTHM_EMBEDDING_DIM
    output_dim: int = N_RHYTHM_FEATURES
    temporal_layers: int = 3
    kernel_size: int = 5
    dropout: float = 0.15

    def __post_init__(self) -> None:
        for name in ("input_dim", "hidden_dim", "embedding_dim", "output_dim", "temporal_layers"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.input_dim != RHYTHM_INPUT_DIM:
            raise ValueError(f"shared rhythm input must be {RHYTHM_INPUT_DIM}D")
        if self.embedding_dim != RHYTHM_EMBEDDING_DIM:
            raise ValueError(f"rhythm fusion embedding must be {RHYTHM_EMBEDDING_DIM}D")
        if self.output_dim != N_RHYTHM_FEATURES:
            raise ValueError(f"rhythm regression output must have {N_RHYTHM_FEATURES} fields")
        if self.kernel_size < 3 or self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd integer >= 3")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "RhythmBranchConfig":
        return cls(**dict(values))


@dataclass
class RhythmBranchOutput:
    embedding: Tensor  # (B,64), sent to fusion
    predictions: Tensor  # (B,10), standardized AcousticBrainz estimates
    availability: Tensor  # (B,), derived only from mel token availability


class _ResidualTemporalBlock(nn.Module):
    def __init__(self, width: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.network = nn.Sequential(
            nn.Conv1d(width, width, kernel_size, padding=padding, dilation=dilation),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(width, width, 1),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(width)

    def forward(self, values: Tensor, mask: Tensor) -> Tensor:
        update = self.network(values.transpose(1, 2)).transpose(1, 2)
        values = self.norm(values + update)
        return values * mask.unsqueeze(-1).to(values.dtype)


class RhythmBranch(nn.Module):
    """Learn rhythm from ordered shared-CNN features, never from AB inputs.

    When ``sequence_window_index`` is supplied, every selected audio window is
    convolved independently. This prevents temporal kernels from crossing gaps
    between non-adjacent full-song windows.
    """

    def __init__(self, config: RhythmBranchConfig | None = None) -> None:
        super().__init__()
        self.config = config or RhythmBranchConfig()
        self.input_projection = nn.Sequential(
            nn.Linear(self.config.input_dim, self.config.hidden_dim),
            nn.LayerNorm(self.config.hidden_dim),
            nn.GELU(),
        )
        self.temporal_blocks = nn.ModuleList(
            _ResidualTemporalBlock(
                self.config.hidden_dim,
                self.config.kernel_size,
                dilation=2**layer,
                dropout=self.config.dropout,
            )
            for layer in range(self.config.temporal_layers)
        )
        self.attention = nn.Linear(self.config.hidden_dim, 1)
        self.embedding_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.embedding_dim),
            nn.LayerNorm(self.config.embedding_dim),
            nn.GELU(),
        )
        self.regression_head = nn.Linear(self.config.embedding_dim, self.config.output_dim)

    def forward(
        self,
        encoded_sequence: Tensor,
        sequence_mask: Tensor,
        sequence_window_index: Tensor | None = None,
    ) -> RhythmBranchOutput:
        if encoded_sequence.ndim != 3 or encoded_sequence.shape[-1] != self.config.input_dim:
            raise ValueError(
                f"encoded_sequence must have shape (B,T,{self.config.input_dim}), "
                f"received {tuple(encoded_sequence.shape)}"
            )
        if sequence_mask.shape != encoded_sequence.shape[:2]:
            raise ValueError("sequence_mask must have shape (B,T)")
        if not torch.is_floating_point(encoded_sequence) or not torch.isfinite(encoded_sequence).all():
            raise ValueError("encoded_sequence must be finite floating point")
        mask = sequence_mask.to(torch.bool)
        if sequence_window_index is not None:
            if sequence_window_index.shape != mask.shape:
                raise ValueError("sequence_window_index must have shape (B,T)")
            if torch.any(mask & (sequence_window_index < 0)):
                raise ValueError("valid tokens must have a non-negative window index")

        values = self.input_projection(encoded_sequence)
        values = values * mask.unsqueeze(-1).to(values.dtype)
        if sequence_window_index is None:
            values = self._encode_segment(values, mask)
        else:
            # Sum disjoint masked segments. A convolution can therefore see zeros,
            # but never features from the next selected (possibly distant) window.
            encoded = torch.zeros_like(values)
            valid_indices = sequence_window_index.masked_select(mask)
            if valid_indices.numel():
                for window in torch.unique(valid_indices).tolist():
                    segment_mask = mask & (sequence_window_index == int(window))
                    encoded = encoded + self._encode_segment(values, segment_mask)
            values = encoded

        availability = mask.any(dim=1)
        scores = self.attention(values).squeeze(-1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        # Softmax over an all-masked row is undefined. Supply a harmless temporary
        # position, then erase the resulting pooled value with availability.
        safe_mask = mask.clone()
        if bool((~availability).any()):
            safe_mask[~availability, 0] = True
            scores = scores.masked_fill(~safe_mask, torch.finfo(scores.dtype).min)
            scores[~availability, 0] = 0
        weights = torch.softmax(scores, dim=1) * mask.to(scores.dtype)
        pooled = (values * weights.unsqueeze(-1)).sum(dim=1)
        embedding = self.embedding_head(pooled)
        embedding = embedding * availability.unsqueeze(-1).to(embedding.dtype)
        predictions = self.regression_head(embedding)
        predictions = predictions * availability.unsqueeze(-1).to(predictions.dtype)
        return RhythmBranchOutput(embedding, predictions, availability)

    def _encode_segment(self, values: Tensor, mask: Tensor) -> Tensor:
        result = values * mask.unsqueeze(-1).to(values.dtype)
        for block in self.temporal_blocks:
            result = block(result, mask)
        return result
