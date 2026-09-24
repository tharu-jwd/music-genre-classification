"""Fixture stand-in for B1: song_repr → 87 logits. Not the real Dehan CNN."""

from __future__ import annotations

import torch
import torch.nn as nn

from concept_fusion.contract import FUSED_DIM, N_GENRE_TAGS
from concept_fusion.validation import ContractError, require_finite, require_tensor


class DirectAudioBaseline(nn.Module):
    """Protocol-matched fixture baseline. Replace with Dehan's CNN for paper B1."""

    def __init__(self, song_repr_dim: int = FUSED_DIM, n_tags: int = N_GENRE_TAGS, dropout: float = 0.1):
        super().__init__()
        if n_tags != N_GENRE_TAGS:
            raise ContractError("genre vocabulary is frozen at 87 tags")
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(song_repr_dim, song_repr_dim),
            nn.ReLU(inplace=True),
            nn.Linear(song_repr_dim, n_tags),
        )

    def forward(self, song_repr: torch.Tensor) -> torch.Tensor:
        x = require_tensor("song_repr", song_repr, ndim=2, last=FUSED_DIM)
        require_finite("song_repr", x)
        return self.net(x)
