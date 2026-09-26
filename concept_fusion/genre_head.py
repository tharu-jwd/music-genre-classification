"""87-logit genre head. Sigmoid is applied exactly once, in metrics — never here."""

from __future__ import annotations

import torch
import torch.nn as nn

from concept_fusion.contract import FUSED_DIM, N_GENRE_TAGS
from concept_fusion.validation import ContractError, require_finite, require_tensor


class GenreHead(nn.Module):
    def __init__(self, fused_dim: int = FUSED_DIM, n_tags: int = N_GENRE_TAGS, dropout: float = 0.1):
        super().__init__()
        if n_tags < 1:
            raise ContractError("genre n_tags must be a positive integer")
        self.n_tags = n_tags
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fused_dim, n_tags),
        )

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        x = require_tensor("fused", fused, ndim=2, last=FUSED_DIM)
        require_finite("fused", x)
        logits = self.net(x)
        if logits.shape[-1] != self.n_tags:
            raise ContractError(f"genre logits last dim {logits.shape[-1]} != {self.n_tags}")
        return logits
