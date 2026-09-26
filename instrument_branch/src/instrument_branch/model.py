"""Song-level instrument head published by the instrument branch.

Matches the notebook contract:

    song_repr (B,128) -> Linear(128,128) -> ReLU -> Dropout(0.1)
                      -> Linear(128,40) -> logits -> sigmoid probabilities
"""

from __future__ import annotations

from typing import TypedDict

import torch
from torch import Tensor, nn

# Official split-0 instrument vocabulary. Fusion reads the same JSON.
N_INSTRUMENT_TAGS = 40


class InstrumentBranchOutput(TypedDict):
    concept_values: Tensor
    logits: Tensor
    supervision_mask: Tensor
    fusion_mask: Tensor
    diagnostics: dict[str, Tensor]


class InstrumentBranch(nn.Module):
    """Pooled song representation to official 40 instrument concepts."""

    def __init__(self, *, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.hidden = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim, N_INSTRUMENT_TAGS)

    def forward(self, pooled_song: Tensor) -> InstrumentBranchOutput:
        hidden = self.hidden(pooled_song)
        logits = self.classifier(hidden)
        return {
            "concept_values": logits.sigmoid(),
            "logits": logits,
            "supervision_mask": torch.ones_like(logits),
            "fusion_mask": torch.ones(len(logits), 1, device=logits.device),
            "diagnostics": {"hidden": hidden.detach()},
        }
