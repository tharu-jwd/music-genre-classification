"""
Member 1 — Stage 2 fusion + multi-label genre classifier scaffold.

Supports:
  - Fusion A: concatenate concept vectors → Linear
  - Fusion B: concatenate → single-head attention over concept tokens → pool

Genre head: 87-tag multi-label, sigmoid logits, BCEWithLogitsLoss.

Wire real feature tensors (instrument 64-d + rhythm/timbre/harmony) by
replacing the PlaceholderConceptBundle loader. Always evaluate on split-0 test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn


NUM_GENRE_TAGS = 87
INSTRUMENT_DIM = 64


@dataclass
class ConceptDims:
    instrument: int = INSTRUMENT_DIM
    rhythm: int = 5
    timbre: int = 6
    harmony: int = 18  # 12 chroma + 6 tonnetz means


class LinearFusion(nn.Module):
    """Variant A: concat all concepts → single linear projection."""

    def __init__(self, dims: ConceptDims, fused_dim: int = 128):
        super().__init__()
        in_dim = dims.instrument + dims.rhythm + dims.timbre + dims.harmony
        self.proj = nn.Sequential(
            nn.Linear(in_dim, fused_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        self.out_dim = fused_dim

    def forward(
        self,
        instrument: torch.Tensor,
        rhythm: torch.Tensor,
        timbre: torch.Tensor,
        harmony: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([instrument, rhythm, timbre, harmony], dim=-1)
        return self.proj(x)


class AttentionFusion(nn.Module):
    """Variant B: treat each concept vector as a token; single-head attention + mean pool."""

    def __init__(self, dims: ConceptDims, token_dim: int = 64, fused_dim: int = 128):
        super().__init__()
        self.instrument_proj = nn.Linear(dims.instrument, token_dim)
        self.rhythm_proj = nn.Linear(dims.rhythm, token_dim)
        self.timbre_proj = nn.Linear(dims.timbre, token_dim)
        self.harmony_proj = nn.Linear(dims.harmony, token_dim)
        self.attn = nn.MultiheadAttention(embed_dim=token_dim, num_heads=1, batch_first=True)
        self.out = nn.Sequential(
            nn.Linear(token_dim, fused_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        self.out_dim = fused_dim

    def forward(
        self,
        instrument: torch.Tensor,
        rhythm: torch.Tensor,
        timbre: torch.Tensor,
        harmony: torch.Tensor,
    ) -> torch.Tensor:
        tokens = torch.stack(
            [
                self.instrument_proj(instrument),
                self.rhythm_proj(rhythm),
                self.timbre_proj(timbre),
                self.harmony_proj(harmony),
            ],
            dim=1,
        )  # (B, 4, token_dim)
        attn_out, attn_weights = self.attn(tokens, tokens, tokens, need_weights=True)
        pooled = attn_out.mean(dim=1)
        return self.out(pooled), attn_weights


class GenreClassifierHead(nn.Module):
    def __init__(self, in_dim: int, num_tags: int = NUM_GENRE_TAGS):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_tags)

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        return self.fc(fused)  # logits; apply BCEWithLogitsLoss


class Stage2Model(nn.Module):
    def __init__(
        self,
        fusion: Literal["linear", "attention"] = "linear",
        dims: ConceptDims | None = None,
        fused_dim: int = 128,
    ):
        super().__init__()
        dims = dims or ConceptDims()
        self.fusion_type = fusion
        if fusion == "linear":
            self.fusion = LinearFusion(dims, fused_dim=fused_dim)
            self.head = GenreClassifierHead(self.fusion.out_dim)
        elif fusion == "attention":
            self.fusion = AttentionFusion(dims, fused_dim=fused_dim)
            self.head = GenreClassifierHead(self.fusion.out_dim)
        else:
            raise ValueError(fusion)

    def forward(self, instrument, rhythm, timbre, harmony):
        if self.fusion_type == "attention":
            fused, attn_weights = self.fusion(instrument, rhythm, timbre, harmony)
            logits = self.head(fused)
            return logits, attn_weights
        fused = self.fusion(instrument, rhythm, timbre, harmony)
        return self.head(fused), None


def bce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return nn.functional.binary_cross_entropy_with_logits(logits, targets)


def demo_forward() -> None:
    """Smoke test with random placeholders (no data required)."""
    b = 4
    dims = ConceptDims()
    batch = {
        "instrument": torch.randn(b, dims.instrument),
        "rhythm": torch.randn(b, dims.rhythm),
        "timbre": torch.randn(b, dims.timbre),
        "harmony": torch.randn(b, dims.harmony),
        "y": torch.randint(0, 2, (b, NUM_GENRE_TAGS)).float(),
    }
    for kind in ("linear", "attention"):
        model = Stage2Model(fusion=kind)
        logits, _ = model(batch["instrument"], batch["rhythm"], batch["timbre"], batch["harmony"])
        loss = bce_loss(logits, batch["y"])
        print(f"{kind}: logits={tuple(logits.shape)} loss={loss.item():.4f}")


if __name__ == "__main__":
    demo_forward()
