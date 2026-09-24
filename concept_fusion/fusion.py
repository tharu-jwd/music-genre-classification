"""Concat, masked gated, and optional self-attention fusion.

Primary: masked gated fusion. Concat is the required simple baseline.
Attention is a secondary comparison. No song_repr shortcut lives here.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from concept_fusion.contract import FUSED_DIM, N_CONCEPTS, TOKEN_DIM
from concept_fusion.types import FusionOutput
from concept_fusion.validation import (
    ContractError,
    require_binary_mask,
    require_finite,
    require_tensor,
)

FusionName = Literal["concat", "gated", "attention"]


def _check_inputs(tokens: torch.Tensor, fusion_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    tok = require_tensor("tokens", tokens, ndim=3, last=TOKEN_DIM)
    if tok.shape[1] != N_CONCEPTS:
        raise ContractError(f"tokens expected (B,{N_CONCEPTS},{TOKEN_DIM}), got {tuple(tok.shape)}")
    mask = require_tensor("fusion_mask", fusion_mask, ndim=2, last=N_CONCEPTS)
    if mask.shape[0] != tok.shape[0]:
        raise ContractError("tokens/fusion_mask batch mismatch")
    require_binary_mask("fusion_mask", mask)
    require_finite("tokens", tok)
    return tok, mask


class ConcatFusion(nn.Module):
    """Zero masked tokens, concat 256-D, project to 128-D. Gates are uniform over enabled."""

    def __init__(self, fused_dim: int = FUSED_DIM, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(TOKEN_DIM)
        self.proj = nn.Sequential(
            nn.Linear(N_CONCEPTS * TOKEN_DIM, fused_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: torch.Tensor, fusion_mask: torch.Tensor) -> FusionOutput:
        tok, mask = _check_inputs(tokens, fusion_mask)
        h = self.norm(tok) * mask.unsqueeze(-1)
        fused = self.proj(h.reshape(h.shape[0], -1))
        enabled = mask.sum(dim=1, keepdim=True).clamp(min=0.0)
        gates = torch.where(enabled > 0, mask / enabled, torch.zeros_like(mask))
        used_null = enabled.squeeze(-1) == 0
        if bool(used_null.any()):
            fused = torch.where(used_null.unsqueeze(-1), torch.zeros_like(fused), fused)
        out = FusionOutput(fused=fused, gates=gates, effective_mask=mask, used_null_token=used_null)
        out.validate(batch=tok.shape[0])
        return out


class MaskedGatedFusion(nn.Module):
    """Global gates over LayerNorm tokens; masked concepts get weight 0.

    All-masked rows use a learned null token projected to 128-D.
    """

    def __init__(self, fused_dim: int = FUSED_DIM, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(TOKEN_DIM)
        self.score = nn.Linear(TOKEN_DIM, 1)
        self.null_token = nn.Parameter(torch.zeros(TOKEN_DIM))
        nn.init.normal_(self.null_token, std=0.02)
        self.out = nn.Sequential(
            nn.Linear(TOKEN_DIM, fused_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: torch.Tensor, fusion_mask: torch.Tensor) -> FusionOutput:
        tok, mask = _check_inputs(tokens, fusion_mask)
        h = self.norm(tok)
        logits = self.score(h).squeeze(-1)  # (B, 4)
        logits = logits.masked_fill(mask < 0.5, -1e9)
        all_off = mask.sum(dim=1) == 0
        # Softmax is well-defined when at least one concept is on.
        alpha = torch.zeros_like(logits)
        if bool((~all_off).any()):
            alpha[~all_off] = F.softmax(logits[~all_off], dim=-1)
        alpha = alpha * mask  # exact zeros on disabled
        pooled = (alpha.unsqueeze(-1) * h).sum(dim=1)
        null = self.norm(self.null_token).unsqueeze(0).expand(tok.shape[0], -1)
        pooled = torch.where(all_off.unsqueeze(-1), null, pooled)
        fused = self.out(pooled)
        out = FusionOutput(
            fused=fused,
            gates=alpha,
            effective_mask=mask,
            used_null_token=all_off,
        )
        out.validate(batch=tok.shape[0])
        return out


class AttentionFusion(nn.Module):
    """Single-head self-attention over LayerNorm tokens; padding mask = disabled concepts."""

    def __init__(self, fused_dim: int = FUSED_DIM, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(TOKEN_DIM)
        self.attn = nn.MultiheadAttention(TOKEN_DIM, num_heads=1, batch_first=True, dropout=0.0)
        self.null_token = nn.Parameter(torch.zeros(TOKEN_DIM))
        nn.init.normal_(self.null_token, std=0.02)
        self.out = nn.Sequential(
            nn.Linear(TOKEN_DIM, fused_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: torch.Tensor, fusion_mask: torch.Tensor) -> FusionOutput:
        tok, mask = _check_inputs(tokens, fusion_mask)
        h = self.norm(tok)
        all_off = mask.sum(dim=1) == 0
        key_pad = mask < 0.5
        # MHA forbids all-True padding rows; swap those to a dummy then overwrite.
        safe_pad = key_pad.clone()
        if bool(all_off.any()):
            safe_pad[all_off] = False
            safe_pad[all_off, 0] = False
        attn_out, attn_w = self.attn(h, h, h, key_padding_mask=safe_pad, need_weights=True)
        # attn_w: (B, 4, 4) average over heads
        gates = attn_w.mean(dim=1) * mask
        denom = gates.sum(dim=1, keepdim=True).clamp(min=1e-8)
        gates = torch.where(mask.sum(dim=1, keepdim=True) > 0, gates / denom, torch.zeros_like(gates))
        pooled = (gates.unsqueeze(-1) * h).sum(dim=1)
        null = self.norm(self.null_token).unsqueeze(0).expand(tok.shape[0], -1)
        pooled = torch.where(all_off.unsqueeze(-1), null, pooled)
        fused = self.out(pooled)
        out = FusionOutput(fused=fused, gates=gates, effective_mask=mask, used_null_token=all_off)
        out.validate(batch=tok.shape[0])
        return out


def build_fusion(name: FusionName, **kwargs) -> nn.Module:
    if name == "concat":
        return ConcatFusion(**kwargs)
    if name == "gated":
        return MaskedGatedFusion(**kwargs)
    if name == "attention":
        return AttentionFusion(**kwargs)
    raise ContractError(f"unknown fusion {name!r}")
