"""Concept dropout: training-only, never all four, same mask used at occlusion time."""

from __future__ import annotations

import torch

from concept_fusion.contract import CONCEPT_DROPOUT_P, N_CONCEPTS
from concept_fusion.validation import require_binary_mask, require_tensor


def apply_concept_dropout(
    fusion_mask: torch.Tensor,
    *,
    p: float = CONCEPT_DROPOUT_P,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Drop each enabled branch independently with probability p.

    Never drops all four. If a row would become all-zero, restore one random
    originally-enabled branch (or the first slot if none were enabled — then
    the all-masked fallback in fusion still applies).
    """
    mask = require_tensor("fusion_mask", fusion_mask, ndim=2, last=N_CONCEPTS)
    require_binary_mask("fusion_mask", mask)
    if p <= 0:
        return mask
    if p >= 1:
        raise ValueError("concept dropout p must be in [0, 1)")
    drop = torch.rand(mask.shape, generator=generator, device=mask.device) < p
    out = mask * (~drop).float()
    all_off = out.sum(dim=1) == 0
    if not bool(all_off.any()):
        return out
    # Restore one concept that was originally on; else leave all-masked.
    orig_on = mask > 0.5
    for b in torch.where(all_off)[0]:
        choices = torch.where(orig_on[b])[0]
        if choices.numel() == 0:
            continue
        idx = choices[int(torch.randint(0, choices.numel(), (1,), generator=generator).item())]
        out[b, idx] = 1.0
    return out
