"""Masked losses for temporal harmony pseudo-supervision."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def masked_soft_target_cross_entropy(
    logits: Tensor,
    targets: Tensor,
    target_mask: Tensor,
    sequence_mask: Tensor,
) -> Tensor:
    """Distribution loss with a differentiable zero for no-supervision batches."""
    if logits.shape != targets.shape or logits.ndim != 3:
        raise ValueError("soft-target logits and targets must share batch x time x class shape")
    if target_mask.shape != logits.shape[:2] or sequence_mask.shape != logits.shape[:2]:
        raise ValueError("soft-target masks must share the logits batch/time axes")
    valid = target_mask.to(torch.bool) & sequence_mask.to(torch.bool)
    if torch.any(valid):
        selected = targets[valid]
        if not torch.isfinite(selected).all() or (selected < 0).any():
            raise ValueError("valid soft targets must be finite and non-negative")
        if not torch.allclose(
            selected.sum(dim=-1),
            torch.ones(len(selected), device=selected.device, dtype=selected.dtype),
            atol=1e-5,
        ):
            raise ValueError("valid soft targets must sum to one")
        losses = -(selected * F.log_softmax(logits[valid], dim=-1)).sum(dim=-1)
        return losses.mean()
    return logits.sum() * 0.0


def masked_hard_classification_loss(
    logits: Tensor,
    labels: Tensor,
    target_mask: Tensor,
    sequence_mask: Tensor,
) -> Tensor:
    """Masked hard-label loss for an accepted chord teacher."""
    if logits.ndim != 3 or labels.shape != logits.shape[:2]:
        raise ValueError("hard-label logits/labels must share batch/time axes")
    if target_mask.shape != labels.shape or sequence_mask.shape != labels.shape:
        raise ValueError("hard-label masks must share the label shape")
    valid = target_mask.to(torch.bool) & sequence_mask.to(torch.bool)
    if torch.any(valid):
        selected = labels[valid]
        if selected.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }:
            raise ValueError("hard labels must use an integer dtype")
        if (selected < 0).any() or (selected >= logits.shape[-1]).any():
            raise ValueError("hard labels fall outside the chord vocabulary")
        return F.cross_entropy(logits[valid], selected.to(torch.long))
    return logits.sum() * 0.0
