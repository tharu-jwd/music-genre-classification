"""Masked rhythm regression losses."""

import torch
import torch.nn.functional as F
from torch import Tensor


def masked_huber_loss(predictions: Tensor, targets: Tensor, validity_mask: Tensor) -> Tensor:
    """Mean Smooth-L1 over observed finite cells; differentiable zero if none exist."""
    if predictions.shape != targets.shape or predictions.shape != validity_mask.shape:
        raise ValueError("predictions, targets, and validity_mask must have identical shapes")
    if predictions.ndim != 2:
        raise ValueError("rhythm tensors must have shape (B,10)")
    observed = validity_mask.to(torch.bool)
    if bool((observed & ~torch.isfinite(targets)).any()):
        raise ValueError("observed rhythm targets must be finite")
    safe_targets = torch.where(observed, targets, torch.zeros_like(targets))
    safe_predictions = torch.where(observed, predictions, torch.zeros_like(predictions))
    raw = F.smooth_l1_loss(safe_predictions, safe_targets, reduction="none")
    count = observed.sum()
    if int(count.item()) == 0:
        return predictions.sum() * 0.0
    return (raw * observed.to(raw.dtype)).sum() / count
