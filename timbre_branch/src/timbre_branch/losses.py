"""Losses for partially observed continuous timbre concepts."""

import torch
import torch.nn.functional as F
from torch import Tensor


def masked_smooth_l1_loss(
    prediction: Tensor,
    target: Tensor,
    valid_mask: Tensor | None = None,
    beta: float = 1.0,
) -> Tensor:
    """Average Smooth L1 over valid descriptor cells only.

    A differentiable zero is returned for an entirely invalid batch so joint
    training can continue without fabricating target values.
    """
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction and target shapes differ: {prediction.shape} vs {target.shape}"
        )
    if valid_mask is None:
        valid_mask = torch.isfinite(target)
    if valid_mask.shape != target.shape:
        raise ValueError("valid_mask must have the same shape as target")

    valid_mask = valid_mask.to(device=prediction.device, dtype=torch.bool)
    finite = torch.isfinite(target) & torch.isfinite(prediction)
    valid_mask = valid_mask & finite
    safe_target = torch.where(valid_mask, target, prediction.detach())
    element_loss = F.smooth_l1_loss(
        prediction, safe_target, reduction="none", beta=beta
    )
    valid_count = valid_mask.sum()
    if int(valid_count.detach().cpu()) == 0:
        return prediction.sum() * 0.0
    return (element_loss * valid_mask.to(element_loss.dtype)).sum() / valid_count
