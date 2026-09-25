"""Adapter from the learned temporal rhythm branch to concept fusion."""

from __future__ import annotations

import torch

from concept_fusion.contract import N_RHYTHM_CONCEPTS, RHYTHM_FEATURES, TOKEN_DIM
from concept_fusion.types import BranchOutput
from concept_fusion.validation import ContractError


def from_rhythm_branch(
    output,
    *,
    supervision_mask: torch.Tensor,
    fusion_mask: torch.Tensor | None = None,
) -> BranchOutput:
    """Expose 10 predictions for primary fusion and retain the embedding for ablation."""
    for name in ("embedding", "predictions", "availability"):
        if not hasattr(output, name):
            raise ContractError(f"rhythm output missing {name}")
    embedding = output.embedding
    predictions = output.predictions
    availability = output.availability
    if not isinstance(embedding, torch.Tensor) or embedding.ndim != 2 or embedding.shape[1] != TOKEN_DIM:
        raise ContractError(f"rhythm embedding must have shape (B,{TOKEN_DIM})")
    batch = embedding.shape[0]
    if (
        not isinstance(predictions, torch.Tensor)
        or predictions.shape != (batch, N_RHYTHM_CONCEPTS)
    ):
        raise ContractError(f"rhythm predictions must have shape (B,{N_RHYTHM_CONCEPTS})")
    if not isinstance(availability, torch.Tensor) or availability.shape != (batch,):
        raise ContractError("rhythm availability must have shape (B,)")
    if supervision_mask.shape != predictions.shape:
        raise ContractError("rhythm supervision_mask must match predictions")
    if fusion_mask is None:
        fusion_mask = availability.to(predictions.dtype).unsqueeze(1)
    if fusion_mask.shape != (batch, 1):
        raise ContractError("rhythm fusion_mask must have shape (B,1)")

    branch = BranchOutput(
        name="rhythm",
        concept_values=predictions,
        supervision_mask=supervision_mask,
        fusion_mask=fusion_mask,
        fusion_token=embedding,
        hidden_token=embedding,
        tag_order=RHYTHM_FEATURES,
    )
    branch.validate(batch=batch, n_concepts=N_RHYTHM_CONCEPTS)
    return branch
