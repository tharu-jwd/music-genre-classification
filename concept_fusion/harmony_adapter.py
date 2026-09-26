"""Adapter from the temporal harmony branch to the shared fusion contract."""

from __future__ import annotations

import torch

from concept_fusion.contract import (
    HARMONY_FEATURES,
    N_HARMONY_CHROMA,
    N_HARMONY_CHORDS,
    N_HARMONY_DESCRIPTORS,
    TOKEN_DIM,
)
from concept_fusion.types import BranchOutput
from concept_fusion.validation import ContractError


def from_temporal_harmony_branch(
    output,
    *,
    chroma_target_mask: torch.Tensor | None = None,
    descriptor_supervision_mask: torch.Tensor | None = None,
    fusion_mask: torch.Tensor | None = None,
    hidden_token: torch.Tensor | None = None,
) -> BranchOutput:
    """Pool predicted chroma probabilities and preserve the embedding for ablation.

    When the branch has a descriptor head, its song-level descriptor predictions
    are the primary fusion input. Otherwise, ``concept_values`` remains the masked
    mean of temporal chroma probabilities for backwards compatibility.
    """
    if getattr(output, "fusion_token", None) is not None:
        raise ContractError("harmony must not supply fusion_token; fusion owns its projection")
    required = (
        "embedding",
        "chroma_logits",
        "chord_logits",
        "availability",
        "prediction_mask",
    )
    missing = [name for name in required if not hasattr(output, name)]
    if missing:
        raise ContractError(f"temporal harmony output missing {missing}")

    embedding = output.embedding
    chroma_logits = output.chroma_logits
    chord_logits = output.chord_logits
    availability = output.availability
    prediction_mask = output.prediction_mask
    if not isinstance(embedding, torch.Tensor) or embedding.ndim != 2:
        raise ContractError("harmony embedding must have shape (B,D)")
    batch = embedding.shape[0]
    if (
        not isinstance(chroma_logits, torch.Tensor)
        or chroma_logits.ndim != 3
        or chroma_logits.shape[0] != batch
        or chroma_logits.shape[-1] != N_HARMONY_CHROMA
    ):
        raise ContractError(f"harmony chroma logits must have shape (B,T,{N_HARMONY_CHROMA})")
    if prediction_mask.shape != chroma_logits.shape[:2]:
        raise ContractError("harmony prediction mask must align with temporal logits")
    if availability.shape != (batch,):
        raise ContractError("harmony availability must have shape (B,)")
    if chord_logits is not None and (
        chord_logits.ndim != 3
        or chord_logits.shape[:2] != prediction_mask.shape
        or chord_logits.shape[-1] != N_HARMONY_CHORDS
    ):
        raise ContractError(f"harmony chord logits must have shape (B,T,{N_HARMONY_CHORDS})")

    valid = prediction_mask.to(dtype=torch.bool)
    if not torch.equal(availability.to(torch.bool), valid.any(dim=1)):
        raise ContractError("harmony availability must equal valid-token availability")
    probabilities = torch.softmax(chroma_logits, dim=-1)
    weights = valid.to(probabilities.dtype)
    pooled = (probabilities * weights.unsqueeze(-1)).sum(dim=1)
    pooled = pooled / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
    pooled = pooled * availability.to(pooled.dtype).unsqueeze(-1)

    descriptor_values = getattr(output, "descriptor_values", None)
    if descriptor_values is not None:
        if descriptor_values.shape != (batch, N_HARMONY_DESCRIPTORS):
            raise ContractError(
                f"harmony descriptor values must have shape (B,{N_HARMONY_DESCRIPTORS})"
            )
        if not torch.isfinite(descriptor_values).all():
            raise ContractError("harmony descriptor values must be finite")
        pooled = descriptor_values
        if descriptor_supervision_mask is None:
            supervision_mask = torch.zeros_like(pooled)
        else:
            if descriptor_supervision_mask.shape != pooled.shape:
                raise ContractError("descriptor supervision mask must match descriptor values")
            supervision_mask = descriptor_supervision_mask.to(pooled.dtype)
    elif chroma_target_mask is None:
        supervision_mask = torch.zeros_like(pooled)
    else:
        if chroma_target_mask.shape != prediction_mask.shape:
            raise ContractError("chroma target mask must align with harmony temporal logits")
        supervised_song = (chroma_target_mask.to(torch.bool) & valid).any(dim=1)
        supervision_mask = supervised_song.to(pooled.dtype).unsqueeze(1).expand(
            batch, N_HARMONY_CHROMA
        )

    if fusion_mask is None:
        fusion_mask = availability.to(pooled.dtype).unsqueeze(1)
    if fusion_mask.shape != (batch, 1):
        raise ContractError("harmony fusion mask must have shape (B,1)")
    if hidden_token is not None and hidden_token.shape != (batch, TOKEN_DIM):
        raise ContractError(f"harmony hidden token must have shape (B,{TOKEN_DIM})")

    branch = BranchOutput(
        name="harmony",
        concept_values=pooled,
        supervision_mask=supervision_mask,
        fusion_mask=fusion_mask,
        fusion_token=None,
        embedding=embedding,
        hidden_token=hidden_token,
        temporal_chroma_logits=chroma_logits,
        temporal_chord_logits=chord_logits,
        temporal_prediction_mask=prediction_mask,
        tag_order=HARMONY_FEATURES if descriptor_values is not None else None,
    )
    branch.validate(batch=batch, n_concepts=pooled.shape[1])
    return branch
