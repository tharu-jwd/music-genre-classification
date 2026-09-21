"""Synthetic branch outputs with exact contract shapes. No silent filling."""

from __future__ import annotations

import torch

from concept_fusion.contract import CONCEPT_ORDER, N_GENRE_TAGS, TOKEN_DIM, ConceptCounts
from concept_fusion.types import BranchBundle, BranchOutput


def make_branch_output(
    name: str,
    *,
    batch: int,
    n_concepts: int,
    seed: int = 0,
    fusion_keep: float = 0.85,
    supervise_keep: float = 0.7,
) -> BranchOutput:
    g = torch.Generator().manual_seed(seed + 17 * CONCEPT_ORDER.index(name))
    values = torch.rand(batch, n_concepts, generator=g)
    if name == "instrument":
        values = values.clamp(0.02, 0.98)
    token = torch.randn(batch, TOKEN_DIM, generator=g)
    token = token / (token.norm(dim=-1, keepdim=True) + 1e-6)
    sup = (torch.rand(batch, n_concepts, generator=g) < supervise_keep).float()
    fus = (torch.rand(batch, 1, generator=g) < fusion_keep).float()
    # Missing supervision is NaN, not zero.
    values = values.masked_fill(sup < 0.5, float("nan"))
    return BranchOutput(
        name=name,
        concept_values=values,
        fusion_token=token,
        supervision_mask=sup,
        fusion_mask=fus,
    )


def make_bundle(
    batch: int = 8,
    *,
    counts: ConceptCounts | None = None,
    seed: int = 0,
    fusion_keep: float = 0.85,
    supervise_keep: float = 0.7,
) -> BranchBundle:
    counts = counts or ConceptCounts()
    branches = {}
    for i, name in enumerate(CONCEPT_ORDER):
        branches[name] = make_branch_output(
            name,
            batch=batch,
            n_concepts=counts.for_name(name),
            seed=seed + 17 * i,
            fusion_keep=fusion_keep,
            supervise_keep=supervise_keep,
        )
    bundle = BranchBundle(branches=branches, counts=counts)
    bundle.validate()
    return bundle


def make_genre_batch(batch: int = 8, *, n_tags: int = N_GENRE_TAGS, seed: int = 1, p: float = 0.08):
    g = torch.Generator().manual_seed(seed)
    y = (torch.rand(batch, n_tags, generator=g) < p).float()
    # Guarantee at least one positive on a few tags for metrics.
    y[:, 0] = 1.0
    y[0, 1] = 1.0
    y[1, 1] = 0.0
    return y


def make_concept_targets(bundle: BranchBundle, *, seed: int = 3) -> dict[str, torch.Tensor]:
    """Finite targets aligned to each branch. Masked cells may stay NaN."""
    g = torch.Generator().manual_seed(seed)
    out: dict[str, torch.Tensor] = {}
    for name in CONCEPT_ORDER:
        pred = bundle.concept_values(name)
        mask = bundle.supervision_mask(name)
        if name == "instrument":
            tgt = (torch.rand(pred.shape, generator=g) < 0.15).float()
        else:
            tgt = torch.randn(pred.shape, generator=g)
        tgt = tgt.masked_fill(mask < 0.5, float("nan"))
        out[name] = tgt
    return out


def make_all_masked_tokens(batch: int = 4, *, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """(B, 4, 64) tokens with fusion_mask all zeros — exercises the null-token fallback."""
    g = torch.Generator().manual_seed(seed)
    tokens = torch.randn(batch, len(CONCEPT_ORDER), TOKEN_DIM, generator=g)
    mask = torch.zeros(batch, len(CONCEPT_ORDER))
    return tokens, mask
