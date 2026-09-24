"""Fusion-owned token assembly for branch outputs with unequal widths."""

from __future__ import annotations

import torch
import torch.nn as nn

from concept_fusion.contract import (
    BRANCHES_WITHOUT_FUSION_TOKEN,
    CONCEPT_ORDER,
    DEFAULT_HARMONY_EMBEDDING_DIM,
    INSTRUMENT_HIDDEN_DIM,
    N_INSTRUMENT_TAGS,
    N_TIMBRE_CONCEPTS,
    TOKEN_DIM,
)
from concept_fusion.types import BranchBundle
from concept_fusion.validation import ContractError, require_finite, require_tensor


class TokenAssembler(nn.Module):
    """Build tokens (B, 4, 64) in CONCEPT_ORDER.

    Owns Linear(40,64) for instrument probabilities, Linear(35,64) for
    standardized timbre concepts, and Linear(D_harmony,64) for the configurable
    harmony song embedding. Applies fusion_mask after each projection.
    """

    def __init__(self, *, harmony_embedding_dim: int = DEFAULT_HARMONY_EMBEDDING_DIM):
        super().__init__()
        if harmony_embedding_dim < 1:
            raise ValueError("harmony_embedding_dim must be positive")
        self.harmony_embedding_dim = harmony_embedding_dim
        self.instrument_projection = nn.Linear(N_INSTRUMENT_TAGS, TOKEN_DIM)
        self.instrument_hidden_projection = nn.Linear(INSTRUMENT_HIDDEN_DIM, TOKEN_DIM)
        self.timbre_projection = nn.Linear(N_TIMBRE_CONCEPTS, TOKEN_DIM)
        self.harmony_projection = nn.Linear(harmony_embedding_dim, TOKEN_DIM)

    def forward(self, bundle: BranchBundle, *, use_hidden: bool = False) -> torch.Tensor:
        bundle.validate()
        tokens = []
        for name in CONCEPT_ORDER:
            br = bundle.branches[name]
            if use_hidden:
                tok = self._hidden_token(br)
            elif name in BRANCHES_WITHOUT_FUSION_TOKEN:
                tok = self._projected_token(br)
            else:
                if br.fusion_token is None:
                    raise ContractError(f"{name} must supply fusion_token (B, {TOKEN_DIM})")
                tok = br.fusion_token
            tokens.append(tok)
        stacked = torch.stack(tokens, dim=1)
        require_tensor("tokens", stacked, ndim=3, last=TOKEN_DIM)
        require_finite("tokens", stacked)
        return stacked

    def _projected_token(self, br) -> torch.Tensor:
        if br.fusion_token is not None:
            raise ContractError(
                f"{br.name} must not supply fusion_token; fusion owns its projection"
            )
        if br.name == "instrument":
            probs = require_tensor(
                "instrument.concept_values", br.concept_values, ndim=2, last=N_INSTRUMENT_TAGS
            )
            require_finite("instrument.concept_values", probs)
            return self.instrument_projection(probs) * br.fusion_mask
        if br.name == "timbre":
            z = require_tensor("timbre.concept_values", br.concept_values, ndim=2, last=N_TIMBRE_CONCEPTS)
            require_finite("timbre.concept_values", z)
            return self.timbre_projection(z) * br.fusion_mask
        if br.name == "harmony":
            if br.embedding is None:
                raise ContractError("harmony output is missing its song embedding")
            embedding = require_tensor(
                "harmony.embedding",
                br.embedding,
                ndim=2,
                last=self.harmony_embedding_dim,
            )
            require_finite("harmony.embedding", embedding)
            return self.harmony_projection(embedding) * br.fusion_mask
        raise ContractError(f"no fusion-owned projection for {br.name}")

    def _hidden_token(self, br) -> torch.Tensor:
        if br.hidden_token is None:
            raise ContractError(f"{br.name} F-Hidden requires hidden_token")
        if br.name == "instrument":
            hid = require_tensor("instrument.hidden_token", br.hidden_token, ndim=2, last=INSTRUMENT_HIDDEN_DIM)
            require_finite("instrument.hidden_token", hid)
            return self.instrument_hidden_projection(hid) * br.fusion_mask
        hid = require_tensor(f"{br.name}.hidden_token", br.hidden_token, ndim=2, last=TOKEN_DIM)
        require_finite(f"{br.name}.hidden_token", hid)
        return hid * br.fusion_mask
