"""Fusion-owned token assembly for branch outputs with unequal widths."""

from __future__ import annotations

import torch
import torch.nn as nn

from concept_fusion.contract import (
    CONCEPT_ORDER,
    DEFAULT_HARMONY_EMBEDDING_DIM,
    FUSION_INPUT_MODES,
    INSTRUMENT_HIDDEN_DIM,
    N_HARMONY_CHROMA,
    N_INSTRUMENT_TAGS,
    N_RHYTHM_CONCEPTS,
    N_TIMBRE_CONCEPTS,
    PRIMARY_FUSION_INPUT_MODE,
    TOKEN_DIM,
    FusionInputMode,
)
from concept_fusion.types import BranchBundle
from concept_fusion.validation import ContractError, require_finite, require_tensor


class TokenAssembler(nn.Module):
    """Build tokens (B, 4, 64) in CONCEPT_ORDER.

    Primary ``predicted_concepts`` mode owns 41→64, 10→64, 35→64, and
    12→64 projections. The harmony input is the configured 12-value predicted
    concept vector (song descriptors in joint vector-dataset training).
    ``embedding_fusion`` preserves the previous rhythm 64D
    embedding and harmony D→64 embedding route. Masks are applied after projection.
    """

    def __init__(
        self,
        *,
        input_mode: FusionInputMode = PRIMARY_FUSION_INPUT_MODE,
        harmony_embedding_dim: int = DEFAULT_HARMONY_EMBEDDING_DIM,
    ):
        super().__init__()
        if input_mode not in FUSION_INPUT_MODES:
            raise ContractError(
                f"unknown fusion input mode {input_mode!r}; expected one of {FUSION_INPUT_MODES}"
            )
        if harmony_embedding_dim < 1:
            raise ValueError("harmony_embedding_dim must be positive")
        self.input_mode = input_mode
        self.harmony_embedding_dim = harmony_embedding_dim
        self.instrument_projection = nn.Linear(N_INSTRUMENT_TAGS, TOKEN_DIM)
        self.rhythm_projection = nn.Linear(N_RHYTHM_CONCEPTS, TOKEN_DIM)
        self.instrument_hidden_projection = nn.Linear(INSTRUMENT_HIDDEN_DIM, TOKEN_DIM)
        self.timbre_projection = nn.Linear(N_TIMBRE_CONCEPTS, TOKEN_DIM)
        self.harmony_chroma_projection = nn.Linear(N_HARMONY_CHROMA, TOKEN_DIM)
        # Retain this exact module for the versioned embedding-fusion ablation.
        self.harmony_projection = nn.Linear(harmony_embedding_dim, TOKEN_DIM)

    def forward(self, bundle: BranchBundle, *, use_hidden: bool = False) -> torch.Tensor:
        bundle.validate()
        tokens = []
        for name in CONCEPT_ORDER:
            br = bundle.branches[name]
            if use_hidden:
                tok = self._hidden_token(br)
            elif self.input_mode == PRIMARY_FUSION_INPUT_MODE:
                tok = self._predicted_concept_token(br)
            else:
                tok = self._embedding_fusion_token(br)
            tokens.append(tok)
        stacked = torch.stack(tokens, dim=1)
        require_tensor("tokens", stacked, ndim=3, last=TOKEN_DIM)
        require_finite("tokens", stacked)
        return stacked

    def _predicted_concept_token(self, br) -> torch.Tensor:
        if br.name == "instrument":
            probs = require_tensor(
                "instrument.concept_values", br.concept_values, ndim=2, last=N_INSTRUMENT_TAGS
            )
            require_finite("instrument.concept_values", probs)
            return self.instrument_projection(probs) * br.fusion_mask
        if br.name == "rhythm":
            values = require_tensor(
                "rhythm.concept_values", br.concept_values, ndim=2, last=N_RHYTHM_CONCEPTS
            )
            require_finite("rhythm.concept_values", values)
            return self.rhythm_projection(values) * br.fusion_mask
        if br.name == "timbre":
            z = require_tensor("timbre.concept_values", br.concept_values, ndim=2, last=N_TIMBRE_CONCEPTS)
            require_finite("timbre.concept_values", z)
            return self.timbre_projection(z) * br.fusion_mask
        if br.name == "harmony":
            harmony_values = require_tensor(
                "harmony.concept_values", br.concept_values, ndim=2, last=N_HARMONY_CHROMA
            )
            require_finite("harmony.concept_values", harmony_values)
            # Keep the historical parameter name for checkpoint compatibility.
            return self.harmony_chroma_projection(harmony_values) * br.fusion_mask
        raise ContractError(f"no predicted-concept projection for {br.name}")

    def _embedding_fusion_token(self, br) -> torch.Tensor:
        if br.name in ("instrument", "timbre"):
            return self._predicted_concept_token(br)
        if br.name == "rhythm":
            if br.fusion_token is None:
                raise ContractError(f"rhythm embedding fusion requires (B,{TOKEN_DIM}) fusion_token")
            token = require_tensor("rhythm.fusion_token", br.fusion_token, ndim=2, last=TOKEN_DIM)
            require_finite("rhythm.fusion_token", token)
            return token * br.fusion_mask
        if br.name == "harmony":
            if br.embedding is None:
                raise ContractError("harmony embedding fusion requires its song embedding")
            embedding = require_tensor(
                "harmony.embedding",
                br.embedding,
                ndim=2,
                last=self.harmony_embedding_dim,
            )
            require_finite("harmony.embedding", embedding)
            return self.harmony_projection(embedding) * br.fusion_mask
        raise ContractError(f"no embedding-fusion route for {br.name}")

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
