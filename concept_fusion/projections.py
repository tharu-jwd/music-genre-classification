"""Fusion-owned token assembly. Instrument and timbre do not supply fusion_token."""

from __future__ import annotations

import torch
import torch.nn as nn

from concept_fusion.contract import (
    BRANCHES_WITHOUT_FUSION_TOKEN,
    CONCEPT_ORDER,
    INSTRUMENT_HIDDEN_DIM,
    N_INSTRUMENT_TAGS,
    N_TIMBRE_CONCEPTS,
    TOKEN_DIM,
)
from concept_fusion.types import BranchBundle
from concept_fusion.validation import ContractError, require_finite, require_tensor


class TokenAssembler(nn.Module):
    """Build tokens (B, 4, 64) in CONCEPT_ORDER.

    Owns Linear(40,64) for instrument probabilities and Linear(35,64) for
    standardized timbre concepts. Applies fusion_mask after each projection.
    """

    def __init__(self):
        super().__init__()
        self.instrument_projection = nn.Linear(N_INSTRUMENT_TAGS, TOKEN_DIM)
        self.instrument_hidden_projection = nn.Linear(INSTRUMENT_HIDDEN_DIM, TOKEN_DIM)
        self.timbre_projection = nn.Linear(N_TIMBRE_CONCEPTS, TOKEN_DIM)

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
                f"{br.name} must not supply fusion_token; fusion owns Linear(C_k,64)"
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
