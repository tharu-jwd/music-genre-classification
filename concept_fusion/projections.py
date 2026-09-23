"""Fusion-owned token assembly. Instrument v2 has no branch fusion_token."""

from __future__ import annotations

import torch
import torch.nn as nn

from concept_fusion.contract import (
    BRANCHES_WITHOUT_FUSION_TOKEN,
    CONCEPT_ORDER,
    INSTRUMENT_HIDDEN_DIM,
    N_INSTRUMENT_TAGS,
    TOKEN_DIM,
)
from concept_fusion.types import BranchBundle
from concept_fusion.validation import ContractError, require_finite, require_tensor


class TokenAssembler(nn.Module):
    """Build tokens (B, 4, 64) in CONCEPT_ORDER.

    Owns `Linear(40, 64)` for instrument probabilities. Applies `fusion_mask`
    after that projection. Other branches still supply their own 64-D tokens.
    """

    def __init__(self):
        super().__init__()
        self.instrument_projection = nn.Linear(N_INSTRUMENT_TAGS, TOKEN_DIM)
        self.instrument_hidden_projection = nn.Linear(INSTRUMENT_HIDDEN_DIM, TOKEN_DIM)

    def forward(self, bundle: BranchBundle, *, use_hidden: bool = False) -> torch.Tensor:
        bundle.validate()
        tokens = []
        for name in CONCEPT_ORDER:
            br = bundle.branches[name]
            if use_hidden:
                tok = self._hidden_token(br)
            elif name in BRANCHES_WITHOUT_FUSION_TOKEN:
                tok = self._instrument_token(br)
            else:
                if br.fusion_token is None:
                    raise ContractError(f"{name} must supply fusion_token (B, {TOKEN_DIM})")
                tok = br.fusion_token
            tokens.append(tok)
        stacked = torch.stack(tokens, dim=1)
        require_tensor("tokens", stacked, ndim=3, last=TOKEN_DIM)
        require_finite("tokens", stacked)
        return stacked

    def _instrument_token(self, br) -> torch.Tensor:
        if br.fusion_token is not None:
            raise ContractError("instrument must not supply fusion_token; fusion owns Linear(40,64)")
        probs = require_tensor("instrument.concept_values", br.concept_values, ndim=2, last=N_INSTRUMENT_TAGS)
        require_finite("instrument.concept_values", probs)
        token = self.instrument_projection(probs)
        return token * br.fusion_mask

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
