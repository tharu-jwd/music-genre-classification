"""Typed branch outputs. Invalid shapes/order/NaNs raise ContractError."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import torch

from concept_fusion.contract import CONCEPT_ORDER, FUSED_DIM, N_CONCEPTS, TOKEN_DIM, ConceptCounts
from concept_fusion.validation import (
    ContractError,
    require_batch,
    require_binary_mask,
    require_finite,
    require_tensor,
)


@dataclass
class BranchOutput:
    name: str
    concept_values: torch.Tensor  # (B, C_k) probabilities / standardized values
    fusion_token: torch.Tensor  # (B, 64)
    supervision_mask: torch.Tensor  # (B, C_k) 1 = target observed
    fusion_mask: torch.Tensor  # (B, 1) 1 = branch enabled for fusion
    hidden_token: torch.Tensor | None = None  # (B, 64) diagnostic / F-Hidden only

    def validate(self, *, batch: int, n_concepts: int) -> None:
        if self.name not in CONCEPT_ORDER:
            raise ContractError(f"unknown branch {self.name!r}; order is {CONCEPT_ORDER}")
        tok = require_tensor("fusion_token", self.fusion_token, ndim=2, last=TOKEN_DIM)
        val = require_tensor("concept_values", self.concept_values, ndim=2)
        sm = require_tensor("supervision_mask", self.supervision_mask, ndim=2)
        fm = require_tensor("fusion_mask", self.fusion_mask, ndim=2, last=1)
        require_batch("fusion_token", tok, batch)
        require_batch("concept_values", val, batch)
        require_batch("supervision_mask", sm, batch)
        require_batch("fusion_mask", fm, batch)
        if val.shape[1] != n_concepts:
            raise ContractError(
                f"{self.name} concept_values C={val.shape[1]} != schema C={n_concepts}"
            )
        if sm.shape != val.shape:
            raise ContractError(
                f"{self.name} supervision_mask {tuple(sm.shape)} != concept_values {tuple(val.shape)}"
            )
        require_binary_mask(f"{self.name}.supervision_mask", sm)
        require_binary_mask(f"{self.name}.fusion_mask", fm)
        # Observed targets must be finite. Masked-out targets may be NaN (missing ≠ zero).
        require_finite(f"{self.name}.concept_values[observed]", val, where=sm)
        require_finite(f"{self.name}.fusion_token", tok)
        if self.hidden_token is not None:
            hid = require_tensor("hidden_token", self.hidden_token, ndim=2, last=TOKEN_DIM)
            require_batch("hidden_token", hid, batch)
            require_finite(f"{self.name}.hidden_token", hid)


@dataclass
class BranchBundle:
    branches: dict[str, BranchOutput]
    counts: ConceptCounts

    @property
    def batch(self) -> int:
        first = next(iter(self.branches.values()))
        return int(first.fusion_token.shape[0])

    def validate(self) -> None:
        if tuple(self.branches.keys()) != CONCEPT_ORDER:
            raise ContractError(
                f"branch dict keys must be exactly {CONCEPT_ORDER} in that order, "
                f"got {tuple(self.branches.keys())}"
            )
        bsz = self.batch
        for name in CONCEPT_ORDER:
            self.branches[name].validate(batch=bsz, n_concepts=self.counts.for_name(name))
            if self.branches[name].name != name:
                raise ContractError(f"key {name!r} does not match BranchOutput.name")

    def tokens(self) -> torch.Tensor:
        """(B, 4, 64) in CONCEPT_ORDER."""
        self.validate()
        return torch.stack([self.branches[n].fusion_token for n in CONCEPT_ORDER], dim=1)

    def fusion_mask(self) -> torch.Tensor:
        """(B, 4)."""
        self.validate()
        return torch.cat([self.branches[n].fusion_mask for n in CONCEPT_ORDER], dim=1)

    def concept_values(self, name: str) -> torch.Tensor:
        return self.branches[name].concept_values

    def supervision_mask(self, name: str) -> torch.Tensor:
        return self.branches[name].supervision_mask

    def hidden_tokens(self) -> torch.Tensor:
        """(B, 4, 64) diagnostic embeddings. Required for F-Hidden only."""
        self.validate()
        toks = []
        for n in CONCEPT_ORDER:
            hid = self.branches[n].hidden_token
            if hid is None:
                raise ContractError("F-Hidden requires hidden_token on every branch")
            toks.append(hid)
        return torch.stack(toks, dim=1)

    def with_enabled_concepts(self, enabled: Sequence[str]) -> "BranchBundle":
        """Zero fusion_mask for branches not in `enabled`. Does not drop tracks."""
        enabled_set = set(enabled)
        unknown = enabled_set - set(CONCEPT_ORDER)
        if unknown:
            raise ContractError(f"unknown enabled concepts {sorted(unknown)}")
        branches = {}
        for name in CONCEPT_ORDER:
            br = self.branches[name]
            keep = 1.0 if name in enabled_set else 0.0
            branches[name] = BranchOutput(
                name=br.name,
                concept_values=br.concept_values,
                fusion_token=br.fusion_token,
                supervision_mask=br.supervision_mask,
                fusion_mask=torch.full_like(br.fusion_mask, keep),
                hidden_token=br.hidden_token,
            )
        out = BranchBundle(branches=branches, counts=self.counts)
        out.validate()
        return out


@dataclass
class FusionOutput:
    fused: torch.Tensor  # (B, 128)
    gates: torch.Tensor  # (B, 4) post-mask weights; disabled concepts are exactly 0
    effective_mask: torch.Tensor  # (B, 4) after concept dropout
    used_null_token: torch.Tensor  # (B,) bool — all-masked fallback fired

    def validate(self, batch: int) -> None:
        require_tensor("fused", self.fused, ndim=2, last=FUSED_DIM)
        require_tensor("gates", self.gates, ndim=2, last=N_CONCEPTS)
        require_tensor("effective_mask", self.effective_mask, ndim=2, last=N_CONCEPTS)
        require_batch("fused", self.fused, batch)
        require_finite("fused", self.fused)
        require_finite("gates", self.gates)
        if not torch.isclose(
            self.gates * (1.0 - self.effective_mask),
            torch.zeros_like(self.gates),
            atol=1e-6,
        ).all():
            raise ContractError("masked concepts must have exactly zero fusion weight")
