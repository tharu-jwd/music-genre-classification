"""Fuse native-width concept predictions without per-branch projections."""
from __future__ import annotations

import torch
from torch import nn

from concept_fusion.contract import CONCEPT_ORDER, ConceptCounts, N_GENRE_TAGS
from concept_fusion.dropout import apply_concept_dropout
from concept_fusion.genre_head import GenreHead
from concept_fusion.types import BranchBundle, FusionOutput


class NativeConceptFusionModel(nn.Module):
    """Selected raw predictions -> concatenation -> 128D MLP -> genre head.

    Instrument probabilities and standardized descriptor predictions retain their
    native widths. The legacy four-slot masks are retained for concept dropout.
    Returned `gates` are uniform availability diagnostics, not learned importance.
    """

    fusion_name = "native_concat"
    fusion_input_mode = "predicted_concepts"
    fusion_contract_version = "native_concept_concat_v1"

    def __init__(self, branches=CONCEPT_ORDER, *, dropout_p=0.15, n_tags=N_GENRE_TAGS):
        super().__init__()
        branches = tuple(branches)
        if not branches or len(set(branches)) != len(branches) or set(branches) - set(CONCEPT_ORDER):
            raise ValueError("Native fusion requires unique, known concept branches")
        if not 0 <= dropout_p < 1:
            raise ValueError("dropout_p must be in [0, 1)")
        self.branches = tuple(name for name in CONCEPT_ORDER if name in branches)
        self.input_dim = sum(ConceptCounts().for_name(name) for name in self.branches)
        self.dropout_p = dropout_p
        self.fusion = nn.Sequential(nn.Linear(self.input_dim, 128), nn.ReLU(), nn.Dropout(0.1))
        self.head = GenreHead(n_tags=n_tags)

    def forward(self, bundle: BranchBundle, *, apply_dropout=None, generator=None):
        bundle.validate()
        mask = bundle.fusion_mask().clone()
        for index, name in enumerate(CONCEPT_ORDER):
            if name not in self.branches:
                mask[:, index] = 0
        use_dropout = self.training if apply_dropout is None else apply_dropout
        if use_dropout:
            mask = apply_concept_dropout(mask, p=self.dropout_p, generator=generator)
        values = torch.cat([
            bundle.concept_values(name) * mask[:, CONCEPT_ORDER.index(name):CONCEPT_ORDER.index(name) + 1]
            for name in self.branches
        ], dim=-1)
        fused = self.fusion(values)
        all_off = mask.sum(dim=1) == 0
        fused = torch.where(all_off[:, None], torch.zeros_like(fused), fused)
        gates = mask / mask.sum(dim=1, keepdim=True).clamp_min(1)
        output = FusionOutput(fused=fused, gates=gates, effective_mask=mask, used_null_token=all_off)
        output.validate(bundle.batch)
        return self.head(fused), output

    def from_bundle(self, bundle: BranchBundle, **kwargs):
        return self(bundle, **kwargs)
