"""Fuse native-width concept predictions without per-branch projections."""
from __future__ import annotations

import torch
from torch import nn

from concept_fusion.contract import CONCEPT_ORDER, ConceptCounts, N_GENRE_TAGS
from concept_fusion.dropout import apply_concept_dropout
from concept_fusion.types import BranchBundle, FusionOutput
from shared_encoder.architecture import activation


class NativeConceptFusionModel(nn.Module):
    """Selected predictions -> concatenation -> linear or MLP genre predictor.

    Instrument probabilities and standardized descriptor predictions retain their
    native widths. The legacy four-slot masks are retained for concept dropout.
    Returned `gates` are uniform availability diagnostics, not learned importance.
    """

    fusion_name = "native_concat"
    fusion_input_mode = "predicted_concepts"
    fusion_contract_version = "native_concept_concat_v2"

    def __init__(self, branches=CONCEPT_ORDER, *, dropout_p=0.0, n_tags=N_GENRE_TAGS,
                 predictor="linear", hidden_dim=64, hidden_dropout=0.0, hidden_activation="gelu"):
        super().__init__()
        branches = tuple(branches)
        if not branches or len(set(branches)) != len(branches) or set(branches) - set(CONCEPT_ORDER):
            raise ValueError("Native fusion requires unique, known concept branches")
        if not 0 <= dropout_p < 1:
            raise ValueError("dropout_p must be in [0, 1)")
        if predictor not in ("linear", "mlp"):
            raise ValueError("predictor must be linear or mlp")
        if hidden_dim < 1 or n_tags < 1 or not 0 <= hidden_dropout < 1:
            raise ValueError("widths must be positive and hidden_dropout in [0, 1)")
        if predictor == "linear" and hidden_dropout != 0:
            raise ValueError("hidden dropout requires the MLP predictor")
        self.branches = tuple(name for name in CONCEPT_ORDER if name in branches)
        self.input_dim = sum(ConceptCounts().for_name(name) for name in self.branches)
        self.dropout_p = dropout_p
        self.predictor = predictor
        self.fused_dim = self.input_dim if predictor == "linear" else hidden_dim
        self.fusion = (nn.Sequential(nn.Identity()) if predictor == "linear" else
                       nn.Sequential(nn.Linear(self.input_dim, hidden_dim), activation(hidden_activation),
                                     nn.Dropout(hidden_dropout)))
        self.head = nn.Linear(self.fused_dim, n_tags)

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
        output.validate(bundle.batch, fused_dim=self.fused_dim)
        return self.head(fused), output

    def from_bundle(self, bundle: BranchBundle, **kwargs):
        return self(bundle, **kwargs)
