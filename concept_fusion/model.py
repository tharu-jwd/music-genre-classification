"""Concept-bottleneck fusion model: tokens + masks → 87 logits. No audio shortcut."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from concept_fusion.contract import (
    CONCEPT_DROPOUT_P,
    DEFAULT_HARMONY_EMBEDDING_DIM,
    FUSED_DIM,
    N_GENRE_TAGS,
)
from concept_fusion.dropout import apply_concept_dropout
from concept_fusion.fusion import FusionName, build_fusion
from concept_fusion.genre_head import GenreHead
from concept_fusion.projections import TokenAssembler
from concept_fusion.types import BranchBundle, FusionOutput
from concept_fusion.validation import ContractError


class ConceptBottleneckModel(nn.Module):
    """Primary proposed model. Fusion inputs are concept tokens only.

    Set `allow_shortcut=True` only for the named F-Shortcut ablation.
    """

    def __init__(
        self,
        fusion: FusionName = "gated",
        *,
        dropout_p: float = CONCEPT_DROPOUT_P,
        fused_dim: int = FUSED_DIM,
        n_tags: int = N_GENRE_TAGS,
        allow_shortcut: bool = False,
        use_hidden: bool = False,
        allow_no_dropout: bool = False,
        song_repr_dim: int = 128,
        harmony_embedding_dim: int = DEFAULT_HARMONY_EMBEDDING_DIM,
    ):
        super().__init__()
        if n_tags != N_GENRE_TAGS:
            raise ContractError("primary genre vocabulary is frozen at 87 tags")
        if use_hidden and allow_shortcut:
            raise ContractError("F-Hidden and F-Shortcut are separate named ablations")
        self.fusion_name = fusion
        self.dropout_p = dropout_p
        self.allow_shortcut = allow_shortcut
        self.use_hidden = use_hidden
        self.allow_no_dropout = allow_no_dropout
        self.assembler = TokenAssembler(harmony_embedding_dim=harmony_embedding_dim)
        self.fusion = build_fusion(fusion, fused_dim=fused_dim)
        self.head = GenreHead(fused_dim=fused_dim, n_tags=n_tags)
        if allow_shortcut:
            self.shortcut = nn.Linear(song_repr_dim, fused_dim)
        else:
            self.shortcut = None

    def forward(
        self,
        tokens: torch.Tensor,
        fusion_mask: torch.Tensor,
        *,
        song_repr: torch.Tensor | None = None,
        apply_dropout: bool | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, FusionOutput]:
        if song_repr is not None and not self.allow_shortcut:
            raise ContractError("primary model forbids song_repr → genre; use F-Shortcut ablation")
        if apply_dropout is None:
            apply_dropout = self.training
        mask = fusion_mask
        if apply_dropout:
            if self.dropout_p <= 0:
                if not self.allow_no_dropout:
                    raise ContractError(
                        "concept dropout is a hard dependency for occlusion faithfulness; "
                        "do not disable it on the primary model"
                    )
            else:
                mask = apply_concept_dropout(mask, p=self.dropout_p, generator=generator)
        fout = self.fusion(tokens, mask)
        fused = fout.fused
        if self.allow_shortcut:
            if song_repr is None:
                raise ContractError("F-Shortcut requires song_repr")
            fused = fused + self.shortcut(song_repr)
        logits = self.head(fused)
        return logits, fout

    def assemble_tokens(self, bundle: BranchBundle) -> torch.Tensor:
        return self.assembler(bundle, use_hidden=self.use_hidden)

    def from_bundle(self, bundle: BranchBundle, **kwargs: Any) -> tuple[torch.Tensor, FusionOutput]:
        return self.forward(self.assemble_tokens(bundle), bundle.fusion_mask(), **kwargs)
