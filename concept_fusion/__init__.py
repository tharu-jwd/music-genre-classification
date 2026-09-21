"""Concept fusion, 87-label genre head, joint loss, metrics, and explainability.

Built against the v0.1 shared architecture contract. Integration with real
branch owners is a shape/mask assertion, not a reshape step.
"""

from concept_fusion.contract import (
    CONCEPT_DROPOUT_P,
    CONCEPT_ORDER,
    FUSED_DIM,
    N_GENRE_TAGS,
    N_INSTRUMENT_TAGS,
    TOKEN_DIM,
    ConceptCounts,
)
from concept_fusion.types import BranchBundle, BranchOutput, FusionOutput
from concept_fusion.model import ConceptBottleneckModel

__all__ = [
    "CONCEPT_DROPOUT_P",
    "CONCEPT_ORDER",
    "FUSED_DIM",
    "N_GENRE_TAGS",
    "N_INSTRUMENT_TAGS",
    "TOKEN_DIM",
    "ConceptCounts",
    "BranchBundle",
    "BranchOutput",
    "FusionOutput",
    "ConceptBottleneckModel",
]
