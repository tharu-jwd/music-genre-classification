"""Concept fusion, 87-label genre head, joint loss, metrics, and explainability.

Built against the v0.3 predicted-concept fusion contract. Integration with real
branch owners is a shape/mask assertion, not a reshape step.
"""

from concept_fusion.contract import (
    CONCEPT_DROPOUT_P,
    CONCEPT_ORDER,
    EMBEDDING_FUSION_INPUT_MODE,
    FUSION_CONTRACT_VERSION,
    FUSED_DIM,
    DEFAULT_HARMONY_EMBEDDING_DIM,
    HARMONY_FEATURES,
    N_GENRE_TAGS,
    N_INSTRUMENT_TAGS,
    N_RHYTHM_CONCEPTS,
    N_HARMONY_CHROMA,
    N_HARMONY_DESCRIPTORS,
    N_HARMONY_CHORDS,
    N_TIMBRE_CONCEPTS,
    TIMBRE_FEATURES,
    RHYTHM_FEATURES,
    PRIMARY_FUSION_INPUT_MODE,
    TOKEN_DIM,
    ConceptCounts,
)
from concept_fusion.types import BranchBundle, BranchOutput, FusionOutput
from concept_fusion.model import ConceptBottleneckModel
from concept_fusion.pipeline import run_all
from concept_fusion.projections import TokenAssembler
from concept_fusion.rhythm_adapter import from_rhythm_branch
from concept_fusion.harmony_adapter import from_temporal_harmony_branch
from concept_fusion.joint_loss import HarmonyTargets

__all__ = [
    "CONCEPT_DROPOUT_P",
    "CONCEPT_ORDER",
    "EMBEDDING_FUSION_INPUT_MODE",
    "FUSION_CONTRACT_VERSION",
    "FUSED_DIM",
    "DEFAULT_HARMONY_EMBEDDING_DIM",
    "HARMONY_FEATURES",
    "N_GENRE_TAGS",
    "N_INSTRUMENT_TAGS",
    "N_RHYTHM_CONCEPTS",
    "N_HARMONY_CHROMA",
    "N_HARMONY_DESCRIPTORS",
    "N_HARMONY_CHORDS",
    "N_TIMBRE_CONCEPTS",
    "TIMBRE_FEATURES",
    "RHYTHM_FEATURES",
    "PRIMARY_FUSION_INPUT_MODE",
    "TOKEN_DIM",
    "ConceptCounts",
    "BranchBundle",
    "BranchOutput",
    "FusionOutput",
    "ConceptBottleneckModel",
    "TokenAssembler",
    "from_rhythm_branch",
    "HarmonyTargets",
    "from_temporal_harmony_branch",
    "run_all",
]
