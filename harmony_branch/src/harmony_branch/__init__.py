"""Temporal harmony branch, targets, alignment, and feature extraction."""

from .alignment import AlignedChroma, align_chroma_to_intervals
from .constants import HARMONY_DESCRIPTORS, HARMONY_V3_DESCRIPTORS, N_HARMONY_DESCRIPTORS
from .descriptors import HarmonyTargetTransform, tonnetz_phi
from .features import ChromaSequence, extract_cqt_chroma, extract_hpcp
from .losses import masked_hard_classification_loss, masked_soft_target_cross_entropy
from .model import ChromaGroundedHarmonyBranch, HarmonyBranchOutput, TemporalHarmonyBranch

__all__ = [
    "AlignedChroma",
    "ChromaGroundedHarmonyBranch",
    "ChromaSequence",
    "HARMONY_DESCRIPTORS",
    "HARMONY_V3_DESCRIPTORS",
    "HarmonyBranchOutput",
    "HarmonyTargetTransform",
    "N_HARMONY_DESCRIPTORS",
    "TemporalHarmonyBranch",
    "tonnetz_phi",
    "align_chroma_to_intervals",
    "extract_cqt_chroma",
    "extract_hpcp",
    "masked_hard_classification_loss",
    "masked_soft_target_cross_entropy",
]
