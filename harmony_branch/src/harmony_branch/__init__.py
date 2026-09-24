"""Temporal harmony branch, targets, alignment, and feature extraction."""

from .alignment import AlignedChroma, align_chroma_to_intervals
from .features import ChromaSequence, extract_cqt_chroma, extract_hpcp
from .losses import masked_hard_classification_loss, masked_soft_target_cross_entropy
from .model import HarmonyBranchOutput, TemporalHarmonyBranch

__all__ = [
    "AlignedChroma",
    "ChromaSequence",
    "HarmonyBranchOutput",
    "TemporalHarmonyBranch",
    "align_chroma_to_intervals",
    "extract_cqt_chroma",
    "extract_hpcp",
    "masked_hard_classification_loss",
    "masked_soft_target_cross_entropy",
]
