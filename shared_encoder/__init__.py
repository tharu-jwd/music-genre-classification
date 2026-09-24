"""Public API for the modular shared CNN audio encoder."""

from .constants import (
    LOGMEL_HOP_LENGTH,
    LOGMEL_SAMPLE_RATE,
    SHARED_ENCODER_ARCHITECTURE,
    SHARED_ENCODER_DIM,
    TEMPORAL_DOWNSAMPLE,
    TEMPORAL_RECEPTIVE_FIELD_FRAMES,
)
from .model import SharedAudioEncoder
from .types import SharedEncoderOutput, TemporalEncoderOutput

__all__ = [
    "LOGMEL_HOP_LENGTH",
    "LOGMEL_SAMPLE_RATE",
    "SHARED_ENCODER_ARCHITECTURE",
    "SHARED_ENCODER_DIM",
    "TEMPORAL_DOWNSAMPLE",
    "TEMPORAL_RECEPTIVE_FIELD_FRAMES",
    "SharedAudioEncoder",
    "SharedEncoderOutput",
    "TemporalEncoderOutput",
]

