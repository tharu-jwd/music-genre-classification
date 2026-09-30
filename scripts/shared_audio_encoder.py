"""Compatibility import for the shared CNN audio encoder.

The implementation lives in the ``shared_encoder`` package. This module used to
hold a full copy, which could silently drift from the trained architecture; it
now re-exports the single implementation under the old import path.
"""

from shared_encoder import (
    LOGMEL_HOP_LENGTH,
    LOGMEL_SAMPLE_RATE,
    SHARED_ENCODER_ARCHITECTURE,
    SHARED_ENCODER_DIM,
    TEMPORAL_DOWNSAMPLE,
    TEMPORAL_RECEPTIVE_FIELD_FRAMES,
    SharedAudioEncoder,
    SharedEncoderOutput,
    TemporalEncoderOutput,
)

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
