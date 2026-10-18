"""Backward-compatible imports for the canonical :mod:`shared_encoder` package.

Keep one implementation so old import paths use the same model and metadata as
the current training scripts.
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
from shared_encoder import __all__
