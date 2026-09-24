"""Compact shared CNN producing pooled and ordered branch representations."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor, nn

from .constants import (
    LOGMEL_HOP_LENGTH,
    LOGMEL_SAMPLE_RATE,
    SHARED_ENCODER_ARCHITECTURE,
    SHARED_ENCODER_DIM,
    TEMPORAL_DOWNSAMPLE,
    TEMPORAL_RECEPTIVE_FIELD_FRAMES,
)
from .geometry import build_temporal_layout
from .types import SharedEncoderOutput
from .validation import input_frame_mask, validate_audio, validate_metadata


class SharedAudioEncoder(nn.Module):
    """2D log-Mel CNN shared by instrument, timbre, rhythm, and harmony.

    Every sampled window is processed independently, preventing convolutions from
    crossing gaps. Metadata-aware calls return every branch representation from the
    same CNN evaluation and autograd graph. ``forward(x)`` without metadata remains
    a window-embedding compatibility path.
    """

    architecture = SHARED_ENCODER_ARCHITECTURE
    temporal_stride_frames = TEMPORAL_DOWNSAMPLE
    temporal_receptive_field_frames = TEMPORAL_RECEPTIVE_FIELD_FRAMES

    def __init__(self, output_dim: int = SHARED_ENCODER_DIM):
        super().__init__()
        if not isinstance(output_dim, int) or isinstance(output_dim, bool) or output_dim < 1:
            raise ValueError("output_dim must be a positive integer")
        self.output_dim = output_dim
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), ceil_mode=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1), ceil_mode=True),
            nn.Conv2d(64, 96, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 96),
            nn.GELU(),
        )
        self.frequency_attention = nn.Conv2d(96, 1, kernel_size=1)
        self.proj = nn.Linear(96, output_dim)

    def _encode_feature_map(self, x: Tensor, valid_frames: Tensor) -> Tensor:
        batch, windows, _, input_frames = validate_audio(x)
        clean = x * input_frame_mask(valid_frames, input_frames).to(x.dtype)
        feature_map = self.cnn(clean.reshape(batch * windows, *clean.shape[2:]))
        return feature_map.reshape(batch, windows, *feature_map.shape[1:])

    def _temporal_projection(self, feature_map: Tensor) -> Tensor:
        batch, windows, channels, frequency, tokens = feature_map.shape
        flat = feature_map.reshape(batch * windows, channels, frequency, tokens)
        frequency_weights = torch.softmax(self.frequency_attention(flat), dim=2)
        temporal = (flat * frequency_weights).sum(dim=2).transpose(1, 2)
        return self.proj(temporal).reshape(batch, windows, tokens, self.output_dim)

    def encode_windows(self, x: Tensor) -> Tensor:
        """Return `(B,W,D)` while assuming every supplied frame is real."""
        batch, windows, _, frames = validate_audio(x)
        valid_frames = torch.full((batch, windows), frames, device=x.device, dtype=torch.long)
        temporal = self._temporal_projection(self._encode_feature_map(x, valid_frames))
        return temporal.mean(dim=2)

    def forward(
        self,
        x: Tensor,
        window_mask: Tensor | None = None,
        window_valid_frames: Tensor | None = None,
        window_start_seconds: Tensor | None = None,
        *,
        sample_rate: int = LOGMEL_SAMPLE_RATE,
        hop_length: int = LOGMEL_HOP_LENGTH,
    ) -> SharedEncoderOutput | Tensor:
        """Encode all branch representations in one shared forward pass."""
        metadata = (window_mask, window_valid_frames, window_start_seconds)
        if all(value is None for value in metadata):
            return self.encode_windows(x)
        if any(value is None for value in metadata):
            raise ValueError(
                "window_mask, window_valid_frames, and window_start_seconds "
                "must be supplied together"
            )

        batch, windows, _, _ = validate_audio(x)
        mask, valid_frames, window_starts = validate_metadata(
            x, window_mask, window_valid_frames, window_start_seconds
        )
        temporal = self._temporal_projection(self._encode_feature_map(x, valid_frames))
        layout = build_temporal_layout(
            valid_frames=valid_frames,
            window_mask=mask,
            window_start_seconds=window_starts,
            token_count=temporal.shape[2],
            sample_rate=sample_rate,
            hop_length=hop_length,
        )
        masked_temporal = temporal * layout.token_mask.unsqueeze(-1).to(temporal.dtype)
        flat_temporal = masked_temporal.reshape(batch, -1, self.output_dim)

        window_denominator = (
            layout.token_mask.sum(dim=2, keepdim=True).clamp_min(1).to(temporal.dtype)
        )
        window_repr = masked_temporal.sum(dim=2) / window_denominator
        window_repr = window_repr * mask.unsqueeze(-1).to(window_repr.dtype)
        song_denominator = (
            layout.flat_mask.sum(dim=1, keepdim=True).clamp_min(1).to(temporal.dtype)
        )
        pooled_song = flat_temporal.sum(dim=1) / song_denominator
        availability = layout.flat_mask.any(dim=1)
        pooled_song = pooled_song * availability.unsqueeze(-1).to(pooled_song.dtype)

        return SharedEncoderOutput(
            encoded_sequence=flat_temporal,
            sequence_times=layout.sequence_times,
            sequence_start_times=layout.sequence_start_times,
            sequence_end_times=layout.sequence_end_times,
            sequence_mask=layout.flat_mask,
            sequence_window_index=layout.sequence_window_index,
            pooled_song=pooled_song,
            window_repr=window_repr,
            availability=availability,
        )

    def encode_temporal(
        self,
        x: Tensor,
        window_mask: Tensor,
        window_valid_frames: Tensor,
        window_start_seconds: Tensor,
        *,
        sample_rate: int = LOGMEL_SAMPLE_RATE,
        hop_length: int = LOGMEL_HOP_LENGTH,
    ) -> SharedEncoderOutput:
        """Compatibility alias for the metadata-aware forward pass."""
        output = self.forward(
            x,
            window_mask,
            window_valid_frames,
            window_start_seconds,
            sample_rate=sample_rate,
            hop_length=hop_length,
        )
        if not isinstance(output, SharedEncoderOutput):  # pragma: no cover
            raise RuntimeError("metadata-aware encoder unexpectedly returned a tensor")
        return output

    def set_trainable(self, trainable: bool) -> "SharedAudioEncoder":
        if not isinstance(trainable, bool):
            raise TypeError("trainable must be bool")
        for parameter in self.parameters():
            parameter.requires_grad_(trainable)
        return self

    def freeze(self) -> "SharedAudioEncoder":
        return self.set_trainable(False)

    def unfreeze(self) -> "SharedAudioEncoder":
        return self.set_trainable(True)

    def load_instrument_pretraining(self, checkpoint: Mapping) -> str:
        """Load a complete v2 encoder saved directly or below a known prefix."""
        state = checkpoint.get("model", checkpoint)
        if not isinstance(state, Mapping):
            raise ValueError("instrument checkpoint must contain a model state mapping")
        declared = checkpoint.get("encoder_architecture")
        if declared is not None and declared != self.architecture:
            raise ValueError(
                f"encoder checkpoint architecture {declared!r} is incompatible with "
                f"{self.architecture!r}"
            )
        expected = self.state_dict()
        formats = {
            "shared_cnn_v2": "",
            "shared_cnn_v2_nested": "enc.",
            "shared_cnn_v2_encoder_nested": "encoder.",
        }
        for name, prefix in formats.items():
            mapped = {}
            for key in expected:
                value = state.get(prefix + key)
                if isinstance(value, Tensor) and tuple(value.shape) == tuple(expected[key].shape):
                    mapped[key] = value
            if set(mapped) == set(expected):
                self.load_state_dict(mapped, strict=True)
                return name
        raise ValueError(
            "checkpoint does not contain a complete shared_cnn_audio_encoder_v2; "
            "legacy two-convolution cnn/proj checkpoints are not shape-compatible"
        )

