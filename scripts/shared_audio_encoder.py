"""Shared log-Mel CNN for pooled and ordered concept-branch representations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn


LOGMEL_SAMPLE_RATE = 12000
LOGMEL_HOP_LENGTH = 256
TEMPORAL_DOWNSAMPLE = 2
TEMPORAL_RECEPTIVE_FIELD_FRAMES = 12
SHARED_ENCODER_DIM = 128
SHARED_ENCODER_ARCHITECTURE = "shared_cnn_audio_encoder_v2"


@dataclass(frozen=True)
class SharedEncoderOutput:
    """All representations derived from one CNN evaluation of the same windows."""

    encoded_sequence: Tensor
    sequence_times: Tensor
    sequence_start_times: Tensor
    sequence_end_times: Tensor
    sequence_mask: Tensor
    sequence_window_index: Tensor
    pooled_song: Tensor
    window_repr: Tensor
    availability: Tensor

    @property
    def song_repr(self) -> Tensor:
        """Instrument/timbre compatibility alias for ``pooled_song``."""
        return self.pooled_song


# Backwards-compatible import name used by older callers.
TemporalEncoderOutput = SharedEncoderOutput


class SharedAudioEncoder(nn.Module):
    """Compact 2D CNN with pooled and fine-grained 128D outputs.

    The CNN evaluates every sampled window independently, so no convolution can
    cross a gap between non-adjacent song regions. Frequency is reduced only after
    the convolution stack using learned, content-dependent attention. Time is
    downsampled exactly once, by two input frames.

    Calling ``forward`` with metadata returns :class:`SharedEncoderOutput`. Calling
    it with only ``x`` retains the old window-embedding convenience interface.
    Production branch integration should always supply metadata.
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
            # The only time downsampling: 2 mel frames per output token.
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), ceil_mode=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            # Reduce frequency cost without changing temporal resolution.
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1), ceil_mode=True),
            nn.Conv2d(64, 96, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 96),
            nn.GELU(),
        )
        self.frequency_attention = nn.Conv2d(96, 1, kernel_size=1)
        self.proj = nn.Linear(96, output_dim)

    @staticmethod
    def _validate_audio(x: Tensor) -> tuple[int, int, int, int]:
        if x.ndim != 5:
            raise ValueError("audio input must have shape batch x windows x channels x mel x time")
        batch, windows, channels, mel_bins, frames = x.shape
        if batch < 1 or windows < 1 or channels != 1 or mel_bins < 1 or frames < 1:
            raise ValueError("audio input has an unsupported empty/channel shape")
        if not torch.is_floating_point(x):
            raise ValueError("audio input must be floating point")
        if not torch.isfinite(x).all():
            raise ValueError("audio input contains non-finite values")
        return batch, windows, mel_bins, frames

    @staticmethod
    def _validate_metadata(
        x: Tensor,
        window_mask: Tensor,
        window_valid_frames: Tensor,
        window_start_seconds: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch, windows, _, input_frames = SharedAudioEncoder._validate_audio(x)
        expected = (batch, windows)
        if tuple(window_mask.shape) != expected:
            raise ValueError(f"window_mask must have shape {expected}")
        if tuple(window_valid_frames.shape) != expected:
            raise ValueError(f"window_valid_frames must have shape {expected}")
        if tuple(window_start_seconds.shape) != expected:
            raise ValueError(f"window_start_seconds must have shape {expected}")

        mask = window_mask.to(device=x.device, dtype=torch.bool)
        valid_frames = window_valid_frames.to(device=x.device, dtype=torch.long)
        starts = window_start_seconds.to(device=x.device)
        if not torch.is_floating_point(starts):
            starts = starts.to(dtype=x.dtype)
        if not torch.isfinite(starts).all() or (starts < 0).any():
            raise ValueError("window_start_seconds must be finite and non-negative")
        if (valid_frames < 0).any() or (valid_frames > input_frames).any():
            raise ValueError("window_valid_frames falls outside the input frame range")
        if not torch.equal(mask, valid_frames > 0):
            raise ValueError("window_mask must be true exactly where valid-frame count is positive")

        # Valid windows need not occupy every slot, but their song-relative starts
        # must remain strictly ordered.
        for row in range(batch):
            valid_starts = starts[row, mask[row]]
            if len(valid_starts) > 1 and torch.any(valid_starts[1:] <= valid_starts[:-1]):
                raise ValueError("real window start times must be strictly increasing")
        return mask, valid_frames, starts

    @staticmethod
    def _input_frame_mask(valid_frames: Tensor, input_frames: int) -> Tensor:
        frame_index = torch.arange(input_frames, device=valid_frames.device)
        return frame_index.view(1, 1, 1, 1, input_frames) < valid_frames[:, :, None, None, None]

    def _encode_feature_map(self, x: Tensor, valid_frames: Tensor) -> Tensor:
        batch, windows, _, input_frames = self._validate_audio(x)
        # Explicitly remove padded values before any normalization or convolution.
        # This makes padding content irrelevant rather than merely excluding it at
        # the final pooling operation.
        clean = x * self._input_frame_mask(valid_frames, input_frames).to(x.dtype)
        feature_map = self.cnn(clean.reshape(batch * windows, *clean.shape[2:]))
        return feature_map.reshape(batch, windows, *feature_map.shape[1:])

    def _temporal_projection(self, feature_map: Tensor) -> Tensor:
        batch, windows, channels, frequency, tokens = feature_map.shape
        flat = feature_map.reshape(batch * windows, channels, frequency, tokens)
        frequency_scores = self.frequency_attention(flat)
        frequency_weights = torch.softmax(frequency_scores, dim=2)
        # Learned frequency aggregation retains content-dependent spectral
        # structure; this is not an unqualified arithmetic mean over mel bands.
        temporal = (flat * frequency_weights).sum(dim=2).transpose(1, 2)
        projected = self.proj(temporal)
        return projected.reshape(batch, windows, tokens, self.output_dim)

    def encode_windows(self, x: Tensor) -> Tensor:
        """Compatibility path returning one representation per complete window.

        This path assumes every input frame is real. Joint training and temporal
        branches must use ``forward``/``encode_temporal`` with explicit metadata.
        """
        batch, windows, _, frames = self._validate_audio(x)
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
        """Encode all branch representations in one shared CNN forward pass."""
        metadata = (window_mask, window_valid_frames, window_start_seconds)
        if all(value is None for value in metadata):
            return self.encode_windows(x)
        if any(value is None for value in metadata):
            raise ValueError(
                "window_mask, window_valid_frames, and window_start_seconds "
                "must be supplied together"
            )
        if sample_rate < 1 or hop_length < 1:
            raise ValueError("sample_rate and hop_length must be positive")

        batch, windows, _, _ = self._validate_audio(x)
        mask, valid_frames, window_starts = self._validate_metadata(
            x, window_mask, window_valid_frames, window_start_seconds
        )
        feature_map = self._encode_feature_map(x, valid_frames)
        temporal = self._temporal_projection(feature_map)
        token_count = temporal.shape[2]

        token_index = torch.arange(token_count, device=x.device).view(1, 1, token_count)
        valid_token_counts = torch.div(
            valid_frames + TEMPORAL_DOWNSAMPLE - 1,
            TEMPORAL_DOWNSAMPLE,
            rounding_mode="floor",
        ).clamp_max(token_count)
        token_mask = mask.unsqueeze(-1) & (token_index < valid_token_counts.unsqueeze(-1))
        masked_temporal = temporal * token_mask.unsqueeze(-1).to(temporal.dtype)

        seconds_per_frame = hop_length / sample_rate
        starts = window_starts.unsqueeze(-1) + (
            token_index * TEMPORAL_DOWNSAMPLE * seconds_per_frame
        )
        relative_ends = torch.minimum(
            (token_index + 1) * TEMPORAL_DOWNSAMPLE,
            valid_frames.unsqueeze(-1),
        ).to(dtype=window_starts.dtype)
        ends = window_starts.unsqueeze(-1) + relative_ends * seconds_per_frame
        centers = (starts + ends) / 2

        flat_mask = token_mask.reshape(batch, -1)
        flat_temporal = masked_temporal.reshape(batch, -1, self.output_dim)
        flat_starts = starts.expand(batch, windows, token_count).reshape(batch, -1)
        flat_ends = ends.reshape(batch, -1)
        flat_centers = centers.reshape(batch, -1)
        flat_starts = flat_starts.masked_fill(~flat_mask, 0)
        flat_ends = flat_ends.masked_fill(~flat_mask, 0)
        flat_centers = flat_centers.masked_fill(~flat_mask, 0)
        window_index = (
            torch.arange(windows, device=x.device)
            .view(1, windows, 1)
            .expand(batch, windows, token_count)
            .reshape(batch, -1)
            .masked_fill(~flat_mask, -1)
        )

        window_denominator = token_mask.sum(dim=2, keepdim=True).clamp_min(1).to(temporal.dtype)
        window_repr = masked_temporal.sum(dim=2) / window_denominator
        window_repr = window_repr * mask.unsqueeze(-1).to(window_repr.dtype)
        song_denominator = flat_mask.sum(dim=1, keepdim=True).clamp_min(1).to(temporal.dtype)
        pooled_song = flat_temporal.sum(dim=1) / song_denominator
        availability = flat_mask.any(dim=1)
        pooled_song = pooled_song * availability.unsqueeze(-1).to(pooled_song.dtype)

        return SharedEncoderOutput(
            encoded_sequence=flat_temporal,
            sequence_times=flat_centers,
            sequence_start_times=flat_starts,
            sequence_end_times=flat_ends,
            sequence_mask=flat_mask,
            sequence_window_index=window_index,
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
        """Compatibility alias for the metadata-aware shared forward pass."""
        output = self.forward(
            x,
            window_mask,
            window_valid_frames,
            window_start_seconds,
            sample_rate=sample_rate,
            hop_length=hop_length,
        )
        if not isinstance(output, SharedEncoderOutput):  # pragma: no cover - type narrowing
            raise RuntimeError("metadata-aware encoder unexpectedly returned a tensor")
        return output

    def set_trainable(self, trainable: bool) -> "SharedAudioEncoder":
        """Explicitly freeze or unfreeze the encoder for controlled experiments."""
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
        """Load a complete v2 encoder saved directly or below ``enc.``.

        The former two-convolution checkpoint is intentionally rejected: it lacks
        normalization, learned frequency aggregation, and the v2 projection shape.
        It must be retrained or explicitly migrated rather than partially loaded.
        """
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
                source_key = prefix + key
                value = state.get(source_key)
                if isinstance(value, Tensor) and tuple(value.shape) == tuple(expected[key].shape):
                    mapped[key] = value
            if set(mapped) == set(expected):
                self.load_state_dict(mapped, strict=True)
                return name
        raise ValueError(
            "checkpoint does not contain a complete shared_cnn_audio_encoder_v2; "
            "legacy two-convolution cnn/proj checkpoints are not shape-compatible"
        )
