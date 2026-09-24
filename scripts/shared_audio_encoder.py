"""Reusable log-Mel encoder with both song-window and fine temporal outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn


LOGMEL_SAMPLE_RATE = 12000
LOGMEL_HOP_LENGTH = 256
TEMPORAL_DOWNSAMPLE = 2


@dataclass(frozen=True)
class TemporalEncoderOutput:
    encoded_sequence: Tensor
    sequence_times: Tensor
    sequence_start_times: Tensor
    sequence_end_times: Tensor
    sequence_mask: Tensor
    sequence_window_index: Tensor
    pooled_song: Tensor
    availability: Tensor


class SharedAudioEncoder(nn.Module):
    """The instrument-pretraining CNN before its time-collapsing average pool.

    ``forward`` preserves the existing window-embedding behavior. ``encode_temporal``
    exposes the same learned convolutional features before time is averaged, so a
    harmony branch can consume ordered within-window tokens instead of one vector per
    approximately 29-second window.
    """

    def __init__(self, output_dim: int = 64):
        super().__init__()
        if not isinstance(output_dim, int) or isinstance(output_dim, bool) or output_dim < 1:
            raise ValueError("output_dim must be a positive integer")
        self.output_dim = output_dim
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
        )
        self.proj = nn.Linear(64, output_dim)

    @staticmethod
    def _validate_audio(x: Tensor) -> tuple[int, int, int]:
        if x.ndim != 5:
            raise ValueError("audio input must have shape batch x windows x channels x mel x time")
        batch, windows, channels, _, frames = x.shape
        if batch < 1 or windows < 1 or channels != 1 or frames < TEMPORAL_DOWNSAMPLE:
            raise ValueError("audio input has an unsupported empty/channel/time shape")
        if not torch.is_floating_point(x):
            raise ValueError("audio input must be floating point")
        if not torch.isfinite(x).all():
            raise ValueError("audio input contains non-finite values")
        return batch, windows, frames

    def _feature_map(self, x: Tensor) -> Tensor:
        batch, windows, _ = self._validate_audio(x)
        feature_map = self._feature_map_unchecked(x, batch, windows)
        return feature_map

    def _feature_map_unchecked(self, x: Tensor, batch: int, windows: int) -> Tensor:
        feature_map = self.cnn(x.reshape(batch * windows, *x.shape[2:]))
        return feature_map.reshape(batch, windows, *feature_map.shape[1:])

    def forward(self, x: Tensor) -> Tensor:
        """Return one embedding per window, compatible with existing pretraining."""
        feature_map = self._feature_map(x)
        pooled = feature_map.mean(dim=(-2, -1))
        return self.proj(pooled)

    def encode_temporal(
        self,
        x: Tensor,
        window_mask: Tensor,
        window_valid_frames: Tensor,
        window_start_seconds: Tensor,
        *,
        sample_rate: int = LOGMEL_SAMPLE_RATE,
        hop_length: int = LOGMEL_HOP_LENGTH,
    ) -> TemporalEncoderOutput:
        """Expose masked time tokens and their exact song-relative intervals."""
        batch, windows, input_frames = self._validate_audio(x)
        expected = (batch, windows)
        if tuple(window_mask.shape) != expected:
            raise ValueError(f"window_mask must have shape {expected}")
        if tuple(window_valid_frames.shape) != expected:
            raise ValueError(f"window_valid_frames must have shape {expected}")
        if tuple(window_start_seconds.shape) != expected:
            raise ValueError(f"window_start_seconds must have shape {expected}")
        if sample_rate < 1 or hop_length < 1:
            raise ValueError("sample_rate and hop_length must be positive")
        if not torch.isfinite(window_start_seconds).all() or (window_start_seconds < 0).any():
            raise ValueError("window_start_seconds must be finite and non-negative")

        mask = window_mask.to(dtype=torch.bool)
        valid_frames = window_valid_frames.to(dtype=torch.long)
        if (valid_frames < 0).any() or (valid_frames > input_frames).any():
            raise ValueError("window_valid_frames falls outside the input frame range")
        if not torch.equal(mask, valid_frames > 0):
            raise ValueError("window_mask must be true exactly where valid-frame count is positive")
        if windows > 1:
            later_real = mask[:, 1:] & mask[:, :-1]
            if torch.any(
                later_real
                & (window_start_seconds[:, 1:] <= window_start_seconds[:, :-1])
            ):
                raise ValueError("real window start times must be strictly increasing")

        feature_map = self._feature_map_unchecked(x, batch, windows)
        # Frequency is pooled, time is retained: B x W x T' x 64.
        temporal = feature_map.mean(dim=-2).transpose(-1, -2)
        temporal = self.proj(temporal)
        token_count = temporal.shape[2]
        token_index = torch.arange(token_count, device=x.device).view(1, 1, token_count)
        valid_token_counts = torch.div(
            valid_frames + TEMPORAL_DOWNSAMPLE - 1,
            TEMPORAL_DOWNSAMPLE,
            rounding_mode="floor",
        ).clamp_max(token_count)
        token_mask = mask.unsqueeze(-1) & (token_index < valid_token_counts.unsqueeze(-1))

        seconds_per_frame = hop_length / sample_rate
        starts = window_start_seconds.unsqueeze(-1) + (
            token_index * TEMPORAL_DOWNSAMPLE * seconds_per_frame
        )
        relative_ends = torch.minimum(
            (token_index + 1) * TEMPORAL_DOWNSAMPLE,
            valid_frames.unsqueeze(-1),
        ).to(dtype=window_start_seconds.dtype)
        ends = window_start_seconds.unsqueeze(-1) + relative_ends * seconds_per_frame
        times = (starts + ends) / 2

        flat_mask = token_mask.reshape(batch, -1)
        flat_temporal = temporal.reshape(batch, -1, self.output_dim)
        flat_temporal = flat_temporal * flat_mask.unsqueeze(-1).to(flat_temporal.dtype)
        flat_starts = starts.expand(batch, windows, token_count).reshape(batch, -1)
        flat_ends = ends.reshape(batch, -1)
        flat_times = times.reshape(batch, -1)
        flat_starts = flat_starts.masked_fill(~flat_mask, 0)
        flat_ends = flat_ends.masked_fill(~flat_mask, 0)
        flat_times = flat_times.masked_fill(~flat_mask, 0)
        window_index = (
            torch.arange(windows, device=x.device)
            .view(1, windows, 1)
            .expand(batch, windows, token_count)
            .reshape(batch, -1)
            .masked_fill(~flat_mask, -1)
        )
        denominator = flat_mask.sum(dim=1, keepdim=True).clamp_min(1).to(flat_temporal.dtype)
        pooled = flat_temporal.sum(dim=1) / denominator
        availability = flat_mask.any(dim=1)
        pooled = pooled * availability.unsqueeze(-1).to(pooled.dtype)
        return TemporalEncoderOutput(
            encoded_sequence=flat_temporal,
            sequence_times=flat_times,
            sequence_start_times=flat_starts,
            sequence_end_times=flat_ends,
            sequence_mask=flat_mask,
            sequence_window_index=window_index,
            pooled_song=pooled,
            availability=availability,
        )

    def load_instrument_pretraining(self, checkpoint: Mapping) -> str:
        """Load either Colab (``cnn.*``) or Kaggle (``enc.cnn.*``) encoder keys."""
        state = checkpoint.get("model", checkpoint)
        if not isinstance(state, Mapping):
            raise ValueError("instrument checkpoint must contain a model state mapping")
        expected = self.state_dict()
        formats = {
            "colab_v1": "",
            "kaggle_v1": "enc.",
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
            "checkpoint does not contain a complete compatible instrument encoder; "
            "expected cnn/proj or enc.cnn/enc.proj weights"
        )
