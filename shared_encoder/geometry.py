"""Temporal masks, source-window identity, and timestamp calculation."""

from dataclasses import dataclass

import torch
from torch import Tensor

from .constants import TEMPORAL_DOWNSAMPLE


@dataclass(frozen=True)
class TemporalLayout:
    token_mask: Tensor
    flat_mask: Tensor
    sequence_start_times: Tensor
    sequence_end_times: Tensor
    sequence_times: Tensor
    sequence_window_index: Tensor


def build_temporal_layout(
    *,
    valid_frames: Tensor,
    window_mask: Tensor,
    window_start_seconds: Tensor,
    token_count: int,
    sample_rate: int,
    hop_length: int,
) -> TemporalLayout:
    """Build window-major token metadata using stride-bin alignment intervals."""
    if sample_rate < 1 or hop_length < 1:
        raise ValueError("sample_rate and hop_length must be positive")
    batch, windows = valid_frames.shape
    token_index = torch.arange(token_count, device=valid_frames.device).view(1, 1, token_count)
    valid_token_counts = torch.div(
        valid_frames + TEMPORAL_DOWNSAMPLE - 1,
        TEMPORAL_DOWNSAMPLE,
        rounding_mode="floor",
    ).clamp_max(token_count)
    token_mask = window_mask.unsqueeze(-1) & (token_index < valid_token_counts.unsqueeze(-1))

    seconds_per_frame = hop_length / sample_rate
    starts = window_start_seconds.unsqueeze(-1) + (
        token_index * TEMPORAL_DOWNSAMPLE * seconds_per_frame
    )
    relative_ends = torch.minimum(
        (token_index + 1) * TEMPORAL_DOWNSAMPLE,
        valid_frames.unsqueeze(-1),
    ).to(dtype=window_start_seconds.dtype)
    ends = window_start_seconds.unsqueeze(-1) + relative_ends * seconds_per_frame
    centers = (starts + ends) / 2

    flat_mask = token_mask.reshape(batch, -1)
    flat_starts = starts.expand(batch, windows, token_count).reshape(batch, -1)
    flat_ends = ends.reshape(batch, -1)
    flat_centers = centers.reshape(batch, -1)
    flat_starts = flat_starts.masked_fill(~flat_mask, 0)
    flat_ends = flat_ends.masked_fill(~flat_mask, 0)
    flat_centers = flat_centers.masked_fill(~flat_mask, 0)
    window_index = (
        torch.arange(windows, device=valid_frames.device)
        .view(1, windows, 1)
        .expand(batch, windows, token_count)
        .reshape(batch, -1)
        .masked_fill(~flat_mask, -1)
    )
    return TemporalLayout(
        token_mask=token_mask,
        flat_mask=flat_mask,
        sequence_start_times=flat_starts,
        sequence_end_times=flat_ends,
        sequence_times=flat_centers,
        sequence_window_index=window_index,
    )

