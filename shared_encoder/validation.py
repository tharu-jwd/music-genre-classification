"""Input, metadata, and padding validation for the shared encoder."""

import torch
from torch import Tensor


def validate_audio(x: Tensor) -> tuple[int, int, int, int]:
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


def validate_metadata(
    x: Tensor,
    window_mask: Tensor,
    window_valid_frames: Tensor,
    window_start_seconds: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    batch, windows, _, input_frames = validate_audio(x)
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

    for row in range(batch):
        valid_starts = starts[row, mask[row]]
        if len(valid_starts) > 1 and torch.any(valid_starts[1:] <= valid_starts[:-1]):
            raise ValueError("real window start times must be strictly increasing")
    return mask, valid_frames, starts


def input_frame_mask(valid_frames: Tensor, input_frames: int) -> Tensor:
    """Return `(B,W,1,1,F)` mask for real leading frames."""
    frame_index = torch.arange(input_frames, device=valid_frames.device)
    return frame_index.view(1, 1, 1, 1, input_frames) < valid_frames[:, :, None, None, None]

