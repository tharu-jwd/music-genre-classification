"""Align temporal chroma pseudo-labels to explicit shared-encoder token intervals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AlignedChroma:
    chroma: np.ndarray
    valid: np.ndarray
    contributing_frames: np.ndarray
    coverage: float

    def validate(self) -> None:
        tokens = len(self.valid)
        if self.chroma.shape != (tokens, 12):
            raise ValueError("aligned chroma must have shape tokens x 12")
        if self.contributing_frames.shape != (tokens,):
            raise ValueError("contributing-frame counts must share the token axis")
        if not np.isfinite(self.chroma).all() or (self.chroma < 0).any():
            raise ValueError("aligned chroma must be finite and non-negative")
        if np.any(self.chroma[~self.valid] != 0):
            raise ValueError("invalid aligned targets must remain zero")
        sums = self.chroma[self.valid].sum(axis=1)
        if sums.size and not np.allclose(sums, 1.0, atol=1e-5):
            raise ValueError("valid aligned chroma rows must sum to one")
        if not 0 <= self.coverage <= 1:
            raise ValueError("coverage must be in [0, 1]")


def align_chroma_to_intervals(
    frame_times_seconds: np.ndarray,
    chroma: np.ndarray,
    frame_valid: np.ndarray,
    token_start_seconds: np.ndarray,
    token_end_seconds: np.ndarray,
    token_mask: np.ndarray,
    *,
    max_token_duration_seconds: float,
) -> AlignedChroma:
    """Average valid chroma samples inside each explicit encoder-token interval.

    The duration limit is mandatory so a caller cannot silently pass 29-second
    window vectors while claiming to model chord progressions. It is a contract
    check, not a hyperparameter search performed by this function.
    """
    times = np.asarray(frame_times_seconds, dtype=np.float64)
    values = np.asarray(chroma, dtype=np.float32)
    valid = np.asarray(frame_valid, dtype=bool)
    starts = np.asarray(token_start_seconds, dtype=np.float64)
    ends = np.asarray(token_end_seconds, dtype=np.float64)
    mask = np.asarray(token_mask, dtype=bool)
    if times.ndim != 1 or values.shape != (len(times), 12) or valid.shape != times.shape:
        raise ValueError("chroma frames require times[N], chroma[N,12], and valid[N]")
    if starts.ndim != 1 or ends.shape != starts.shape or mask.shape != starts.shape:
        raise ValueError("token starts, ends, and mask must be equal-length vectors")
    if not np.isfinite(max_token_duration_seconds) or max_token_duration_seconds <= 0:
        raise ValueError("max_token_duration_seconds must be finite and positive")
    if not np.isfinite(times).all() or not np.isfinite(values).all():
        raise ValueError("chroma frames contain non-finite values")
    if len(times) and np.any(np.diff(times) <= 0):
        raise ValueError("chroma frame times must be strictly increasing")
    if (values < 0).any():
        raise ValueError("chroma values must be non-negative")
    valid_sums = values[valid].sum(axis=1)
    if valid_sums.size and not np.allclose(valid_sums, 1.0, atol=1e-5):
        raise ValueError("valid source chroma frames must sum to one")
    if np.any(values[~valid] != 0):
        raise ValueError("invalid source chroma frames must be zero")
    if not np.isfinite(starts).all() or not np.isfinite(ends).all():
        raise ValueError("token intervals contain non-finite values")
    if np.any(mask & ((starts < 0) | (ends <= starts))):
        raise ValueError("valid token intervals must have finite positive duration")
    durations = ends - starts
    if np.any(mask & (durations > max_token_duration_seconds + 1e-9)):
        raise ValueError(
            "encoder token interval is too coarse for the registered temporal target"
        )

    output = np.zeros((len(starts), 12), dtype=np.float32)
    output_valid = np.zeros(len(starts), dtype=bool)
    counts = np.zeros(len(starts), dtype=np.int64)
    for index in np.flatnonzero(mask):
        left = int(np.searchsorted(times, starts[index], side="left"))
        right = int(np.searchsorted(times, ends[index], side="left"))
        selected = valid[left:right]
        counts[index] = int(selected.sum())
        if counts[index] == 0:
            continue
        mean = values[left:right][selected].mean(axis=0)
        total = float(mean.sum())
        if np.isfinite(total) and total > np.finfo(np.float32).eps:
            output[index] = mean / total
            output_valid[index] = True
    requested = int(mask.sum())
    result = AlignedChroma(
        chroma=output,
        valid=output_valid,
        contributing_frames=counts,
        coverage=float(output_valid.sum() / requested) if requested else 0.0,
    )
    result.validate()
    return result
