"""Descriptor math shared by the target pipeline and the harmony model.

Two exact identities of the extractor are used by the model:

* ``chroma_<pitch>_mean`` is the mean of per-frame L1-normalized chroma, so it is
  a probability distribution over the 12 pitch classes.
* ``tonnetz_<k>_mean`` equals ``TONNETZ_PHI @ chroma_mean`` because librosa's
  Tonnetz is a fixed linear map of L1-normalized chroma (max error 0.0 on all
  7,324 rows of ``data/harmony_df.csv``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor

from .constants import (
    DESCRIPTOR_TRANSFORMS,
    HARMONY_DESCRIPTORS,
    HARMONY_SCHEMA_VERSION,
    LOG1M_EPSILON,
    LOG_EPSILON,
)

TRANSFORM_CODES = {"identity": 0, "log": 1, "log1m": 2}


def tonnetz_phi() -> np.ndarray:
    """Return librosa's 6x12 Tonnetz projection (fifths, minor and major thirds)."""
    dim_map = np.linspace(0, 12, num=12, endpoint=False)
    scale = np.asarray([7 / 6, 7 / 6, 3 / 2, 3 / 2, 2 / 3, 2 / 3])
    angles = np.multiply.outer(scale, dim_map)
    angles[::2] -= 0.5
    radii = np.asarray([1.0, 1.0, 1.0, 1.0, 0.5, 0.5])
    return radii[:, None] * np.cos(np.pi * angles)


def transform_codes(names: tuple[str, ...] = HARMONY_DESCRIPTORS) -> np.ndarray:
    return np.asarray([TRANSFORM_CODES[DESCRIPTOR_TRANSFORMS[name]] for name in names])


def forward_transform_np(raw: np.ndarray, codes: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float64)
    out = raw.copy()
    log = codes == TRANSFORM_CODES["log"]
    log1m = codes == TRANSFORM_CODES["log1m"]
    out[..., log] = np.log(np.maximum(raw[..., log], 0.0) + LOG_EPSILON)
    out[..., log1m] = -np.log(np.maximum(1.0 - raw[..., log1m], 0.0) + LOG1M_EPSILON)
    return out


def inverse_transform_np(values: np.ndarray, codes: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    out = values.copy()
    log = codes == TRANSFORM_CODES["log"]
    log1m = codes == TRANSFORM_CODES["log1m"]
    out[..., log] = np.exp(values[..., log]) - LOG_EPSILON
    out[..., log1m] = 1.0 + LOG1M_EPSILON - np.exp(-values[..., log1m])
    return out


def forward_transform_torch(raw: Tensor, codes: Tensor) -> Tensor:
    log = codes == TRANSFORM_CODES["log"]
    log1m = codes == TRANSFORM_CODES["log1m"]
    logged = torch.log(raw.clamp_min(0.0) + LOG_EPSILON)
    logged_complement = -torch.log((1.0 - raw).clamp_min(0.0) + LOG1M_EPSILON)
    return torch.where(log, logged, torch.where(log1m, logged_complement, raw))


def inverse_transform_torch(values: Tensor, codes: Tensor) -> Tensor:
    log = codes == TRANSFORM_CODES["log"]
    log1m = codes == TRANSFORM_CODES["log1m"]
    # Clamp before exp so unselected branches of torch.where cannot overflow.
    exp = torch.exp(values.clamp(max=30.0)) - LOG_EPSILON
    exp_complement = 1.0 + LOG1M_EPSILON - torch.exp(-values.clamp(min=-30.0))
    return torch.where(log, exp, torch.where(log1m, exp_complement, values))


@dataclass
class HarmonyTargetTransform:
    """Named variance-stabilizing transform followed by a train-only z-score."""

    mean: np.ndarray | None = None
    scale: np.ndarray | None = None
    count: np.ndarray | None = None
    epsilon: float = 1e-8

    @property
    def codes(self) -> np.ndarray:
        return transform_codes()

    def fit(self, values: np.ndarray, valid_mask: np.ndarray | None = None) -> "HarmonyTargetTransform":
        values = self._as_matrix(values)
        mask = np.isfinite(values)
        if valid_mask is not None:
            valid_mask = np.asarray(valid_mask, dtype=bool)
            if valid_mask.shape != values.shape:
                raise ValueError("harmony valid_mask must match values")
            mask &= valid_mask
        count = mask.sum(axis=0)
        if np.any(count == 0):
            missing = [HARMONY_DESCRIPTORS[i] for i in np.flatnonzero(count == 0)]
            raise ValueError(f"No valid training harmony targets for: {missing}")
        transformed = forward_transform_np(np.where(mask, values, 0.5), self.codes)
        safe = np.where(mask, transformed, 0.0)
        mean = safe.sum(axis=0) / count
        variance = np.where(mask, (transformed - mean) ** 2, 0.0).sum(axis=0) / count
        scale = np.sqrt(variance)
        self.mean = mean.astype(np.float64)
        self.scale = np.where(scale < self.epsilon, 1.0, scale).astype(np.float64)
        self.count = count.astype(np.int64)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        self._check_fitted()
        values = self._as_matrix(values)
        return ((forward_transform_np(values, self.codes) - self.mean) / self.scale).astype(np.float32)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        self._check_fitted()
        values = self._as_matrix(values)
        return inverse_transform_np(values * self.scale + self.mean, self.codes)

    def state_dict(self) -> dict[str, Any]:
        self._check_fitted()
        return {
            "schema_version": HARMONY_SCHEMA_VERSION,
            "feature_names": list(HARMONY_DESCRIPTORS),
            "transforms": [DESCRIPTOR_TRANSFORMS[name] for name in HARMONY_DESCRIPTORS],
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "count": self.count.tolist(),
            "epsilon": self.epsilon,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "HarmonyTargetTransform":
        if tuple(state["feature_names"]) != HARMONY_DESCRIPTORS:
            raise ValueError("Checkpoint feature order does not match the harmony contract")
        expected = [DESCRIPTOR_TRANSFORMS[name] for name in HARMONY_DESCRIPTORS]
        if list(state.get("transforms", expected)) != expected:
            raise ValueError("Checkpoint descriptor transforms do not match the harmony contract")
        result = cls(epsilon=float(state.get("epsilon", 1e-8)))
        result.mean = np.asarray(state["mean"], dtype=np.float64)
        result.scale = np.asarray(state["scale"], dtype=np.float64)
        result.count = np.asarray(state["count"], dtype=np.int64)
        result._check_fitted()
        return result

    @staticmethod
    def _as_matrix(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(HARMONY_DESCRIPTORS):
            raise ValueError(
                f"Expected harmony targets [samples, {len(HARMONY_DESCRIPTORS)}], got {values.shape}"
            )
        return values

    def _check_fitted(self) -> None:
        if self.mean is None or self.scale is None or self.count is None:
            raise RuntimeError("HarmonyTargetTransform has not been fitted")
        if self.mean.shape != (len(HARMONY_DESCRIPTORS),):
            raise ValueError("Transform state does not contain exactly 45 descriptors")
