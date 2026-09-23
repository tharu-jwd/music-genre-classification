"""Target loading and leakage-safe standardization for the 35 concepts."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .constants import FEATURE_COLUMNS


def load_timbre_targets(path: str | Path) -> pd.DataFrame:
    """Load and validate one target row per track in the declared feature order."""
    frame = pd.read_csv(path, dtype={"TRACK_ID": str})
    required = {"TRACK_ID", *FEATURE_COLUMNS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Timbre target CSV is missing columns: {missing}")
    if frame["TRACK_ID"].isna().any() or frame["TRACK_ID"].duplicated().any():
        raise ValueError("TRACK_ID must be non-null and unique in the timbre target CSV")
    if "extraction_status" in frame and not frame["extraction_status"].eq("ok").all():
        bad = int((frame["extraction_status"] != "ok").sum())
        raise ValueError(f"Timbre target CSV contains {bad} non-OK rows")
    values = frame.loc[:, FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    frame.loc[:, FEATURE_COLUMNS] = values
    return frame


@dataclass
class TimbreStandardizer:
    """Per-concept z-score parameters fitted exclusively on training rows."""

    mean: np.ndarray | None = None
    scale: np.ndarray | None = None
    count: np.ndarray | None = None
    epsilon: float = 1e-8

    def fit(self, values: np.ndarray, valid_mask: np.ndarray | None = None) -> "TimbreStandardizer":
        values = self._as_matrix(values)
        mask = np.isfinite(values)
        if valid_mask is not None:
            valid_mask = np.asarray(valid_mask, dtype=bool)
            if valid_mask.shape != values.shape:
                raise ValueError("valid_mask must match values")
            mask &= valid_mask

        count = mask.sum(axis=0)
        if np.any(count == 0):
            missing = [FEATURE_COLUMNS[i] for i in np.flatnonzero(count == 0)]
            raise ValueError(f"No valid training targets for: {missing}")

        safe = np.where(mask, values, 0.0)
        mean = safe.sum(axis=0) / count
        variance = np.where(mask, (values - mean) ** 2, 0.0).sum(axis=0) / count
        scale = np.sqrt(variance)
        scale = np.where(scale < self.epsilon, 1.0, scale)

        self.mean = mean.astype(np.float64)
        self.scale = scale.astype(np.float64)
        self.count = count.astype(np.int64)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        self._check_fitted()
        values = self._as_matrix(values)
        return ((values - self.mean) / self.scale).astype(np.float32)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        self._check_fitted()
        values = self._as_matrix(values)
        return (values * self.scale + self.mean).astype(np.float64)

    def state_dict(self) -> dict[str, Any]:
        self._check_fitted()
        return {
            "feature_names": list(FEATURE_COLUMNS),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "count": self.count.tolist(),
            "epsilon": self.epsilon,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "TimbreStandardizer":
        if tuple(state["feature_names"]) != FEATURE_COLUMNS:
            raise ValueError("Checkpoint feature order does not match the timbre contract")
        result = cls(epsilon=float(state.get("epsilon", 1e-8)))
        result.mean = np.asarray(state["mean"], dtype=np.float64)
        result.scale = np.asarray(state["scale"], dtype=np.float64)
        result.count = np.asarray(state["count"], dtype=np.int64)
        result._check_fitted()
        return result

    @staticmethod
    def _as_matrix(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(FEATURE_COLUMNS):
            raise ValueError(
                f"Expected [samples, {len(FEATURE_COLUMNS)}], received {values.shape}"
            )
        return values

    def _check_fitted(self) -> None:
        if self.mean is None or self.scale is None or self.count is None:
            raise RuntimeError("TimbreStandardizer has not been fitted")
        expected = (len(FEATURE_COLUMNS),)
        if self.mean.shape != expected or self.scale.shape != expected or self.count.shape != expected:
            raise ValueError("Scaler state does not contain exactly 35 concepts")


def align_embeddings_and_targets(
    track_ids: Sequence[str],
    embeddings: np.ndarray,
    targets: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align encoder embeddings with target rows by TRACK_ID, never by row order."""
    track_ids = np.asarray(track_ids, dtype=str)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[1] != 128:
        raise ValueError(f"Embeddings must have shape [samples, 128], got {embeddings.shape}")
    if len(track_ids) != len(embeddings):
        raise ValueError("track_ids and embeddings must contain the same number of rows")
    if len(np.unique(track_ids)) != len(track_ids):
        raise ValueError("Embedding TRACK_ID values must be unique")

    indexed = targets.set_index("TRACK_ID", verify_integrity=True)
    missing = sorted(set(track_ids).difference(indexed.index))
    if missing:
        raise ValueError(f"No timbre target for {len(missing)} embedding tracks; first={missing[0]}")
    ordered = indexed.loc[track_ids, FEATURE_COLUMNS]
    target_values = ordered.to_numpy(dtype=np.float64)
    valid_mask = np.isfinite(target_values)
    return embeddings, target_values, valid_mask
