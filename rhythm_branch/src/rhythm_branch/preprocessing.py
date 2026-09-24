"""Leakage-safe rhythm target alignment, normalization, and coverage auditing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .constants import RHYTHM_FEATURES, RHYTHM_SCHEMA_VERSION


def _stable_track_id(raw: object) -> str:
    value = str(raw).strip()
    match = re.fullmatch(r"(?:track_)?(\d+)", value, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"cannot normalize track ID {raw!r}")
    return f"{int(match.group(1)):07d}"


def load_rhythm_targets(path: str | Path) -> pd.DataFrame:
    """Load exactly one AcousticBrainz target row per stable track ID and split."""
    frame = pd.read_csv(path, dtype={"song_id": str, "split": str})
    required = {"song_id", "split", *RHYTHM_FEATURES}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"rhythm target CSV is missing columns: {missing}")
    frame = frame.copy()
    frame["song_id"] = frame["song_id"].map(_stable_track_id)
    if frame[["song_id", "split"]].isna().any().any():
        raise ValueError("rhythm song_id and split must be non-null")
    duplicated = frame.duplicated(["song_id"], keep=False)
    if duplicated.any():
        first = frame.loc[duplicated, ["song_id", "split"]].iloc[0].to_dict()
        raise ValueError(f"duplicate rhythm target row: {first}")
    if "source" in frame and not frame["source"].eq("acousticbrainz").all():
        raise ValueError("rhythm targets must declare source=acousticbrainz")
    frame.loc[:, RHYTHM_FEATURES] = frame.loc[:, RHYTHM_FEATURES].apply(
        pd.to_numeric, errors="coerce"
    )
    return frame


def align_manifest_and_targets(
    manifest: pd.DataFrame,
    targets: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Left-join targets by stable ID and official split without dropping mel rows."""
    required = {"song_id", "split"}
    if not required.issubset(manifest.columns):
        raise ValueError("manifest must contain song_id and split")
    examples = manifest.copy()
    examples["song_id"] = examples["song_id"].map(_stable_track_id)
    if examples.duplicated(["song_id"]).any():
        raise ValueError("manifest contains duplicate song_id examples")
    joined = examples.merge(
        targets.loc[:, ["song_id", "split", *RHYTHM_FEATURES]],
        on=["song_id", "split"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    values = joined.loc[:, RHYTHM_FEATURES].to_numpy(dtype=np.float64)
    validity = np.isfinite(values)
    return joined, values, validity


def audit_target_coverage(
    joined: pd.DataFrame,
    values: np.ndarray,
    validity: np.ndarray,
    *,
    input_scope: str,
) -> dict[str, Any]:
    """Report coverage and explicit interval exclusions.

    AcousticBrainz ``beats_count`` describes the analyzed recording. It is masked
    when the model input is an excerpt or a capped/sampled set of windows.
    """
    if input_scope not in {"full_recording", "centered_excerpt", "sampled_windows"}:
        raise ValueError("input_scope must be full_recording, centered_excerpt, or sampled_windows")
    excluded = [] if input_scope == "full_recording" else ["beats_count"]
    audited = validity.copy()
    for name in excluded:
        audited[:, RHYTHM_FEATURES.index(name)] = False
    total = len(joined)
    coverage = {
        name: (float(audited[:, index].mean()) if total else 0.0)
        for index, name in enumerate(RHYTHM_FEATURES)
    }
    exact_interval_match = input_scope == "full_recording"
    interval_audit = {
        name: {
            "target_scope": "whole_recording",
            "model_input_scope": input_scope,
            "exact_interval_match": exact_interval_match,
            "status": (
                "compatible"
                if exact_interval_match
                else (
                    "excluded_length_dependent_count"
                    if name == "beats_count"
                    else "retained_song_level_target_from_same_recording"
                )
            ),
        }
        for name in RHYTHM_FEATURES
    }
    return {
        "schema_version": RHYTHM_SCHEMA_VERSION,
        "input_scope": input_scope,
        "rows": total,
        "matched_rows": int((joined["_merge"] == "both").sum()) if "_merge" in joined else None,
        "field_coverage": coverage,
        "field_interval_audit": interval_audit,
        "excluded_fields": excluded,
        "exclusion_reason": (
            "beats_count is a whole-recording count and is not interval-compatible"
            if excluded
            else None
        ),
        "validity_mask": audited,
    }


@dataclass
class RhythmStandardizer:
    """Per-field z-score parameters fitted on training observations only."""

    mean: np.ndarray | None = None
    scale: np.ndarray | None = None
    count: np.ndarray | None = None
    epsilon: float = 1e-8

    def fit(
        self,
        values: np.ndarray,
        validity_mask: np.ndarray,
        *,
        excluded_fields: Sequence[str] = (),
    ) -> "RhythmStandardizer":
        matrix = self._matrix(values)
        mask = np.asarray(validity_mask, dtype=bool)
        if mask.shape != matrix.shape:
            raise ValueError("validity_mask must match rhythm values")
        mask &= np.isfinite(matrix)
        unknown = set(excluded_fields).difference(RHYTHM_FEATURES)
        if unknown:
            raise ValueError(f"unknown excluded rhythm fields: {sorted(unknown)}")
        excluded_indices = np.asarray(
            [name in set(excluded_fields) for name in RHYTHM_FEATURES], dtype=bool
        )
        mask[:, excluded_indices] = False
        count = mask.sum(axis=0)
        missing_required = (count == 0) & ~excluded_indices
        if np.any(missing_required):
            names = [RHYTHM_FEATURES[index] for index in np.flatnonzero(missing_required)]
            raise ValueError(f"no valid training targets for: {names}")
        safe_count = np.maximum(count, 1)
        safe = np.where(mask, matrix, 0.0)
        mean = safe.sum(axis=0) / safe_count
        variance = np.where(mask, (matrix - mean) ** 2, 0.0).sum(axis=0) / safe_count
        scale = np.sqrt(variance)
        self.mean = mean.astype(np.float64)
        self.scale = np.where(scale < self.epsilon, 1.0, scale).astype(np.float64)
        self.count = count.astype(np.int64)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        self._check()
        return ((self._matrix(values) - self.mean) / self.scale).astype(np.float32)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        self._check()
        return (self._matrix(values) * self.scale + self.mean).astype(np.float64)

    def state_dict(self) -> dict[str, Any]:
        self._check()
        return {
            "schema_version": RHYTHM_SCHEMA_VERSION,
            "feature_names": list(RHYTHM_FEATURES),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "count": self.count.tolist(),
            "epsilon": self.epsilon,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "RhythmStandardizer":
        if state.get("schema_version") != RHYTHM_SCHEMA_VERSION:
            raise ValueError("checkpoint rhythm schema version does not match")
        if tuple(state.get("feature_names", ())) != RHYTHM_FEATURES:
            raise ValueError("checkpoint rhythm feature order does not match")
        result = cls(epsilon=float(state.get("epsilon", 1e-8)))
        result.mean = np.asarray(state["mean"], dtype=np.float64)
        result.scale = np.asarray(state["scale"], dtype=np.float64)
        result.count = np.asarray(state["count"], dtype=np.int64)
        result._check()
        return result

    @staticmethod
    def _matrix(values: np.ndarray) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(RHYTHM_FEATURES):
            raise ValueError(f"expected (N,{len(RHYTHM_FEATURES)}) rhythm values")
        return matrix

    def _check(self) -> None:
        expected = (len(RHYTHM_FEATURES),)
        if self.mean is None or self.scale is None or self.count is None:
            raise RuntimeError("RhythmStandardizer has not been fitted")
        if self.mean.shape != expected or self.scale.shape != expected or self.count.shape != expected:
            raise ValueError("invalid rhythm standardizer state")


def fit_training_standardizer(
    joined: pd.DataFrame,
    values: np.ndarray,
    validity_mask: np.ndarray,
    *,
    excluded_fields: Sequence[str] = (),
) -> RhythmStandardizer:
    """Fit normalization exclusively on rows whose official split is ``train``."""
    if "split" not in joined:
        raise ValueError("joined target table must contain the official split")
    train_rows = joined["split"].astype(str).eq("train").to_numpy()
    if not train_rows.any():
        raise ValueError("no official training rows are available for rhythm normalization")
    return RhythmStandardizer().fit(
        np.asarray(values)[train_rows],
        np.asarray(validity_mask)[train_rows],
        excluded_fields=excluded_fields,
    )
