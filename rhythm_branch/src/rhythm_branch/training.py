"""Checkpoint helpers for standalone rhythm-branch screening."""

from pathlib import Path
from typing import Any

import torch

from .constants import RHYTHM_FEATURES, RHYTHM_SCHEMA_VERSION
from .model import RhythmBranch, RhythmBranchConfig
from .preprocessing import RhythmStandardizer


def save_checkpoint(
    path: str | Path,
    model: RhythmBranch,
    standardizer: RhythmStandardizer,
    *,
    epoch: int,
    metrics: dict[str, Any],
    seed: int,
    input_scope: str,
    excluded_fields: list[str] | None = None,
    optimizer: torch.optim.Optimizer | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format_version": 1,
        "architecture": "temporal_rhythm_branch_v1",
        "schema_version": RHYTHM_SCHEMA_VERSION,
        "model_config": model.config.to_dict(),
        "model_state_dict": model.state_dict(),
        "feature_names": list(RHYTHM_FEATURES),
        "standardizer": standardizer.state_dict(),
        "input_scope": input_scope,
        "excluded_fields": list(excluded_fields or []),
        "epoch": int(epoch),
        "metrics": metrics,
        "seed": int(seed),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path, device: torch.device | str = "cpu"
) -> tuple[RhythmBranch, RhythmStandardizer, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("schema_version") != RHYTHM_SCHEMA_VERSION:
        raise ValueError("checkpoint rhythm schema version does not match")
    if tuple(payload.get("feature_names", ())) != RHYTHM_FEATURES:
        raise ValueError("checkpoint rhythm feature order does not match")
    model = RhythmBranch(RhythmBranchConfig.from_dict(payload["model_config"]))
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    return model, RhythmStandardizer.from_state_dict(payload["standardizer"]), payload
