"""Training, evaluation, and reproducible checkpoint helpers."""

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .constants import FEATURE_COLUMNS
from .losses import masked_smooth_l1_loss
from .model import TimbreBranch, TimbreBranchConfig
from .preprocessing import TimbreStandardizer


def train_one_epoch(
    model: TimbreBranch,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
) -> float:
    model.train()
    total_loss = 0.0
    total_rows = 0
    for h_audio, targets, mask, _track_ids in loader:
        h_audio = h_audio.to(device)
        targets = targets.to(device)
        mask = mask.to(device)
        optimizer.zero_grad(set_to_none=True)
        predictions = model(h_audio)
        loss = masked_smooth_l1_loss(predictions, targets, mask)
        loss.backward()
        optimizer.step()
        rows = h_audio.shape[0]
        total_loss += float(loss.detach()) * rows
        total_rows += rows
    if total_rows == 0:
        raise ValueError("Training loader is empty")
    return total_loss / total_rows


@torch.no_grad()
def evaluate(model: TimbreBranch, loader, device: torch.device | str) -> dict[str, Any]:
    model.eval()
    predictions, targets, masks = [], [], []
    for h_audio, target, mask, _track_ids in loader:
        predictions.append(model(h_audio.to(device)).cpu())
        targets.append(target.cpu())
        masks.append(mask.cpu())
    if not predictions:
        raise ValueError("Evaluation loader is empty")
    prediction = torch.cat(predictions).numpy()
    target = torch.cat(targets).numpy()
    mask = torch.cat(masks).numpy().astype(bool)

    error = prediction - target
    per_mae, per_rmse = [], []
    for column in range(len(FEATURE_COLUMNS)):
        valid = mask[:, column] & np.isfinite(error[:, column])
        if not valid.any():
            per_mae.append(float("nan"))
            per_rmse.append(float("nan"))
            continue
        column_error = error[valid, column]
        per_mae.append(float(np.mean(np.abs(column_error))))
        per_rmse.append(float(np.sqrt(np.mean(column_error ** 2))))
    valid = mask & np.isfinite(error)
    return {
        "macro_mae_standardized": float(np.nanmean(per_mae)),
        "macro_rmse_standardized": float(np.nanmean(per_rmse)),
        "cell_mae_standardized": float(np.mean(np.abs(error[valid]))),
        "per_descriptor_mae_standardized": dict(zip(FEATURE_COLUMNS, per_mae)),
        "per_descriptor_rmse_standardized": dict(zip(FEATURE_COLUMNS, per_rmse)),
        "valid_cells": int(valid.sum()),
    }


def save_checkpoint(
    path: str | Path,
    model: TimbreBranch,
    standardizer: TimbreStandardizer,
    *,
    epoch: int,
    metrics: dict[str, Any],
    seed: int,
    optimizer: torch.optim.Optimizer | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "architecture": "strict_timbre_concept_bottleneck",
        "model_config": model.config.to_dict(),
        "model_state_dict": model.state_dict(),
        "feature_names": list(FEATURE_COLUMNS),
        "standardizer": standardizer.state_dict(),
        "epoch": int(epoch),
        "metrics": metrics,
        "seed": int(seed),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[TimbreBranch, TimbreStandardizer, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if tuple(payload["feature_names"]) != FEATURE_COLUMNS:
        raise ValueError("Checkpoint feature order does not match the 35-D contract")
    model = TimbreBranch(TimbreBranchConfig.from_dict(payload["model_config"]))
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    standardizer = TimbreStandardizer.from_state_dict(payload["standardizer"])
    return model, standardizer, payload
