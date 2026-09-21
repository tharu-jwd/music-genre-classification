"""Actual-model compute: params, checkpoint bytes, synced inference latency."""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn as nn


def parameter_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def checkpoint_size_bytes(path: Path) -> int:
    return int(path.stat().st_size) if path.exists() else 0


@torch.no_grad()
def synced_infer_ms(
    model: nn.Module,
    tokens: torch.Tensor,
    fusion_mask: torch.Tensor,
    *,
    warmup: int = 10,
    steps: int = 50,
) -> float:
    model.eval()
    device = next(model.parameters()).device
    tokens = tokens.to(device)
    fusion_mask = fusion_mask.to(device)
    use_cuda = device.type == "cuda"
    for _ in range(warmup):
        _ = model(tokens, fusion_mask, apply_dropout=False)
    if use_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        _ = model(tokens, fusion_mask, apply_dropout=False)
    if use_cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / steps * 1000.0
