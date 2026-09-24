"""Runtime validation that rejects rather than reshapes."""

from __future__ import annotations

import torch


class ContractError(ValueError):
    """Branch or fusion tensor violated the shared architecture contract."""


def require_tensor(name: str, x: object, *, ndim: int, last: int | None = None) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        raise ContractError(f"{name} must be a torch.Tensor, got {type(x).__name__}")
    if x.ndim != ndim:
        raise ContractError(f"{name} expected ndim={ndim}, got shape {tuple(x.shape)}")
    if last is not None and x.shape[-1] != last:
        raise ContractError(f"{name} expected last dim {last}, got shape {tuple(x.shape)}")
    if x.dtype not in (torch.float32, torch.float64):
        raise ContractError(f"{name} expected float32/64, got {x.dtype}")
    return x


def require_finite(name: str, x: torch.Tensor, *, where: torch.Tensor | None = None) -> None:
    chk = x if where is None else x[where.bool()]
    if chk.numel() == 0:
        return
    if not torch.isfinite(chk).all():
        raise ContractError(f"{name} contains NaN/Inf (contract forbids silent imputation)")


def require_binary_mask(name: str, x: torch.Tensor) -> None:
    allowed = (x == 0) | (x == 1)
    if not bool(allowed.all()):
        raise ContractError(f"{name} must be 0/1, got unique={torch.unique(x).tolist()}")


def require_batch(name: str, x: torch.Tensor, batch: int) -> None:
    if x.shape[0] != batch:
        raise ContractError(f"{name} batch {x.shape[0]} != {batch}")
