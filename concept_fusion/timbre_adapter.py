"""Accept Senindu's published timbre z_timbre (B,35). Reject fusion_token."""

from __future__ import annotations

import torch

from concept_fusion.contract import N_TIMBRE_CONCEPTS, TIMBRE_FEATURES
from concept_fusion.types import BranchOutput
from concept_fusion.validation import ContractError


def from_timbre_branch(
    output: dict | torch.Tensor,
    *,
    supervision_mask: torch.Tensor | None = None,
    fusion_mask: torch.Tensor | None = None,
    name: str = "timbre",
) -> BranchOutput:
    """Map timbre v2 output onto BranchOutput.

    Accepts either the raw ``z_timbre`` tensor or the inference dict with
    ``z_timbre`` / ``d_hat_standardized``. Fusion consumes only those 35
    standardized concepts — never ``h_audio``.
    """
    if isinstance(output, dict):
        if output.get("fusion_token") is not None:
            raise ContractError("timbre v2 must not return fusion_token; fusion owns Linear(35,64)")
        if "h_audio" in output and output["h_audio"] is not None:
            raise ContractError("fusion must not consume h_audio; that bypasses the timbre bottleneck")
        z = output.get("z_timbre", output.get("d_hat_standardized", output.get("concept_values")))
        if z is None:
            raise ContractError("timbre output missing z_timbre / d_hat_standardized")
        if supervision_mask is None:
            supervision_mask = output.get("supervision_mask")
        if fusion_mask is None:
            fusion_mask = output.get("fusion_mask")
        names = output.get("feature_names")
        if names is not None and tuple(names) != TIMBRE_FEATURES:
            raise ContractError("timbre feature_names must match published FEATURE_COLUMNS order")
    else:
        z = output

    if not isinstance(z, torch.Tensor) or z.ndim != 2 or z.shape[1] != N_TIMBRE_CONCEPTS:
        raise ContractError(f"timbre z_timbre must be (B,{N_TIMBRE_CONCEPTS})")
    batch = z.shape[0]
    if supervision_mask is None:
        supervision_mask = torch.ones(batch, N_TIMBRE_CONCEPTS, dtype=z.dtype, device=z.device)
    if fusion_mask is None:
        fusion_mask = torch.ones(batch, 1, dtype=z.dtype, device=z.device)
    return BranchOutput(
        name=name,
        concept_values=z,
        fusion_token=None,
        supervision_mask=supervision_mask,
        fusion_mask=fusion_mask,
        tag_order=TIMBRE_FEATURES,
    )
