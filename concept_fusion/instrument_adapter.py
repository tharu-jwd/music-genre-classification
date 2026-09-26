"""Accept Anupama's published instrument-branch dict. Reject fusion_token."""

from __future__ import annotations

import torch

from concept_fusion.contract import INSTRUMENT_HIDDEN_DIM, INSTRUMENT_TAGS, N_INSTRUMENT_TAGS
from concept_fusion.types import BranchOutput
from concept_fusion.validation import ContractError


def from_instrument_branch(output: dict, *, name: str = "instrument") -> BranchOutput:
    """Map the v2 instrument notebook dict onto BranchOutput.

    Required keys: concept_values (B,41), logits (B,41),
    supervision_mask (B,41), fusion_mask (B,1).
    Optional: diagnostics.hidden (B,128), detached.
    """
    if "fusion_token" in output and output["fusion_token"] is not None:
        raise ContractError(
            "instrument v2 must not return fusion_token; fusion owns Linear(41,64)"
        )
    for key in ("concept_values", "logits", "supervision_mask", "fusion_mask"):
        if key not in output:
            raise ContractError(f"instrument output missing {key}")
    hidden = None
    diag = output.get("diagnostics") or {}
    if "hidden" in diag and diag["hidden"] is not None:
        hidden = diag["hidden"]
        if hidden.shape[-1] != INSTRUMENT_HIDDEN_DIM:
            raise ContractError(
                f"instrument hidden last dim {hidden.shape[-1]} != {INSTRUMENT_HIDDEN_DIM}"
            )
        hidden = hidden.detach()
    probs = output["concept_values"]
    if tuple(probs.shape[1:]) != (N_INSTRUMENT_TAGS,):
        raise ContractError(f"instrument concept_values must be (B,{N_INSTRUMENT_TAGS})")
    return BranchOutput(
        name=name,
        concept_values=probs,
        fusion_token=None,
        supervision_mask=output["supervision_mask"],
        fusion_mask=output["fusion_mask"],
        hidden_token=hidden,
        logits=output["logits"],
        tag_order=INSTRUMENT_TAGS,
    )
