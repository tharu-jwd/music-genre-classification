"""Faithfulness interventions. Occlusion uses the same fusion_mask as training dropout.

Hard rule: if concept dropout was not used in training, occlusion claims are invalid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from concept_fusion.contract import CONCEPT_ORDER, N_CONCEPTS
from concept_fusion.metrics import sigmoid_np
from concept_fusion.validation import ContractError


@dataclass
class OcclusionResult:
    original_probs: np.ndarray
    masked_probs: np.ndarray
    delta: np.ndarray  # masked - original (B, 87)
    abs_mean_delta: float
    concept: str


def occlude_concept(
    model,
    tokens: torch.Tensor,
    fusion_mask: torch.Tensor,
    concept: str,
    *,
    dropout_was_trained: bool,
) -> OcclusionResult:
    if not dropout_was_trained:
        raise ContractError(
            "occlusion-based faithfulness is forbidden unless concept dropout "
            "was enabled during training (otherwise deltas measure OOD shift)"
        )
    if concept not in CONCEPT_ORDER:
        raise ContractError(concept)
    idx = CONCEPT_ORDER.index(concept)
    model.eval()
    with torch.no_grad():
        orig_logits, _ = model(tokens, fusion_mask, apply_dropout=False)
        mask = fusion_mask.clone()
        mask[:, idx] = 0.0
        new_logits, fout = model(tokens, mask, apply_dropout=False)
        if not torch.isclose(fout.gates[:, idx], torch.zeros_like(fout.gates[:, idx]), atol=1e-6).all():
            raise ContractError("occluded concept gate was not exactly zero")
    p0 = sigmoid_np(orig_logits)
    p1 = sigmoid_np(new_logits)
    delta = p1 - p0
    return OcclusionResult(
        original_probs=p0,
        masked_probs=p1,
        delta=delta,
        abs_mean_delta=float(np.abs(delta).mean()),
        concept=concept,
    )


def occlude_each_concept(model, tokens, fusion_mask, *, dropout_was_trained: bool) -> dict[str, OcclusionResult]:
    return {
        c: occlude_concept(model, tokens, fusion_mask, c, dropout_was_trained=dropout_was_trained)
        for c in CONCEPT_ORDER
    }


def gate_vs_occlusion_correlation(gates: torch.Tensor, occ: dict[str, OcclusionResult]) -> dict[str, float]:
    """Spearman-like rank correlation between mean gate and mean |delta| across concepts (n=4)."""
    g = gates.detach().cpu().numpy().mean(axis=0)
    d = np.array([occ[c].abs_mean_delta for c in CONCEPT_ORDER])
    # Pearson on ranks (n=4)
    rg = np.argsort(np.argsort(-g))
    rd = np.argsort(np.argsort(-d))
    if rg.std() < 1e-12 or rd.std() < 1e-12:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(rg, rd)[0, 1])
    return {
        "rank_correlation_gate_vs_occlusion": corr,
        "mean_gates": {c: float(g[i]) for i, c in enumerate(CONCEPT_ORDER)},
        "mean_abs_delta": {c: float(d[i]) for i, c in enumerate(CONCEPT_ORDER)},
        "n_concepts": N_CONCEPTS,
    }
