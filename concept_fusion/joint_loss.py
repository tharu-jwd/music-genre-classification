"""Joint genre + masked concept losses. Missing supervision never drops a track."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from concept_fusion.contract import CONCEPT_ORDER, ConceptCounts
from concept_fusion.types import BranchBundle
from concept_fusion.validation import ContractError, require_finite, require_tensor


@dataclass
class LossWeights:
    genre: float = 1.0
    instrument: float = 1.0
    rhythm: float = 1.0
    timbre: float = 1.0
    harmony: float = 1.0
    use_kendall: bool = False


@dataclass
class LossBreakdown:
    total: torch.Tensor
    terms: dict[str, float] = field(default_factory=dict)
    n_observed: dict[str, int] = field(default_factory=dict)


def _masked_mean(loss_elem: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, int]:
    m = mask.float()
    n = int(m.sum().item())
    if n == 0:
        return loss_elem.new_zeros(()), 0
    return (loss_elem * m).sum() / m.sum(), n


class JointLossOrchestrator(torch.nn.Module):
    """L = L_genre + Σ λ_k L_k, each L_k averaged over observed elements only.

    Instrument: BCE on predicted probabilities vs 0/1 (or soft) targets.
    Rhythm/timbre/harmony: Smooth L1 on standardized values.
    NaN targets are allowed only where supervision_mask is 0.
    """

    def __init__(self, counts: ConceptCounts | None = None, weights: LossWeights | None = None):
        super().__init__()
        self.counts = counts or ConceptCounts()
        self.weights = weights or LossWeights()
        if self.weights.use_kendall:
            self.log_vars = torch.nn.Parameter(torch.zeros(5))
        else:
            self.register_parameter("log_vars", None)

    def forward(
        self,
        genre_logits: torch.Tensor,
        genre_targets: torch.Tensor,
        bundle: BranchBundle,
        concept_targets: dict[str, torch.Tensor],
    ) -> LossBreakdown:
        logits = require_tensor("genre_logits", genre_logits, ndim=2)
        y = require_tensor("genre_targets", genre_targets, ndim=2)
        if logits.shape != y.shape:
            raise ContractError(f"genre logits {tuple(logits.shape)} != targets {tuple(y.shape)}")
        require_finite("genre_logits", logits)
        require_finite("genre_targets", y)

        terms: dict[str, torch.Tensor] = {
            "genre": F.binary_cross_entropy_with_logits(logits, y, reduction="mean")
        }
        n_obs = {"genre": int(y.numel())}

        for name in CONCEPT_ORDER:
            pred = bundle.concept_values(name)
            mask = bundle.supervision_mask(name)
            if name not in concept_targets:
                raise ContractError(f"missing concept_targets[{name}]")
            tgt = require_tensor(f"concept_targets[{name}]", concept_targets[name], ndim=2)
            if tgt.shape != pred.shape:
                raise ContractError(f"{name} target {tuple(tgt.shape)} != pred {tuple(pred.shape)}")
            obs = mask > 0.5
            require_finite(f"concept_targets[{name}][observed]", tgt, where=mask)
            dummy = torch.zeros_like(pred)
            pred_f = torch.where(obs, pred, dummy)
            tgt_f = torch.where(obs, tgt, dummy)
            if name == "instrument":
                pred_f = pred_f.clamp(1e-6, 1 - 1e-6)
                raw = F.binary_cross_entropy(pred_f, tgt_f, reduction="none")
            else:
                raw = F.smooth_l1_loss(pred_f, tgt_f, reduction="none")
            l_k, n_k = _masked_mean(raw, mask)
            terms[name] = l_k
            n_obs[name] = n_k

        if self.weights.use_kendall:
            ordered = ["genre", *CONCEPT_ORDER]
            total = logits.new_zeros(())
            for i, name in enumerate(ordered):
                s = self.log_vars[i]
                total = total + 0.5 * torch.exp(-s) * terms[name] + 0.5 * s
        else:
            w = self.weights
            total = (
                w.genre * terms["genre"]
                + w.instrument * terms["instrument"]
                + w.rhythm * terms["rhythm"]
                + w.timbre * terms["timbre"]
                + w.harmony * terms["harmony"]
            )

        return LossBreakdown(
            total=total,
            terms={k: float(v.detach().item()) for k, v in terms.items()},
            n_observed=n_obs,
        )
