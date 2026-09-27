"""Joint genre + masked concept losses. Missing supervision never drops a track."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from concept_fusion.contract import CONCEPT_ORDER, N_HARMONY_CHROMA, N_HARMONY_CHORDS, ConceptCounts
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


@dataclass
class SongHarmonyTargets:
    """Song-level chroma distributions; no temporal alignment is implied."""

    chroma: torch.Tensor  # (B,12)
    chroma_mask: torch.Tensor  # (B,)

    def indexed(self, index: torch.Tensor) -> "SongHarmonyTargets":
        return SongHarmonyTargets(self.chroma[index], self.chroma_mask[index])


@dataclass
class HarmonyTargets:
    """Temporal pseudo-supervision aligned to harmony branch tokens."""

    chroma: torch.Tensor  # (B,T,12), non-negative distributions
    chroma_mask: torch.Tensor  # (B,T)
    chord_labels: torch.Tensor | None = None  # (B,T), class indices
    chord_mask: torch.Tensor | None = None  # (B,T)

    def indexed(self, index: torch.Tensor) -> "HarmonyTargets":
        return HarmonyTargets(
            chroma=self.chroma[index],
            chroma_mask=self.chroma_mask[index],
            chord_labels=None if self.chord_labels is None else self.chord_labels[index],
            chord_mask=None if self.chord_mask is None else self.chord_mask[index],
        )


def _masked_mean(loss_elem: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, int]:
    m = mask.float()
    n = int(m.sum().item())
    if n == 0:
        return loss_elem.new_zeros(()), 0
    return (loss_elem * m).sum() / m.sum(), n


class JointLossOrchestrator(torch.nn.Module):
    """L = L_genre + Σ λ_k L_k, each L_k averaged over observed elements only.

    Instrument: BCE-with-logits when the branch supplies logits (v2); else BCE on probabilities.
    Rhythm/timbre: Smooth L1 on standardized values.
    Harmony: masked Smooth L1 for song descriptors, or temporal chroma soft-target
    loss plus optional chord CE when aligned chroma/chord targets are supplied.
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
        concept_targets: dict[str, torch.Tensor | HarmonyTargets | SongHarmonyTargets],
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
            if name == "harmony":
                harmony_targets = concept_targets[name]
                if isinstance(harmony_targets, torch.Tensor):
                    tgt = require_tensor("concept_targets[harmony]", harmony_targets, ndim=2)
                    if tgt.shape != pred.shape:
                        raise ContractError(
                            f"harmony target {tuple(tgt.shape)} != pred {tuple(pred.shape)}"
                        )
                    obs = mask > 0.5
                    require_finite("concept_targets[harmony][observed]", tgt, where=mask)
                    pred_f = torch.where(obs, pred, torch.zeros_like(pred))
                    tgt_f = torch.where(obs, tgt, torch.zeros_like(tgt))
                    raw = F.smooth_l1_loss(pred_f, tgt_f, reduction="none")
                    terms[name], n_obs[name] = _masked_mean(raw, mask)
                    continue
                if isinstance(harmony_targets, SongHarmonyTargets):
                    target = harmony_targets.chroma
                    if target.shape != pred.shape or harmony_targets.chroma_mask.shape != pred.shape[:1]:
                        raise ContractError("song harmony targets must have shapes (B,12) and (B,)")
                    valid = harmony_targets.chroma_mask.bool()
                    valid = valid & bundle.branches[name].temporal_prediction_mask.any(dim=1)
                    n_obs[name] = int(valid.sum().item())
                    if valid.any():
                        selected = target[valid]
                        require_finite("song harmony targets", selected)
                        if bool((selected < 0).any()) or not torch.allclose(
                            selected.sum(-1), torch.ones_like(selected[:, 0]), atol=1e-5
                        ):
                            raise ContractError("song chroma targets must be non-negative and sum to one")
                        terms[name] = -(selected * pred[valid].clamp_min(1e-8).log()).sum(-1).mean()
                    else:
                        terms[name] = pred.sum() * 0.0
                    continue
                if not isinstance(harmony_targets, HarmonyTargets):
                    raise ContractError(
                        "harmony targets must preserve temporal chroma using HarmonyTargets"
                    )
                harmony_loss, harmony_counts, harmony_terms = _temporal_harmony_loss(
                    bundle.branches[name], harmony_targets
                )
                terms[name] = harmony_loss
                terms.update(harmony_terms)
                n_obs.update(harmony_counts)
                continue
            tgt = require_tensor(f"concept_targets[{name}]", concept_targets[name], ndim=2)
            if tgt.shape != pred.shape:
                raise ContractError(f"{name} target {tuple(tgt.shape)} != pred {tuple(pred.shape)}")
            obs = mask > 0.5
            require_finite(f"concept_targets[{name}][observed]", tgt, where=mask)
            dummy = torch.zeros_like(pred)
            pred_f = torch.where(obs, pred, dummy)
            tgt_f = torch.where(obs, tgt, dummy)
            if name == "instrument":
                inst_logits = bundle.branches[name].logits
                if inst_logits is not None:
                    lg = require_tensor("instrument.logits", inst_logits, ndim=2)
                    if lg.shape != pred.shape:
                        raise ContractError("instrument logits must match concept_values")
                    lg_f = torch.where(obs, lg, torch.zeros_like(lg))
                    raw = F.binary_cross_entropy_with_logits(lg_f, tgt_f, reduction="none")
                else:
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


def _temporal_harmony_loss(
    branch,
    targets: HarmonyTargets,
) -> tuple[torch.Tensor, dict[str, int], dict[str, torch.Tensor]]:
    logits = branch.temporal_chroma_logits
    prediction_mask = branch.temporal_prediction_mask
    if logits is None or prediction_mask is None:
        raise ContractError("harmony branch is missing temporal chroma predictions")
    chroma = require_tensor("harmony_targets.chroma", targets.chroma, ndim=3)
    chroma_mask = targets.chroma_mask
    if not isinstance(chroma_mask, torch.Tensor) or chroma_mask.ndim != 2:
        raise ContractError("harmony chroma mask must be a 2D tensor")
    if logits.shape != chroma.shape or logits.shape[-1] != N_HARMONY_CHROMA:
        raise ContractError("harmony chroma targets must match temporal logits (B,T,12)")
    if chroma_mask.shape != logits.shape[:2]:
        raise ContractError("harmony chroma mask must match temporal logits")
    valid = chroma_mask.to(torch.bool) & prediction_mask.to(torch.bool)
    n_chroma = int(valid.sum().item())
    if n_chroma:
        selected = chroma[valid]
        require_finite("harmony chroma targets[observed]", selected)
        if bool((selected < 0).any()):
            raise ContractError("observed harmony chroma targets must be non-negative")
        if not torch.allclose(
            selected.sum(dim=-1),
            torch.ones(n_chroma, device=selected.device, dtype=selected.dtype),
            atol=1e-5,
        ):
            raise ContractError("observed harmony chroma targets must sum to one")
        chroma_loss = -(selected * F.log_softmax(logits[valid], dim=-1)).sum(-1).mean()
    else:
        chroma_loss = logits.sum() * 0.0

    chord_logits = branch.temporal_chord_logits
    if targets.chord_labels is None and targets.chord_mask is None:
        chord_loss = logits.sum() * 0.0
        n_chord = 0
    else:
        if targets.chord_labels is None or targets.chord_mask is None:
            raise ContractError("harmony chord labels and mask must be supplied together")
        if chord_logits is None:
            raise ContractError("chord targets were supplied but the harmony chord head is disabled")
        labels = targets.chord_labels
        chord_mask = targets.chord_mask
        if not isinstance(labels, torch.Tensor) or labels.ndim != 2:
            raise ContractError("harmony chord labels must be a 2D tensor")
        if not isinstance(chord_mask, torch.Tensor) or chord_mask.ndim != 2:
            raise ContractError("harmony chord mask must be a 2D tensor")
        if labels.shape != chord_logits.shape[:2] or chord_mask.shape != labels.shape:
            raise ContractError("harmony chord labels/mask must match temporal chord logits")
        chord_valid = chord_mask.to(torch.bool) & prediction_mask.to(torch.bool)
        n_chord = int(chord_valid.sum().item())
        if n_chord:
            selected_labels = labels[chord_valid]
            if selected_labels.dtype not in {
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
            }:
                raise ContractError("observed harmony chord labels must use an integer dtype")
            if bool((selected_labels < 0).any()) or bool(
                (selected_labels >= N_HARMONY_CHORDS).any()
            ):
                raise ContractError("observed harmony chord labels are outside the 25-class schema")
            chord_loss = F.cross_entropy(chord_logits[chord_valid], selected_labels.long())
        else:
            chord_loss = chord_logits.sum() * 0.0

    total = chroma_loss + chord_loss
    return (
        total,
        {
            "harmony": n_chroma + n_chord,
            "harmony_chroma_frames": n_chroma,
            "harmony_chord_frames": n_chord,
        },
        {"harmony_chroma": chroma_loss, "harmony_chord": chord_loss},
    )
