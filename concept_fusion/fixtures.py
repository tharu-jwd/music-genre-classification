"""Synthetic branch outputs with exact contract shapes. No silent filling."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from concept_fusion.contract import (
    CONCEPT_ORDER,
    DEFAULT_HARMONY_EMBEDDING_DIM,
    FUSED_DIM,
    INSTRUMENT_HIDDEN_DIM,
    INSTRUMENT_TAGS,
    N_GENRE_TAGS,
    N_HARMONY_CHORDS,
    TIMBRE_FEATURES,
    TOKEN_DIM,
    ConceptCounts,
)
from concept_fusion.types import BranchBundle, BranchOutput
from concept_fusion.joint_loss import HarmonyTargets


def make_branch_output(
    name: str,
    *,
    batch: int,
    n_concepts: int,
    seed: int = 0,
    fusion_keep: float = 0.85,
    supervise_keep: float = 0.7,
) -> BranchOutput:
    g = torch.Generator().manual_seed(seed + 17 * CONCEPT_ORDER.index(name))
    values = torch.rand(batch, n_concepts, generator=g)
    sup = (torch.rand(batch, n_concepts, generator=g) < supervise_keep).float()
    fus = (torch.rand(batch, 1, generator=g) < fusion_keep).float()
    if name == "instrument":
        values = values.clamp(0.02, 0.98)
        logits = torch.logit(values)
        hidden = torch.randn(batch, INSTRUMENT_HIDDEN_DIM, generator=g)
        hidden = hidden / (hidden.norm(dim=-1, keepdim=True) + 1e-6)
        return BranchOutput(
            name=name,
            concept_values=values,
            fusion_token=None,
            supervision_mask=sup,
            fusion_mask=fus,
            hidden_token=hidden.detach(),
            logits=logits,
            tag_order=INSTRUMENT_TAGS,
        )
    if name == "timbre":
        values = torch.randn(batch, n_concepts, generator=g)
        hidden = torch.randn(batch, TOKEN_DIM, generator=g)
        hidden = hidden / (hidden.norm(dim=-1, keepdim=True) + 1e-6)
        return BranchOutput(
            name=name,
            concept_values=values,
            fusion_token=None,
            supervision_mask=sup,
            fusion_mask=fus,
            hidden_token=hidden,
            tag_order=TIMBRE_FEATURES,
        )
    if name == "harmony":
        time_steps = 6
        temporal_logits = torch.randn(batch, time_steps, n_concepts, generator=g)
        prediction_mask = torch.rand(batch, time_steps, generator=g) < 0.9
        chord_logits = torch.randn(batch, time_steps, N_HARMONY_CHORDS, generator=g)
        probabilities = torch.softmax(temporal_logits, dim=-1)
        weights = prediction_mask.to(probabilities.dtype)
        values = (probabilities * weights.unsqueeze(-1)).sum(1)
        values = values / weights.sum(1, keepdim=True).clamp_min(1.0)
        embedding = torch.randn(batch, DEFAULT_HARMONY_EMBEDDING_DIM, generator=g)
        hidden = torch.randn(batch, TOKEN_DIM, generator=g)
        return BranchOutput(
            name=name,
            concept_values=values,
            fusion_token=None,
            embedding=embedding,
            supervision_mask=sup,
            fusion_mask=fus,
            hidden_token=hidden,
            temporal_chroma_logits=temporal_logits,
            temporal_chord_logits=chord_logits,
            temporal_prediction_mask=prediction_mask,
        )
    token = torch.randn(batch, TOKEN_DIM, generator=g)
    token = token / (token.norm(dim=-1, keepdim=True) + 1e-6)
    # Predictions stay finite when targets are missing. Target tensors carry NaNs.
    hidden = torch.randn(batch, TOKEN_DIM, generator=g)
    hidden = hidden / (hidden.norm(dim=-1, keepdim=True) + 1e-6)
    return BranchOutput(
        name=name,
        concept_values=values,
        fusion_token=token,
        supervision_mask=sup,
        fusion_mask=fus,
        hidden_token=hidden,
    )


def make_bundle(
    batch: int = 8,
    *,
    counts: ConceptCounts | None = None,
    seed: int = 0,
    fusion_keep: float = 0.85,
    supervise_keep: float = 0.7,
) -> BranchBundle:
    counts = counts or ConceptCounts()
    branches = {}
    for i, name in enumerate(CONCEPT_ORDER):
        branches[name] = make_branch_output(
            name,
            batch=batch,
            n_concepts=counts.for_name(name),
            seed=seed + 17 * i,
            fusion_keep=fusion_keep,
            supervise_keep=supervise_keep,
        )
    bundle = BranchBundle(branches=branches, counts=counts)
    bundle.validate()
    return bundle


def make_genre_batch(batch: int = 8, *, n_tags: int = N_GENRE_TAGS, seed: int = 1, p: float = 0.08):
    g = torch.Generator().manual_seed(seed)
    y = (torch.rand(batch, n_tags, generator=g) < p).float()
    # Guarantee at least one positive on a few tags for metrics.
    y[:, 0] = 1.0
    y[0, 1] = 1.0
    y[1, 1] = 0.0
    return y


def make_concept_targets(
    bundle: BranchBundle, *, seed: int = 3
) -> dict[str, torch.Tensor | HarmonyTargets]:
    """Finite targets aligned to each branch. Masked cells may stay NaN."""
    g = torch.Generator().manual_seed(seed)
    out: dict[str, torch.Tensor | HarmonyTargets] = {}
    for name in CONCEPT_ORDER:
        pred = bundle.concept_values(name)
        mask = bundle.supervision_mask(name)
        if name == "harmony":
            branch = bundle.branches[name]
            assert branch.temporal_chroma_logits is not None
            assert branch.temporal_prediction_mask is not None
            shape = branch.temporal_chroma_logits.shape
            raw = torch.rand(shape, generator=g)
            chroma = raw / raw.sum(dim=-1, keepdim=True)
            chroma_mask = (
                (torch.rand(shape[:2], generator=g) < 0.7)
                & branch.temporal_prediction_mask.to(torch.bool)
            )
            chord_labels = torch.randint(N_HARMONY_CHORDS, shape[:2], generator=g)
            chord_mask = (
                (torch.rand(shape[:2], generator=g) < 0.6)
                & branch.temporal_prediction_mask.to(torch.bool)
            )
            out[name] = HarmonyTargets(chroma, chroma_mask, chord_labels, chord_mask)
            continue
        if name == "instrument":
            tgt = (torch.rand(pred.shape, generator=g) < 0.15).float()
        else:
            tgt = torch.randn(pred.shape, generator=g)
        tgt = tgt.masked_fill(mask < 0.5, float("nan"))
        out[name] = tgt
    return out


def make_all_masked_tokens(batch: int = 4, *, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """(B, 4, 64) tokens with fusion_mask all zeros — exercises the null-token fallback."""
    g = torch.Generator().manual_seed(seed)
    tokens = torch.randn(batch, len(CONCEPT_ORDER), TOKEN_DIM, generator=g)
    mask = torch.zeros(batch, len(CONCEPT_ORDER))
    return tokens, mask


def make_song_repr(batch: int, *, seed: int = 4) -> torch.Tensor:
    """Fixture stand-in for Dehan song_repr (B, 128). Used by B1 and F-Shortcut."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, FUSED_DIM, generator=g)
    return x / (x.norm(dim=-1, keepdim=True) + 1e-6)


def _index_bundle(bundle: BranchBundle, idx: torch.Tensor) -> BranchBundle:
    branches = {}
    for name in CONCEPT_ORDER:
        br = bundle.branches[name]
        hid = br.hidden_token[idx] if br.hidden_token is not None else None
        tok = br.fusion_token[idx] if br.fusion_token is not None else None
        lg = br.logits[idx] if br.logits is not None else None
        branches[name] = BranchOutput(
            name=br.name,
            concept_values=br.concept_values[idx],
            fusion_token=tok,
            embedding=None if br.embedding is None else br.embedding[idx],
            supervision_mask=br.supervision_mask[idx],
            fusion_mask=br.fusion_mask[idx],
            hidden_token=hid,
            logits=lg,
            tag_order=br.tag_order,
            temporal_chroma_logits=(
                None if br.temporal_chroma_logits is None else br.temporal_chroma_logits[idx]
            ),
            temporal_chord_logits=(
                None if br.temporal_chord_logits is None else br.temporal_chord_logits[idx]
            ),
            temporal_prediction_mask=(
                None
                if br.temporal_prediction_mask is None
                else br.temporal_prediction_mask[idx]
            ),
        )
    out = BranchBundle(branches=branches, counts=bundle.counts)
    out.validate()
    return out


def _index_targets(
    targets: dict[str, torch.Tensor | HarmonyTargets], idx: torch.Tensor
) -> dict[str, torch.Tensor | HarmonyTargets]:
    return {
        key: value.indexed(idx) if isinstance(value, HarmonyTargets) else value[idx]
        for key, value in targets.items()
    }


@dataclass
class FixtureSplit:
    song_ids: list[str]
    bundle: BranchBundle
    genre: torch.Tensor
    concept_targets: dict[str, torch.Tensor | HarmonyTargets]
    song_repr: torch.Tensor


@dataclass
class FixtureCohort:
    """Pairwise-disjoint train/val/test fixture tracks. Same IDs for every experiment."""

    train: FixtureSplit
    val: FixtureSplit
    test: FixtureSplit

    def all_ids(self) -> list[str]:
        return [*self.train.song_ids, *self.val.song_ids, *self.test.song_ids]


def make_cohort(
    *,
    n_train: int = 48,
    n_val: int = 24,
    n_test: int = 24,
    seed: int = 0,
    fusion_keep: float = 1.0,
    supervise_keep: float = 0.7,
) -> FixtureCohort:
    """One frozen fixture cohort. Every comparison model sees the same test IDs."""
    n = n_train + n_val + n_test
    bundle = make_bundle(n, seed=seed, fusion_keep=fusion_keep, supervise_keep=supervise_keep)
    y = make_genre_batch(n, seed=seed + 1)
    targets = make_concept_targets(bundle, seed=seed + 2)
    song_repr = make_song_repr(n, seed=seed + 3)
    song_ids = [f"{i:07d}" for i in range(n)]
    cuts = (0, n_train, n_train + n_val, n)

    def split(a: int, b: int) -> FixtureSplit:
        idx = torch.arange(a, b)
        return FixtureSplit(
            song_ids=song_ids[a:b],
            bundle=_index_bundle(bundle, idx),
            genre=y[idx],
            concept_targets=_index_targets(targets, idx),
            song_repr=song_repr[idx],
        )

    cohort = FixtureCohort(train=split(cuts[0], cuts[1]), val=split(cuts[1], cuts[2]), test=split(cuts[2], cuts[3]))
    ids = cohort.all_ids()
    if len(ids) != len(set(ids)):
        raise RuntimeError("fixture cohort IDs are not unique")
    if set(cohort.train.song_ids) & set(cohort.val.song_ids):
        raise RuntimeError("train/val IDs overlap")
    if set(cohort.train.song_ids) & set(cohort.test.song_ids):
        raise RuntimeError("train/test IDs overlap")
    if set(cohort.val.song_ids) & set(cohort.test.song_ids):
        raise RuntimeError("val/test IDs overlap")
    return cohort
