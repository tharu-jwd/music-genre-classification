"""End-to-end contract tests for temporal harmony and concept fusion."""

from __future__ import annotations

import torch

from concept_fusion.contract import (
    CONCEPT_ORDER,
    EMBEDDING_FUSION_INPUT_MODE,
    N_HARMONY_CHORDS,
)
from concept_fusion.fixtures import make_bundle, make_genre_batch
from concept_fusion.harmony_adapter import from_temporal_harmony_branch
from concept_fusion.joint_loss import HarmonyTargets, JointLossOrchestrator
from concept_fusion.model import ConceptBottleneckModel
from concept_fusion.types import BranchBundle
from harmony_branch.model import TemporalHarmonyBranch


def _inputs(batch: int = 4, *, available: bool = True):
    sequence = torch.randn(batch, 8, 6)
    mask = torch.ones(batch, 8, dtype=torch.bool) if available else torch.zeros(
        batch, 8, dtype=torch.bool
    )
    window_index = torch.tensor([[0] * 4 + [1] * 4] * batch)
    window_index = window_index.masked_fill(~mask, -1)
    return sequence, mask, window_index


def _real_harmony_bundle(
    batch: int = 4,
    *,
    target_mask: torch.Tensor | None = None,
    embedding_dim: int = 32,
):
    branch_model = TemporalHarmonyBranch(
        6,
        embedding_dim=embedding_dim,
        hidden_dim=12,
        chord_classes=N_HARMONY_CHORDS,
        dropout=0,
    )
    sequence, mask, window_index = _inputs(batch)
    raw = branch_model(
        sequence,
        mask,
        window_index,
        windows=2,
        tokens_per_window=4,
    )
    if target_mask is None:
        target_mask = mask.clone()
    harmony = from_temporal_harmony_branch(raw, chroma_target_mask=target_mask)
    fixture = make_bundle(batch, seed=11, fusion_keep=1.0)
    branches = dict(fixture.branches)
    branches["harmony"] = harmony
    bundle = BranchBundle(branches=branches, counts=fixture.counts)
    bundle.validate()
    return branch_model, bundle, raw, mask


def _targets(mask: torch.Tensor) -> HarmonyTargets:
    chroma = torch.rand(*mask.shape, 12)
    chroma = chroma / chroma.sum(dim=-1, keepdim=True)
    chords = torch.randint(N_HARMONY_CHORDS, mask.shape)
    return HarmonyTargets(chroma, mask, chords, mask)


def test_adapter_preserves_predictions_and_embedding_ablation_is_selectable():
    _, bundle, raw, _ = _real_harmony_bundle(embedding_dim=13)
    harmony = bundle.branches["harmony"]
    assert harmony.fusion_token is None
    assert harmony.embedding is raw.embedding
    assert harmony.temporal_chroma_logits is raw.chroma_logits
    assert harmony.temporal_chord_logits is raw.chord_logits
    model = ConceptBottleneckModel(
        "gated",
        harmony_embedding_dim=13,
        fusion_input_mode=EMBEDDING_FUSION_INPUT_MODE,
    )
    tokens = model.assemble_tokens(bundle)
    assert tokens.shape == (4, len(CONCEPT_ORDER), 64)


def test_missing_supervision_does_not_remove_available_harmony_from_fusion():
    no_targets = torch.zeros(4, 8, dtype=torch.bool)
    _, bundle, _, _ = _real_harmony_bundle(target_mask=no_targets)
    harmony = bundle.branches["harmony"]
    assert torch.count_nonzero(harmony.supervision_mask) == 0
    assert bool((harmony.fusion_mask == 1).all())
    assert torch.count_nonzero(ConceptBottleneckModel().assemble_tokens(bundle)[:, 3]) > 0
    no_temporal_targets = HarmonyTargets(
        chroma=torch.zeros(4, 8, 12),
        chroma_mask=no_targets,
    )
    fixed_targets = {
        name: torch.zeros_like(bundle.concept_values(name))
        for name in ("instrument", "rhythm", "timbre")
    }
    breakdown = JointLossOrchestrator()(
        torch.zeros(4, 6, requires_grad=True),
        make_genre_batch(4),
        bundle,
        {**fixed_targets, "harmony": no_temporal_targets},
    )
    assert breakdown.n_observed["harmony"] == 0
    assert torch.isfinite(breakdown.total)


def test_unavailable_harmony_is_masked_and_projects_to_exact_zero():
    branch_model = TemporalHarmonyBranch(6, embedding_dim=32, hidden_dim=12, dropout=0)
    sequence, mask, window_index = _inputs(3, available=False)
    raw = branch_model(sequence, mask, window_index, windows=2, tokens_per_window=4)
    harmony = from_temporal_harmony_branch(raw)
    fixture = make_bundle(3, seed=4, fusion_keep=1.0)
    branches = dict(fixture.branches)
    branches["harmony"] = harmony
    bundle = BranchBundle(branches, fixture.counts)
    model = ConceptBottleneckModel()
    tokens = model.assemble_tokens(bundle)
    assert torch.count_nonzero(harmony.fusion_mask) == 0
    assert torch.count_nonzero(tokens[:, 3]) == 0


def test_temporal_losses_and_genre_loss_backpropagate_through_harmony():
    branch_model, bundle, _, mask = _real_harmony_bundle()
    model = ConceptBottleneckModel("gated")
    logits, _ = model.from_bundle(bundle, apply_dropout=False)
    targets = {
        "instrument": torch.zeros_like(bundle.concept_values("instrument")),
        "rhythm": torch.zeros_like(bundle.concept_values("rhythm")),
        "timbre": torch.zeros_like(bundle.concept_values("timbre")),
        "harmony": _targets(mask),
    }
    breakdown = JointLossOrchestrator()(
        logits, make_genre_batch(4), bundle, targets
    )
    breakdown.total.backward()
    assert breakdown.n_observed["harmony_chroma_frames"] == int(mask.sum())
    assert breakdown.n_observed["harmony_chord_frames"] == int(mask.sum())
    assert branch_model.embedding_projection.weight.grad is not None
    assert branch_model.chroma_head.weight.grad is not None
    assert model.assembler.harmony_chroma_projection.weight.grad is not None


def test_checkpoint_restore_and_harmony_removal(tmp_path):
    branch_model = TemporalHarmonyBranch(
        6, embedding_dim=32, hidden_dim=12, chord_classes=N_HARMONY_CHORDS, dropout=0
    )
    sequence, mask, window_index = _inputs(4)
    raw = branch_model(sequence, mask, window_index, windows=2, tokens_per_window=4)
    fixture = make_bundle(4, seed=11, fusion_keep=1.0)
    branches = dict(fixture.branches)
    branches["harmony"] = from_temporal_harmony_branch(raw, chroma_target_mask=mask)
    bundle = BranchBundle(branches, fixture.counts)
    model = ConceptBottleneckModel("gated")
    branch_model.eval()
    model.eval()
    with torch.no_grad():
        reference, _ = model.from_bundle(bundle, apply_dropout=False)
    checkpoint = tmp_path / "harmony_fusion.pt"
    torch.save({"branch": branch_model.state_dict(), "fusion": model.state_dict()}, checkpoint)

    restored_branch = TemporalHarmonyBranch(
        6, embedding_dim=32, hidden_dim=12, chord_classes=N_HARMONY_CHORDS, dropout=0
    )
    restored_model = ConceptBottleneckModel("gated")
    payload = torch.load(checkpoint, weights_only=True)
    restored_branch.load_state_dict(payload["branch"])
    restored_model.load_state_dict(payload["fusion"])
    restored_branch.eval()
    restored_model.eval()
    with torch.no_grad():
        restored_raw = restored_branch(
            sequence, mask, window_index, windows=2, tokens_per_window=4
        )
        torch.testing.assert_close(restored_raw.embedding, raw.embedding)
        restored_branches = dict(fixture.branches)
        restored_branches["harmony"] = from_temporal_harmony_branch(
            restored_raw, chroma_target_mask=mask
        )
        restored_bundle = BranchBundle(restored_branches, fixture.counts)
        actual, _ = restored_model.from_bundle(restored_bundle, apply_dropout=False)
    torch.testing.assert_close(actual, reference)

    without_harmony = bundle.with_enabled_concepts(("instrument", "rhythm", "timbre"))
    _, fusion = model.from_bundle(without_harmony, apply_dropout=False)
    assert torch.count_nonzero(without_harmony.fusion_mask()[:, 3]) == 0
    assert torch.count_nonzero(fusion.gates[:, 3]) == 0
