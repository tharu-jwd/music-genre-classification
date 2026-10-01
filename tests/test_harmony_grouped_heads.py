"""Contract tests for the Harmony V3 grouped-head, balanced-loss screen."""

from __future__ import annotations

import torch

from concept_fusion.joint_loss import _group_balanced_masked_mean, _mixed_harmony_masked_mean
from harmony_branch.constants import DESCRIPTOR_GROUPS, HARMONY_DESCRIPTORS
from harmony_branch.descriptors import HarmonyTargetTransform
from harmony_branch.model import ChromaGroundedHarmonyBranch


def test_grouped_heads_preserve_the_public_45_descriptor_contract():
    model = ChromaGroundedHarmonyBranch(
        128, hidden_dim=16, embedding_dim=8, temporal_layers=1,
        grouped_descriptor_heads=True,
    )
    model.set_target_transform(HarmonyTargetTransform().fit(torch.rand(8, 45).numpy()))
    sequence = torch.randn(2, 8, 128)
    mask = torch.ones(2, 8, dtype=torch.bool)
    window_index = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]]).expand(2, -1)

    output = model(sequence, mask, window_index, windows=2, tokens_per_window=4)

    assert output.descriptor_values.shape == (2, 45)
    assert tuple(model.descriptor_heads) == (
        "chroma_std", "tonnetz_std", "tonal_dynamics",
    )
    output.descriptor_values.square().mean().backward()
    assert all(
        parameter.grad is not None
        for head in model.descriptor_heads.values()
        for parameter in head.parameters()
    )


def test_feature_balanced_loss_weights_each_semantic_group_equally():
    # Smooth L1 of residuals 1..5 is respectively 0.5, 1.5, 2.5, 3.5, 4.5.
    residuals = torch.zeros(1, 45, requires_grad=True)
    groups = tuple(tuple(names) for names in DESCRIPTOR_GROUPS.values())
    index = {name: i for i, name in enumerate(HARMONY_DESCRIPTORS)}
    with torch.no_grad():
        for value, names in enumerate(groups, start=1):
            residuals[:, [index[name] for name in names]] = float(value)
    raw = torch.nn.functional.smooth_l1_loss(residuals, torch.zeros_like(residuals), reduction="none")
    indices = tuple(tuple(index[name] for name in names) for names in groups)

    loss, observed = _group_balanced_masked_mean(raw, torch.ones_like(raw), indices)

    torch.testing.assert_close(loss, torch.tensor(2.5))
    assert observed == 45
    loss.backward()
    assert torch.isfinite(residuals.grad).all()


def test_mixed_loss_uses_the_requested_75_25_blend():
    # Group-balanced loss is 2.5. Ordinary per-feature loss is lower because
    # the two 12-feature chroma groups have greater representation.
    raw = torch.tensor([[0.5] * 12 + [1.5] * 12 + [2.5] * 6 + [3.5] * 6 + [4.5] * 9])
    groups = tuple(tuple(names) for names in DESCRIPTOR_GROUPS.values())
    index = {name: i for i, name in enumerate(HARMONY_DESCRIPTORS)}
    indices = tuple(tuple(index[name] for name in names) for names in groups)
    mask = torch.ones_like(raw)
    ordinary = raw.mean()
    balanced, _ = _group_balanced_masked_mean(raw, mask, indices)
    mixed, observed = _mixed_harmony_masked_mean(raw, mask, indices, 0.25)

    torch.testing.assert_close(mixed, 0.75 * ordinary + 0.25 * balanced)
    assert observed == 45
