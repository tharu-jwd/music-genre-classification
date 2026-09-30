"""Contract tests for the strictly isolated I1 integration-adapter experiment."""

import numpy as np
import pytest
import torch

from integration_adapters import I1BranchAdapters, ZeroInitializedResidualAdapter
from scripts import train_joint as joint
from scripts.train_joint import TimbreV1Standardizer


def test_all_i1_adapters_are_exact_identity_at_initialization():
    torch.manual_seed(5)
    adapters = I1BranchAdapters().eval()
    pooled = torch.randn(3, 128)
    sequence = torch.randn(3, 7, 128)
    mask = torch.tensor([
        [True, True, True, True, True, True, True],
        [True, True, True, False, False, False, False],
        [True, False, False, False, False, False, False],
    ])

    torch.testing.assert_close(adapters.pooled("instrument", pooled), pooled)
    torch.testing.assert_close(adapters.pooled("timbre", pooled), pooled)
    expected_temporal = sequence * mask.unsqueeze(-1)
    torch.testing.assert_close(adapters.temporal("rhythm", sequence, mask), expected_temporal)
    torch.testing.assert_close(adapters.temporal("harmony", sequence, mask), expected_temporal)


def test_temporal_adapter_preserves_shape_order_and_padding_mask_after_learning():
    adapter = ZeroInitializedResidualAdapter().eval()
    adapter.residual_scale.data.fill_(0.4)
    sequence = torch.randn(2, 5, 128)
    mask = torch.tensor([[True, True, True, False, False], [True, True, True, True, True]])
    output = adapter(sequence, mask)

    assert output.shape == sequence.shape
    assert torch.count_nonzero(output[0, 3:]) == 0
    assert torch.isfinite(output).all()


def test_i1_uses_four_independent_branch_private_parameter_sets():
    adapters = I1BranchAdapters()
    branch_ids = {
        name: {id(parameter) for parameter in getattr(adapters, name).parameters()}
        for name in ("instrument", "timbre", "rhythm", "harmony")
    }
    for name, ids in branch_ids.items():
        for other_name, other_ids in branch_ids.items():
            if name != other_name:
                assert ids.isdisjoint(other_ids)


def test_zero_scale_allows_scale_gradient_before_internal_adapter_gradients():
    adapter = ZeroInitializedResidualAdapter()
    inputs = torch.randn(4, 128, requires_grad=True)
    adapter(inputs).square().mean().backward()

    assert adapter.residual_scale.grad is not None
    assert torch.isfinite(adapter.residual_scale.grad)
    assert inputs.grad is not None and inputs.grad.abs().sum() > 0


def test_timbre_v1_standardizer_is_raw_zscore_without_flatness_log_transform():
    values = np.arange(105, dtype=np.float64).reshape(3, 35) + 1.0
    standardizer = TimbreV1Standardizer().fit(values)
    transformed = standardizer.transform(values)
    restored = standardizer.inverse_transform(transformed)

    np.testing.assert_allclose(restored, values, rtol=1e-6, atol=1e-6)
    state = standardizer.state_dict()
    assert state["preprocessing_version"] == "timbre_v1_raw_zscore_v1"
    assert state["target_transforms"] == {"all_features": "identity"}


def test_i1_and_its_control_are_declared_as_distinct_modes():
    i1 = joint.TrainConfig(experiment_i1=True)
    control = joint.TrainConfig(experiment_i1_control=True)
    assert i1.experiment_i1 and not i1.experiment_i1_control
    assert control.experiment_i1_control and not control.experiment_i1
    assert i1.epochs == 30
    assert control.epochs == 30


def test_i1_rejects_a_non_30_epoch_budget_before_loading_data():
    with pytest.raises(ValueError, match="exactly 30 epochs"):
        joint.train(joint.TrainConfig(experiment_i1=True, epochs=5))


def test_i1_full_forward_backward_smoke_preserves_branch_contracts():
    torch.manual_seed(13)
    encoder = joint.SharedAudioEncoder()
    instrument = joint.InstrumentHead()
    timbre = joint.TimbreBranch()
    rhythm = joint.RhythmBranch()
    harmony = joint.TemporalHarmonyBranch(128, chord_classes=None, descriptor_dim=12)
    adapters = I1BranchAdapters()
    fusion = joint.ConceptBottleneckModel(fusion="gated", dropout_p=0.0)

    mel = torch.randn(2, 2, 1, 16, 12)
    window_mask = torch.tensor([[True, True], [True, False]])
    valid_frames = torch.tensor([[12, 8], [7, 0]])
    starts = torch.tensor([[0.0, 15.0], [0.0, 0.0]])
    encoded = encoder(mel, window_mask, valid_frames, starts)
    kwargs = {
        "instr_tgt": torch.zeros(2, 41),
        "timbre_tgt": torch.zeros(2, 35),
        "timbre_msk": torch.ones(2, 35, dtype=torch.bool),
        "rhythm_tgt": torch.zeros(2, 10),
        "rhythm_msk": torch.ones(2, 10, dtype=torch.bool),
        "harmony_tgt": torch.zeros(2, 12),
        "harmony_msk": torch.ones(2, 12, dtype=torch.bool),
        "device": torch.device("cpu"),
    }
    bundle, targets = joint._build_bundle(
        encoded, instrument, timbre, rhythm, harmony, adapters, **kwargs
    )
    logits, _ = fusion.from_bundle(bundle, apply_dropout=False)
    loss = joint.JointLossOrchestrator()(logits, torch.zeros_like(logits), bundle, targets)
    loss.total.backward()

    assert logits.shape == (2, 6)
    assert bundle.concept_values("instrument").shape == (2, 41)
    assert bundle.concept_values("rhythm").shape == (2, 10)
    assert bundle.concept_values("timbre").shape == (2, 35)
    assert bundle.concept_values("harmony").shape == (2, 12)
    assert all(
        getattr(adapters, name).residual_scale.grad is not None
        for name in ("instrument", "rhythm", "timbre", "harmony")
    )
    assert encoder.cnn[0].weight.grad is not None
