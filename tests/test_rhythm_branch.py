import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "rhythm_branch" / "src"))

from concept_fusion.fixtures import make_bundle
from concept_fusion.model import ConceptBottleneckModel
from concept_fusion.rhythm_adapter import from_rhythm_branch
from rhythm_branch.constants import RHYTHM_FEATURES
from rhythm_branch.losses import masked_huber_loss
from rhythm_branch.model import RhythmBranch, RhythmBranchConfig
from rhythm_branch.preprocessing import (
    RhythmStandardizer,
    align_manifest_and_targets,
    audit_target_coverage,
    fit_training_standardizer,
    load_rhythm_targets,
)
from rhythm_branch.training import load_checkpoint, save_checkpoint
from shared_encoder import SharedAudioEncoder


def _encoded(batch=2, windows=2, frames=12):
    encoder = SharedAudioEncoder(output_dim=128)
    mel = torch.randn(batch, windows, 1, 8, frames)
    mask = torch.ones(batch, windows, dtype=torch.bool)
    valid = torch.full((batch, windows), frames, dtype=torch.long)
    starts = torch.arange(windows).float().repeat(batch, 1) * 30
    return encoder, mel, encoder.encode_temporal(mel, mask, valid, starts)


@pytest.mark.parametrize("frames", [8, 15])
def test_output_shapes_and_variable_temporal_length(frames):
    _encoder, _mel, temporal = _encoded(frames=frames)
    branch = RhythmBranch(RhythmBranchConfig(dropout=0))
    output = branch(
        temporal.encoded_sequence,
        temporal.sequence_mask,
        temporal.sequence_window_index,
    )
    assert output.embedding.shape == (2, 64)
    assert output.predictions.shape == (2, 10)
    assert output.availability.shape == (2,)
    assert torch.isfinite(output.embedding).all()
    assert torch.isfinite(output.predictions).all()


def test_attention_mean_std_pooling_ignores_padding():
    torch.manual_seed(4)
    branch = RhythmBranch(RhythmBranchConfig(dropout=0)).eval()
    valid_tokens = torch.randn(2, 3, 128)
    short_mask = torch.ones(2, 3, dtype=torch.bool)
    short = branch(valid_tokens, short_mask)

    padded_tokens = torch.cat((valid_tokens, torch.randn(2, 4, 128) * 1000), dim=1)
    padded_mask = torch.cat((short_mask, torch.zeros(2, 4, dtype=torch.bool)), dim=1)
    padded = branch(padded_tokens, padded_mask)

    torch.testing.assert_close(short.embedding, padded.embedding)
    torch.testing.assert_close(short.predictions, padded.predictions)
    assert torch.equal(short.availability, padded.availability)


def test_attention_mean_std_pooling_concatenates_expected_summaries():
    torch.manual_seed(8)
    branch = RhythmBranch(RhythmBranchConfig(dropout=0)).eval()
    captured = {}

    def capture_temporal(_module, _inputs, output):
        captured["temporal"] = output.detach()

    def capture_summary(_module, inputs):
        captured["summary"] = inputs[0].detach()

    temporal_hook = branch.temporal_blocks[-1].register_forward_hook(capture_temporal)
    summary_hook = branch.summary_projection.register_forward_pre_hook(capture_summary)
    tokens = torch.randn(2, 5, 128)
    mask = torch.tensor(
        [[True, True, True, False, False], [True, True, True, True, True]]
    )
    branch(tokens, mask)
    temporal_hook.remove()
    summary_hook.remove()

    values = captured["temporal"]
    valid = mask.unsqueeze(-1).to(values.dtype)
    scores = branch.attention(values).squeeze(-1)
    scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    weights = torch.softmax(scores, dim=1) * mask.to(scores.dtype)
    attention_mean = (values * weights.unsqueeze(-1)).sum(dim=1)
    count = valid.sum(dim=1)
    masked_mean = (values * valid).sum(dim=1) / count
    variance = ((values - masked_mean.unsqueeze(1)) * valid).square().sum(dim=1) / count
    masked_std = torch.sqrt(variance)

    assert branch.summary_projection.weight.shape == (64, 192)
    torch.testing.assert_close(
        captured["summary"],
        torch.cat((attention_mean, masked_mean, masked_std), dim=-1),
    )


def test_attention_mean_std_pooling_handles_single_and_all_masked_sequences():
    torch.manual_seed(5)
    branch = RhythmBranch(RhythmBranchConfig(dropout=0))
    tokens = torch.randn(2, 4, 128, requires_grad=True)
    mask = torch.tensor([[True, False, False, False], [False, False, False, False]])
    output = branch(tokens, mask)

    assert torch.isfinite(output.embedding).all()
    assert torch.isfinite(output.predictions).all()
    assert output.availability.tolist() == [True, False]
    assert torch.count_nonzero(output.embedding[1]) == 0
    assert torch.count_nonzero(output.predictions[1]) == 0

    output.embedding.sum().backward()
    assert tokens.grad is not None
    assert torch.isfinite(tokens.grad).all()
    assert torch.count_nonzero(tokens.grad[1]) == 0


def test_config_loads_statistics_pooling_checkpoint_and_rejects_attention_only():
    config = RhythmBranchConfig.from_dict(
        {"dropout": 0, "pooling_mode": "attention_mean_std"}
    )
    assert config.dropout == 0
    assert config.pooling_mode == "attention_mean_std"
    with pytest.raises(ValueError, match="attention-only"):
        RhythmBranchConfig.from_dict({"pooling_mode": "attention"})
    with pytest.raises(ValueError, match="attention-only"):
        RhythmBranchConfig.from_dict({"dropout": 0})


def test_missing_targets_and_all_missing_batch_are_safe():
    predictions = torch.randn(3, 10, requires_grad=True)
    targets = torch.randn(3, 10)
    targets[0, 2] = float("nan")
    mask = torch.ones(3, 10, dtype=torch.bool)
    mask[0, 2] = False
    loss = masked_huber_loss(predictions, targets, mask)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(predictions.grad).all()

    predictions = torch.randn(3, 10, requires_grad=True)
    empty = masked_huber_loss(
        predictions,
        torch.full((3, 10), float("nan")),
        torch.zeros(3, 10, dtype=torch.bool),
    )
    assert empty.item() == 0
    empty.backward()
    assert torch.equal(predictions.grad, torch.zeros_like(predictions))


def test_rhythm_and_genre_gradients_reach_branch_and_shared_cnn():
    encoder, mel, temporal = _encoded(batch=3)
    branch = RhythmBranch(RhythmBranchConfig(dropout=0))
    output = branch(
        temporal.encoded_sequence,
        temporal.sequence_mask,
        temporal.sequence_window_index,
    )
    rhythm_loss = masked_huber_loss(
        output.predictions,
        torch.randn(3, 10),
        torch.ones(3, 10, dtype=torch.bool),
    )
    rhythm_loss.backward()
    assert encoder.cnn[0].weight.grad is not None
    assert encoder.cnn[0].weight.grad.abs().sum() > 0
    assert branch.temporal_blocks[0].network[0].weight.grad.abs().sum() > 0

    encoder.zero_grad(set_to_none=True)
    branch.zero_grad(set_to_none=True)
    temporal = encoder.encode_temporal(
        mel,
        torch.ones(3, 2, dtype=torch.bool),
        torch.full((3, 2), 12),
        torch.tensor([[0.0, 30.0]]).expand(3, -1),
    )
    output = branch(
        temporal.encoded_sequence,
        temporal.sequence_mask,
        temporal.sequence_window_index,
    )
    rhythm = from_rhythm_branch(output, supervision_mask=torch.ones(3, 10))
    bundle = make_bundle(batch=3)
    bundle.branches["rhythm"] = rhythm
    model = ConceptBottleneckModel(dropout_p=0, allow_no_dropout=True).eval()
    genre_logits, _ = model.from_bundle(bundle, apply_dropout=False)
    torch.nn.functional.binary_cross_entropy_with_logits(
        genre_logits, torch.rand_like(genre_logits)
    ).backward()
    assert branch.regression_head.weight.grad.abs().sum() > 0
    assert branch.embedding_head[0].weight.grad.abs().sum() > 0
    assert encoder.cnn[0].weight.grad.abs().sum() > 0


def test_target_join_scaling_and_interval_audit(tmp_path):
    rows = []
    for index, split in enumerate(("train", "train", "validation"), start=1):
        row = {"song_id": str(index), "split": split}
        row.update({name: float(index + column) for column, name in enumerate(RHYTHM_FEATURES)})
        rows.append(row)
    target_path = tmp_path / "rhythm.csv"
    pd.DataFrame(rows).to_csv(target_path, index=False)
    targets = load_rhythm_targets(target_path)
    manifest = pd.DataFrame(
        {"song_id": ["0000001", "2", "0000003", "0000004"],
         "split": ["train", "train", "validation", "test"]}
    )
    joined, values, mask = align_manifest_and_targets(manifest, targets)
    assert joined["_merge"].tolist()[-1] == "left_only"
    assert not mask[-1].any()
    audit = audit_target_coverage(
        joined, values, mask, input_scope="sampled_windows"
    )
    assert audit["excluded_fields"] == ["beats_count"]
    assert audit["field_interval_audit"]["bpm"]["exact_interval_match"] is False
    assert (
        audit["field_interval_audit"]["beats_count"]["status"]
        == "excluded_length_dependent_count"
    )
    assert not audit["validity_mask"][:, RHYTHM_FEATURES.index("beats_count")].any()

    scaler = fit_training_standardizer(
        joined,
        values,
        audit["validity_mask"],
        excluded_fields=audit["excluded_fields"],
    )
    standardized = scaler.transform(values[:2])
    observed_columns = [index for index, name in enumerate(RHYTHM_FEATURES) if name != "beats_count"]
    np.testing.assert_allclose(
        np.nanmean(standardized[:, observed_columns], axis=0), 0.0, atol=1e-6
    )
    assert scaler.count[RHYTHM_FEATURES.index("beats_count")] == 0
    assert tuple(scaler.state_dict()["feature_names"]) == RHYTHM_FEATURES


def test_duplicate_target_rows_are_rejected(tmp_path):
    row = {"song_id": "1", "split": "train", **{name: 1 for name in RHYTHM_FEATURES}}
    path = tmp_path / "duplicate.csv"
    pd.DataFrame([row, row]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicate rhythm target"):
        load_rhythm_targets(path)


def test_checkpoint_preserves_schema_and_train_statistics(tmp_path):
    values = np.arange(40, dtype=np.float64).reshape(4, 10)
    scaler = RhythmStandardizer().fit(values, np.ones_like(values, dtype=bool))
    model = RhythmBranch(RhythmBranchConfig(dropout=0))
    path = tmp_path / "rhythm.pt"
    save_checkpoint(
        path,
        model,
        scaler,
        epoch=2,
        metrics={"smoke_only": True},
        seed=7,
        input_scope="full_recording",
    )
    restored, restored_scaler, payload = load_checkpoint(path)
    assert tuple(payload["feature_names"]) == RHYTHM_FEATURES
    assert payload["model_config"]["pooling_mode"] == "attention_mean_std"
    np.testing.assert_allclose(restored_scaler.mean, scaler.mean)
    for expected, actual in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(expected, actual)
