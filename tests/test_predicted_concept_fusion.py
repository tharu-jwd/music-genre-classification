from types import SimpleNamespace
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "harmony_branch" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "rhythm_branch" / "src"))

from concept_fusion.contract import (
    EMBEDDING_FUSION_INPUT_MODE,
    PRIMARY_FUSION_INPUT_MODE,
)
from concept_fusion.fixtures import make_bundle
from concept_fusion.harmony_adapter import from_temporal_harmony_branch
from concept_fusion.model import ConceptBottleneckModel
from concept_fusion.rhythm_adapter import from_rhythm_branch
from concept_fusion.types import BranchBundle
from concept_fusion.validation import ContractError
from harmony_branch.model import TemporalHarmonyBranch
from rhythm_branch.model import RhythmBranch, RhythmBranchConfig
from shared_encoder import SharedAudioEncoder


def test_both_fusion_input_modes_have_fixed_shape_and_order():
    bundle = make_bundle(3, seed=8, fusion_keep=1.0)
    primary = ConceptBottleneckModel(fusion_input_mode=PRIMARY_FUSION_INPUT_MODE)
    embedding = ConceptBottleneckModel(fusion_input_mode=EMBEDDING_FUSION_INPUT_MODE)

    primary_tokens = primary.assemble_tokens(bundle)
    embedding_tokens = embedding.assemble_tokens(bundle)
    assert primary_tokens.shape == embedding_tokens.shape == (3, 4, 64)
    torch.testing.assert_close(
        primary_tokens[:, 1],
        primary.assembler.rhythm_projection(bundle.concept_values("rhythm")),
    )
    torch.testing.assert_close(
        primary_tokens[:, 3],
        primary.assembler.harmony_chroma_projection(bundle.concept_values("harmony")),
    )
    torch.testing.assert_close(embedding_tokens[:, 1], bundle.branches["rhythm"].fusion_token)
    torch.testing.assert_close(
        embedding_tokens[:, 3],
        embedding.assembler.harmony_projection(bundle.branches["harmony"].embedding),
    )


def test_invalid_fusion_input_mode_is_rejected():
    with pytest.raises(ContractError, match="fusion input mode"):
        ConceptBottleneckModel(fusion_input_mode="silent-migration")  # type: ignore[arg-type]


def test_harmony_pools_probabilities_over_valid_tokens_only():
    logits = torch.tensor(
        [[
            [4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [100.0, -100.0, -100.0, -100.0, -100.0, -100.0,
             -100.0, -100.0, -100.0, -100.0, -100.0, -100.0],
        ]]
    )
    mask = torch.tensor([[True, True, False]])
    raw = SimpleNamespace(
        embedding=torch.randn(1, 32),
        chroma_logits=logits,
        chord_logits=None,
        availability=torch.tensor([True]),
        prediction_mask=mask,
    )
    branch = from_temporal_harmony_branch(raw, chroma_target_mask=torch.zeros_like(mask))
    expected = torch.softmax(logits[:, :2], dim=-1).mean(dim=1)
    torch.testing.assert_close(branch.concept_values, expected)
    assert torch.count_nonzero(branch.supervision_mask) == 0
    assert branch.fusion_mask.item() == 1


def test_harmony_descriptor_predictions_replace_chroma_for_primary_fusion():
    mask = torch.tensor([[True, True]])
    descriptors = torch.randn(1, 12)
    supervision = torch.ones(1, 12)
    raw = SimpleNamespace(
        embedding=torch.randn(1, 32),
        chroma_logits=torch.randn(1, 2, 12),
        descriptor_values=descriptors,
        chord_logits=None,
        availability=torch.tensor([True]),
        prediction_mask=mask,
    )
    branch = from_temporal_harmony_branch(
        raw, descriptor_supervision_mask=supervision
    )
    torch.testing.assert_close(branch.concept_values, descriptors)
    torch.testing.assert_close(branch.supervision_mask, supervision)


def test_all_masked_harmony_is_zero_and_unavailable():
    logits = torch.randn(2, 4, 12)
    mask = torch.zeros(2, 4, dtype=torch.bool)
    raw = SimpleNamespace(
        embedding=torch.zeros(2, 32),
        chroma_logits=logits,
        chord_logits=None,
        availability=torch.zeros(2, dtype=torch.bool),
        prediction_mask=mask,
    )
    branch = from_temporal_harmony_branch(raw)
    assert torch.count_nonzero(branch.concept_values) == 0
    assert torch.count_nonzero(branch.fusion_mask) == 0


def test_missing_rhythm_targets_do_not_disable_available_fusion():
    output = SimpleNamespace(
        embedding=torch.randn(2, 64),
        predictions=torch.randn(2, 10),
        availability=torch.ones(2, dtype=torch.bool),
    )
    branch = from_rhythm_branch(output, supervision_mask=torch.zeros(2, 10))
    assert torch.count_nonzero(branch.supervision_mask) == 0
    assert bool((branch.fusion_mask == 1).all())


def _live_bundle(batch: int = 2):
    encoder = SharedAudioEncoder()
    mel = torch.randn(batch, 2, 1, 8, 12)
    window_mask = torch.ones(batch, 2, dtype=torch.bool)
    valid_frames = torch.full((batch, 2), 12, dtype=torch.long)
    starts = torch.tensor([[0.0, 30.0]]).expand(batch, -1)
    encoded = encoder(mel, window_mask, valid_frames, starts)

    rhythm_model = RhythmBranch(RhythmBranchConfig(dropout=0))
    rhythm_raw = rhythm_model(
        encoded.encoded_sequence,
        encoded.sequence_mask,
        encoded.sequence_window_index,
    )
    rhythm = from_rhythm_branch(rhythm_raw, supervision_mask=torch.zeros(batch, 10))

    tokens_per_window = encoded.encoded_sequence.shape[1] // 2
    harmony_model = TemporalHarmonyBranch(128, dropout=0)
    harmony_raw = harmony_model(
        encoded.encoded_sequence,
        encoded.sequence_mask,
        encoded.sequence_window_index,
        windows=2,
        tokens_per_window=tokens_per_window,
    )
    harmony = from_temporal_harmony_branch(
        harmony_raw,
        chroma_target_mask=torch.zeros_like(encoded.sequence_mask),
    )

    fixture = make_bundle(batch, seed=14, fusion_keep=1.0)
    branches = dict(fixture.branches)
    branches["rhythm"] = rhythm
    branches["harmony"] = harmony
    bundle = BranchBundle(branches, fixture.counts)
    bundle.validate()
    return encoder, rhythm_model, harmony_model, bundle


def test_genre_loss_reaches_prediction_heads_and_shared_encoder_in_primary_mode():
    encoder, rhythm, harmony, bundle = _live_bundle()
    model = ConceptBottleneckModel(
        dropout_p=0,
        allow_no_dropout=True,
        fusion_input_mode=PRIMARY_FUSION_INPUT_MODE,
    )
    logits, _ = model.from_bundle(bundle, apply_dropout=False)
    torch.nn.functional.binary_cross_entropy_with_logits(
        logits, torch.rand_like(logits)
    ).backward()

    assert rhythm.regression_head.weight.grad is not None
    assert rhythm.regression_head.weight.grad.abs().sum() > 0
    assert harmony.chroma_head.weight.grad is not None
    assert harmony.chroma_head.weight.grad.abs().sum() > 0
    assert encoder.cnn[0].weight.grad is not None
    assert encoder.cnn[0].weight.grad.abs().sum() > 0


def test_embedding_fusion_ablation_bypasses_prediction_heads_for_genre():
    _, rhythm, harmony, bundle = _live_bundle()
    model = ConceptBottleneckModel(
        dropout_p=0,
        allow_no_dropout=True,
        fusion_input_mode=EMBEDDING_FUSION_INPUT_MODE,
    )
    logits, _ = model.from_bundle(bundle, apply_dropout=False)
    logits.sum().backward()
    assert rhythm.regression_head.weight.grad is None
    assert harmony.chroma_head.weight.grad is None
    assert rhythm.embedding_head[0].weight.grad is not None
    assert harmony.embedding_projection.weight.grad is not None
