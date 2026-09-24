"""Autograd integration from every concept path into one shared CNN."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "rhythm_branch" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "timbre_branch" / "src"))

from concept_fusion.contract import ConceptCounts
from concept_fusion.harmony_adapter import from_temporal_harmony_branch
from concept_fusion.instrument_adapter import from_instrument_branch
from concept_fusion.model import ConceptBottleneckModel
from concept_fusion.rhythm_adapter import from_rhythm_branch
from concept_fusion.timbre_adapter import from_timbre_branch
from concept_fusion.types import BranchBundle
from harmony_branch.model import TemporalHarmonyBranch
from rhythm_branch.model import RhythmBranch, RhythmBranchConfig
from shared_encoder import SharedAudioEncoder
from timbre_branch.model import TimbreBranch


class _InstrumentHead(nn.Module):
    """Importable equivalent of the notebook-owned instrument v2 head."""

    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.1))
        self.classifier = nn.Linear(128, 40)

    def forward(self, pooled_song: torch.Tensor) -> dict:
        hidden = self.hidden(pooled_song)
        logits = self.classifier(hidden)
        return {
            "concept_values": logits.sigmoid(),
            "logits": logits,
            "supervision_mask": torch.ones_like(logits),
            "fusion_mask": torch.ones(len(logits), 1, device=logits.device),
            "diagnostics": {"hidden": hidden.detach()},
        }


def _live_stack():
    torch.manual_seed(23)
    encoder = SharedAudioEncoder()
    instrument_head = _InstrumentHead()
    timbre_head = TimbreBranch()
    rhythm_head = RhythmBranch(RhythmBranchConfig(dropout=0))
    harmony_head = TemporalHarmonyBranch(
        128,
        embedding_dim=32,
        hidden_dim=32,
        temporal_layers=2,
        chord_classes=None,
        dropout=0,
    )

    mel = torch.randn(2, 2, 1, 16, 10)
    window_mask = torch.tensor([[True, True], [True, False]])
    valid_frames = torch.tensor([[10, 7], [5, 0]])
    starts = torch.tensor([[0.0, 40.0], [3.0, 0.0]])
    encoded = encoder(mel, window_mask, valid_frames, starts)

    instrument_raw = instrument_head(encoded.pooled_song)
    timbre_raw = timbre_head(encoded.pooled_song)
    rhythm_raw = rhythm_head(
        encoded.encoded_sequence,
        encoded.sequence_mask,
        encoded.sequence_window_index,
    )
    harmony_raw = harmony_head(
        encoded.encoded_sequence,
        encoded.sequence_mask,
        encoded.sequence_window_index,
        windows=2,
        tokens_per_window=encoded.encoded_sequence.shape[1] // 2,
    )

    branches = {
        "instrument": from_instrument_branch(instrument_raw),
        "rhythm": from_rhythm_branch(
            rhythm_raw,
            supervision_mask=torch.ones_like(rhythm_raw.predictions),
        ),
        "timbre": from_timbre_branch(timbre_raw),
        "harmony": from_temporal_harmony_branch(
            harmony_raw,
            chroma_target_mask=encoded.sequence_mask,
        ),
    }
    bundle = BranchBundle(branches=branches, counts=ConceptCounts())
    bundle.validate()
    return encoder, encoded, instrument_raw, timbre_raw, rhythm_raw, harmony_raw, bundle


def _assert_encoder_gradient(loss: torch.Tensor, encoder: SharedAudioEncoder) -> None:
    loss.backward()
    gradient = encoder.cnn[0].weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0


def test_one_encoder_output_satisfies_all_four_branch_interfaces():
    _, encoded, instrument, timbre, rhythm, harmony, bundle = _live_stack()
    assert encoded.pooled_song.shape == (2, 128)
    assert encoded.window_repr.shape == (2, 2, 128)
    assert encoded.encoded_sequence.shape == (2, 10, 128)
    assert instrument["concept_values"].shape == (2, 40)
    assert timbre.shape == (2, 35)
    assert rhythm.embedding.shape == (2, 64)
    assert rhythm.predictions.shape == (2, 10)
    assert harmony.embedding.shape == (2, 32)
    assert harmony.chroma_logits.shape == (2, 10, 12)
    bundle.validate()


def test_each_branch_loss_path_reaches_shared_cnn():
    for branch_name in ("instrument", "timbre", "rhythm", "harmony"):
        encoder, _, instrument, timbre, rhythm, harmony, _ = _live_stack()
        if branch_name == "instrument":
            loss = F.binary_cross_entropy_with_logits(
                instrument["logits"], torch.rand_like(instrument["logits"])
            )
        elif branch_name == "timbre":
            loss = F.smooth_l1_loss(timbre, torch.randn_like(timbre))
        elif branch_name == "rhythm":
            loss = F.smooth_l1_loss(rhythm.predictions, torch.randn_like(rhythm.predictions))
        else:
            valid_logits = harmony.chroma_logits[harmony.prediction_mask]
            targets = torch.softmax(torch.randn_like(valid_logits), dim=-1)
            loss = -(targets * F.log_softmax(valid_logits, dim=-1)).sum(dim=-1).mean()
        _assert_encoder_gradient(loss, encoder)


def test_genre_head_through_concept_fusion_reaches_shared_cnn_without_shortcut():
    encoder, encoded, _, _, _, _, bundle = _live_stack()
    model = ConceptBottleneckModel(dropout_p=0, allow_no_dropout=True).eval()
    genre_logits, _ = model.from_bundle(bundle, apply_dropout=False)
    assert model.shortcut is None
    assert genre_logits.shape == (2, 87)
    loss = F.binary_cross_entropy_with_logits(genre_logits, torch.rand_like(genre_logits))
    _assert_encoder_gradient(loss, encoder)
    assert encoded.pooled_song.grad_fn is not None
    assert encoded.encoded_sequence.grad_fn is not None
