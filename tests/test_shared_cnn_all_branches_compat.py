"""All four concept branches and fusion on one shared CNN, as the trainer wires them.

Complements test_shared_encoder_branch_integration.py (tiny tensors) with the
real cache geometries, the trainer's collate/bundle path, batch-padding
invariance, masking, per-branch gradients, concept removal, and the joint loss.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from scripts import train_joint as j  # first: puts every branch package on sys.path

from concept_fusion.contract import CONCEPT_ORDER
from harmony_branch.model import ChromaGroundedHarmonyBranch
from rhythm_branch.model import RhythmBranchConfig
from shared_encoder import SharedAudioEncoder

GEOMETRIES = [
    ("stacked_cache_16k_512", 128, 469, 16000, 512),
    ("fallback_2d_12k_256", 96, 1366, 12000, 256),
]
IDS = [g[0] for g in GEOMETRIES]


def _models(seed=0):
    torch.manual_seed(seed)
    models = dict(
        encoder=SharedAudioEncoder(),
        instrument=j.InstrumentHead(),
        timbre=j.TimbreBranch(),
        rhythm=j.RhythmBranch(RhythmBranchConfig(dropout=0.0)),
        harmony=ChromaGroundedHarmonyBranch(128, dropout=0.0),
        fusion=j.ConceptBottleneckModel(harmony_embedding_dim=64),
    )
    for model in models.values():
        model.eval()
    return models


def _targets(batch):
    return dict(
        instr_tgt=torch.zeros(batch, 41), timbre_tgt=torch.randn(batch, 35),
        timbre_msk=torch.ones(batch, 35, dtype=torch.bool),
        rhythm_tgt=torch.randn(batch, 10), rhythm_msk=torch.ones(batch, 10, dtype=torch.bool),
        harmony_tgt=torch.randn(batch, 45), harmony_msk=torch.ones(batch, 45, dtype=torch.bool),
        device=torch.device("cpu"),
    )


def _forward(models, mel, valid, starts, *, sr=16000, hop=512, targets=None):
    encoded = models["encoder"](mel, valid > 0, valid, starts, sample_rate=sr, hop_length=hop)
    bundle, concept_targets = j._build_bundle(
        encoded, models["instrument"], models["timbre"], models["rhythm"], models["harmony"],
        **(targets or _targets(mel.shape[0])),
    )
    logits, fusion = models["fusion"].from_bundle(bundle, apply_dropout=False)
    return encoded, bundle, concept_targets, logits, fusion


def _item(windows, mels, frames, valid_last, seed):
    mel = torch.randn(windows, 1, mels, frames, generator=torch.Generator().manual_seed(seed))
    valid = torch.full((windows,), frames, dtype=torch.long)
    valid[-1] = valid_last
    mel[-1, :, :, valid_last:] = 0
    starts = torch.arange(windows, dtype=torch.float32) * 15.0
    rest = [torch.zeros(6), torch.zeros(41), torch.zeros(35), torch.ones(35, dtype=torch.bool),
            torch.zeros(10), torch.ones(10, dtype=torch.bool), torch.zeros(45),
            torch.ones(45, dtype=torch.bool)]
    return ((mel, valid, starts), *rest, f"{seed:07d}")


@pytest.mark.parametrize("name,mels,frames,sr,hop", GEOMETRIES, ids=IDS)
def test_every_branch_and_fusion_run_on_real_geometries(name, mels, frames, sr, hop):
    models = _models()
    valid = torch.tensor([[frames, frames // 4], [frames // 2, 0]])
    starts = torch.tensor([[0.0, 15.0], [0.0, 0.0]])
    mel = torch.randn(2, 2, 1, mels, frames)
    with torch.no_grad():
        encoded, bundle, _, logits, fusion = _forward(models, mel, valid, starts, sr=sr, hop=hop)
    assert encoded.pooled_song.shape == (2, 128)
    assert tuple(bundle.branches) == CONCEPT_ORDER
    widths = {n: bundle.concept_values(n).shape[1] for n in CONCEPT_ORDER}
    assert widths == {"instrument": 41, "rhythm": 10, "timbre": 35, "harmony": 45}
    assert logits.shape == (2, 6) and torch.isfinite(logits).all()
    assert bundle.fusion_mask().tolist() == [[1.0] * 4, [1.0] * 4]
    torch.testing.assert_close(fusion.gates.sum(dim=1), torch.ones(2))


def test_every_output_is_independent_of_batch_neighbours_and_padding():
    models = _models()
    short = _item(2, 128, 40, 23, seed=5)
    long = _item(4, 128, 40, 11, seed=6)
    alone = j.collate_fn([short])
    batched = j.collate_fn([short, long])
    with torch.no_grad():
        _, left, _, left_logits, _ = _forward(models, alone[0], alone[2], alone[3], targets=_targets(1))
        _, right, _, right_logits, _ = _forward(models, batched[0], batched[2], batched[3])
    for name in CONCEPT_ORDER:
        torch.testing.assert_close(
            left.concept_values(name)[0], right.concept_values(name)[0], atol=1e-5, rtol=1e-4,
            msg=lambda m, n=name: f"{n}: {m}",
        )
    torch.testing.assert_close(left_logits[0], right_logits[0], atol=1e-5, rtol=1e-4)


def test_audio_beyond_valid_frames_changes_no_branch():
    models = _models()
    valid = torch.tensor([[40, 13]])
    starts = torch.tensor([[0.0, 15.0]])
    mel = torch.randn(1, 2, 1, 128, 40)
    noisy = mel.clone()
    noisy[0, 1, :, :, 13:] = 1e3
    with torch.no_grad():
        _, left, _, left_logits, _ = _forward(models, mel, valid, starts, targets=_targets(1))
        _, right, _, right_logits, _ = _forward(models, noisy, valid, starts, targets=_targets(1))
    for name in CONCEPT_ORDER:
        torch.testing.assert_close(left.concept_values(name), right.concept_values(name))
    torch.testing.assert_close(left_logits, right_logits)


def test_joint_loss_trains_every_branch_and_the_shared_cnn():
    models = _models()
    for model in models.values():
        model.train()
    valid = torch.tensor([[40, 25], [40, 0]])
    starts = torch.tensor([[0.0, 15.0], [0.0, 0.0]])
    _, bundle, targets, logits, _ = _forward(models, torch.randn(2, 2, 1, 128, 40), valid, starts)
    loss = j.JointLossOrchestrator()(logits, torch.tensor([[1.0, 0, 0, 0, 0, 0]] * 2), bundle, targets)
    assert torch.isfinite(loss.total)
    assert {"genre", *CONCEPT_ORDER} <= set(loss.terms)
    loss.total.backward()
    for name in ("encoder", "instrument", "timbre", "rhythm", "harmony", "fusion"):
        grads = [p.grad for p in models[name].parameters() if p.grad is not None]
        assert grads and any(g.abs().sum() > 0 for g in grads), name


@pytest.mark.parametrize("branch", CONCEPT_ORDER)
def test_genre_loss_alone_reaches_the_cnn_through_each_branch(branch):
    models = _models()
    valid = torch.tensor([[40, 25]])
    starts = torch.tensor([[0.0, 15.0]])
    encoded = models["encoder"](torch.randn(1, 2, 1, 128, 40), valid > 0, valid, starts)
    bundle, _ = j._build_bundle(encoded, models["instrument"], models["timbre"], models["rhythm"],
                                models["harmony"], **_targets(1))
    only = bundle.with_enabled_concepts((branch,))
    logits, fusion = models["fusion"].from_bundle(only, apply_dropout=False)
    assert fusion.gates[0].tolist() == [1.0 if n == branch else 0.0 for n in CONCEPT_ORDER]
    F.binary_cross_entropy_with_logits(logits, torch.rand_like(logits)).backward()
    assert models["encoder"].cnn[0].weight.grad.abs().sum() > 0
    head = models[branch]
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters())


def test_fused_stack_checkpoint_round_trip(tmp_path):
    models = _models(seed=9)
    valid = torch.tensor([[40, 30]])
    starts = torch.tensor([[0.0, 15.0]])
    mel = torch.randn(1, 2, 1, 128, 40)
    with torch.no_grad():
        *_, reference, _ = _forward(models, mel, valid, starts, targets=_targets(1))
    path = tmp_path / "stack.pt"
    torch.save({k: m.state_dict() for k, m in models.items()}, path)
    restored = _models(seed=123)
    payload = torch.load(path, weights_only=True)
    for key, model in restored.items():
        model.load_state_dict(payload[key])
    with torch.no_grad():
        *_, actual, _ = _forward(restored, mel, valid, starts, targets=_targets(1))
    torch.testing.assert_close(actual, reference)
