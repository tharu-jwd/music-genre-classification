"""Compatibility of the shared CNN audio encoder with the Harmony v4 branch.

Covers the real cache geometries, the trainer's batching/padding, masking,
window isolation, gradients, frozen/pretrained encoders, checkpoints, and the
full trainer data path on stacked log-mel files.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from concept_fusion.contract import HARMONY_FEATURES
from harmony_branch.constants import CHROMA_MEAN_FEATURES, TONNETZ_MEAN_FEATURES
from harmony_branch.descriptors import HarmonyTargetTransform, tonnetz_phi
from harmony_branch.model import ChromaGroundedHarmonyBranch
from scripts import train_joint as j
from shared_encoder import SHARED_ENCODER_ARCHITECTURE, SharedAudioEncoder

ROOT = Path(__file__).resolve().parents[1]
HARMONY_TABLE = ROOT / "data" / "harmony_df.csv"
CHROMA = [HARMONY_FEATURES.index(n) for n in CHROMA_MEAN_FEATURES]
TONNETZ = [HARMONY_FEATURES.index(n) for n in TONNETZ_MEAN_FEATURES]

# (name, mel bins, frames per window, sample rate, hop) of the two real caches.
GEOMETRIES = [
    ("stacked_cache_16k_512", 128, 469, 16000, 512),
    ("fallback_2d_12k_256", 96, 1366, 12000, 256),
]


def _real_transform() -> HarmonyTargetTransform:
    if HARMONY_TABLE.is_file():
        table = pd.read_csv(HARMONY_TABLE)[list(HARMONY_FEATURES)].to_numpy()
        return HarmonyTargetTransform().fit(table[:2000])
    rng = np.random.default_rng(0)
    chroma = rng.dirichlet(np.ones(12), size=200)
    values = rng.uniform(0.05, 0.9, size=(200, 45))
    values[:, CHROMA] = chroma
    values[:, TONNETZ] = chroma @ tonnetz_phi().T
    return HarmonyTargetTransform().fit(values)


def _stack(seed=0, dropout=0.0):
    torch.manual_seed(seed)
    encoder = SharedAudioEncoder()
    harmony = ChromaGroundedHarmonyBranch(128, dropout=dropout)
    harmony.set_target_transform(_real_transform())
    return encoder.eval(), harmony.eval()


def _run(encoder, harmony, mel, valid, starts, *, sample_rate=16000, hop_length=512):
    encoded = encoder(mel, valid > 0, valid, starts, sample_rate=sample_rate, hop_length=hop_length)
    windows = encoded.window_repr.shape[1]
    out = harmony(
        encoded.encoded_sequence,
        encoded.sequence_mask,
        encoded.sequence_window_index,
        windows=windows,
        tokens_per_window=encoded.encoded_sequence.shape[1] // windows,
    )
    return encoded, out


def _audio(batch, windows, mels, frames, *, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(batch, windows, 1, mels, frames, generator=g)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,mels,frames,sr,hop", GEOMETRIES, ids=[g[0] for g in GEOMETRIES])
def test_real_cache_geometries_feed_the_v4_branch(name, mels, frames, sr, hop):
    encoder, harmony = _stack()
    # Song 0: two full windows + a short last one. Song 1: one window, then padding.
    valid = torch.tensor([[frames, frames, frames // 3], [frames // 2, 0, 0]])
    starts = torch.tensor([[0.0, 15.0, 30.0], [0.0, 0.0, 0.0]])
    mel = _audio(2, 3, mels, frames)
    with torch.no_grad():
        encoded, out = _run(encoder, harmony, mel, valid, starts, sample_rate=sr, hop_length=hop)

    tokens_per_window = math.ceil(frames / 2)
    assert encoded.encoded_sequence.shape == (2, 3 * tokens_per_window, 128)
    expected_valid = [
        2 * tokens_per_window + math.ceil((frames // 3) / 2),
        math.ceil((frames // 2) / 2),
    ]
    assert encoded.sequence_mask.sum(dim=1).tolist() == expected_valid
    assert out.chroma_logits.shape == (2, 3 * tokens_per_window, 12)
    assert out.descriptor_values.shape == (2, 45)
    assert torch.isfinite(out.descriptor_values).all()
    assert torch.isfinite(out.descriptor_raw).all()
    raw = out.descriptor_raw.double()
    torch.testing.assert_close(raw[:, CHROMA].sum(-1), torch.ones(2, dtype=torch.float64))
    phi = torch.tensor(tonnetz_phi())
    torch.testing.assert_close(raw[:, TONNETZ], raw[:, CHROMA] @ phi.T, atol=1e-5, rtol=1e-4)


def test_trainer_token_layout_matches_the_branch_contract():
    encoder, _ = _stack()
    valid = torch.tensor([[40, 17], [9, 0]])
    encoded = encoder(_audio(2, 2, 128, 40), valid > 0, valid, torch.tensor([[0.0, 15.0], [0.0, 0.0]]))
    windows = encoded.window_repr.shape[1]
    tokens_per_window = encoded.encoded_sequence.shape[1] // windows
    assert windows * tokens_per_window == encoded.encoded_sequence.shape[1]
    index = encoded.sequence_window_index
    mask = encoded.sequence_mask
    assert torch.all(index[~mask] == -1)
    expected = torch.arange(windows).repeat_interleave(tokens_per_window).expand_as(index)
    assert torch.equal(index[mask], expected[mask])


# ---------------------------------------------------------------------------
# Batching, padding and masking
# ---------------------------------------------------------------------------

def _dataset_item(windows: int, frames: int, valid_last: int, seed: int):
    mel = torch.randn(windows, 1, 128, frames, generator=torch.Generator().manual_seed(seed))
    valid = torch.full((windows,), frames, dtype=torch.long)
    valid[-1] = valid_last
    mel[-1, :, :, valid_last:] = 0
    starts = torch.arange(windows, dtype=torch.float32) * 15.0
    zeros = [torch.zeros(6), torch.zeros(41), torch.zeros(35), torch.ones(35, dtype=torch.bool),
             torch.zeros(10), torch.ones(10, dtype=torch.bool), torch.zeros(45),
             torch.ones(45, dtype=torch.bool)]
    return ((mel, valid, starts), *zeros, f"{seed:07d}")


def test_song_predictions_do_not_depend_on_batch_neighbours_or_window_padding():
    encoder, harmony = _stack()
    short = _dataset_item(2, 40, 23, seed=5)
    long = _dataset_item(4, 40, 11, seed=6)
    alone = j.collate_fn([short])
    batched = j.collate_fn([short, long])  # short song is padded from 2 to 4 windows
    assert batched[0].shape[1] == 4
    with torch.no_grad():
        _, left = _run(encoder, harmony, alone[0], alone[2], alone[3])
        _, right = _run(encoder, harmony, batched[0], batched[2], batched[3])
    torch.testing.assert_close(left.descriptor_values[0], right.descriptor_values[0], atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(left.embedding[0], right.embedding[0], atol=1e-5, rtol=1e-4)


def test_audio_beyond_valid_frames_cannot_change_predictions():
    encoder, harmony = _stack()
    valid = torch.tensor([[40, 13]])
    starts = torch.tensor([[0.0, 15.0]])
    mel = _audio(1, 2, 128, 40)
    noisy = mel.clone()
    noisy[0, 1, :, :, 13:] = 1e3
    with torch.no_grad():
        _, left = _run(encoder, harmony, mel, valid, starts)
        _, right = _run(encoder, harmony, noisy, valid, starts)
    torch.testing.assert_close(left.descriptor_values, right.descriptor_values)


def test_a_window_never_sees_audio_from_another_window():
    encoder, harmony = _stack()
    valid = torch.tensor([[40, 40]])
    starts = torch.tensor([[0.0, 15.0]])
    mel = _audio(1, 2, 128, 40)
    changed = mel.clone()
    changed[0, 1] = torch.randn_like(changed[0, 1])
    with torch.no_grad():
        encoded, left = _run(encoder, harmony, mel, valid, starts)
        _, right = _run(encoder, harmony, changed, valid, starts)
    first = encoded.sequence_window_index[0] == 0
    torch.testing.assert_close(left.chroma_logits[0, first], right.chroma_logits[0, first])
    assert not torch.allclose(left.descriptor_values, right.descriptor_values)


def test_song_with_no_audio_is_unavailable_and_exactly_zero():
    encoder, harmony = _stack()
    valid = torch.tensor([[40, 0], [0, 0]])
    starts = torch.zeros(2, 2)
    with torch.no_grad():
        encoded, out = _run(encoder, harmony, _audio(2, 2, 128, 40), valid, starts)
    assert encoded.availability.tolist() == [True, False]
    assert out.availability.tolist() == [True, False]
    assert out.descriptor_values[1].abs().sum() == 0
    assert out.descriptor_raw[1].abs().sum() == 0


# ---------------------------------------------------------------------------
# Gradients, frozen and pretrained encoders
# ---------------------------------------------------------------------------

def test_both_descriptor_groups_train_the_shared_cnn():
    exact = CHROMA + TONNETZ
    learned = [i for i in range(45) if i not in exact]
    for columns in (exact, learned):
        encoder, harmony = _stack()
        encoder.train()
        harmony.train()
        valid = torch.tensor([[40, 25]])
        _, out = _run(encoder, harmony, _audio(1, 2, 128, 40), valid, torch.tensor([[0.0, 15.0]]))
        F.smooth_l1_loss(out.descriptor_values[:, columns], torch.randn(1, len(columns))).backward()
        grad = encoder.cnn[0].weight.grad
        assert grad is not None and grad.abs().sum() > 0


def test_frozen_encoder_still_trains_the_harmony_branch():
    encoder, harmony = _stack()
    encoder.freeze()
    harmony.train()
    valid = torch.tensor([[40, 25]])
    _, out = _run(encoder, harmony, _audio(1, 2, 128, 40), valid, torch.tensor([[0.0, 15.0]]))
    out.descriptor_values.square().sum().backward()
    assert all(p.grad is None for p in encoder.parameters())
    assert harmony.chroma_head[-1].weight.grad.abs().sum() > 0
    assert harmony.descriptor_head[-1].weight.grad.abs().sum() > 0


@pytest.mark.parametrize("prefix", ["", "enc.", "encoder."])
def test_pretrained_encoder_checkpoints_load_and_drive_harmony(prefix):
    source, harmony = _stack(seed=3)
    state = {prefix + k: v for k, v in source.state_dict().items()}
    target = SharedAudioEncoder().eval()
    target.load_instrument_pretraining({"model": state, "encoder_architecture": SHARED_ENCODER_ARCHITECTURE})
    valid = torch.tensor([[40, 30]])
    starts = torch.tensor([[0.0, 15.0]])
    mel = _audio(1, 2, 128, 40)
    with torch.no_grad():
        _, left = _run(source, harmony, mel, valid, starts)
        _, right = _run(target, harmony, mel, valid, starts)
    torch.testing.assert_close(left.descriptor_values, right.descriptor_values)


def test_encoder_and_harmony_checkpoint_round_trip(tmp_path):
    encoder, harmony = _stack(seed=4)
    path = tmp_path / "stack.pt"
    torch.save({"encoder": encoder.state_dict(), "harmony": harmony.state_dict()}, path)
    payload = torch.load(path, weights_only=True)
    encoder2 = SharedAudioEncoder().eval()
    harmony2 = ChromaGroundedHarmonyBranch(128).eval()  # buffers restore the transform
    encoder2.load_state_dict(payload["encoder"])
    harmony2.load_state_dict(payload["harmony"])
    valid = torch.tensor([[40, 30]])
    starts = torch.tensor([[0.0, 15.0]])
    mel = _audio(1, 2, 128, 40)
    with torch.no_grad():
        _, left = _run(encoder, harmony, mel, valid, starts)
        _, right = _run(encoder2, harmony2, mel, valid, starts)
    torch.testing.assert_close(left.descriptor_values, right.descriptor_values)
    torch.testing.assert_close(left.descriptor_raw, right.descriptor_raw)


def test_legacy_script_encoder_is_the_shared_implementation():
    from scripts.shared_audio_encoder import SharedAudioEncoder as Legacy

    assert Legacy is SharedAudioEncoder


# ---------------------------------------------------------------------------
# Full trainer path on stacked log-mel files
# ---------------------------------------------------------------------------

def _write_stacked_project(root: Path, n_tracks: int = 6, frames: int = 40) -> Path:
    data = root / "data"
    songs = data / "logmel_songs" / "0"
    songs.mkdir(parents=True)
    ids = [f"{i:07d}" for i in range(n_tracks)]
    window_seconds = frames * 512 / 16000
    durations = []
    rng = np.random.default_rng(0)
    for i, track in enumerate(ids):
        windows = 2 + i % 3
        np.save(songs / f"{track}.npy", rng.normal(size=(windows, 128, frames)).astype("float32"))
        durations.append((windows - 0.4) * window_seconds)  # short final window
    (data / "logmel_config.json").write_text(json.dumps({
        "sample_rate": 16000, "hop_length": 512, "window_seconds": window_seconds,
        "n_mels": 128, "center": True,
    }))
    pd.DataFrame({"TRACK_ID": ids, "track_duration_sec": durations}).to_csv(
        data / "logmel_audit.csv", index=False)
    pd.DataFrame({"TRACK_ID": ids, "logmel_path": [str(songs / f"{t}.npy") for t in ids]}).to_csv(
        data / "logmel_metadata.csv", index=False)
    pd.DataFrame({"TRACK_ID": ids, "split": ["train"] * 2 + ["validation"] * 2 + ["test"] * 2}).to_csv(
        data / "track_split_assignments.csv", index=False)
    for name, columns in [("genres", j.GENRE_TAGS), ("instrument", j.INSTRUMENT_TAGS),
                          ("timbre", j.TIMBRE_FEATURES), ("rhythm", j.RHYTHM_FEATURES)]:
        frame = pd.DataFrame(np.tile(np.arange(n_tracks)[:, None] % 2, (1, len(columns))), columns=columns)
        frame.insert(0, "TRACK_ID", ids)
        frame.to_csv(data / f"{name}_df.csv", index=False)
    if HARMONY_TABLE.is_file():
        harmony = pd.read_csv(HARMONY_TABLE).head(n_tracks)[list(HARMONY_FEATURES)]
    else:
        harmony = pd.DataFrame(np.full((n_tracks, 45), 0.3), columns=list(HARMONY_FEATURES))
    harmony.insert(0, "TRACK_ID", ids)
    harmony.to_csv(data / "harmony_df.csv", index=False)
    return data


def test_trainer_trains_evaluates_and_reloads_on_stacked_cache(tmp_path, monkeypatch):
    data = _write_stacked_project(tmp_path)
    monkeypatch.setattr(j, "ROOT", tmp_path)
    j.train(j.TrainConfig(epochs=1, batch_size=2, device="cpu", data_dir=data, max_windows=3))

    results = json.loads((tmp_path / "results/joint/results.json").read_text())
    harmony = results["test_branch_metrics"]["harmony"]
    assert set(harmony["per_feature"]) == set(HARMONY_FEATURES)
    assert harmony["n_observed"] == 2 * 45
    assert set(harmony["group_macro_r2"]) == {
        "chroma_mean", "chroma_std", "tonnetz_mean", "tonnetz_std", "tonal_dynamics"}
    assert results["harmony_supervised"] is True

    checkpoint = torch.load(tmp_path / "results/joint/best.pt", weights_only=False)
    assert checkpoint["mel_config"]["n_mels"] == 128
    encoder = SharedAudioEncoder()
    harmony_head = ChromaGroundedHarmonyBranch(128)
    encoder.load_state_dict(checkpoint["encoder"])
    harmony_head.load_state_dict(checkpoint["harmony_head"])
    transform = HarmonyTargetTransform.from_state_dict(checkpoint["harmony_standardizer"])
    np.testing.assert_allclose(harmony_head.target_mean.numpy(), transform.mean, rtol=1e-6)

    # The dataset path: last window is cut at the audited duration.
    train_ds, *_ = j.build_datasets(data, max_windows=3)
    (mel, valid, _starts), *_ = train_ds[0]
    assert mel.shape[-2:] == (128, 40)
    assert 0 < int(valid[-1]) < 40


# ---------------------------------------------------------------------------
# Full-size song
# ---------------------------------------------------------------------------

def test_full_size_twelve_window_song_forward_and_backward():
    encoder, harmony = _stack()
    encoder.train()
    harmony.train()
    frames = 469
    valid = torch.full((1, 12), frames)
    valid[0, -1] = 200
    starts = (torch.arange(12, dtype=torch.float32) * 15.0).unsqueeze(0)
    encoded, out = _run(encoder, harmony, _audio(1, 12, 128, frames), valid, starts)
    assert encoded.encoded_sequence.shape == (1, 12 * 235, 128)
    assert int(encoded.sequence_mask.sum()) == 11 * 235 + 100
    loss = F.smooth_l1_loss(out.descriptor_values, torch.zeros_like(out.descriptor_values))
    loss.backward()
    assert torch.isfinite(loss)
    assert encoder.cnn[0].weight.grad.abs().sum() > 0
