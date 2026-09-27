"""Real trainer wiring: learned harmony, target isolation, and checkpoint reload."""
import json

import numpy as np
import pandas as pd
import pytest
import torch

from scripts import train_joint as j


def test_combined_dataset_schema_has_one_vector_per_branch():
    schema = j.dataset_schema()
    assert schema["logmel_input"] == "path"
    assert {name: len(columns) for name, columns in schema["branch_vectors"].items()} == {
        "instrument_vector": 41, "rhythm_vector": 10,
        "timbre_vector": 35, "harmony_vector": 12,
    }
    assert len(schema["fusion_target"]["genre"]) == 6


def test_harmony_supervision_reaches_branch_and_encoder_without_target_leakage():
    torch.manual_seed(7)
    encoder = j.SharedAudioEncoder().eval()
    heads = [j.InstrumentHead(), j.TimbreBranch(), j.RhythmBranch(),
             j.TemporalHarmonyBranch(128, descriptor_dim=12)]
    for head in heads:
        head.eval()
    encoded = encoder(torch.randn(2, 2, 1, 16, 12), torch.ones(2, 2, dtype=torch.bool),
                      torch.full((2, 2), 12), torch.tensor([[0., 10.], [0., 10.]]))
    kwargs = dict(instr_tgt=torch.zeros(2, 41), timbre_tgt=torch.zeros(2, 35),
                  timbre_msk=torch.ones(2, 35, dtype=torch.bool),
                  rhythm_tgt=torch.zeros(2, 10), rhythm_msk=torch.ones(2, 10, dtype=torch.bool),
                  harmony_tgt=torch.randn(2, 12),
                  harmony_msk=torch.ones(2, 12, dtype=torch.bool), device=torch.device('cpu'))
    bundle, targets = j._build_bundle(encoded, *heads, **kwargs)
    fusion = j.ConceptBottleneckModel().eval()
    logits, _ = fusion.from_bundle(bundle, apply_dropout=False)
    kwargs['harmony_tgt'] = torch.roll(kwargs['harmony_tgt'], 1, -1)
    other, _ = j._build_bundle(encoded, *heads, **kwargs)
    other_logits, _ = fusion.from_bundle(other, apply_dropout=False)
    torch.testing.assert_close(logits, other_logits)
    loss = j.JointLossOrchestrator(weights=j.LossWeights(genre=0, instrument=0,
                                  timbre=0, rhythm=0, harmony=1))
    result = loss(logits, torch.zeros_like(logits), bundle, targets)
    result.total.backward()
    for module in (encoder, heads[-1]):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    bundle.branches['harmony'].supervision_mask.zero_()
    masked = loss(logits, torch.zeros_like(logits), bundle, targets)
    assert torch.isfinite(masked.total)
    assert masked.n_observed['harmony'] == 0


def test_training_saves_and_reloads_learned_harmony(tmp_path, monkeypatch):
    data = tmp_path / 'data'
    data.mkdir()
    ids = [str(i) for i in range(6)]
    paths = []
    for i in range(6):
        path = data / f'{i}.npy'
        np.save(path, np.random.default_rng(i).normal(size=(96, 25)).astype('float32'))
        paths.append(str(path))
    pd.DataFrame({'TRACK_ID': ids, 'logmel_path': paths}).to_csv(data / 'logmel_metadata.csv', index=False)
    pd.DataFrame({'TRACK_ID': ids, 'split': ['train'] * 2 + ['validation'] * 2 + ['test'] * 2}).to_csv(data / 'track_split_assignments.csv', index=False)
    for name, columns in [('genres', j.GENRE_TAGS), ('instrument', j.INSTRUMENT_TAGS),
                          ('timbre', j.TIMBRE_FEATURES), ('rhythm', j.RHYTHM_FEATURES),
                          ('harmony', j.HARMONY_FEATURES)]:
        values = np.ones((6, len(columns))) if name == 'harmony' else np.tile(np.arange(6)[:, None] % 2, (1, len(columns)))
        frame = pd.DataFrame(values, columns=columns)
        frame.insert(0, 'TRACK_ID', ids)
        frame.to_csv(data / f'{name}_df.csv', index=False)
    monkeypatch.setattr(j, 'ROOT', tmp_path)
    j.train(j.TrainConfig(epochs=1, batch_size=2, device='cpu', window_frames=16, max_windows=2))
    checkpoint = torch.load(tmp_path / 'results/joint/best.pt', weights_only=False)
    assert checkpoint['harmony_head']
    assert checkpoint['harmony_strategy'] == 'standardized_song_descriptor_regression'
    assert checkpoint['harmony_standardizer']['feature_names'] == list(j.HARMONY_FEATURES)
    assert checkpoint['n_instrument_tags'] == 41
    assert json.loads((tmp_path / 'results/joint/results.json').read_text())['n_test'] == 2


def test_relocates_colab_logmel_path(tmp_path):
    assert j.resolve_logmel_path('/content/drive/old/dataset/logmel_songs/74/171074.npy',
                                 tmp_path, tmp_path) == str(tmp_path / '74/171074.npy')


def test_combined_dataset_requires_harmony_descriptors(tmp_path):
    ids = [f"track_{i:07d}" for i in range(3)]
    columns = [*j.INSTRUMENT_TAGS, *j.TIMBRE_FEATURES, *j.RHYTHM_FEATURES, *j.GENRE_TAGS]
    frame = pd.DataFrame(np.ones((3, len(columns))), columns=columns)
    frame.insert(0, "logmel_path", [f"logmel_songs/0/{i}.npy" for i in range(3)])
    frame.insert(0, "TRACK_ID", ids)
    dataset_csv = tmp_path / "full_dataset.csv"
    split_csv = tmp_path / "splits.csv"
    frame.to_csv(dataset_csv, index=False)
    pd.DataFrame({"track_id": ids, "split": ["train", "validation", "test"]}).to_csv(
        split_csv, index=False
    )

    with pytest.raises(ValueError, match="tonal_concentration_mean"):
        j.build_datasets(
            tmp_path, dataset_csv=dataset_csv, split_csv=split_csv,
            logmel_root=tmp_path / "logmel_songs",
        )


def test_compact_json_vectors_expand_for_training(tmp_path):
    ids = [f"track_{i:07d}" for i in range(3)]
    row = {
        "instrument_vector": json.dumps([0] * 41),
        "rhythm_vector": json.dumps([1.0] * 10),
        "timbre_vector": json.dumps([2.0] * 35),
        "harmony_vector": json.dumps([3.0] * 12),
        "genre": json.dumps([0, 1, 0, 0, 0, 0]),
    }
    dataset = pd.DataFrame([
        {"track_id": track_id, "path": f"logmel_songs/0/{i}.npy", **row}
        for i, track_id in enumerate(ids)
    ])
    dataset_csv, split_csv = tmp_path / "dataset.csv", tmp_path / "splits.csv"
    dataset.to_csv(dataset_csv, index=False)
    pd.DataFrame({"track_id": ids, "split": ["train", "validation", "test"]}).to_csv(
        split_csv, index=False
    )

    train, val, test, *_ = j.build_datasets(
        tmp_path, dataset_csv=dataset_csv, split_csv=split_csv,
        logmel_root=tmp_path / "logmel_songs",
    )

    assert [len(train), len(val), len(test)] == [1, 1, 1]
    assert train.instrument.shape == (1, 41)
    assert train.genre.tolist() == [[0, 1, 0, 0, 0, 0]]
    assert train.harmony.shape == (1, 12)
    assert train.harmony_mask.all()


def test_stored_windows_preserve_boundaries_and_mask_final_padding(tmp_path):
    path = tmp_path / 'stack.npy'
    np.save(path, np.ones((3, 128, 469), dtype=np.float32))
    ds = j.MultiTargetDataset(['1'], [str(path)], np.zeros((1, 6)), np.zeros((1, 41)),
        np.zeros((1, 35)), np.ones((1, 35), bool), np.zeros((1, 10)),
        np.ones((1, 10), bool), np.ones((1, 12)) / 12, max_windows=2,
        mel_config={'sample_rate': 16000, 'hop_length': 512, 'window_seconds': 15,
                    'n_mels': 128, 'center': True}, durations={'1': 31.0})
    (mel, valid, starts), *_ = ds[0]
    assert mel.shape == (2, 1, 128, 469)
    assert starts.tolist() == [0., 30.]
    assert valid.tolist() == [469, 32]
    assert not mel[1, :, :, 32:].any()
