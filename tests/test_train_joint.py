"""Real trainer wiring: learned harmony, target isolation, and checkpoint reload."""
import json

import numpy as np
import pandas as pd
import torch

from scripts import train_joint as j


def test_harmony_supervision_reaches_branch_and_encoder_without_target_leakage():
    torch.manual_seed(7)
    encoder = j.SharedAudioEncoder().eval()
    heads = [j.InstrumentBranch(), j.TimbreBranch(), j.RhythmBranch(),
             j.TemporalHarmonyBranch(128)]
    for head in heads:
        head.eval()
    encoded = encoder(torch.randn(2, 2, 1, 16, 12), torch.ones(2, 2, dtype=torch.bool),
                      torch.full((2, 2), 12), torch.tensor([[0., 10.], [0., 10.]]))
    kwargs = dict(instr_tgt=torch.zeros(2, 40), timbre_tgt=torch.zeros(2, 35),
                  timbre_msk=torch.ones(2, 35, dtype=torch.bool),
                  rhythm_tgt=torch.zeros(2, 10), rhythm_msk=torch.ones(2, 10, dtype=torch.bool),
                  harmony_chroma=torch.softmax(torch.randn(2, 12), -1), device=torch.device('cpu'))
    bundle, targets = j._build_bundle(encoded, *heads, **kwargs)
    fusion = j.ConceptBottleneckModel().eval()
    logits, _ = fusion.from_bundle(bundle, apply_dropout=False)
    kwargs['harmony_chroma'] = torch.roll(kwargs['harmony_chroma'], 1, -1)
    other, _ = j._build_bundle(encoded, *heads, **kwargs)
    other_logits, _ = fusion.from_bundle(other, apply_dropout=False)
    torch.testing.assert_close(logits, other_logits)
    loss = j.JointLossOrchestrator(weights=j.LossWeights(genre=0, instrument=0,
                                  timbre=0, rhythm=0, harmony=1))
    result = loss(logits, torch.zeros_like(logits), bundle, targets)
    result.total.backward()
    for module in (encoder, heads[-1]):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    targets['harmony'].chroma[:] = float('nan')
    targets['harmony'].chroma_mask[:] = False
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
                          ('harmony', j.CHROMA_COLS)]:
        values = np.ones((6, len(columns))) if name == 'harmony' else np.tile(np.arange(6)[:, None] % 2, (1, len(columns)))
        frame = pd.DataFrame(values, columns=columns)
        frame.insert(0, 'TRACK_ID', ids)
        frame.to_csv(data / f'{name}_df.csv', index=False)
    monkeypatch.setattr(j, 'ROOT', tmp_path)
    j.train(j.TrainConfig(epochs=1, batch_size=2, device='cpu', window_frames=16, max_windows=2))
    checkpoint = torch.load(tmp_path / 'results/joint/best.pt', weights_only=False)
    assert checkpoint['harmony_head']
    assert checkpoint['harmony_strategy'] == 'predicted_chroma_song_mean_supervision'
    assert checkpoint['n_instrument_tags'] == 40
    assert checkpoint['beats_count_masked'] is True
    assert checkpoint['val_thresholds']['per_tag']
    assert json.loads((tmp_path / 'results/joint/results.json').read_text())['n_test'] == 2


def test_relocates_colab_logmel_path(tmp_path):
    assert j.resolve_logmel_path('/content/drive/old/dataset/logmel_songs/74/171074.npy',
                                 tmp_path, tmp_path) == str(tmp_path / '74/171074.npy')


def test_combined_dataset_masks_missing_chroma_targets(tmp_path):
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

    train, val, test, *_ = j.build_datasets(
        tmp_path, dataset_csv=dataset_csv, split_csv=split_csv,
        logmel_root=tmp_path / "logmel_songs"
    )

    assert [len(train), len(val), len(test)] == [1, 1, 1]
    assert train.harmony.shape == (1, 12)
    assert torch.isnan(train.harmony).all()


def test_combined_dataset_takes_chroma_targets_from_harmony_table(tmp_path):
    ids = [f"track_{i:07d}" for i in range(3)]
    columns = [*j.INSTRUMENT_TAGS, *j.TIMBRE_FEATURES, *j.RHYTHM_FEATURES, *j.GENRE_TAGS]
    frame = pd.DataFrame(np.ones((3, len(columns))), columns=columns)
    frame.insert(0, "logmel_path", [f"logmel_songs/0/{i}.npy" for i in range(3)])
    frame.insert(0, "TRACK_ID", ids)
    frame["chroma_entropy_mean"] = 0.5  # tonal summaries are not chroma bins
    frame.to_csv(tmp_path / "full_dataset.csv", index=False)
    pd.DataFrame({"track_id": ids, "split": ["train", "validation", "test"]}).to_csv(
        tmp_path / "track_split_assignments.csv", index=False
    )
    # Rows 0 and 2 have chroma; row 1 (validation) is absent from the harmony table.
    chroma = np.arange(1, 13, dtype=float)
    harmony = pd.DataFrame([chroma, chroma[::-1]], columns=j.CHROMA_COLS)
    harmony.insert(0, "TRACK_ID", ["0", "track_0000002"])
    harmony.to_csv(tmp_path / "harmony_df.csv", index=False)

    train, val, test, *_ = j.build_datasets(
        tmp_path, dataset_csv=tmp_path / "full_dataset.csv",
        logmel_root=tmp_path / "logmel_songs",
    )

    torch.testing.assert_close(train.harmony[0], torch.tensor(chroma / chroma.sum(), dtype=torch.float32))
    torch.testing.assert_close(test.harmony[0], torch.tensor(chroma[::-1] / chroma.sum(), dtype=torch.float32))
    assert torch.isnan(val.harmony).all()


def test_stored_windows_preserve_boundaries_and_mask_final_padding(tmp_path):
    path = tmp_path / 'stack.npy'
    np.save(path, np.ones((3, 128, 469), dtype=np.float32))
    ds = j.MultiTargetDataset(['1'], [str(path)], np.zeros((1, 6)), np.zeros((1, 40)),
        np.zeros((1, 35)), np.ones((1, 35), bool), np.zeros((1, 10)),
        np.ones((1, 10), bool), np.ones((1, 12)) / 12, max_windows=2,
        mel_config={'sample_rate': 16000, 'hop_length': 512, 'window_seconds': 15,
                    'n_mels': 128, 'center': True}, durations={'1': 31.0})
    (mel, valid, starts), *_ = ds[0]
    assert mel.shape == (2, 1, 128, 469)
    assert starts.tolist() == [0., 30.]
    assert valid.tolist() == [469, 32]
    assert not mel[1, :, :, 32:].any()
