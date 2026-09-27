"""End-to-end baseline checkpoint and metric protocol checks."""
import json
import numpy as np
import pandas as pd
import torch
from scripts import train_joint as j
from scripts import train_cnn as c


def test_cnn_training_and_saved_predictions(tmp_path, monkeypatch):
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
    c.train(j.TrainConfig(epochs=1, batch_size=2, device='cpu',
                          window_frames=16, max_windows=2, out_dir='results/cnn'))
    out = tmp_path / 'results/cnn'
    results = json.loads((out / 'results.json').read_text())
    pred = np.load(out / 'test_predictions.npz')
    assert pred['probabilities'].shape == (2, 6)
    assert results['test_macro_ap'] == j.macro_average_precision(
        torch.from_numpy(pred['probabilities']), torch.from_numpy(pred['targets']))
    assert results['n_test'] == 2
    assert len(results['test_per_genre_ap']) == 6
    checkpoint = torch.load(out / 'best.pt', weights_only=False)
    model = c.GenreCNN()
    model.load_state_dict(checkpoint['model_state'])
    assert all(key.startswith(('encoder.', 'head.')) for key in checkpoint['model_state'])


def test_ap_excludes_absent_genres():
    truth = torch.tensor([[1., 0.], [0., 0.], [1., 0.]])
    probs = torch.tensor([[0.9, 0.1], [0.8, 0.5], [0.7, 0.9]])
    assert abs(j.macro_average_precision(probs, truth) - (1 + 2/3)/2) < 1e-6
