"""Branch ablations preserve cohorts, target isolation and gradient flow."""
import json
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from scripts import train_joint as j


STAGES = [(), ('instrument',), ('instrument', 'timbre'),
          ('instrument', 'timbre', 'rhythm'), j.CONCEPT_ORDER,
          ('harmony',), ('rhythm', 'timbre')]


def write_dataset(path, branches, vector=False):
    path.mkdir()
    rows = []
    for i in range(6):
        audio = path / f'{i}.npy'
        np.save(audio, np.random.default_rng(i).normal(size=(96, 16)).astype('float32'))
        row = {'TRACK_ID': str(i), 'logmel_path': str(audio)}
        for name, columns in j.VECTOR_GROUPS:
            if name == 'genre' or name.removesuffix('_vector') in branches:
                values = [float(i % 2)] * len(columns)
                row.update({name: json.dumps(values)} if vector else dict(zip(columns, values)))
        rows.append(row)
    pd.DataFrame(rows).to_csv(path / 'dataset.csv', index=False)
    pd.DataFrame({'TRACK_ID': [str(i) for i in range(6)],
                  'split': ['train'] * 2 + ['validation'] * 2 + ['test'] * 2
                  }).to_csv(path / 'track_split_assignments.csv', index=False)


@pytest.mark.parametrize('branches', STAGES)
def test_selected_stages_train_reload_and_export_without_disabled_targets(tmp_path, branches):
    data = tmp_path / 'data'
    write_dataset(data, branches)
    out = tmp_path / 'run'
    j.train(j.TrainConfig(branches=branches, epochs=1, batch_size=2, device='cpu',
                         window_frames=16, max_windows=1, data_dir=data,
                         dataset_csv=data / 'dataset.csv', out_dir=str(out)))
    result = json.loads((out / 'results.json').read_text())
    checkpoint = torch.load(out / 'best.pt', weights_only=False)
    assert set(result['branches']) == set(branches)
    assert result['split_track_ids'] == {
        name: [j._norm_id(str(i)) for i in indices]
        for name, indices in [('train', (0, 1)), ('validation', (2, 3)), ('test', (4, 5))]
    }
    assert checkpoint['branches'] == result['branches']
    assert result['seed'] == 42
    for split in ('validation', 'test'):
        predictions = np.load(out / f'{split}_predictions.npz')
        assert predictions['probabilities'].shape == (2, 6)
        assert np.isfinite(predictions['probabilities']).all()
    if branches:
        assert set(result['test_branch_metrics']) == set(branches)
        for name in j.CONCEPT_ORDER:
            assert (checkpoint[f'{name}_head'] is not None) == (name in branches)
            if name not in branches:
                assert result['log'][0]['train_terms'][name] == 0
                if name != 'instrument':
                    assert checkpoint[f'{name}_standardizer'] is None
    else:
        assert all(k.startswith(('encoder.', 'head.')) for k in checkpoint['model_state'])


@pytest.mark.parametrize('branches', [('instrument',), ('harmony',), ()])
def test_partial_vector_datasets_ignore_disabled_vectors(tmp_path, branches):
    data = tmp_path / 'data'
    write_dataset(data, branches, vector=True)
    frame = pd.read_csv(data / 'dataset.csv')
    frame['timbre_vector'] = 'invalid unused JSON'
    frame.to_csv(data / 'dataset.csv', index=False)
    datasets = j.build_datasets(data, dataset_csv=data / 'dataset.csv', branches=branches)
    assert [len(ds) for ds in datasets[:3]] == [2, 2, 2]


@pytest.mark.parametrize('branches', STAGES[1:])
def test_only_selected_heads_receive_genre_gradients(branches):
    j.seed_run(9)
    encoder = j.SharedAudioEncoder()
    factories = [j.InstrumentHead, j.TimbreBranch, j.RhythmBranch,
                 lambda: j.TemporalHarmonyBranch(128, descriptor_dim=12)]
    names = ('instrument', 'timbre', 'rhythm', 'harmony')
    heads = [factory() if name in branches else None for name, factory in zip(names, factories)]
    encoded = encoder(torch.randn(2, 1, 1, 16, 12), torch.ones(2, 1, dtype=torch.bool),
                      torch.full((2, 1), 12), torch.zeros(2, 1))
    kwargs = dict(instr_tgt=torch.zeros(2, 41), timbre_tgt=torch.zeros(2, 35),
                  timbre_msk=torch.ones(2, 35), rhythm_tgt=torch.zeros(2, 10),
                  rhythm_msk=torch.ones(2, 10), harmony_tgt=torch.zeros(2, 12),
                  harmony_msk=torch.ones(2, 12), device=torch.device('cpu'))
    bundle, _ = j._build_bundle(encoded, *heads, **kwargs)
    fusion = j.ConceptBottleneckModel().eval()
    logits, output = fusion.from_bundle(bundle, apply_dropout=False)
    for i, name in enumerate(j.CONCEPT_ORDER):
        if name not in branches:
            assert not output.gates[:, i].any()
            assert not bundle.supervision_mask(name).any()
    torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.ones_like(logits)).backward()
    for module in [encoder, *(head for head in heads if head is not None)]:
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())


@pytest.mark.parametrize('selection', [('instrument', 'instrument'), ('unknown',), ('none', 'rhythm')])
def test_invalid_selections_fail(selection):
    with pytest.raises(ValueError):
        j.normalize_branches(selection)


def test_cli_passes_selection_and_seed(monkeypatch):
    seen = []
    monkeypatch.setattr(j, 'train', seen.append)
    monkeypatch.setattr(sys, 'argv', ['train_joint.py', '--branches', 'timbre', 'instrument', '--seed', '5'])
    j.main()
    assert seen[0].branches == ('instrument', 'timbre')
    assert seen[0].seed == 5
    assert seen[0].out_dir == 'results/instrument-timbre'


def test_legacy_instrument_only_needs_no_other_target_csvs(tmp_path):
    data = tmp_path / 'data'
    write_dataset(data, ('instrument',))
    frame = pd.read_csv(data / 'dataset.csv')
    frame[['TRACK_ID', 'logmel_path']].to_csv(data / 'logmel_metadata.csv', index=False)
    frame[['TRACK_ID', *j.GENRE_TAGS]].to_csv(data / 'genres_df.csv', index=False)
    frame[['TRACK_ID', *j.INSTRUMENT_TAGS]].to_csv(data / 'instrument_df.csv', index=False)
    datasets = j.build_datasets(data, branches=('instrument',))
    assert [len(ds) for ds in datasets[:3]] == [2, 2, 2]
    assert datasets[3:] == (None, None, None)


def test_skip_test_saves_validation_predictions_only(tmp_path):
    data = tmp_path / 'data'
    write_dataset(data, ('instrument',))
    out = tmp_path / 'run'
    j.train(j.TrainConfig(branches=('instrument',), epochs=1, batch_size=2, device='cpu',
                         window_frames=16, max_windows=1, data_dir=data, evaluate_test=False,
                         dataset_csv=data / 'dataset.csv', out_dir=str(out)))
    result = json.loads((out / 'results.json').read_text())
    assert not result['test_evaluated']
    assert result['test_macro_ap'] is None
    assert (out / 'validation_predictions.npz').is_file()
    assert not (out / 'test_predictions.npz').exists()


@pytest.mark.parametrize('args', [['--branches', 'none'], ['--model', 'cnn']])
def test_cli_routes_cnn_only(monkeypatch, args):
    from scripts import train_cnn
    seen = []
    monkeypatch.setattr(train_cnn, 'train', seen.append)
    monkeypatch.setattr(sys, 'argv', ['train_joint.py', *args])
    j.main()
    assert seen[0].branches == ()
    assert seen[0].out_dir == 'results/cnn'


@pytest.mark.parametrize('args', [
    ['--branches', 'none', 'instrument'],
    ['--model', 'cnn', '--branches', 'instrument'],
    ['--branches', 'instrument', '--require-harmony-targets'],
])
def test_cli_rejects_conflicting_options(monkeypatch, args):
    monkeypatch.setattr(sys, 'argv', ['train_joint.py', *args])
    with pytest.raises(SystemExit) as error:
        j.main()
    assert error.value.code == 2
