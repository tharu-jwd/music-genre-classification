"""Check config changes reach real training and checkpoint-only evaluation."""
import json

import numpy as np
import pytest
import torch

from shared_encoder.architecture import load_architecture, configure_training
from scripts import train_joint as j
from scripts.train_cnn import GenreCNN
from scripts.evaluate_checkpoint import evaluate_checkpoint
from scripts.paired_bootstrap_compare import load_predictions
from scripts.paper_experiments import load_manifest
from tests.test_modular_training import write_dataset


def test_default_architecture_preserves_current_parameter_counts():
    cfg = j.TrainConfig(architecture=load_architecture())
    configure_training(cfg)
    modules = j.build_concept_modules(cfg, "cpu")
    assert sum(p.numel() for m in modules.values() for p in m.parameters() if p.requires_grad) == 177333
    assert sum(p.numel() for p in GenreCNN(cfg.architecture).parameters()) == 104199
    manifest = load_manifest("experiments/paper21-modal.json")
    assert all(run["config"]["architecture"] == cfg.architecture for run in manifest["runs"])


@pytest.mark.parametrize("model", ["concept", "cnn"])
def test_custom_architecture_trains_and_restores_without_original_json(tmp_path, model):
    architecture = load_architecture()
    architecture["shared_encoder"].update(channels=[16, 32, 64], kernel_size=5, conv_bias=True, activation="silu")
    architecture["instrument"].update(hidden_dim=32, dropout=0.1, activation="relu")
    architecture["timbre"].update(hidden_dim=48, dropout=0.2)
    architecture["rhythm"].update(hidden_dim=32, temporal_layers=3, kernel_size=5, dilation_base=1, activation="relu")
    architecture["harmony"].update(hidden_dim=80, embedding_dim=48, temporal_layers=1, kernel_size=5,
                                   dilation_base=3, activation="silu")
    architecture["genre_predictor"].update(predictor="mlp", hidden_dim=32, hidden_dropout=0.1, activation="relu")
    architecture["direct_cnn"].update(hidden_dim=48, dropout=0.2, activation="silu")
    path = tmp_path / "architecture.json"
    path.write_text(json.dumps(architecture), encoding="utf-8")
    data = tmp_path / "data"
    write_dataset(data, j.CONCEPT_ORDER)
    out = tmp_path / model
    cfg = j.TrainConfig(architecture=load_architecture(path), branches=j.CONCEPT_ORDER if model == "concept" else (),
        data_dir=data, dataset_csv=data / "dataset.csv", out_dir=str(out), device="cpu", epochs=1,
        batch_size=2, max_windows=1, window_frames=16, evaluate_test=False)
    j.train(cfg)
    results = json.loads((out / "results.json").read_text())
    assert results["architecture"] == architecture
    checkpoint = torch.load(out / "best.pt", weights_only=False)
    assert checkpoint["architecture"] == architecture
    path.unlink()
    evaluate_checkpoint(out / "best.pt", split="validation", out_dir=tmp_path / "reloaded")
    original = load_predictions(out / "validation_predictions.npz")
    reloaded = load_predictions(tmp_path / "reloaded" / "validation_predictions.npz")
    np.testing.assert_allclose(original["scores"], reloaded["scores"], atol=1e-7)


@pytest.mark.parametrize("section,key,value", [
    ("instrument", "output_dim", 42),
    ("shared_encoder", "pool_sizes", [[2, 2], [2, 2]]),
    ("shared_encoder", "channels", [30, 64, 96]),
    ("rhythm", "kernel_size", 4),
    ("timbre", "dropout", 1),
    ("harmony", "chord_head_enabled", True),
])
def test_invalid_architecture_changes_fail_before_training(section, key, value):
    architecture = load_architecture()
    architecture[section][key] = value
    with pytest.raises(ValueError):
        load_architecture(architecture)


def test_unknown_architecture_key_is_not_silently_ignored():
    architecture = load_architecture()
    architecture["instrument"]["dropuot"] = 0.1
    with pytest.raises(ValueError, match="unknown fields"):
        load_architecture(architecture)
