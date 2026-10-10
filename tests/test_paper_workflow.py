"""Real train/reload/export checks and paper protocol validation."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from scripts import train_joint as j
from scripts.evaluate_checkpoint import evaluate_checkpoint, restore
from scripts.paper_experiments import make_manifest, load_manifest, selected_runs
from scripts.paper_artifacts import write_json, sha256
from scripts.paired_bootstrap_compare import load_predictions, macro_metric
from scripts.analyze_paper_runs import analyze, paired_seed_average_bootstrap
from tests.test_modular_training import write_dataset


@pytest.mark.parametrize("branches,weight", [((), 0), (j.CONCEPT_ORDER, 1), (j.CONCEPT_ORDER, 0)])
def test_checkpoint_only_evaluation_reproduces_training_and_preserves_weights(tmp_path, branches, weight):
    data = tmp_path / "data"
    write_dataset(data, j.CONCEPT_ORDER)
    frame = pd.read_csv(data / 'dataset.csv', dtype={"TRACK_ID": str})
    frame.loc[0, j.INSTRUMENT_TAGS[0]] = np.nan
    frame.loc[4, j.INSTRUMENT_TAGS[0]] = np.nan
    frame.to_csv(data / 'dataset.csv', index=False)
    out = tmp_path / "run"
    cfg = j.TrainConfig(branches=branches, lambda_aux=weight, data_dir=data,
        dataset_csv=data / "dataset.csv", out_dir=str(out), device="cpu", epochs=1,
        batch_size=2, max_windows=1, window_frames=16, evaluate_test=False)
    j.train(cfg)
    before = (out / "best.pt").read_bytes()
    if branches:
        trained = torch.load(out / "best.pt", weights_only=False)
        assert trained["harmony_supervised"] == bool(weight)
        assert trained["harmony_targets_available"]
        _, _, _, restored = restore(out / "best.pt")
        assert not restored["harmony_head"].chroma_head.weight.requires_grad
    report = evaluate_checkpoint(out / "best.pt", split="validation", out_dir=tmp_path / "reevaluated")
    original = load_predictions(out / "validation_predictions.npz")
    reloaded = load_predictions(tmp_path / "reevaluated" / "validation_predictions.npz")
    np.testing.assert_allclose(original["scores"], reloaded["scores"], atol=1e-7)
    assert report["metrics"]["macro_ap"] == pytest.approx(macro_metric(reloaded["targets"], reloaded["scores"], "macro_pr_auc"))
    assert before == (out / "best.pt").read_bytes()
    evaluate_checkpoint(out / "best.pt", split="test")
    with np.load(out / "test_predictions.npz", allow_pickle=False) as exported:
        for branch in branches:
            assert exported[f"{branch}_mask"].shape == exported[f"{branch}_predictions"].shape
            if branch != "instrument":
                assert f"{branch}_predictions_raw" in exported.files
            assert f"{branch}_baseline_predictions" in exported.files
    assert json.loads((out / "evaluation_test.json").read_text())["prediction_sha256"] == sha256(out / "test_predictions.npz")
    with pytest.raises(FileExistsError):
        evaluate_checkpoint(out / "best.pt", split="test")
    with pytest.raises(FileExistsError):
        j.train(cfg)
    if branches:
        results = json.loads((out / "evaluation_test.json").read_text())
        assert results["metrics"]["branches"]["instrument"]["n_observed"] == 81
    # Changing target data after training must not silently refit/evaluate.
    frame.loc[0, j.GENRE_TAGS[0]] = 1 - frame.loc[0, j.GENRE_TAGS[0]]
    frame.to_csv(data / "dataset.csv", index=False)
    with pytest.raises(ValueError, match="differs from training"):
        restore(out / "best.pt")


def test_fixed_manifest_has_21_unique_runs_and_equal_protocol(tmp_path):
    path = tmp_path / "manifest.json"
    manifest = make_manifest(out_root="/runs", data_dir="/data/dataset",
                             dataset_csv="/data/dataset/vector-dataset-normalized.csv")
    write_json(path, manifest)
    loaded = load_manifest(path)
    assert len(loaded["runs"]) == 21
    assert len(selected_runs(loaded, phase="core")) == 9
    assert len(selected_runs(loaded, phase="ablation", seed=42)) == 4
    assert loaded["runs"][0]["config"]["out_dir"].startswith("/runs/")
    loaded["runs"][0]["config"]["epochs"] = 99
    write_json(path, loaded)
    with pytest.raises(ValueError, match="protocol"):
        load_manifest(path)


def test_analysis_aligns_tracks_and_does_not_ensemble_seeds(tmp_path):
    from scripts.paper_artifacts import export_predictions
    manifest = make_manifest(out_root=str(tmp_path / "runs"), epochs=1)
    path = tmp_path / "manifest.json"
    write_json(path, manifest)
    manifest = load_manifest(path)
    truth = np.tile([[1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1]], (4, 1))
    ids = np.asarray([str(index) for index in range(len(truth))])
    for run in manifest["runs"]:
        folder = Path(run["config"]["out_dir"])
        folder.mkdir(parents=True)
        # Permuted exports must be aligned by ID, including across seeds.
        order = np.arange(len(truth))[::-1] if run["seed"] == 123 else np.arange(len(truth))
        scores = (0.8 * truth + 0.1)[order]
        concepts = {}
        for name in run["config"]["branches"]:
            width = j.ConceptCounts().for_name(name)
            concepts[name] = dict(predictions=np.zeros((len(truth), width)),
                targets=np.zeros((len(truth), width)), mask=np.ones((len(truth), width), bool),
                names=np.asarray([str(i) for i in range(width)]),
                baseline_predictions=np.zeros((len(truth), width)))
        export_predictions(folder / "test_predictions.npz", scores, truth[order], ids[order], j.GENRE_TAGS, concepts)
        write_json(folder / "results.json", dict(metric_version=manifest["metric_version"],
            seed=run["seed"], branches=run["config"]["branches"], config=run["config"],
            parameter_count=100, best_epoch=1, log=[]))
        # Teammate handoffs omit checkpoints; the test report binds the predictions.
        write_json(folder / "evaluation_test.json", dict(checkpoint_sha256="not-downloaded",
            prediction_sha256=sha256(folder / "test_predictions.npz")))
    tables = analyze(path, out_dir=tmp_path / "analysis", n_resamples=100, plots=True)
    assert len(tables["summary"]) == 7
    assert (tables["summary"]["n_seeds"] == 3).all()
    assert np.allclose(tables["paired_differences"]["delta_macro_ap"], 0)
    assert (tmp_path / "analysis" / "genre_performance.pdf").exists()
    assert (tmp_path / "analysis" / "branch_removal_heatmap.pdf").exists()
    # Bootstrap applies AP per seed, rather than AP of an averaged score vector.
    pairs = [(truth, np.full_like(truth, 0.5, dtype=float), 0.8 * truth + 0.1)] * 3
    result = paired_seed_average_bootstrap(pairs, n_resamples=100)
    assert result["mean_seed_difference"] == pytest.approx(0.5)
    with (folder / "test_predictions.npz").open("ab") as artifact:
        artifact.write(b"changed")
    with pytest.raises(ValueError, match="Prediction file differs"):
        analyze(path, out_dir=tmp_path / "bad-analysis", plots=False)


def test_teammate_archives_combine_without_overwriting_or_unexpected_paths(tmp_path):
    import zipfile
    from scripts.paper_handoff import FILES, import_archives

    manifest = make_manifest()
    archives = []
    for seed in (42, 123, 456):
        archive_path = tmp_path / f"seed-{seed}.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            for run in selected_runs(manifest, seed=seed):
                for filename in FILES:
                    archive.writestr(f"{run['run_id']}/{filename}", "test payload")
        archives.append(archive_path)
    output = tmp_path / "combined"
    import_archives(manifest, archives, output)
    assert len(list(output.iterdir())) == 21
    assert len(list(output.rglob("*.*"))) == 63
    with pytest.raises(FileExistsError):
        import_archives(manifest, archives, output)
    with pytest.raises(ValueError, match="overlapping"):
        import_archives(manifest, [archives[0], archives[0]], tmp_path / "duplicate")
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as archive:
        archive.writestr("../outside.json", "invalid")
    with pytest.raises(ValueError, match="Unexpected"):
        import_archives(manifest, [bad], tmp_path / "unsafe")
    assert not (tmp_path / "outside.json").exists()


def test_seed_download_streams_only_selected_reports_and_predictions(tmp_path, monkeypatch):
    import sys
    import types
    import zipfile
    from scripts.paper_handoff import FILES, download_seed

    calls = []
    class FakeVolume:
        @staticmethod
        def from_name(name):
            assert name == "music-genre-runs"
            return FakeVolume()

        def read_file(self, path):
            calls.append(path)
            yield b"first "
            yield b"second"

    monkeypatch.setitem(sys.modules, "modal", types.SimpleNamespace(Volume=FakeVolume))
    output = tmp_path / "seed-123.zip"
    download_seed(make_manifest(), 123, output)
    assert len(calls) == 21
    assert all("-s123/" in path and path.split("/")[-1] in FILES for path in calls)
    with zipfile.ZipFile(output) as archive:
        assert all(archive.read(name) == b"first second" for name in archive.namelist())
    with pytest.raises(FileExistsError):
        download_seed(make_manifest(), 123, output)
    calls.clear()
    individual = tmp_path / "full-s42.zip"
    download_seed(make_manifest(), None, individual, run_name="paper-v2-full-s42")
    assert calls == [f"/paper-v2-full-s42/{filename}" for filename in FILES]
    from scripts.paper_handoff import import_archives
    import_archives(make_manifest(), [individual], tmp_path / "one-run")
    assert len(list((tmp_path / "one-run").iterdir())) == 1
    assert len(list((tmp_path / "one-run").rglob("*.*"))) == 3


def test_pilot_is_bounded_and_uses_no_test_audio(tmp_path):
    from scripts.pilot_paper_run import pilot
    data = tmp_path / "data"
    write_dataset(data, j.CONCEPT_ORDER)
    (data / "4.npy").unlink()
    (data / "5.npy").unlink()
    report = pilot(data_dir=data, dataset_csv=data / "dataset.csv", device="cpu",
                   batches=1, warmup=1, batch_size=2, max_windows=1, window_frames=16,
                   output=tmp_path / "pilot.json")
    assert set(report["configurations"]) == {"cnn", "full", "no-supervision"}
    assert report["configurations"]["full"]["parameter_count"] == report["configurations"]["no-supervision"]["parameter_count"]
    assert not list(tmp_path.rglob("*.pt"))
