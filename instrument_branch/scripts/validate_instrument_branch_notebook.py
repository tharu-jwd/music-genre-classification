"""Validate the standalone notebook or audit the representative cohort."""
import argparse
import json
from pathlib import Path
import tempfile

import nbformat
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]


def notebook_scope():
    path = ROOT / "notebooks/03_instrument_branch.ipynb"
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    scope = {}
    for cell in notebook.cells:
        if cell.cell_type == "code" and cell.id != "contract-tests":
            exec(compile(cell.source, f"{path}:{cell.id}", "exec"), scope)
    return scope, notebook


def synthetic_test(scope):
    with tempfile.TemporaryDirectory(prefix="instrument-test-") as temporary:
        root = Path(temporary)
        config = dict(scope["CFG"], output=str(root), manifest=str(root / "manifest.csv"),
                      genre_labels_csv=str(root / "genres_df.csv"),
                      instrument_labels_csv=str(root / "instrument_df.csv"),
                      representations=str(root / "repr.npz"), encoder_provenance=str(root / "encoder.json"),
                      epochs=2, seeds=[17, 42, 73], batch_size=20)
        manifest = []
        genres = []
        for split_index, split in enumerate(scope["SPLITS"]):
            for j in range(87):
                sid = str(split_index * 1000 + j + 1).zfill(7)
                manifest.append({"track_id": "track_" + sid, "split": split})
                genre_row = {"TRACK_ID": "track_" + sid}
                genre_row.update({name: int(k == j % 6) for k, name in enumerate(scope["GENRE_TAGS"])})
                genres.append(genre_row)
        frame = pd.DataFrame(manifest)
        frame.to_csv(config["manifest"], index=False)
        pd.DataFrame(genres).to_csv(config["genre_labels_csv"], index=False)
        labels = []
        for split_index in range(len(scope["SPLITS"])):
            for j in range(41):
                row = {"TRACK_ID": "track_" + str(split_index * 1000 + j + 1).zfill(7)}
                row.update({name: int(k == j) for k, name in enumerate(scope["VOCAB"])})
                labels.append(row)
        pd.DataFrame(labels).to_csv(config["instrument_labels_csv"], index=False)
        rng = np.random.default_rng(42)
        np.savez(config["representations"], song_ids=frame.track_id.to_numpy(dtype=str),
                 song_repr=rng.normal(size=(len(frame), 128)).astype(np.float32),
                 window_repr=rng.normal(size=(len(frame), 2, 128)).astype(np.float32))
        scope["write_json"](config["encoder_provenance"], {
            "encoder_sha256": "synthetic-test-only", "normalization_fit_split": "train", "training_split": "train"})
        _, y, mask, _ = scope["audit_dataset"](config)
        assert mask[:41].all() and not mask[41:87].any()
        ukulele = scope["VOCAB"].index("ukulele")
        assert y[ukulele, ukulele] == 1 and mask[ukulele, ukulele] == 1
        summaries = scope["run_training"](config)
        assert len(summaries) == 3
        model, checkpoint = scope["load_branch"](root / "seed_17/instrument.pt")
        assert checkpoint["fusion_projection_owner"] == "fusion"
        assert checkpoint["branch_version"] == scope["BRANCH_VERSION"]
        assert not any("projection" in key for key in checkpoint["state_dict"])
        assert checkpoint["observed_train_counts"] == [41] * 41
        assert checkpoint["target_prevalence"] == [float(np.float32(1 / 41))] * 41
        scope["evaluate_locked_test"](config, root / "seed_17/instrument.pt")
        # Changing test inputs/targets must not alter fitted weights or thresholds.
        with np.load(config["representations"]) as data:
            arrays = dict(data)
        arrays["song_repr"][frame.split.to_numpy() == "test"] = 1000
        np.savez(config["representations"], **arrays)
        config["seeds"] = [17]
        scope["run_training"](config)
        _, repeated = scope["load_branch"](root / "seed_17/instrument.pt")
        assert repeated["thresholds"] == checkpoint["thresholds"]
        assert all(torch.equal(checkpoint["state_dict"][k], repeated["state_dict"][k]) for k in checkpoint["state_dict"])
        # Reject duplicate IDs and tracks without six-genre labels.
        bad = pd.concat([frame, frame.iloc[:1]])
        bad.to_csv(config["manifest"], index=False)
        try:
            scope["audit_dataset"](config)
        except ValueError:
            pass
        else:
            raise AssertionError("Duplicate manifest IDs accepted")
        frame.loc[0, "track_id"] = "track_9999999"
        frame.to_csv(config["manifest"], index=False)
        try:
            scope["audit_dataset"](config)
        except ValueError:
            pass
        else:
            raise AssertionError("Unlabeled cohort track accepted")
    print("Synthetic workflow passed; synthetic metrics/checkpoints were discarded.")


def cohort_audit(scope):
    data = ROOT.parent / "data"
    root = data / "instrument_audit"
    config = dict(scope["CFG"], output=str(root),
                  manifest=str(data / "track_split_assignments.csv"),
                  genre_labels_csv=str(data / "genres_df.csv"),
                  instrument_labels_csv=str(data / "instrument_df.csv"))
    scope["audit_dataset"](config)
    report = {
        "scope": "Representative six-genre cohort with 41 local instrument tags",
        "coverage": pd.read_csv(root / "coverage.csv").to_dict("records"),
        "per_tag": pd.read_csv(root / "coverage_per_tag.csv").to_dict("records"),
        "provenance": json.loads((root / "annotation_provenance.json").read_text()),
    }
    scope["write_json"](ROOT / "docs/instrument-annotation-audit.json", report)
    scope["write_json"](ROOT / "docs/instrument-vocabulary.json", scope["TAGS"])
    print(pd.read_csv(root / "coverage.csv").to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort-audit", action="store_true", help="Audit local six-genre and 41-instrument cohort")
    args = parser.parse_args()
    torch.set_num_threads(1)
    scope, notebook = notebook_scope()
    if args.cohort_audit:
        cohort_audit(scope)
    else:
        tests = next(cell for cell in notebook.cells if cell.id == "contract-tests")
        exec(compile(tests.source, "contract-tests", "exec"), scope)
        synthetic_test(scope)
