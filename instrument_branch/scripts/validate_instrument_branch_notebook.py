"""Validate the standalone notebook or produce an official annotation audit."""
import argparse
import csv
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
                      representations=str(root / "repr.npz"), encoder_provenance=str(root / "encoder.json"),
                      epochs=2, seeds=[17, 42, 73], batch_size=20)
        annotation_dir = root / "annotations"
        annotation_dir.mkdir()
        manifest = []
        for split_index, split in enumerate(scope["SPLITS"]):
            for category, count in [("genre", 87), ("instrument", 40)]:
                path = annotation_dir / f"autotagging_{category}-{split}.tsv"
                with path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle, delimiter="\t")
                    writer.writerow(["TRACK_ID", "ARTIST_ID", "ALBUM_ID", "PATH", "DURATION", "TAGS"])
                    for j in range(count):
                        sid = str(split_index * 1000 + j + 1).zfill(7)
                        tag = scope["TAGS"][j] if category == "instrument" else f"genre---synthetic{j:02d}"
                        writer.writerow(["track_" + sid, "artist_1", "album_1", f"00/{int(sid)}.mp3", "30", tag])
                        if category == "genre":
                            manifest.append({"song_id": sid, "split": split})
        frame = pd.DataFrame(manifest)
        frame.to_csv(config["manifest"], index=False)
        rng = np.random.default_rng(42)
        np.savez(config["representations"], song_ids=frame.song_id.to_numpy(dtype=str),
                 song_repr=rng.normal(size=(len(frame), 128)).astype(np.float32),
                 window_repr=rng.normal(size=(len(frame), 2, 128)).astype(np.float32))
        scope["write_json"](config["encoder_provenance"], {
            "encoder_sha256": "synthetic-test-only", "normalization_fit_split": "train", "training_split": "train"})
        _, y, mask, _ = scope["audit_dataset"](config)
        assert mask[:40].all() and not mask[40:87].any()
        summaries = scope["run_training"](config)
        assert len(summaries) == 3
        model, checkpoint = scope["load_branch"](root / "seed_17/instrument.pt")
        assert checkpoint["fusion_projection_owner"] == "fusion"
        assert checkpoint["branch_version"] == scope["BRANCH_VERSION"]
        assert not any("projection" in key for key in checkpoint["state_dict"])
        assert checkpoint["observed_train_counts"] == [40] * 40
        assert checkpoint["target_prevalence"] == [float(np.float32(1 / 40))] * 40
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
        # Reject duplicate IDs and partition mismatches before any optimization.
        bad = pd.concat([frame, frame.iloc[:1]])
        bad.to_csv(config["manifest"], index=False)
        try:
            scope["audit_dataset"](config)
        except ValueError:
            pass
        else:
            raise AssertionError("Duplicate manifest IDs accepted")
        frame.loc[0, "split"] = "test"
        frame.to_csv(config["manifest"], index=False)
        try:
            scope["audit_dataset"](config)
        except ValueError:
            pass
        else:
            raise AssertionError("Wrong split accepted")
    print("Synthetic workflow passed; synthetic metrics/checkpoints were discarded.")


def official_audit(scope):
    root = ROOT.parent / "data/instrument_audit"
    records = []
    for split in scope["SPLITS"]:
        name = f"autotagging_genre-{split}.tsv"
        path = scope["fetch"](scope["ANN_URL"] + name, root / "annotations" / name)
        rows = scope["read_official"](path, "genre---")
        records.extend({"song_id": sid, "split": split} for sid in rows)
    manifest = root / "official_genre_manifest.csv"
    pd.DataFrame(records).to_csv(manifest, index=False)
    config = dict(scope["CFG"], output=str(root), manifest=str(manifest))
    scope["audit_dataset"](config)
    report = {
        "scope": "Complete official split-0 genre manifest, not local downloaded audio coverage",
        "coverage": pd.read_csv(root / "coverage.csv").to_dict("records"),
        "per_tag": pd.read_csv(root / "coverage_per_tag.csv").to_dict("records"),
        "provenance": json.loads((root / "annotation_provenance.json").read_text()),
    }
    scope["write_json"](ROOT / "docs/instrument-annotation-audit.json", report)
    scope["write_json"](ROOT / "docs/instrument-vocabulary.json", scope["TAGS"])
    print(pd.read_csv(root / "coverage.csv").to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-audit", action="store_true", help="Download official TSVs and save coverage, no model training")
    args = parser.parse_args()
    torch.set_num_threads(1)
    scope, notebook = notebook_scope()
    if args.official_audit:
        official_audit(scope)
    else:
        tests = next(cell for cell in notebook.cells if cell.id == "contract-tests")
        exec(compile(tests.source, "contract-tests", "exec"), scope)
        synthetic_test(scope)
