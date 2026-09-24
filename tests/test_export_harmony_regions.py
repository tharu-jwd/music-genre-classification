import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.export_harmony_regions import export_regions


def write_fixture(root: Path):
    audio = root / "audio" / "0000001.wav"
    mel = root / "mels" / "0000001.npy"
    audio.parent.mkdir()
    mel.parent.mkdir()
    audio.write_bytes(b"fixture")
    np.save(mel, np.ones((96, 1400), dtype=np.float32))
    manifest = root / "manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "song_id", "split", "audio_path", "logmel_path", "waveform_available",
        ))
        writer.writeheader()
        writer.writerow({
            "song_id": "0000001",
            "split": "train",
            "audio_path": "audio/0000001.wav",
            "logmel_path": "mels/0000001.npy",
            "waveform_available": "true",
        })
    splits = {"train": ["0000001"], "validation": ["0000002"]}
    canonical = json.dumps(splits, sort_keys=True, separators=(",", ":")).encode()
    cohort = root / "cohort.json"
    cohort.write_text(json.dumps({
        "schema_version": "experiment_cohort_v1",
        "source_manifest": {
            "path": str(manifest),
            "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        },
        "selection": {},
        "splits": splits,
        "counts": {"train": 1, "validation": 1},
        "cohort_sha256": hashlib.sha256(canonical).hexdigest(),
    }))
    return manifest, cohort


class ExportHarmonyRegionsTest(unittest.TestCase):
    def test_exports_exact_window_boundaries(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest, cohort = write_fixture(root)
            # Add the required validation fixture to keep the cohort schema meaningful.
            (root / "audio" / "0000002.wav").write_bytes(b"fixture")
            np.save(root / "mels" / "0000002.npy", np.ones((96, 100), dtype=np.float32))
            with manifest.open("a") as handle:
                handle.write("0000002,validation,audio/0000002.wav,mels/0000002.npy,true\n")
            cohort_data = json.loads(cohort.read_text())
            cohort_data["source_manifest"]["sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
            cohort.write_text(json.dumps(cohort_data))

            artifact = export_regions(manifest, cohort, root=root)
            self.assertEqual(len(artifact["songs"]), 2)
            first_regions = artifact["songs"][0]["regions"]
            self.assertEqual([item["frame_start"] for item in first_regions], [0, 1366])
            self.assertEqual(first_regions[1]["valid_frames"], 34)

    def test_test_split_is_rejected_by_default(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest, cohort = write_fixture(root)
            data = json.loads(cohort.read_text())
            data["splits"] = {"train": ["0000001"], "validation": ["0000002"], "test": ["0000003"]}
            data["counts"] = {split: len(ids) for split, ids in data["splits"].items()}
            canonical = json.dumps(data["splits"], sort_keys=True, separators=(",", ":")).encode()
            data["cohort_sha256"] = hashlib.sha256(canonical).hexdigest()
            cohort.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "test IDs are forbidden"):
                export_regions(manifest, cohort, root=root)

    def test_hard_song_cap_cannot_be_raised(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest, cohort = write_fixture(root)
            with self.assertRaisesRegex(ValueError, r"max_songs must be in \[1, 32\]"):
                export_regions(manifest, cohort, root=root, max_songs=33)

    def test_region_subsampling_is_even_and_preserves_original_window_indices(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest, cohort = write_fixture(root)
            (root / "audio" / "0000002.wav").write_bytes(b"fixture")
            np.save(root / "mels" / "0000002.npy", np.ones((96, 100), dtype=np.float32))
            # Ten complete model windows produce original indices 0 through 9.
            np.save(root / "mels" / "0000001.npy", np.ones((96, 13660), dtype=np.float32))
            with manifest.open("a") as handle:
                handle.write("0000002,validation,audio/0000002.wav,mels/0000002.npy,true\n")
            cohort_data = json.loads(cohort.read_text())
            cohort_data["source_manifest"]["sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
            cohort.write_text(json.dumps(cohort_data))

            artifact = export_regions(manifest, cohort, root=root, regions_per_song=3)
            first = artifact["songs"][0]
            self.assertEqual(first["total_model_regions"], 10)
            self.assertEqual([region["window_index"] for region in first["regions"]], [0, 4, 9])
            self.assertEqual(
                artifact["region_selection"]["method"],
                "evenly_spaced_model_windows_v1",
            )


if __name__ == "__main__":
    unittest.main()
