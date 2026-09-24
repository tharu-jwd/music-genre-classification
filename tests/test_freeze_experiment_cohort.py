import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.freeze_experiment_cohort import freeze_cohort, parse_limits


def write_manifest(path: Path) -> None:
    rows = [
        {"song_id": "0000001", "split": "train", "harmony_available": "true"},
        {"song_id": "0000002", "split": "train", "harmony_available": "false"},
        {"song_id": "0000003", "split": "train", "harmony_available": "1"},
        {"song_id": "0000004", "split": "validation", "harmony_available": "true"},
        {"song_id": "0000005", "split": "validation", "harmony_available": "true"},
        {"song_id": "0000006", "split": "test", "harmony_available": "true"},
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


class FreezeExperimentCohortTest(unittest.TestCase):
    def test_selection_is_deterministic_and_filters_availability(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest = Path(temporary_directory) / "manifest.csv"
            write_manifest(manifest)
            first = freeze_cohort(
                manifest,
                splits=("train", "validation"),
                required_available=("harmony_available",),
                limits={"train": 1, "validation": 1},
                seed=17,
            )
            second = freeze_cohort(
                manifest,
                splits=("train", "validation"),
                required_available=("harmony_available",),
                limits={"train": 1, "validation": 1},
                seed=17,
            )
            self.assertEqual(first["splits"], second["splits"])
            self.assertEqual(first["cohort_sha256"], second["cohort_sha256"])
            self.assertNotIn("0000002", first["splits"]["train"])
            self.assertEqual(first["counts"], {"train": 1, "validation": 1})

    def test_source_manifest_fingerprint_changes_with_manifest(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest = Path(temporary_directory) / "manifest.csv"
            write_manifest(manifest)
            before = freeze_cohort(manifest)
            manifest.write_text(manifest.read_text() + "0000007,test,true\n")
            after = freeze_cohort(manifest)
            self.assertNotEqual(
                before["source_manifest"]["sha256"],
                after["source_manifest"]["sha256"],
            )

    def test_invalid_availability_value_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest = Path(temporary_directory) / "manifest.csv"
            write_manifest(manifest)
            manifest.write_text(manifest.read_text().replace("false", "unknown"))
            with self.assertRaisesRegex(ValueError, "must be true/false"):
                freeze_cohort(manifest, required_available=("harmony_available",))

    def test_parse_limits_rejects_duplicate_split(self):
        with self.assertRaisesRegex(ValueError, "invalid or duplicate"):
            parse_limits(["train=10", "train=20"])


if __name__ == "__main__":
    unittest.main()
