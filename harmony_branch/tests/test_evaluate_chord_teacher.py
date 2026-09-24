import json
import tempfile
import unittest
from pathlib import Path

from evaluate_chord_teacher import CANONICAL_CLASSES, evaluate_manifest


def write_manifest(root: Path, reference: str, estimate: str, *, track_id="song-1") -> Path:
    (root / "reference.lab").write_text(reference)
    (root / "estimate.lab").write_text(estimate)
    path = root / "benchmark.json"
    path.write_text(json.dumps({
        "schema_version": "chord_teacher_benchmark_v1",
        "benchmark": {
            "name": "fixture chords",
            "source": "existing human annotation fixture",
            "license": "test-only",
        },
        "teacher": {
            "name": "fixture teacher",
            "version": "1",
            "settings": {"vocabulary": "major_minor_no_chord"},
            "license": "test-only",
            "runtime_seconds": 0.5,
            "device": "cpu",
            "confidence": "none",
        },
        "tracks": [{
            "track_id": track_id,
            "reference": "reference.lab",
            "estimate": "estimate.lab",
        }],
    }))
    return path


class EvaluateChordTeacherTest(unittest.TestCase):
    def test_perfect_major_minor_no_chord_score(self):
        reference = "0 1 C:maj\n1 2 A:min\n2 3 N\n"
        estimate = "0 1 C\n1 2 A:min/1\n2 3 N\n"
        with tempfile.TemporaryDirectory() as directory:
            manifest = write_manifest(Path(directory), reference, estimate)
            report = evaluate_manifest(manifest)
        self.assertEqual(report["aggregate"]["weighted_chord_accuracy_majmin"], 1.0)
        self.assertEqual(report["aggregate"]["comparable_duration_fraction"], 1.0)
        self.assertEqual(report["aggregate"]["no_chord_f1"], 1.0)
        self.assertEqual(tuple(report["per_reference_class"]), CANONICAL_CLASSES)
        self.assertFalse(report["automatic_selection"])

    def test_out_of_gamut_reference_is_reported_as_uncovered(self):
        reference = "0 1 C:sus4\n1 2 C:maj\n"
        estimate = "0 2 C:maj\n"
        with tempfile.TemporaryDirectory() as directory:
            report = evaluate_manifest(write_manifest(Path(directory), reference, estimate))
        self.assertAlmostEqual(report["aggregate"]["comparable_duration_fraction"], 0.5)
        self.assertEqual(report["aggregate"]["weighted_chord_accuracy_majmin"], 1.0)

    def test_missing_estimate_tail_is_filled_with_no_chord(self):
        reference = "0 1 C:maj\n1 2 N\n"
        estimate = "0 1 C:maj\n"
        with tempfile.TemporaryDirectory() as directory:
            report = evaluate_manifest(write_manifest(Path(directory), reference, estimate))
        self.assertEqual(report["aggregate"]["weighted_chord_accuracy_majmin"], 1.0)
        self.assertEqual(report["aggregate"]["no_chord_recall"], 1.0)

    def test_duplicate_track_ids_are_rejected(self):
        labels = "0 1 C:maj\n"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = write_manifest(root, labels, labels)
            value = json.loads(manifest.read_text())
            value["tracks"].append(dict(value["tracks"][0]))
            manifest.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "duplicate track_id"):
                evaluate_manifest(manifest)

    def test_runtime_and_source_hashes_are_recorded(self):
        reference = "0 2 G:maj\n"
        estimate = "0 2 G\n"
        with tempfile.TemporaryDirectory() as directory:
            report = evaluate_manifest(write_manifest(Path(directory), reference, estimate))
        self.assertEqual(report["teacher"]["audio_to_runtime_ratio"], 4.0)
        self.assertRegex(report["source_manifest_sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(report["track_results"][0]["reference_sha256"], r"^[0-9a-f]{64}$")

    def test_identical_reference_and_estimate_are_rejected_as_possible_leakage(self):
        labels = "0 1 C:maj\n"
        with tempfile.TemporaryDirectory() as directory:
            manifest = write_manifest(Path(directory), labels, labels)
            with self.assertRaisesRegex(ValueError, "possible leakage"):
                evaluate_manifest(manifest)

    def test_teacher_settings_must_be_structured_and_nonempty(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = write_manifest(root, "0 1 C:maj\n", "0 1 C\n")
            value = json.loads(manifest.read_text())
            value["teacher"]["settings"] = {}
            manifest.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "settings must be a non-empty object"):
                evaluate_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
