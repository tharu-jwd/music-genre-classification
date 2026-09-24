import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from decide_harmony_extractor import decide, main, verify_source_artifact


def report(cqt=None, hpcp=None, agreement=0.9):
    base = {
        "runtime_seconds": 10.0,
        "frames": 100,
        "valid_frames": 80,
        "failures": 0,
        "valid_fraction": 0.8,
    }
    return {
        "schema_version": "harmony_extractor_benchmark_v1",
        "purpose": "bounded_cpu_chroma_candidate_comparison",
        "input_mode": "aligned_regions",
        "automatic_selection": False,
        "source_artifact": "/frozen/regions.json",
        "source_artifact_sha256": "a" * 64,
        "sample_rate": 22050,
        "limits": {"max_files": 32, "max_regions": 384, "max_total_seconds": 600.0},
        "files": 1,
        "analysis_units": 1,
        "total_audio_seconds": 29.1,
        "items": [{"split": "validation"}],
        "mean_temporal_chroma_cosine_agreement": agreement,
        "aligned_valid_frame_pairs": 80,
        "aggregate": {
            "cqt": {**base, **(cqt or {})},
            "hpcp_harmonic": {**base, **(hpcp or {})},
        },
    }


class DecideHarmonyExtractorTest(unittest.TestCase):
    def test_prefers_simpler_cqt_when_both_are_comparable(self):
        result = decide(report())
        self.assertEqual(result["decision"], "select")
        self.assertEqual(result["selected_extractor"], "cqt")

    def test_hpcp_is_fallback_when_cqt_fails(self):
        result = decide(report(cqt={"failures": 1}))
        self.assertEqual(result["selected_extractor"], "hpcp_harmonic")

    def test_disagreement_stops_instead_of_triggering_more_experiments(self):
        result = decide(report(agreement=0.5))
        self.assertEqual(result["decision"], "stop")
        self.assertIsNone(result["selected_extractor"])
        self.assertIn("do not spend GPU", result["next_action"])

    def test_material_cqt_disadvantage_requires_review(self):
        result = decide(report(
            cqt={"runtime_seconds": 25.0, "valid_fraction": 0.6},
            hpcp={"runtime_seconds": 10.0, "valid_fraction": 0.8},
        ))
        self.assertEqual(result["decision"], "stop")
        self.assertIn("review", result["rationale"])

    def test_whole_file_report_cannot_drive_registered_decision(self):
        value = report()
        value["input_mode"] = "whole_files"
        with self.assertRaisesRegex(ValueError, "aligned_regions"):
            decide(value)

    def test_malformed_alignment_artifact_hash_is_rejected(self):
        value = report()
        value["source_artifact_sha256"] = "not-a-sha256"
        with self.assertRaisesRegex(ValueError, "lowercase SHA-256"):
            decide(value)

    def test_resource_caps_are_enforced_again_at_decision_time(self):
        value = report()
        value["total_audio_seconds"] = 600.1
        with self.assertRaisesRegex(ValueError, "at most 600 seconds"):
            decide(value)

    def test_raised_or_inconsistent_report_limits_are_rejected(self):
        value = report()
        value["limits"]["max_files"] = 33
        with self.assertRaisesRegex(ValueError, "max_files exceeds"):
            decide(value)

        value = report()
        value["limits"]["max_regions"] = 0
        with self.assertRaisesRegex(ValueError, "max_regions exceeds"):
            decide(value)

        value = report()
        value["limits"]["max_total_seconds"] = 10.0
        value["total_audio_seconds"] = 11.0
        with self.assertRaisesRegex(ValueError, "measurements exceed"):
            decide(value)

    def test_nonregistered_sample_rate_is_rejected(self):
        value = report()
        value["sample_rate"] = 44100
        with self.assertRaisesRegex(ValueError, "sample_rate must be 22050"):
            decide(value)

    def test_report_without_temporally_aligned_frames_is_rejected(self):
        value = report()
        value["aligned_valid_frame_pairs"] = 0
        with self.assertRaisesRegex(ValueError, "no aligned valid frame pairs"):
            decide(value)

    def test_sparse_temporal_alignment_stops_selection(self):
        value = report()
        value["aligned_valid_frame_pairs"] = 20
        result = decide(value)
        self.assertEqual(result["decision"], "stop")
        self.assertFalse(result["alignment_coverage_gate_passed"])
        self.assertIn("temporal_alignment", result["rationale"])

    def test_written_decision_hashes_the_exact_source_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "report.json"
            output = root / "decision.json"
            artifact = root / "regions.json"
            artifact.write_text("{}")
            value = report()
            value["source_artifact"] = str(artifact)
            value["source_artifact_sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
            source.write_text(json.dumps(value))
            with mock.patch(
                "sys.argv", ["decide_harmony_extractor.py", str(source), "--output", str(output)]
            ):
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(main(), 0)
            saved = json.loads(output.read_text())
            self.assertRegex(saved["source_report_sha256"], r"^[0-9a-f]{64}$")
            self.assertEqual(saved["verified_source_artifact"], str(artifact.resolve()))

    def test_source_artifact_is_rehashed_before_cli_decision(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "regions.json"
            artifact.write_text("original")
            value = report()
            value["source_artifact"] = str(artifact)
            value["source_artifact_sha256"] = "a" * 64
            with self.assertRaisesRegex(ValueError, "hash changed"):
                verify_source_artifact(value)


if __name__ == "__main__":
    unittest.main()
