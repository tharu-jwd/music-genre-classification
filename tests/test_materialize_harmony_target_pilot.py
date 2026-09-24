import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from scripts.decide_harmony_extractor import decide
from scripts.materialize_harmony_target_pilot import materialize_pilot


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_gate(root: Path, region_path: Path, *, decision_value="select") -> Path:
    stats = {
        "runtime_seconds": 1.0,
        "frames": 100,
        "valid_frames": 80,
        "failures": 0,
        "valid_fraction": 0.8,
    }
    report = {
        "schema_version": "harmony_extractor_benchmark_v1",
        "purpose": "bounded_cpu_chroma_candidate_comparison",
        "input_mode": "aligned_regions",
        "automatic_selection": False,
        "source_artifact": str(region_path),
        "source_artifact_sha256": sha256(region_path),
        "sample_rate": 22050,
        "limits": {"max_files": 32, "max_regions": 384, "max_total_seconds": 600.0},
        "files": 1,
        "analysis_units": 1,
        "total_audio_seconds": 1.0,
        "items": [{"split": "train"}],
        "mean_temporal_chroma_cosine_agreement": 0.95,
        "aligned_valid_frame_pairs": 80,
        "aggregate": {"cqt": stats, "hpcp_harmonic": stats},
    }
    report_path = root / "benchmark.json"
    report_path.write_text(json.dumps(report))
    decision = decide(report)
    decision["source_report"] = str(report_path.resolve())
    decision["source_report_sha256"] = sha256(report_path)
    if decision_value != "select":
        decision["decision"] = decision_value
        decision["selected_extractor"] = None
    decision_path = root / "decision.json"
    decision_path.write_text(json.dumps(decision))
    return decision_path


def write_region(root: Path, audio_path: Path, *, split="train", end=1.0) -> Path:
    path = root / "regions.json"
    path.write_text(json.dumps({
        "schema_version": "harmony_audio_regions_v1",
        "cohort_sha256": "c" * 64,
        "songs": [{
            "song_id": "0000001",
            "split": split,
            "audio_path": str(audio_path),
            "regions": [{
                "window_index": 0,
                "start_seconds": 0.0,
                "end_seconds": end,
            }],
        }],
    }))
    return path


class MaterializeHarmonyTargetPilotTest(unittest.TestCase):
    def test_materializes_selected_chroma_as_masked_pseudo_labels(self):
        sample_rate = 22050
        time = np.arange(sample_rate, dtype=np.float32) / sample_rate
        audio = (0.2 * np.sin(2 * np.pi * 440 * time)).astype(np.float32)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio_path = root / "tone.wav"
            sf.write(audio_path, audio, sample_rate)
            region_path = write_region(root, audio_path)
            decision_path = write_gate(root, region_path)
            output = root / "targets"

            result = materialize_pilot(region_path, decision_path, output)
            arrays = np.load(output / result["items"][0]["artifact"], allow_pickle=False)
            repeated = materialize_pilot(region_path, decision_path, root / "targets-repeat")

            self.assertEqual(result["status"], "ready")
            self.assertEqual(result["schema_version"], "harmony_target_pilot_v1")
            self.assertEqual(result["target_variant"], "temporal_chroma_v1")
            self.assertFalse(result["contains_genre_labels"])
            self.assertFalse(result["contains_chord_targets"])
            self.assertEqual(arrays["chroma"].shape[1], 12)
            self.assertEqual(arrays["chroma_valid"].dtype, np.bool_)
            self.assertTrue(np.all(np.diff(arrays["frame_times_seconds"]) > 0))
            self.assertLessEqual(float(arrays["frame_times_seconds"].max()), 1.0)
            self.assertRegex(result["items"][0]["array_content_sha256"], r"^[0-9a-f]{64}$")
            self.assertEqual(
                result["items"][0]["array_content_sha256"],
                repeated["items"][0]["array_content_sha256"],
            )
            self.assertTrue((output / "index.json").is_file())

    def test_rejects_an_unselected_extractor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio_path = root / "audio.wav"
            audio_path.write_bytes(b"not decoded")
            region_path = write_region(root, audio_path)
            decision_path = write_gate(root, region_path, decision_value="stop")
            with self.assertRaisesRegex(ValueError, "successful extractor selection"):
                materialize_pilot(region_path, decision_path, root / "targets")

    def test_rejects_test_split_before_audio_decode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio_path = root / "audio.wav"
            audio_path.write_bytes(b"not decoded")
            region_path = write_region(root, audio_path, split="test")
            decision_path = write_gate(root, region_path)
            with self.assertRaisesRegex(ValueError, "train and validation"):
                materialize_pilot(region_path, decision_path, root / "targets")

    def test_rejects_changed_region_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio_path = root / "audio.wav"
            audio_path.write_bytes(b"not decoded")
            region_path = write_region(root, audio_path)
            decision_path = write_gate(root, region_path)
            region_path.write_text(region_path.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "does not match"):
                materialize_pilot(region_path, decision_path, root / "targets")

    def test_records_decode_failure_without_writing_fake_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio_path = root / "broken.wav"
            audio_path.write_bytes(b"not audio")
            region_path = write_region(root, audio_path)
            decision_path = write_gate(root, region_path)
            output = root / "targets"
            result = materialize_pilot(region_path, decision_path, output)
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["materialized_regions"], 0)
            self.assertEqual(result["failure_count"], 1)
            self.assertEqual(list(output.glob("*.npz")), [])


if __name__ == "__main__":
    unittest.main()
