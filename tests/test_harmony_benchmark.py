import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf

from scripts.benchmark_harmony_extractors import (
    _temporal_chroma_agreement,
    benchmark_paths,
    benchmark_region_artifact,
)


class HarmonyBenchmarkTest(unittest.TestCase):
    def test_temporal_agreement_detects_reversed_order_with_the_same_global_mean(self):
        forward = SimpleNamespace(
            timestamps=np.array([0.0, 1.0]),
            chroma=np.eye(12, dtype=np.float32)[[0, 2]],
            valid=np.array([True, True]),
            hop_length=1,
            sample_rate=1,
        )
        reversed_order = SimpleNamespace(
            timestamps=np.array([0.0, 1.0]),
            chroma=np.eye(12, dtype=np.float32)[[2, 0]],
            valid=np.array([True, True]),
            hop_length=1,
            sample_rate=1,
        )
        self.assertTrue(np.allclose(forward.chroma.mean(0), reversed_order.chroma.mean(0)))
        agreements, pairs = _temporal_chroma_agreement(forward, reversed_order)
        self.assertEqual(pairs, 2)
        self.assertEqual(agreements, [0.0, 0.0])

    @staticmethod
    def write_region_artifact(path: Path, audio_path: Path, *, split="train"):
        path.write_text(json.dumps({
            "schema_version": "harmony_audio_regions_v1",
            "songs": [{
                "song_id": "0000001",
                "split": split,
                "audio_path": str(audio_path),
                "regions": [
                    {"window_index": 0, "start_seconds": 0.0, "end_seconds": 0.5},
                    {"window_index": 1, "start_seconds": 1.0, "end_seconds": 1.5},
                ],
            }],
        }))

    def test_caps_are_enforced_before_extraction(self):
        with self.assertRaisesRegex(ValueError, "cap is 1"):
            benchmark_paths([Path("a.wav"), Path("b.wav")], max_files=1)

    def test_registered_hard_caps_cannot_be_raised(self):
        with self.assertRaisesRegex(ValueError, r"max_files must be in \[1, 32\]"):
            benchmark_paths([Path("never-decoded.wav")], max_files=33)
        with self.assertRaisesRegex(ValueError, r"max_total_seconds must be in"):
            benchmark_paths([Path("never-decoded.wav")], max_total_seconds=601.0)

    def test_small_audio_comparison_reports_both_candidates(self):
        sample_rate = 22050
        time = np.arange(sample_rate // 2, dtype=np.float32) / sample_rate
        audio = (0.2 * np.sin(2 * np.pi * 440 * time)).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tone.wav"
            sf.write(path, audio, sample_rate)
            report = benchmark_paths([path], max_files=1, max_total_seconds=1.0)
        self.assertEqual(report["files"], 1)
        self.assertEqual(report["schema_version"], "harmony_extractor_benchmark_v1")
        self.assertFalse(report["automatic_selection"])
        self.assertEqual(set(report["aggregate"]), {"cqt", "hpcp_harmonic"})
        self.assertEqual(report["aggregate"]["cqt"]["failures"], 0)
        self.assertEqual(report["aggregate"]["hpcp_harmonic"]["failures"], 0)
        self.assertGreater(report["mean_temporal_chroma_cosine_agreement"], 0.9)
        self.assertGreater(report["aligned_valid_frame_pairs"], 0)

    def test_region_artifact_decodes_only_registered_model_regions(self):
        sample_rate = 22050
        time = np.arange(sample_rate * 2, dtype=np.float32) / sample_rate
        audio = (0.2 * np.sin(2 * np.pi * 440 * time)).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_path = root / "tone.wav"
            artifact_path = root / "regions.json"
            sf.write(audio_path, audio, sample_rate)
            self.write_region_artifact(artifact_path, audio_path)
            report = benchmark_region_artifact(
                artifact_path, max_files=1, max_regions=2, max_total_seconds=1.1
            )
        self.assertEqual(report["input_mode"], "aligned_regions")
        self.assertEqual(report["files"], 1)
        self.assertEqual(report["analysis_units"], 2)
        self.assertAlmostEqual(report["total_audio_seconds"], 1.0, places=2)
        self.assertGreater(report["aligned_valid_frame_pairs"], 0)
        self.assertEqual(
            [item["source_region_seconds"] for item in report["items"]],
            [[0.0, 0.5], [1.0, 1.5]],
        )

    def test_region_duration_cap_is_checked_before_decode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_path = root / "not_audio.wav"
            audio_path.write_bytes(b"exists but should not be decoded")
            artifact_path = root / "regions.json"
            self.write_region_artifact(artifact_path, audio_path)
            with self.assertRaisesRegex(ValueError, "aligned regions total"):
                benchmark_region_artifact(
                    artifact_path, max_files=1, max_regions=2, max_total_seconds=0.5
                )

    def test_region_benchmark_rejects_test_songs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_path = root / "exists.wav"
            audio_path.write_bytes(b"fixture")
            artifact_path = root / "regions.json"
            self.write_region_artifact(artifact_path, audio_path, split="test")
            with self.assertRaisesRegex(ValueError, "train and validation"):
                benchmark_region_artifact(artifact_path, max_files=1)

    def test_registered_region_sample_rate_is_frozen(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact_path = Path(tmp) / "regions.json"
            artifact_path.write_text(json.dumps({
                "schema_version": "harmony_audio_regions_v1",
                "songs": [{"song_id": "0000001", "split": "train", "regions": []}],
            }))
            with self.assertRaisesRegex(ValueError, "sample_rate must be 22050"):
                benchmark_region_artifact(artifact_path, sample_rate=44100)


if __name__ == "__main__":
    unittest.main()
