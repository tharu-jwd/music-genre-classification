import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf

from generate_essentia_chord_estimates import (
    FRAME_SIZE,
    HOP_LENGTH,
    TEACHER_NAME,
    canonicalize_essentia_label,
    frame_labels_to_intervals,
    generate,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class GenerateEssentiaChordEstimatesTest(unittest.TestCase):
    def test_essentia_documented_frame_hop_contract_is_respected(self):
        self.assertEqual(FRAME_SIZE, 2 * HOP_LENGTH)

    def test_label_mapping_covers_flats_minor_and_unknown(self):
        self.assertEqual(canonicalize_essentia_label("Bb"), "A#:maj")
        self.assertEqual(canonicalize_essentia_label("Ebm"), "D#:min")
        self.assertEqual(canonicalize_essentia_label("X"), "N")

    def test_frame_intervals_merge_and_cover_exact_duration(self):
        result = frame_labels_to_intervals(
            ["C:maj", "C:maj", "G:maj"], np.array([0.1, 0.2, 0.3]), 0.25
        )
        self.assertEqual([item[2] for item in result], ["C:maj", "G:maj"])
        np.testing.assert_allclose(
            [(item[0], item[1]) for item in result], [(0.0, 0.2), (0.2, 0.25)]
        )

    def test_end_to_end_silence_is_no_chord_without_fake_confidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "silence.wav"
            reference = root / "reference.lab"
            sf.write(audio, np.zeros(22050, dtype=np.float32), 22050)
            reference.write_text("0 1 C:maj\n")
            source = root / "source.json"
            source.write_text(json.dumps({
                "schema_version": "chord_teacher_source_v1",
                "benchmark": {
                    "name": "fixture",
                    "source": "test fixture",
                    "license": "test-only",
                },
                "tracks": [{
                    "track_id": "silent-track",
                    "audio": audio.name,
                    "audio_sha256": sha256(audio),
                    "reference": reference.name,
                    "reference_sha256": sha256(reference),
                }],
            }))
            output = root / "run"
            report = generate(source, output)
            manifest = json.loads((output / "evaluation-manifest.json").read_text())
            estimate = output / manifest["tracks"][0]["estimate"]

            self.assertEqual(report["status"], "ready_for_evaluation")
            self.assertEqual(manifest["teacher"]["name"], TEACHER_NAME)
            self.assertEqual(manifest["teacher"]["confidence"], "none")
            self.assertEqual(estimate.read_text().split()[-1], "N")
            self.assertFalse(report["tracks"][0]["essentia_strength_is_confidence"])

    def test_hash_mismatch_fails_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "audio.wav"
            reference = root / "reference.lab"
            sf.write(audio, np.zeros(1000, dtype=np.float32), 22050)
            reference.write_text("0 1 N\n")
            source = root / "source.json"
            source.write_text(json.dumps({
                "schema_version": "chord_teacher_source_v1",
                "benchmark": {"name": "x", "source": "x", "license": "x"},
                "tracks": [{
                    "track_id": "x",
                    "audio": "audio.wav",
                    "audio_sha256": "0" * 64,
                    "reference": "reference.lab",
                    "reference_sha256": sha256(reference),
                }],
            }))
            output = root / "run"
            with self.assertRaisesRegex(ValueError, "does not match"):
                generate(source, output)
            self.assertFalse(output.exists())

    def test_inference_failure_removes_staging_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "audio.wav"
            reference = root / "reference.lab"
            sf.write(audio, np.zeros(22050, dtype=np.float32), 22050)
            reference.write_text("0 1 N\n")
            source = root / "source.json"
            source.write_text(json.dumps({
                "schema_version": "chord_teacher_source_v1",
                "benchmark": {"name": "x", "source": "x", "license": "x"},
                "tracks": [{
                    "track_id": "x",
                    "audio": "audio.wav",
                    "audio_sha256": sha256(audio),
                    "reference": "reference.lab",
                    "reference_sha256": sha256(reference),
                }],
            }))
            output = root / "run"
            with mock.patch(
                "generate_essentia_chord_estimates.estimate_chords",
                side_effect=RuntimeError("inference failed"),
            ):
                with self.assertRaisesRegex(RuntimeError, "inference failed"):
                    generate(source, output)
            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob(".run.*")), [])


if __name__ == "__main__":
    unittest.main()
