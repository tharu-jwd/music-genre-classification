import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from scripts.prepare_chord_benchmark_source import prepare_source_manifest
from scripts import generate_essentia_chord_estimates as teacher_runner
from scripts import prepare_chord_benchmark_source as source_builder


class PrepareChordBenchmarkSourceTest(unittest.TestCase):
    def test_resource_contract_matches_teacher_runner(self):
        self.assertEqual(source_builder.SOURCE_SCHEMA, teacher_runner.SOURCE_SCHEMA)
        self.assertEqual(source_builder.MAX_TRACKS, teacher_runner.MAX_TRACKS)
        self.assertEqual(
            source_builder.MAX_TOTAL_AUDIO_SECONDS,
            teacher_runner.MAX_TOTAL_AUDIO_SECONDS,
        )

    def write_fixture(self, root: Path, *, duplicate: bool = False) -> Path:
        rows = []
        for index in range(2):
            audio = root / f"audio-{index}.wav"
            reference = root / f"reference-{index}.lab"
            sf.write(audio, np.zeros(8000, dtype=np.float32), 8000)
            reference.write_text("0 1 N\n")
            rows.append({
                "track_id": "track-0" if duplicate else f"track-{index}",
                "audio": audio.name,
                "reference": reference.name,
            })
        mapping = root / "mapping.csv"
        with mapping.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("track_id", "audio", "reference"))
            writer.writeheader()
            writer.writerows(rows)
        return mapping

    def test_freezes_paths_hashes_and_resource_summary_without_label_parsing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mapping = self.write_fixture(root)
            output = root / "source.json"
            result = prepare_source_manifest(
                mapping,
                output,
                root=root,
                benchmark_name="existing fixture",
                benchmark_source="published benchmark",
                benchmark_license="test-only",
            )
            saved = json.loads(output.read_text())
            self.assertEqual(result["schema_version"], "chord_teacher_source_v1")
            self.assertEqual(saved["track_count"], 2)
            self.assertEqual(saved["total_audio_seconds"], 2.0)
            self.assertEqual(
                saved["mapping_csv_sha256"], hashlib.sha256(mapping.read_bytes()).hexdigest()
            )
            self.assertRegex(saved["tracks"][0]["audio_sha256"], r"^[0-9a-f]{64}$")
            self.assertEqual(
                saved["reference_usage"],
                "opaque_hash_and_evaluation_only_not_teacher_inference",
            )

    def test_duplicate_track_ids_fail_before_output_is_created(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mapping = self.write_fixture(root, duplicate=True)
            output = root / "source.json"
            with self.assertRaisesRegex(ValueError, "duplicate track_id"):
                prepare_source_manifest(
                    mapping,
                    output,
                    root=root,
                    benchmark_name="fixture",
                    benchmark_source="fixture",
                    benchmark_license="fixture",
                )
            self.assertFalse(output.exists())

    def test_existing_manifest_is_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mapping = self.write_fixture(root)
            output = root / "source.json"
            output.write_text("already registered")
            with self.assertRaisesRegex(FileExistsError, "immutable"):
                prepare_source_manifest(
                    mapping,
                    output,
                    root=root,
                    benchmark_name="fixture",
                    benchmark_source="fixture",
                    benchmark_license="fixture",
                )


if __name__ == "__main__":
    unittest.main()
