import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from benchmark_harmony_extractors import benchmark_region_artifact
from build_harmony_screen_dataset import build_screen_dataset
from cache_harmony_encoder_pilot import cache_encoder_pilot
from decide_harmony_branch_screen import decide as decide_branch
from decide_harmony_extractor import decide as decide_extractor
from export_harmony_regions import export_regions
from scripts.freeze_experiment_cohort import freeze_cohort
from materialize_harmony_target_pilot import materialize_pilot
from screen_temporal_harmony_branch import screen_branch
from shared_encoder import SHARED_ENCODER_ARCHITECTURE, SharedAudioEncoder


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class HarmonyCPULadderIntegrationTest(unittest.TestCase):
    def test_real_artifacts_connect_from_audio_to_preregistered_branch_decision(self):
        """Exercise the complete cheap ladder on two tiny songs without CUDA."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample_rate = 22050
            seconds = 1.4
            time = np.arange(round(sample_rate * seconds), dtype=np.float32) / sample_rate
            chords = (
                (261.6256, 329.6276, 391.9954),  # C major
                (220.0, 261.6256, 329.6276),  # A minor
            )
            rows = []
            for index, (split, frequencies) in enumerate(
                zip(("train", "validation"), chords), start=1
            ):
                song_id = f"{index:07d}"
                audio = sum(
                    np.sin(2 * np.pi * frequency * time) for frequency in frequencies
                ).astype(np.float32) * 0.1
                audio_path = root / f"{song_id}.wav"
                logmel_path = root / f"{song_id}.npy"
                sf.write(audio_path, audio, sample_rate)
                np.save(logmel_path, np.full((96, 60), index, dtype=np.float32))
                rows.append({
                    "song_id": song_id,
                    "split": split,
                    "audio_path": audio_path.name,
                    "logmel_path": logmel_path.name,
                    "waveform_available": "true",
                })

            manifest_path = root / "manifest.csv"
            with manifest_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0])
                writer.writeheader()
                writer.writerows(rows)
            cohort_path = root / "cohort.json"
            cohort_path.write_text(json.dumps(
                freeze_cohort(manifest_path, splits=("train", "validation"))
            ))

            region_path = root / "regions.json"
            region_path.write_text(json.dumps(export_regions(
                manifest_path,
                cohort_path,
                root=root,
                max_songs=2,
                regions_per_song=1,
            )))
            extractor_report = benchmark_region_artifact(
                region_path,
                max_files=2,
                max_regions=2,
                max_total_seconds=10,
            )
            extractor_report_path = root / "extractor-report.json"
            extractor_report_path.write_text(json.dumps(extractor_report))
            extractor_decision = decide_extractor(extractor_report)
            extractor_decision.update({
                "source_report": str(extractor_report_path.resolve()),
                "source_report_sha256": sha256(extractor_report_path),
            })
            extractor_decision_path = root / "extractor-decision.json"
            extractor_decision_path.write_text(json.dumps(extractor_decision))
            self.assertEqual(extractor_decision["decision"], "select")

            target_dir = root / "targets"
            targets = materialize_pilot(region_path, extractor_decision_path, target_dir)
            self.assertEqual(targets["status"], "ready")

            encoder = SharedAudioEncoder()
            checkpoint_path = root / "instrument.pt"
            torch.save({
                "model": encoder.state_dict(),
                "encoder_architecture": SHARED_ENCODER_ARCHITECTURE,
                "best_macro_map": 0.5,
                "tags": ["instrument---guitar"],
                "training_config": {
                    "input_schema": "mtg_full_audio_logmel_windows_v1",
                    "max_windows": 12,
                },
            }, checkpoint_path)
            feature_dir = root / "features"
            features = cache_encoder_pilot(
                manifest_path,
                cohort_path,
                checkpoint_path,
                feature_dir,
                root=root,
                max_songs=2,
                max_cpu_seconds=60,
                cpu_threads=1,
            )
            self.assertEqual(features["status"], "ready")
            self.assertEqual(features["device"], "cpu")

            dataset_dir = root / "screen-dataset"
            dataset = build_screen_dataset(feature_dir, target_dir, dataset_dir)
            self.assertEqual(dataset["status"], "ready")
            self.assertGreater(dataset["coverage"], 0)

            # This is frozen after the dataset exists and before the screen report.
            policy = {
                "schema_version": "harmony_branch_screen_policy_v1",
                "registered_at": "2026-09-24T10:00:00+05:30",
                "approved_by": "integration-test",
                "reports_not_seen": True,
                "cohort_sha256": dataset["cohort_sha256"],
                "screen_dataset_sha256": sha256(dataset_dir / "index.json"),
                "thresholds": {
                    "max_validation_cross_entropy": 10.0,
                    "min_validation_cosine_similarity": 0.0,
                    "min_cross_entropy_improvement_over_training_mean": 1e-6,
                    "max_cpu_wall_seconds": 30.0,
                    "max_parameter_count": 1_000_000,
                },
            }
            run_dir = root / "screen-run"
            screen = screen_branch(
                dataset_dir,
                run_dir,
                max_epochs=1,
                max_cpu_seconds=30,
                cpu_threads=1,
            )
            self.assertEqual(screen["status"], "ready")
            branch_decision = decide_branch(policy, screen)
            self.assertEqual(
                branch_decision["schema_version"], "harmony_branch_screen_decision_v1"
            )
            self.assertEqual(branch_decision["target_variant"], "temporal_chroma_v1")
            self.assertIn(
                branch_decision["decision"], {"advance_to_gpu_registration", "stop_before_gpu"}
            )


if __name__ == "__main__":
    unittest.main()
