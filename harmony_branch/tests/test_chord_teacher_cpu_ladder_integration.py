import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from decide_chord_teacher import decide, policy_template_for_source
from evaluate_chord_teacher import evaluate_manifest
from generate_essentia_chord_estimates import TEACHER_NAME, generate
from prepare_chord_benchmark_source import prepare_source_manifest


class ChordTeacherCPULadderIntegrationTest(unittest.TestCase):
    def test_registered_mapping_runs_through_inference_evaluation_and_decision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample_rate = 22050
            seconds = 1.4
            time = np.arange(round(sample_rate * seconds), dtype=np.float32) / sample_rate
            signals = (
                sum(np.sin(2 * np.pi * f * time) for f in (261.6256, 329.6276, 391.9954)),
                sum(np.sin(2 * np.pi * f * time) for f in (220.0, 261.6256, 329.6276)),
                np.zeros_like(time),
            )
            labels = ("C:maj", "A:min", "N")
            rows = []
            for index, (signal, label) in enumerate(zip(signals, labels), start=1):
                audio = root / f"track-{index}.wav"
                reference = root / f"track-{index}.lab"
                sf.write(audio, np.asarray(signal, dtype=np.float32) * 0.1, sample_rate)
                reference.write_text(f"0 {seconds} {label}\n")
                rows.append({
                    "track_id": f"track-{index}",
                    "audio": audio.name,
                    "reference": reference.name,
                })
            mapping = root / "mapping.csv"
            with mapping.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=("track_id", "audio", "reference"))
                writer.writeheader()
                writer.writerows(rows)

            source_path = root / "source.json"
            prepare_source_manifest(
                mapping,
                source_path,
                root=root,
                benchmark_name="existing fixture benchmark",
                benchmark_source="test fixture",
                benchmark_license="test-only",
            )

            # Freeze thresholds and the exact reference hashes before inference.
            policy = policy_template_for_source(source_path)
            policy.update({
                "registered_at": "2026-09-24T10:00:00+05:30",
                "approved_by": "integration-test",
                "reports_not_seen": True,
                "candidates": [{"name": TEACHER_NAME, "role": "simple_baseline"}],
                "allowed_devices": ["cpu"],
                "thresholds": {
                    "min_weighted_chord_accuracy_majmin": 0.1,
                    "min_comparable_duration_fraction": 1.0,
                    "min_no_chord_f1": 1.0,
                    "min_audio_to_runtime_ratio": 0.0001,
                    "min_wca_improvement_over_simple": 0.01,
                    "max_coverage_drop_vs_simple": 0.0,
                    "max_no_chord_f1_drop_vs_simple": 0.0,
                    "require_confidence": False,
                },
            })

            run_dir = root / "teacher-run"
            generation = generate(source_path, run_dir)
            self.assertEqual(generation["status"], "ready_for_evaluation")
            report = evaluate_manifest(run_dir / "evaluation-manifest.json")
            self.assertEqual(report["teacher"]["name"], TEACHER_NAME)
            self.assertEqual(report["tracks"], 3)
            self.assertEqual(report["aggregate"]["comparable_duration_fraction"], 1.0)
            self.assertEqual(report["aggregate"]["no_chord_f1"], 1.0)

            decision = decide(policy, {TEACHER_NAME: report})
            self.assertEqual(decision["schema_version"], "chord_teacher_decision_v1")
            self.assertEqual(decision["decision"], "select")
            self.assertEqual(decision["selected_teacher"], TEACHER_NAME)


if __name__ == "__main__":
    unittest.main()
