import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.decide_harmony_branch_screen import (
    decide,
    policy_template_for_dataset,
    validate_policy,
    verify_report_files,
)


def policy():
    return {
        "schema_version": "harmony_branch_screen_policy_v1",
        "registered_at": "2026-09-24T10:00:00+05:30",
        "approved_by": "team-review",
        "reports_not_seen": True,
        "cohort_sha256": "c" * 64,
        "screen_dataset_sha256": "d" * 64,
        "thresholds": {
            "max_validation_cross_entropy": 1.1,
            "min_validation_cosine_similarity": 0.7,
            "min_cross_entropy_improvement_over_training_mean": 0.1,
            "max_cpu_wall_seconds": 60.0,
            "max_parameter_count": 100000,
        },
    }


def report():
    return {
        "schema_version": "harmony_branch_screen_v1",
        "status": "ready",
        "automatic_selection": False,
        "device": "cpu",
        "cohort_sha256": "c" * 64,
        "source_dataset_sha256": "d" * 64,
        "configuration": {
            "seed": 42,
            "embedding_dim": 32,
            "hidden_dim": 64,
            "temporal_layers": 2,
            "dropout": 0.1,
            "learning_rate": 1e-3,
            "patience": 3,
            "max_epochs": 20,
            "max_cpu_seconds": 300,
            "cpu_threads": 4,
        },
        "best_validation": {
            "cross_entropy": 1.0,
            "mean_cosine_similarity": 0.8,
            "valid_tokens": 20,
        },
        "baselines": {"training_mean": {"cross_entropy": 1.2}},
        "elapsed_cpu_wall_seconds": 10.0,
        "parameter_count": 50000,
    }


class DecideHarmonyBranchScreenTest(unittest.TestCase):
    def test_advances_only_after_all_preregistered_cpu_gates_pass(self):
        result = decide(policy(), report())
        self.assertEqual(result["decision"], "advance_to_gpu_registration")
        self.assertEqual(result["target_variant"], "temporal_chroma_v1")
        self.assertEqual(result["cohort_sha256"], "c" * 64)
        self.assertEqual(result["screen_dataset_sha256"], "d" * 64)
        self.assertTrue(all(result["checks"].values()))
        self.assertIn("does not itself authorize CUDA", result["next_action"])

    def test_small_improvement_stops_before_gpu(self):
        value = report()
        value["best_validation"]["cross_entropy"] = 1.15
        result = decide(policy(), value)
        self.assertEqual(result["decision"], "stop_before_gpu")
        self.assertFalse(result["checks"]["improvement_over_training_mean"])

    def test_improvement_threshold_must_be_positive(self):
        value = policy()
        value["thresholds"]["min_cross_entropy_improvement_over_training_mean"] = 0
        with self.assertRaisesRegex(ValueError, "must be positive"):
            validate_policy(value)

    def test_policy_must_be_registered_before_results(self):
        value = policy()
        value["reports_not_seen"] = False
        with self.assertRaisesRegex(ValueError, "reports_not_seen"):
            validate_policy(value)

    def test_report_must_use_exact_fixed_reference_configuration(self):
        value = report()
        value["configuration"]["embedding_dim"] = 64
        with self.assertRaisesRegex(ValueError, "fixed reference configuration"):
            decide(policy(), value)

    def test_report_must_match_registered_dataset(self):
        value = report()
        value["source_dataset_sha256"] = "e" * 64
        with self.assertRaisesRegex(ValueError, "dataset does not match"):
            decide(policy(), value)

    def test_report_files_are_rehashed_before_cli_decision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset.json"
            checkpoint = root / "best.pt"
            report_path = root / "report.json"
            dataset.write_text("dataset")
            checkpoint.write_bytes(b"checkpoint")
            value = report()
            value["source_dataset"] = str(dataset)
            value["source_dataset_sha256"] = hashlib.sha256(dataset.read_bytes()).hexdigest()
            value["checkpoint"] = "best.pt"
            value["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            verified = verify_report_files(value, report_path)
            self.assertEqual(verified["checkpoint"], str(checkpoint.resolve()))
            checkpoint.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "checkpoint is missing or changed"):
                verify_report_files(value, report_path)

    def test_policy_template_is_prefilled_from_exact_ready_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.json"
            path.write_text(json.dumps({
                "schema_version": "harmony_screen_dataset_v1",
                "status": "ready",
                "cohort_sha256": "c" * 64,
            }))
            template = policy_template_for_dataset(path)
            self.assertEqual(template["cohort_sha256"], "c" * 64)
            self.assertEqual(
                template["screen_dataset_sha256"],
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )


if __name__ == "__main__":
    unittest.main()
