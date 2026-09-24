import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.decide_chord_teacher import decide, policy_template_for_source, validate_policy


def policy(*, require_confidence=False):
    return {
        "schema_version": "chord_teacher_policy_v1",
        "registered_at": "2026-09-24T09:00:00+05:30",
        "approved_by": "team-review",
        "reports_not_seen": True,
        "benchmark_name": "fixture benchmark",
        "benchmark_source_manifest_sha256": "f" * 64,
        "benchmark_tracks": [{"track_id": "track-1", "reference_sha256": "a" * 64}],
        "candidates": [
            {"name": "simple", "role": "simple_baseline"},
            {"name": "alternative", "role": "alternative"},
        ],
        "allowed_devices": ["cpu"],
        "thresholds": {
            "min_weighted_chord_accuracy_majmin": 0.60,
            "min_comparable_duration_fraction": 0.90,
            "min_no_chord_f1": 0.50,
            "min_audio_to_runtime_ratio": 1.0,
            "min_wca_improvement_over_simple": 0.05,
            "max_coverage_drop_vs_simple": 0.02,
            "max_no_chord_f1_drop_vs_simple": 0.03,
            "require_confidence": require_confidence,
        },
    }


def report(
    name,
    *,
    wca=0.70,
    coverage=0.95,
    no_chord_f1=0.80,
    speed=2.0,
    confidence="none",
    reference_hash="a" * 64,
):
    return {
        "schema_version": "chord_teacher_evaluation_v1",
        "purpose": "external_human_annotated_chord_teacher_evaluation",
        "automatic_selection": False,
        "benchmark": {"name": "fixture benchmark"},
        "teacher": {
            "name": name,
            "device": "cpu",
            "confidence": confidence,
            "audio_to_runtime_ratio": speed,
        },
        "tracks": 1,
        "aggregate": {
            "weighted_chord_accuracy_majmin": wca,
            "comparable_duration_fraction": coverage,
            "no_chord_f1": no_chord_f1,
        },
        "track_results": [{
            "track_id": "track-1",
            "reference_sha256": reference_hash,
            "duration_seconds": 120.0,
        }],
    }


class DecideChordTeacherTest(unittest.TestCase):
    def test_simple_teacher_is_preferred_for_a_marginal_difference(self):
        result = decide(policy(), {
            "simple": report("simple", wca=0.70),
            "alternative": report("alternative", wca=0.73),
        })
        self.assertEqual(result["decision"], "select")
        self.assertEqual(result["selected_teacher"], "simple")

    def test_alternative_must_materially_improve_without_regression(self):
        result = decide(policy(), {
            "simple": report("simple", wca=0.70, coverage=0.95, no_chord_f1=0.80),
            "alternative": report(
                "alternative", wca=0.77, coverage=0.94, no_chord_f1=0.78
            ),
        })
        self.assertEqual(result["selected_teacher"], "alternative")

    def test_chord_supervision_is_rejected_when_both_fail(self):
        result = decide(policy(), {
            "simple": report("simple", wca=0.40),
            "alternative": report("alternative", wca=0.50),
        })
        self.assertEqual(result["decision"], "reject_chord_supervision")
        self.assertIsNone(result["selected_teacher"])
        self.assertIn("do not expand teachers", result["next_action"])

    def test_confidence_requirement_is_enforced(self):
        result = decide(policy(require_confidence=True), {
            "simple": report("simple", confidence="none"),
            "alternative": report("alternative", confidence="none"),
        })
        self.assertEqual(result["decision"], "reject_chord_supervision")
        self.assertIn("confidence_required_but_unavailable", result["ineligibility_reasons"]["simple"])

    def test_reports_must_share_exact_reference_tracks(self):
        with self.assertRaisesRegex(ValueError, "preregistered benchmark tracks"):
            decide(policy(), {
                "simple": report("simple"),
                "alternative": report("alternative", reference_hash="b" * 64),
            })

    def test_policy_allows_at_most_one_alternative(self):
        value = policy()
        value["candidates"].append({"name": "third", "role": "alternative"})
        with self.assertRaisesRegex(ValueError, "one or two candidates"):
            validate_policy(value)

    def test_material_improvement_threshold_must_be_positive(self):
        value = policy()
        value["thresholds"]["min_wca_improvement_over_simple"] = 0.0
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            validate_policy(value)

    def test_quality_thresholds_cannot_be_zero_rubber_stamps(self):
        for field in (
            "min_weighted_chord_accuracy_majmin",
            "min_comparable_duration_fraction",
            "min_no_chord_f1",
        ):
            with self.subTest(field=field):
                value = policy()
                value["thresholds"][field] = 0.0
                with self.assertRaisesRegex(ValueError, "greater than zero"):
                    validate_policy(value)

    def test_missing_no_chord_metric_rejects_instead_of_crashing(self):
        result = decide(policy(), {
            "simple": report("simple", no_chord_f1=None),
            "alternative": report("alternative", no_chord_f1=None),
        })
        self.assertEqual(result["decision"], "reject_chord_supervision")
        self.assertIn("no_chord_f1_below_minimum", result["ineligibility_reasons"]["simple"])

    def test_decision_has_a_versioned_schema(self):
        value = policy()
        value["candidates"] = [{"name": "simple", "role": "simple_baseline"}]
        result = decide(value, {"simple": report("simple")})
        self.assertEqual(result["schema_version"], "chord_teacher_decision_v1")

    def test_policy_template_is_pinned_to_source_manifest_tracks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.json"
            path.write_text(json.dumps({
                "schema_version": "chord_teacher_source_v1",
                "benchmark": {"name": "fixture benchmark"},
                "tracks": [{"track_id": "track-1", "reference_sha256": "a" * 64}],
            }))
            template = policy_template_for_source(path)
            self.assertEqual(template["benchmark_name"], "fixture benchmark")
            self.assertEqual(template["benchmark_tracks"][0]["track_id"], "track-1")
            self.assertEqual(
                template["benchmark_source_manifest_sha256"],
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )


if __name__ == "__main__":
    unittest.main()
