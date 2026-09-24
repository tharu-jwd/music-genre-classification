import unittest

from scripts.gpu_run_contract import template_record
from scripts.manage_gpu_budget import complete_run, load_gpu_budget_from_object, new_ledger, reserve_run


def approved_record():
    record = template_record()
    record.update({
        "status": "approved",
        "code_commit": "0123456789abcdef",
        "cohort_artifact": "results/cohorts/harmony_screen_seed42.json",
        "artifact_dir": "results/runs/H1_temporal_chroma_seed42",
        "reused_artifacts": [{
            "path": "features/cached_encoder/harmony_screen_seed42.npz",
            "sha256": "e" * 64,
        }],
        "cheaper_checks": [{
            "name": "CPU extractor gate",
            "status": "passed",
            "evidence": "Extractor decision selected an accepted target generator.",
            "artifact": "results/evaluations/harmony_extractor_decision.json",
            "sha256": "c" * 64,
        }],
        "control_artifact": "results/predictions/H0_validation.npz",
        "control_artifact_sha256": "b" * 64,
        "approved_by": "team-review",
        "approved_at": "2026-09-23T10:00:00+05:30",
        "budget_remaining_before_gpu_hours": 4.0,
        "budget_ledger": "gpu_budget.json",
    })
    return record


class ManageGpuBudgetTest(unittest.TestCase):
    def test_reserve_then_complete_accounts_actual_hours(self):
        ledger = new_ledger(4.0)
        ledger = reserve_run(ledger, approved_record())
        _, reserved = load_gpu_budget_from_object(ledger)
        self.assertEqual(reserved["reserved_gpu_hours"], 2.0)
        self.assertEqual(reserved["unreserved_gpu_hours"], 2.0)

        ledger = complete_run(ledger, "H1_temporal_chroma_seed42", 1.25)
        _, completed = load_gpu_budget_from_object(ledger)
        self.assertEqual(completed["used_gpu_hours"], 1.25)
        self.assertEqual(completed["reserved_gpu_hours"], 0.0)
        self.assertEqual(completed["unreserved_gpu_hours"], 2.75)

    def test_second_reservation_uses_current_unreserved_snapshot(self):
        ledger = reserve_run(new_ledger(4.0), approved_record())
        second = approved_record()
        second["run_id"] = "H2_temporal_chroma_chords_seed42"
        second["experiment_id"] = "H2_temporal_chroma_chords"
        second["control_run_id"] = "H1_temporal_chroma_seed42"
        second["budget_remaining_before_gpu_hours"] = 2.0
        ledger = reserve_run(ledger, second)
        _, summary = load_gpu_budget_from_object(ledger)
        self.assertEqual(summary["unreserved_gpu_hours"], 0.0)

    def test_duplicate_experiment_attempt_is_rejected_even_with_new_run_id(self):
        ledger = reserve_run(new_ledger(4.0), approved_record())
        duplicate = approved_record()
        duplicate["run_id"] = "H1_temporal_chroma_duplicate"
        duplicate["budget_remaining_before_gpu_hours"] = 2.0
        with self.assertRaisesRegex(ValueError, "experiment attempts must be unique"):
            reserve_run(ledger, duplicate)

    def test_stale_budget_snapshot_is_rejected(self):
        record = approved_record()
        record["budget_remaining_before_gpu_hours"] = 3.0
        with self.assertRaisesRegex(ValueError, "does not equal"):
            reserve_run(new_ledger(4.0), record)


if __name__ == "__main__":
    unittest.main()
