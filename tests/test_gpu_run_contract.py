import json
import hashlib
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from scripts.gpu_run_contract import (
    GPUApprovalError,
    apply_approved_cohort,
    approved_run_limits,
    require_gpu_run_approval,
    template_record,
    validate_gpu_run_record,
    validate_repeat_evidence,
    verify_gpu_run_artifacts,
    write_gpu_termination_ledger,
)


def write_cohort(path: Path, splits=None, manifest_path=None):
    splits = splits or {"train": ["0000001"], "validation": ["0000002"]}
    canonical = json.dumps(splits, sort_keys=True, separators=(",", ":")).encode()
    path.write_text(json.dumps({
        "schema_version": "experiment_cohort_v1",
        "source_manifest": {
            "path": str(manifest_path or "manifest.csv"),
            "sha256": (
                hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()
                if manifest_path else "a" * 64
            ),
        },
        "selection": {"method": "sha256_rank_v1", "seed": 42},
        "splits": splits,
        "counts": {split: len(ids) for split, ids in splits.items()},
        "cohort_sha256": hashlib.sha256(canonical).hexdigest(),
    }))


def write_budget(path: Path, record):
    path.write_text(json.dumps({
        "schema_version": "gpu_budget_v1",
        "total_gpu_hours": 6.0,
        "completed_runs": [],
        "reservations": [{
            "run_id": record["run_id"],
            "experiment_id": record["experiment_id"],
            "attempt": record["attempt"],
            "estimated_gpu_hours": record["estimated_gpu_hours"],
            "code_commit": record["code_commit"],
        }],
    }))


def write_harmony_gate(path: Path, *, decision="advance_to_gpu_registration"):
    path.write_text(json.dumps({
        "schema_version": "harmony_branch_screen_decision_v1",
        "target_variant": "temporal_chroma_v1",
        "cohort_sha256": "c" * 64,
        "screen_dataset_sha256": "d" * 64,
        "source_policy_sha256": "e" * 64,
        "source_report_sha256": "f" * 64,
        "decision": decision,
        "checks": {
            "validation_cross_entropy": decision == "advance_to_gpu_registration",
            "validation_cosine_similarity": decision == "advance_to_gpu_registration",
            "improvement_over_training_mean": decision == "advance_to_gpu_registration",
            "cpu_wall_time": True,
            "parameter_count": True,
        },
    }))


class FakeSeries(list):
    def astype(self, _type):
        return FakeSeries(str(value) for value in self)

    def duplicated(self):
        seen = set()
        result = []
        for value in self:
            result.append(value in seen)
            seen.add(value)
        return FakeSeries(result)

    def isin(self, values):
        return FakeSeries(value in values for value in self)

    def any(self):
        return any(self)

    def tolist(self):
        return list(self)


class FakeManifest:
    def __init__(self, rows):
        self.rows = [dict(row) for row in rows]
        self.columns = list(self.rows[0]) if self.rows else []

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, key):
        if isinstance(key, str):
            return FakeSeries(row[key] for row in self.rows)
        return FakeManifest(row for row, keep in zip(self.rows, key) if keep)

    def copy(self):
        return FakeManifest(self.rows)

    def reset_index(self, drop=False):
        return self


def approved_record():
    record = template_record()
    record.update({
        "status": "approved",
        "code_commit": "0123456789abcdef",
        "cohort_artifact": "results/cohorts/dev.json",
        "reused_artifacts": [{
            "path": "features/cached_encoder/dev.npz",
            "sha256": "e" * 64,
        }],
        "comparison_id": "H1_vs_H0_temporal_chroma_v1",
        "control_run_id": "H0_no_harmony_seed42",
        "control_artifact": "results/predictions/H0_validation.npz",
        "control_artifact_sha256": "b" * 64,
        "cheaper_checks": [{
            "name": "CPU harmony branch gate",
            "status": "passed",
            "evidence": "The fixed CPU harmony branch passed its preregistered gate.",
            "artifact": "results/evaluations/harmony_branch_screen_decision.json",
            "sha256": "c" * 64,
        }],
        "budget_remaining_before_gpu_hours": 6.0,
        "budget_ledger": "results/gpu_budget.json",
        "approved_by": "team-review",
        "approved_at": "2026-09-23T10:00:00+05:30",
    })
    return record


class GPURunContractTest(unittest.TestCase):
    def test_complete_approved_record_passes(self):
        record = validate_gpu_run_record(approved_record(), expected_job="harmony_branch_screening")
        self.assertEqual(record["run_id"], "H1_temporal_chroma_seed42")

    def test_planned_record_cannot_start_gpu(self):
        with self.assertRaisesRegex(GPUApprovalError, "status='approved'"):
            validate_gpu_run_record(template_record())

    def test_template_is_structurally_valid_as_a_plan(self):
        planned = validate_gpu_run_record(template_record(), require_approved=False)
        self.assertEqual(planned["attempt"], 1)

    def test_failed_cheaper_check_is_rejected(self):
        record = approved_record()
        record["cheaper_checks"][0]["status"] = "failed"
        with self.assertRaisesRegex(GPUApprovalError, "not passed"):
            validate_gpu_run_record(record)

    def test_approved_comparison_requires_positive_practical_effect(self):
        record = approved_record()
        record["validation"]["min_effect"] = 0.0
        with self.assertRaisesRegex(GPUApprovalError, "positive validation.min_effect"):
            validate_gpu_run_record(record)

    def test_approved_comparison_requires_pinned_control(self):
        record = approved_record()
        del record["control_artifact_sha256"]
        with self.assertRaisesRegex(GPUApprovalError, "control_artifact_sha256"):
            validate_gpu_run_record(record)

    def test_only_two_explicit_attempts_are_permitted(self):
        record = approved_record()
        record["attempt"] = 3
        with self.assertRaisesRegex(GPUApprovalError, "attempt must be 1 or 2"):
            validate_gpu_run_record(record)

    def test_second_attempt_requires_hashed_repeat_evidence(self):
        record = approved_record()
        record["attempt"] = 2
        record["repeat_of"] = "H1_temporal_chroma_seed42_first"
        record["repeat_reason"] = "Paired validation interval was ambiguous."
        with self.assertRaisesRegex(GPUApprovalError, "repeat_evidence_artifact"):
            validate_gpu_run_record(record)

    def test_second_attempt_with_registered_evidence_is_structurally_valid(self):
        record = approved_record()
        record.update({
            "run_id": "H1_temporal_chroma_seed43",
            "attempt": 2,
            "repeat_of": "H1_temporal_chroma_seed42",
            "repeat_reason": "The registered paired comparison was ambiguous.",
            "repeat_evidence_artifact": "results/evaluations/H1_vs_H0.json",
            "repeat_evidence_sha256": "d" * 64,
        })
        validated = validate_gpu_run_record(record)
        self.assertEqual(validated["attempt"], 2)

    def test_repeat_requires_semantically_ambiguous_paired_report(self):
        record = approved_record()
        record.update({
            "run_id": "H1_temporal_chroma_seed43",
            "attempt": 2,
            "repeat_of": "H1_temporal_chroma_seed42",
            "repeat_reason": "The registered paired comparison was ambiguous.",
            "repeat_evidence_artifact": "paired.json",
            "repeat_evidence_sha256": "d" * 64,
        })
        record = validate_gpu_run_record(record)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "paired.json"
            evidence = {
                "schema_version": "paired_bootstrap_comparison_v1",
                "decision": "ambiguous",
                "comparison_id": record["comparison_id"],
                "control_run_id": record["control_run_id"],
                "candidate_run_id": record["repeat_of"],
                "metric": record["validation"]["metric"],
                "min_effect": record["validation"]["min_effect"],
                "confidence": record["validation"]["confidence"],
                "seed": record["validation"]["bootstrap_seed"],
                "control_sha256": record["control_artifact_sha256"],
                "candidate_sha256": "a" * 64,
            }
            path.write_text(json.dumps(evidence))
            validated = validate_repeat_evidence(record, path)
            self.assertEqual(validated["decision"], "ambiguous")

            evidence["decision"] = "stop"
            path.write_text(json.dumps(evidence))
            with self.assertRaisesRegex(GPUApprovalError, "decision='ambiguous'"):
                validate_repeat_evidence(record, path)

            path.write_text(json.dumps({"status": "ambiguous"}))
            with self.assertRaisesRegex(GPUApprovalError, "must be an ambiguous"):
                validate_repeat_evidence(record, path)

    def test_failed_run_termination_can_justify_one_repeat(self):
        record = approved_record()
        record.update({
            "run_id": "H1_temporal_chroma_seed42_retry",
            "attempt": 2,
            "repeat_of": "H1_temporal_chroma_seed42",
            "repeat_reason": "The first run reached its cap before validation.",
            "repeat_evidence_artifact": "termination.json",
            "repeat_evidence_sha256": "d" * 64,
        })
        record = validate_gpu_run_record(record)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "termination.json"
            path.write_text(json.dumps({
                "schema_version": "gpu_run_termination_v1",
                "status": "terminated_without_result",
                "reason": "no_complete_validation_epoch",
                "gpu_run": {"run_id": record["repeat_of"]},
            }))
            validated = validate_repeat_evidence(record, path)
            self.assertEqual(validated["reason"], "no_complete_validation_epoch")

    def test_approved_record_rejects_template_placeholders(self):
        record = approved_record()
        record["cheaper_checks"][0]["artifact"] = "REPLACE_WITH_REPORT_PATH"
        with self.assertRaisesRegex(GPUApprovalError, "template placeholder"):
            validate_gpu_run_record(record)

    def test_test_access_is_rejected_before_final_evaluation(self):
        record = approved_record()
        record["evaluate_test"] = True
        record["test_authorization_reason"] = "finalist"
        with self.assertRaisesRegex(GPUApprovalError, "only at final_evaluation"):
            validate_gpu_run_record(record)

    def test_wall_cap_override_requires_reason(self):
        record = approved_record()
        record["max_wall_minutes"] = 180
        record["estimated_gpu_hours"] = 2.5
        with self.assertRaisesRegex(GPUApprovalError, "wall_cap_override_reason"):
            validate_gpu_run_record(record)

    def test_approved_run_must_fit_remaining_budget(self):
        record = approved_record()
        record["budget_remaining_before_gpu_hours"] = 1.0
        with self.assertRaisesRegex(GPUApprovalError, "must cover"):
            validate_gpu_run_record(record)

    def test_mismatched_job_is_rejected(self):
        with self.assertRaisesRegex(GPUApprovalError, "does not approve"):
            validate_gpu_run_record(approved_record(), expected_job="direct_cnn")

    def test_unimplemented_joint_training_job_cannot_reserve_gpu(self):
        record = approved_record()
        record["job"] = "joint_training"
        record["stage"] = "joint_training"
        with self.assertRaisesRegex(GPUApprovalError, "unsupported job"):
            validate_gpu_run_record(record)

    def test_job_cannot_claim_an_unrelated_stage(self):
        record = approved_record()
        record["stage"] = "joint_training"
        with self.assertRaisesRegex(GPUApprovalError, "cannot run at stage"):
            validate_gpu_run_record(record)

    def test_parent_traversal_artifact_path_is_rejected(self):
        record = approved_record()
        record["artifact_dir"] = "../results"
        with self.assertRaisesRegex(GPUApprovalError, "parent traversal"):
            validate_gpu_run_record(record)

    def test_one_integer_seed_is_required(self):
        record = approved_record()
        record["seed"] = [42, 43]
        with self.assertRaisesRegex(GPUApprovalError, "one integer"):
            validate_gpu_run_record(record)

    def test_approved_record_requires_real_commit_hash(self):
        record = approved_record()
        record["code_commit"] = "working-tree"
        with self.assertRaisesRegex(GPUApprovalError, "Git hash"):
            validate_gpu_run_record(record)

    def test_approval_time_requires_timezone(self):
        record = approved_record()
        record["approved_at"] = "2026-09-23T10:00:00"
        with self.assertRaisesRegex(GPUApprovalError, "timezone"):
            validate_gpu_run_record(record)

    def test_cuda_is_locked_without_an_approval_record(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(GPUApprovalError, "CUDA training is locked"):
                require_gpu_run_approval("cuda:0", "harmony_branch_screening")

    def test_cuda_accepts_existing_frozen_cohort_and_records_resolved_paths(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            cohort = root / "cohort.json"
            write_cohort(cohort)
            record = approved_record()
            record["cohort_artifact"] = "cohort.json"
            control = root / "control.npz"
            control.write_bytes(b"frozen-control-predictions")
            record["control_artifact"] = "control.npz"
            record["control_artifact_sha256"] = hashlib.sha256(control.read_bytes()).hexdigest()
            gate = root / "branch-screen-decision.json"
            write_harmony_gate(gate)
            record["cheaper_checks"][0]["artifact"] = "branch-screen-decision.json"
            record["cheaper_checks"][0]["sha256"] = hashlib.sha256(gate.read_bytes()).hexdigest()
            reused = root / "cached-encoder.npz"
            reused.write_bytes(b"cached-features")
            record["reused_artifacts"] = [{
                "path": "cached-encoder.npz",
                "sha256": hashlib.sha256(reused.read_bytes()).hexdigest(),
            }]
            record["budget_ledger"] = "budget.json"
            write_budget(root / "budget.json", record)
            record_path = root / "approved.json"
            record_path.write_text(json.dumps(record))
            with mock.patch.dict(
                os.environ,
                {"GPU_RUN_RECORD": str(record_path), "EVALUATE_TEST": "0"},
                clear=True,
            ):
                loaded = require_gpu_run_approval("cuda:0", "harmony_branch_screening")
            self.assertEqual(loaded["record_path"], str(record_path.resolve()))
            self.assertEqual(loaded["resolved_cohort_artifact"], str(cohort.resolve()))
            self.assertEqual(loaded["cohort_counts"], {"train": 1, "validation": 1})
            self.assertEqual(loaded["resolved_control_artifact"], str(control.resolve()))
            self.assertEqual(loaded["resolved_cheaper_check_artifacts"], [str(gate.resolve())])
            self.assertEqual(
                loaded["harmony_cpu_gate"]["decision"], "advance_to_gpu_registration"
            )
            self.assertEqual(loaded["resolved_reused_artifacts"], [str(reused.resolve())])

            verified, _, prerequisites = verify_gpu_run_artifacts(record, record_path)
            self.assertEqual(verified["run_id"], record["run_id"])
            self.assertEqual(prerequisites["control"], str(control.resolve()))

            reused.write_bytes(b"changed-cached-features")
            with mock.patch.dict(
                os.environ,
                {"GPU_RUN_RECORD": str(record_path), "EVALUATE_TEST": "0"},
                clear=True,
            ):
                with self.assertRaisesRegex(GPUApprovalError, "reused artifact SHA-256"):
                    require_gpu_run_approval("cuda", "harmony_branch_screening")

    def test_cuda_rejects_changed_control_artifact(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            cohort = root / "cohort.json"
            write_cohort(cohort)
            control = root / "control.npz"
            control.write_bytes(b"changed")
            record = approved_record()
            record["cohort_artifact"] = "cohort.json"
            record["control_artifact"] = "control.npz"
            record["budget_ledger"] = "budget.json"
            write_budget(root / "budget.json", record)
            record_path = root / "approved.json"
            record_path.write_text(json.dumps(record))
            with mock.patch.dict(
                os.environ,
                {"GPU_RUN_RECORD": str(record_path), "EVALUATE_TEST": "0"},
                clear=True,
            ):
                with self.assertRaisesRegex(GPUApprovalError, "control artifact SHA-256"):
                    require_gpu_run_approval("cuda", "harmony_branch_screening")

    def test_cuda_rejects_changed_cheaper_check_artifact(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            cohort = root / "cohort.json"
            write_cohort(cohort)
            control = root / "control.npz"
            control.write_bytes(b"control")
            gate = root / "gate.json"
            gate.write_bytes(b"changed-gate")
            record = approved_record()
            record["cohort_artifact"] = "cohort.json"
            record["control_artifact"] = "control.npz"
            record["control_artifact_sha256"] = hashlib.sha256(control.read_bytes()).hexdigest()
            record["cheaper_checks"][0]["artifact"] = "gate.json"
            record["budget_ledger"] = "budget.json"
            write_budget(root / "budget.json", record)
            record_path = root / "approved.json"
            record_path.write_text(json.dumps(record))
            with mock.patch.dict(
                os.environ,
                {"GPU_RUN_RECORD": str(record_path), "EVALUATE_TEST": "0"},
                clear=True,
            ):
                with self.assertRaisesRegex(GPUApprovalError, "cheaper-check artifact SHA-256"):
                    require_gpu_run_approval("cuda", "harmony_branch_screening")

    def test_harmony_gpu_rejects_hashed_but_nonsemantic_gate_file(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            write_cohort(root / "cohort.json")
            control = root / "control.npz"
            control.write_bytes(b"control")
            gate = root / "gate.json"
            gate.write_text(json.dumps({"status": "passed"}))
            reused = root / "cached.npz"
            reused.write_bytes(b"cached")
            record = approved_record()
            record["cohort_artifact"] = "cohort.json"
            record["control_artifact"] = "control.npz"
            record["control_artifact_sha256"] = hashlib.sha256(control.read_bytes()).hexdigest()
            record["cheaper_checks"][0]["artifact"] = "gate.json"
            record["cheaper_checks"][0]["sha256"] = hashlib.sha256(gate.read_bytes()).hexdigest()
            record["reused_artifacts"] = [{
                "path": "cached.npz",
                "sha256": hashlib.sha256(reused.read_bytes()).hexdigest(),
            }]
            record["budget_ledger"] = "budget.json"
            write_budget(root / "budget.json", record)
            record_path = root / "approved.json"
            record_path.write_text(json.dumps(record))
            with mock.patch.dict(
                os.environ,
                {"GPU_RUN_RECORD": str(record_path), "EVALUATE_TEST": "0"},
                clear=True,
            ):
                with self.assertRaisesRegex(
                    GPUApprovalError, "exactly one harmony_branch_screen_decision"
                ):
                    require_gpu_run_approval("cuda", "harmony_branch_screening")

    def test_harmony_gpu_rejects_cpu_gate_that_said_stop(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            write_cohort(root / "cohort.json")
            control = root / "control.npz"
            control.write_bytes(b"control")
            gate = root / "gate.json"
            write_harmony_gate(gate, decision="stop_before_gpu")
            reused = root / "cached.npz"
            reused.write_bytes(b"cached")
            record = approved_record()
            record["cohort_artifact"] = "cohort.json"
            record["control_artifact"] = "control.npz"
            record["control_artifact_sha256"] = hashlib.sha256(control.read_bytes()).hexdigest()
            record["cheaper_checks"][0]["artifact"] = "gate.json"
            record["cheaper_checks"][0]["sha256"] = hashlib.sha256(gate.read_bytes()).hexdigest()
            record["reused_artifacts"] = [{
                "path": "cached.npz",
                "sha256": hashlib.sha256(reused.read_bytes()).hexdigest(),
            }]
            record["budget_ledger"] = "budget.json"
            write_budget(root / "budget.json", record)
            record_path = root / "approved.json"
            record_path.write_text(json.dumps(record))
            with mock.patch.dict(
                os.environ,
                {"GPU_RUN_RECORD": str(record_path), "EVALUATE_TEST": "0"},
                clear=True,
            ):
                with self.assertRaisesRegex(GPUApprovalError, "did not advance"):
                    require_gpu_run_approval("cuda", "harmony_branch_screening")

    def test_approved_record_requires_hashes_for_reused_artifacts(self):
        record = approved_record()
        record["reused_artifacts"][0]["sha256"] = "not-a-hash"
        with self.assertRaisesRegex(GPUApprovalError, r"reused_artifacts\[0\].sha256"):
            validate_gpu_run_record(record)

    def test_malformed_cohort_cannot_unlock_cuda(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            cohort = root / "cohort.json"
            cohort.write_text("{}")
            record = approved_record()
            record["cohort_artifact"] = "cohort.json"
            record["budget_ledger"] = "budget.json"
            write_budget(root / "budget.json", record)
            record_path = root / "approved.json"
            record_path.write_text(json.dumps(record))
            with mock.patch.dict(os.environ, {"GPU_RUN_RECORD": str(record_path)}, clear=True):
                with self.assertRaisesRegex(GPUApprovalError, "cohort schema_version"):
                    require_gpu_run_approval("cuda", "harmony_branch_screening")

    def test_approved_cohort_filters_manifest_and_checks_split(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            cohort = Path(temporary_directory) / "cohort.json"
            manifest = FakeManifest([
                {"song_id": "0000001", "split": "train"},
                {"song_id": "0000002", "split": "validation"},
                {"song_id": "0000003", "split": "test"},
            ])
            manifest_path = Path(temporary_directory) / "manifest.csv"
            manifest_path.write_text(
                "song_id,split\n0000001,train\n0000002,validation\n0000003,test\n"
            )
            write_cohort(cohort, manifest_path=manifest_path)
            selected = apply_approved_cohort(
                manifest,
                {"resolved_cohort_artifact": str(cohort)},
                manifest_path,
            )
            self.assertEqual(selected["song_id"].tolist(), ["0000001", "0000002"])

            manifest.rows[1]["split"] = "test"
            with self.assertRaisesRegex(GPUApprovalError, "split mismatch"):
                apply_approved_cohort(
                    manifest,
                    {"resolved_cohort_artifact": str(cohort)},
                    manifest_path,
                )

    def test_approved_limits_cannot_be_exceeded_by_environment_request(self):
        record = approved_record()
        record["max_epochs"] = 4
        record["max_wall_minutes"] = 30
        self.assertEqual(
            approved_run_limits(record, default_epochs=10, requested_wall_minutes=120),
            (4, 30.0),
        )

    def test_non_positive_requested_wall_time_is_rejected(self):
        with self.assertRaisesRegex(GPUApprovalError, "must be positive"):
            approved_run_limits(None, default_epochs=10, requested_wall_minutes=0)

    def test_deliberate_termination_persists_consumed_time(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "runtime.json"
            payload = write_gpu_termination_ledger(
                path,
                device="cpu",
                started_at=time.perf_counter(),
                record=None,
                reason="no_complete_validation_epoch",
                max_epochs=2,
                max_wall_minutes=1,
            )
            self.assertTrue(path.is_file())
            self.assertEqual(payload["status"], "terminated_without_result")
            self.assertEqual(payload["gpu_wall_hours"], 0.0)


if __name__ == "__main__":
    unittest.main()
