"""Validation contract for scarce-GPU experiment approval records."""

NOTEBOOK_GPU_RUN_CONTRACT = r'''
import json
import os
import re
import hashlib
import time
from datetime import datetime
from pathlib import Path, PurePath


GPU_RUN_SCHEMA_VERSION = "gpu_run_request_v1"
GPU_JOBS = {
    "direct_cnn", "instrument_pretraining", "descriptor_fusion",
    "harmony_branch_screening",
}
GPU_STAGES = {"baseline", "pretraining", "branch_screening", "joint_training", "final_evaluation"}
JOB_STAGES = {
    "direct_cnn": {"baseline", "final_evaluation"},
    "instrument_pretraining": {"pretraining", "final_evaluation"},
    "descriptor_fusion": {"baseline", "final_evaluation"},
    "harmony_branch_screening": {"branch_screening"},
}
COHORT_SCHEMA_VERSION = "experiment_cohort_v1"
BUDGET_SCHEMA_VERSION = "gpu_budget_v1"


class GPUApprovalError(ValueError):
    pass


def _required_text(record, field):
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise GPUApprovalError(f"{field} must be a non-empty string")
    return value.strip()


def _reject_placeholder(value, field):
    text = str(value).strip()
    if "REPLACE_" in text.upper():
        raise GPUApprovalError(f"{field} still contains a template placeholder")
    return text


def _safe_artifact_path(value, field):
    text = _required_text({field: value}, field)
    path = PurePath(text)
    if text in {"/", ".", "~"} or ".." in path.parts:
        raise GPUApprovalError(f"{field} is too broad or contains parent traversal")
    return text


def _load_cohort_artifact(path):
    try:
        cohort = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise GPUApprovalError(f"cannot read cohort artifact {path}: {error}") from error
    if not isinstance(cohort, dict) or cohort.get("schema_version") != COHORT_SCHEMA_VERSION:
        raise GPUApprovalError(f"cohort schema_version must be {COHORT_SCHEMA_VERSION!r}")
    source = cohort.get("source_manifest")
    if not isinstance(source, dict) or not re.fullmatch(r"[0-9a-f]{64}", str(source.get("sha256", ""))):
        raise GPUApprovalError("cohort source_manifest.sha256 must be a lowercase SHA-256")
    splits = cohort.get("splits")
    if not isinstance(splits, dict) or not splits:
        raise GPUApprovalError("cohort splits must be a non-empty object")
    unexpected = set(splits) - {"train", "validation", "test"}
    if unexpected:
        raise GPUApprovalError(f"cohort has unsupported splits: {sorted(unexpected)}")
    if not splits.get("train") or not splits.get("validation"):
        raise GPUApprovalError("cohort must contain non-empty train and validation splits")
    seen = set()
    normalized_splits = {}
    for split, song_ids in splits.items():
        if not isinstance(song_ids, list) or any(
            not isinstance(song_id, str) or not re.fullmatch(r"\d{7}", song_id)
            for song_id in song_ids
        ):
            raise GPUApprovalError(f"cohort {split} IDs must be seven-digit strings")
        if len(song_ids) != len(set(song_ids)):
            raise GPUApprovalError(f"cohort {split} contains duplicate song IDs")
        overlap = seen.intersection(song_ids)
        if overlap:
            raise GPUApprovalError(f"cohort song IDs occur in multiple splits: {sorted(overlap)[:3]}")
        seen.update(song_ids)
        normalized_splits[split] = list(song_ids)
    counts = cohort.get("counts")
    expected_counts = {split: len(ids) for split, ids in normalized_splits.items()}
    if counts != expected_counts:
        raise GPUApprovalError(f"cohort counts do not match IDs: expected {expected_counts}")
    canonical = json.dumps(normalized_splits, sort_keys=True, separators=(",", ":")).encode()
    expected_hash = hashlib.sha256(canonical).hexdigest()
    if cohort.get("cohort_sha256") != expected_hash:
        raise GPUApprovalError("cohort_sha256 does not match the frozen split IDs")
    return cohort


def _load_gpu_budget(path, record=None):
    try:
        ledger = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise GPUApprovalError(f"cannot read GPU budget ledger {path}: {error}") from error
    if not isinstance(ledger, dict) or ledger.get("schema_version") != BUDGET_SCHEMA_VERSION:
        raise GPUApprovalError(f"budget schema_version must be {BUDGET_SCHEMA_VERSION!r}")
    total = ledger.get("total_gpu_hours")
    if not isinstance(total, (int, float)) or isinstance(total, bool) or total <= 0:
        raise GPUApprovalError("budget total_gpu_hours must be positive")

    completed = ledger.get("completed_runs", [])
    reservations = ledger.get("reservations", [])
    if not isinstance(completed, list) or not isinstance(reservations, list):
        raise GPUApprovalError("budget completed_runs and reservations must be lists")
    completed_ids = set()
    completed_attempts = set()
    used = 0.0
    for item in completed:
        if not isinstance(item, dict):
            raise GPUApprovalError("each completed GPU run must be an object")
        run_id = _required_text(item, "run_id")
        experiment_id = _required_text(item, "experiment_id")
        attempt = item.get("attempt")
        actual = item.get("actual_gpu_hours")
        experiment_attempt = (experiment_id, attempt)
        if (
            run_id in completed_ids
            or attempt not in {1, 2}
            or experiment_attempt in completed_attempts
            or not isinstance(actual, (int, float))
            or isinstance(actual, bool)
            or actual < 0
        ):
            raise GPUApprovalError("completed GPU runs need unique IDs and non-negative actual hours")
        completed_ids.add(run_id)
        completed_attempts.add(experiment_attempt)
        used += float(actual)

    reservation_by_id = {}
    reserved_attempts = set()
    reserved = 0.0
    for item in reservations:
        if not isinstance(item, dict):
            raise GPUApprovalError("each GPU reservation must be an object")
        run_id = _required_text(item, "run_id")
        experiment_id = _required_text(item, "experiment_id")
        attempt = item.get("attempt")
        experiment_attempt = (experiment_id, attempt)
        estimate = item.get("estimated_gpu_hours")
        commit = _required_text(item, "code_commit")
        if (
            run_id in completed_ids
            or run_id in reservation_by_id
            or attempt not in {1, 2}
            or experiment_attempt in completed_attempts
            or experiment_attempt in reserved_attempts
        ):
            raise GPUApprovalError(
                "GPU run IDs and experiment attempts must be unique across the ledger"
            )
        if not isinstance(estimate, (int, float)) or isinstance(estimate, bool) or estimate <= 0:
            raise GPUApprovalError("GPU reservations require positive estimated_gpu_hours")
        if not re.fullmatch(r"[0-9a-fA-F]{7,40}", commit):
            raise GPUApprovalError("GPU reservation code_commit must be a Git hash")
        reservation_by_id[run_id] = item
        reserved_attempts.add(experiment_attempt)
        reserved += float(estimate)
    summary = {
        "total_gpu_hours": float(total),
        "used_gpu_hours": used,
        "reserved_gpu_hours": reserved,
        "unreserved_gpu_hours": float(total) - used - reserved,
        "overcommitted": used + reserved > float(total) + 1e-9,
    }
    if record is not None:
        reservation = reservation_by_id.get(record["run_id"])
        if reservation is None:
            raise GPUApprovalError("approved run has no matching GPU budget reservation")
        if (
            float(reservation["estimated_gpu_hours"]) != float(record["estimated_gpu_hours"])
            or reservation["code_commit"].lower() != record["code_commit"].lower()
            or reservation["experiment_id"] != record["experiment_id"]
            or reservation["attempt"] != record["attempt"]
        ):
            raise GPUApprovalError(
                "GPU budget reservation does not match run experiment, attempt, estimate, and commit"
            )
        available_before = summary["unreserved_gpu_hours"] + float(record["estimated_gpu_hours"])
        if abs(float(record["budget_remaining_before_gpu_hours"]) - available_before) > 1e-9:
            raise GPUApprovalError("run budget snapshot does not match the GPU budget ledger")
        summary["available_before_this_run_gpu_hours"] = available_before
    return ledger, summary


def _validate_required_gate_artifacts(record, artifact_paths):
    """Require semantic CPU evidence before the only registered harmony GPU screen."""
    if record["job"] != "harmony_branch_screening":
        return None
    if record["stage"] != "branch_screening":
        raise GPUApprovalError("harmony_branch_screening must use stage='branch_screening'")
    if record["experiment_id"] != "H1_temporal_chroma":
        raise GPUApprovalError(
            "only the registered H1_temporal_chroma experiment may use harmony GPU screening"
        )
    if record["seed"] != 42:
        raise GPUApprovalError("the first harmony GPU screen must use registered seed 42")

    decisions = []
    for value in artifact_paths:
        try:
            candidate = json.loads(Path(value).read_text())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if (
            isinstance(candidate, dict)
            and candidate.get("schema_version") == "harmony_branch_screen_decision_v1"
        ):
            decisions.append(candidate)
    if len(decisions) != 1:
        raise GPUApprovalError(
            "harmony GPU screening requires exactly one harmony_branch_screen_decision_v1 artifact"
        )
    decision = decisions[0]
    if decision.get("decision") != "advance_to_gpu_registration":
        raise GPUApprovalError("harmony CPU branch gate did not advance to GPU registration")
    if decision.get("target_variant") != "temporal_chroma_v1":
        raise GPUApprovalError("harmony CPU gate does not approve temporal_chroma_v1")
    checks = decision.get("checks")
    if not isinstance(checks, dict) or not checks or not all(value is True for value in checks.values()):
        raise GPUApprovalError("harmony CPU branch gate does not contain all passing checks")
    for field in ("cohort_sha256", "screen_dataset_sha256", "source_policy_sha256", "source_report_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(decision.get(field, ""))):
            raise GPUApprovalError(f"harmony CPU branch gate has invalid {field}")
    return decision


def _validate_repeat_evidence(record, evidence_path):
    """Allow one repeat only for a registered ambiguous comparison or recorded failure."""
    if record["attempt"] != 2:
        return None
    try:
        evidence = json.loads(Path(evidence_path).read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GPUApprovalError(f"repeat evidence is not valid JSON: {error}") from error
    if not isinstance(evidence, dict):
        raise GPUApprovalError("repeat evidence must be a JSON object")

    if evidence.get("schema_version") == "paired_bootstrap_comparison_v1":
        if evidence.get("decision") != "ambiguous":
            raise GPUApprovalError("paired repeat evidence must have decision='ambiguous'")
        expected = {
            "comparison_id": record["comparison_id"],
            "control_run_id": record["control_run_id"],
            "candidate_run_id": record["repeat_of"],
            "metric": record["validation"]["metric"],
            "min_effect": record["validation"]["min_effect"],
            "confidence": record["validation"]["confidence"],
            "seed": record["validation"]["bootstrap_seed"],
            "control_sha256": record["control_artifact_sha256"],
        }
        for field, value in expected.items():
            if evidence.get(field) != value:
                raise GPUApprovalError(f"repeat evidence {field} does not match the run request")
        if not re.fullmatch(r"[0-9a-f]{64}", str(evidence.get("candidate_sha256", ""))):
            raise GPUApprovalError("repeat evidence candidate_sha256 is invalid")
        return evidence

    if evidence.get("schema_version") == "gpu_run_termination_v1":
        prior = evidence.get("gpu_run")
        if evidence.get("status") != "terminated_without_result" or not isinstance(prior, dict):
            raise GPUApprovalError("termination repeat evidence is not a failed run record")
        if prior.get("run_id") != record["repeat_of"]:
            raise GPUApprovalError("termination evidence does not describe repeat_of")
        reason = evidence.get("reason")
        if reason not in {
            "no_complete_validation_epoch",
            "embedding_export_cap_reached",
            "runtime_exception",
        }:
            raise GPUApprovalError("termination evidence has an unsupported failure reason")
        return evidence

    raise GPUApprovalError(
        "repeat evidence must be an ambiguous paired comparison or GPU termination ledger"
    )


def validate_gpu_run_record(record, *, expected_job=None, require_approved=True):
    if not isinstance(record, dict):
        raise GPUApprovalError("run record must be a JSON object")
    if record.get("schema_version") != GPU_RUN_SCHEMA_VERSION:
        raise GPUApprovalError(f"schema_version must be {GPU_RUN_SCHEMA_VERSION!r}")

    run_id = _required_text(record, "run_id")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{2,63}", run_id):
        raise GPUApprovalError("run_id must be 3-64 safe filename characters")
    experiment_id = _required_text(record, "experiment_id")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{2,63}", experiment_id):
        raise GPUApprovalError("experiment_id must be 3-64 safe filename characters")
    attempt = record.get("attempt")
    if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt not in {1, 2}:
        raise GPUApprovalError("attempt must be 1 or 2; broader repeat sweeps are forbidden")
    repeat_of = record.get("repeat_of")
    if attempt == 1 and repeat_of not in {None, ""}:
        raise GPUApprovalError("attempt 1 must not declare repeat_of")
    if attempt == 2:
        repeat_of = _required_text(record, "repeat_of")
        if repeat_of == run_id:
            raise GPUApprovalError("repeat_of must differ from run_id")
        _required_text(record, "repeat_reason")
        repeat_artifact = _safe_artifact_path(
            record.get("repeat_evidence_artifact"), "repeat_evidence_artifact"
        )
        repeat_hash = _required_text(record, "repeat_evidence_sha256")
        if record.get("status") == "approved" and not re.fullmatch(r"[0-9a-f]{64}", repeat_hash):
            raise GPUApprovalError("repeat_evidence_sha256 must be a lowercase SHA-256")
    job = _required_text(record, "job")
    if job not in GPU_JOBS:
        raise GPUApprovalError(f"unsupported job: {job!r}")
    if expected_job is not None and job != expected_job:
        raise GPUApprovalError(f"record job {job!r} does not approve {expected_job!r}")
    stage = _required_text(record, "stage")
    if stage not in GPU_STAGES:
        raise GPUApprovalError(f"unsupported stage: {stage!r}")
    if stage not in JOB_STAGES[job]:
        raise GPUApprovalError(f"job {job!r} cannot run at stage {stage!r}")

    status = _required_text(record, "status")
    if status not in {"planned", "approved"}:
        raise GPUApprovalError("status must be 'planned' or 'approved'")
    if require_approved and status != "approved":
        raise GPUApprovalError("GPU execution requires status='approved'")
    _required_text(record, "question")
    _required_text(record, "single_change")
    code_commit = _required_text(record, "code_commit")
    cohort_artifact = _safe_artifact_path(record.get("cohort_artifact"), "cohort_artifact")
    artifact_dir = _safe_artifact_path(record.get("artifact_dir"), "artifact_dir")

    checks = record.get("cheaper_checks")
    if not isinstance(checks, list) or not checks:
        raise GPUApprovalError("cheaper_checks must contain at least one passed CPU/preflight check")
    for index, check in enumerate(checks):
        if not isinstance(check, dict):
            raise GPUApprovalError(f"cheaper_checks[{index}] must be an object")
        _required_text(check, "name")
        _required_text(check, "evidence")
        if check.get("status") != "passed":
            raise GPUApprovalError(f"cheaper_checks[{index}] is not passed")
        evidence_artifact = _safe_artifact_path(
            check.get("artifact"), f"cheaper_checks[{index}].artifact"
        )
        evidence_hash = _required_text(check, "sha256")
        if status == "approved" and not re.fullmatch(r"[0-9a-f]{64}", evidence_hash):
            raise GPUApprovalError(
                f"cheaper_checks[{index}].sha256 must be a lowercase SHA-256"
            )

    reused = record.get("reused_artifacts")
    if not isinstance(reused, list):
        raise GPUApprovalError("reused_artifacts must be a list (possibly empty)")
    normalized_reused = []
    for index, item in enumerate(reused):
        if not isinstance(item, dict):
            raise GPUApprovalError(f"reused_artifacts[{index}] must be an object")
        reused_path = _safe_artifact_path(
            item.get("path"), f"reused_artifacts[{index}].path"
        )
        reused_hash = _required_text(item, "sha256")
        if status == "approved" and not re.fullmatch(r"[0-9a-f]{64}", reused_hash):
            raise GPUApprovalError(
                f"reused_artifacts[{index}].sha256 must be a lowercase SHA-256"
            )
        normalized_reused.append({"path": reused_path, "sha256": reused_hash})

    max_epochs = record.get("max_epochs")
    if not isinstance(max_epochs, int) or isinstance(max_epochs, bool) or not 1 <= max_epochs <= 100:
        raise GPUApprovalError("max_epochs must be an integer in [1, 100]")
    max_wall_minutes = record.get("max_wall_minutes")
    if not isinstance(max_wall_minutes, (int, float)) or isinstance(max_wall_minutes, bool) or max_wall_minutes <= 0:
        raise GPUApprovalError("max_wall_minutes must be positive")
    if max_wall_minutes > 120 and not str(record.get("wall_cap_override_reason", "")).strip():
        raise GPUApprovalError("runs above 120 minutes require wall_cap_override_reason")
    estimated_gpu_hours = record.get("estimated_gpu_hours")
    if not isinstance(estimated_gpu_hours, (int, float)) or isinstance(estimated_gpu_hours, bool):
        raise GPUApprovalError("estimated_gpu_hours must be numeric")
    if estimated_gpu_hours <= 0 or estimated_gpu_hours > max_wall_minutes / 60:
        raise GPUApprovalError("estimated_gpu_hours must be positive and no larger than the wall cap")
    if status == "approved":
        remaining_gpu_hours = record.get("budget_remaining_before_gpu_hours")
        if (
            not isinstance(remaining_gpu_hours, (int, float))
            or isinstance(remaining_gpu_hours, bool)
            or remaining_gpu_hours < estimated_gpu_hours
        ):
            raise GPUApprovalError(
                "budget_remaining_before_gpu_hours must cover estimated_gpu_hours"
            )
        _safe_artifact_path(record.get("budget_ledger"), "budget_ledger")
    seed = record.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise GPUApprovalError("seed must be one integer")

    validation = record.get("validation")
    if not isinstance(validation, dict):
        raise GPUApprovalError("validation must be an object")
    if validation.get("split") != "validation":
        raise GPUApprovalError("model selection must use the validation split")
    _required_text(validation, "metric")
    min_effect = validation.get("min_effect")
    if not isinstance(min_effect, (int, float)) or isinstance(min_effect, bool) or min_effect < 0:
        raise GPUApprovalError("validation.min_effect must be non-negative")
    comparison_stage = stage in {"branch_screening", "joint_training"}
    if comparison_stage and status == "approved" and min_effect <= 0:
        raise GPUApprovalError(
            "approved branch/joint comparisons require a positive validation.min_effect"
        )
    confidence = validation.get("confidence")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not 0.5 < confidence < 1:
        raise GPUApprovalError("validation.confidence must be between 0.5 and 1")
    if not isinstance(validation.get("bootstrap_seed"), int) or isinstance(validation.get("bootstrap_seed"), bool):
        raise GPUApprovalError("validation.bootstrap_seed must be one integer")

    evaluate_test = record.get("evaluate_test")
    if not isinstance(evaluate_test, bool):
        raise GPUApprovalError("evaluate_test must be boolean")
    if evaluate_test and stage != "final_evaluation":
        raise GPUApprovalError("test evaluation is allowed only at final_evaluation")
    if evaluate_test:
        _required_text(record, "test_authorization_reason")

    if status == "approved":
        if not re.fullmatch(r"[0-9a-fA-F]{7,40}", code_commit):
            raise GPUApprovalError("approved code_commit must be a 7-40 character Git hash")
        _required_text(record, "approved_by")
        approved_at = _required_text(record, "approved_at")
        try:
            parsed_approval_time = datetime.fromisoformat(approved_at.replace("Z", "+00:00"))
        except ValueError as error:
            raise GPUApprovalError("approved_at must be an ISO-8601 timestamp") from error
        if parsed_approval_time.tzinfo is None:
            raise GPUApprovalError("approved_at must include a timezone")
        for field in ("question", "single_change", "cohort_artifact", "artifact_dir"):
            _reject_placeholder(record[field], field)
        for index, check in enumerate(checks):
            _reject_placeholder(check["evidence"], f"cheaper_checks[{index}].evidence")
            _reject_placeholder(check["artifact"], f"cheaper_checks[{index}].artifact")
        for index, item in enumerate(normalized_reused):
            _reject_placeholder(item["path"], f"reused_artifacts[{index}].path")
        if attempt == 2:
            for field, value in (
                ("repeat_of", repeat_of),
                ("repeat_reason", record["repeat_reason"]),
                ("repeat_evidence_artifact", repeat_artifact),
            ):
                _reject_placeholder(value, field)

    if comparison_stage:
        comparison_id = _required_text(record, "comparison_id")
        control_run_id = _required_text(record, "control_run_id")
        if control_run_id == run_id:
            raise GPUApprovalError("control_run_id must differ from run_id")
        control_artifact = _safe_artifact_path(
            record.get("control_artifact"), "control_artifact"
        )
        control_hash = _required_text(record, "control_artifact_sha256")
        if status == "approved" and not re.fullmatch(r"[0-9a-f]{64}", control_hash):
            raise GPUApprovalError("control_artifact_sha256 must be a lowercase SHA-256")
        if status == "approved":
            for field, value in (
                ("comparison_id", comparison_id),
                ("control_run_id", control_run_id),
                ("control_artifact", control_artifact),
            ):
                _reject_placeholder(value, field)

    normalized = dict(record)
    normalized["cohort_artifact"] = cohort_artifact
    normalized["artifact_dir"] = artifact_dir
    normalized["reused_artifacts"] = normalized_reused
    if attempt == 2:
        normalized["repeat_evidence_artifact"] = repeat_artifact
    if comparison_stage:
        normalized["control_artifact"] = control_artifact
    return normalized


def require_gpu_run_approval(device, expected_job):
    if str(device).split(":", 1)[0] != "cuda":
        print("CPU execution: GPU approval record is not required.")
        return None
    record_path = os.environ.get("GPU_RUN_RECORD", "").strip()
    if not record_path:
        raise GPUApprovalError(
            "CUDA training is locked. Set GPU_RUN_RECORD to an approved JSON record after CPU gates pass."
        )
    path = Path(record_path)
    if not path.is_file():
        raise GPUApprovalError(f"GPU_RUN_RECORD does not exist: {path}")
    record = validate_gpu_run_record(json.loads(path.read_text()), expected_job=expected_job)
    cohort = Path(record["cohort_artifact"])
    if not cohort.is_absolute():
        cohort = path.parent / cohort
    if not cohort.is_file():
        raise GPUApprovalError(f"approved cohort artifact does not exist: {cohort}")
    cohort_data = _load_cohort_artifact(cohort)
    if record["evaluate_test"] and not cohort_data["splits"].get("test"):
        raise GPUApprovalError("final test evaluation requires a non-empty frozen test cohort")
    if record.get("control_artifact"):
        control = Path(record["control_artifact"])
        if not control.is_absolute():
            control = path.parent / control
        if not control.is_file():
            raise GPUApprovalError(f"approved control artifact does not exist: {control}")
        digest = hashlib.sha256()
        with control.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != record["control_artifact_sha256"]:
            raise GPUApprovalError("approved control artifact SHA-256 does not match")
        record["resolved_control_artifact"] = str(control.resolve())
    resolved_check_artifacts = []
    for index, check in enumerate(record["cheaper_checks"]):
        evidence = Path(check["artifact"])
        if not evidence.is_absolute():
            evidence = path.parent / evidence
        if not evidence.is_file():
            raise GPUApprovalError(
                f"approved cheaper-check artifact does not exist: {evidence}"
            )
        digest = hashlib.sha256()
        with evidence.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != check["sha256"]:
            raise GPUApprovalError(
                f"approved cheaper-check artifact SHA-256 does not match at index {index}"
            )
        resolved_check_artifacts.append(str(evidence.resolve()))
    record["resolved_cheaper_check_artifacts"] = resolved_check_artifacts
    gate_decision = _validate_required_gate_artifacts(record, resolved_check_artifacts)
    if gate_decision is not None:
        record["harmony_cpu_gate"] = gate_decision
    resolved_reused_artifacts = []
    for index, item in enumerate(record["reused_artifacts"]):
        reused_path = Path(item["path"])
        if not reused_path.is_absolute():
            reused_path = path.parent / reused_path
        if not reused_path.is_file():
            raise GPUApprovalError(f"approved reused artifact does not exist: {reused_path}")
        digest = hashlib.sha256()
        with reused_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != item["sha256"]:
            raise GPUApprovalError(
                f"approved reused artifact SHA-256 does not match at index {index}"
            )
        resolved_reused_artifacts.append(str(reused_path.resolve()))
    record["resolved_reused_artifacts"] = resolved_reused_artifacts
    if record["attempt"] == 2:
        repeat_evidence = Path(record["repeat_evidence_artifact"])
        if not repeat_evidence.is_absolute():
            repeat_evidence = path.parent / repeat_evidence
        if not repeat_evidence.is_file():
            raise GPUApprovalError(f"approved repeat evidence does not exist: {repeat_evidence}")
        digest = hashlib.sha256()
        with repeat_evidence.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != record["repeat_evidence_sha256"]:
            raise GPUApprovalError("approved repeat evidence SHA-256 does not match")
        _validate_repeat_evidence(record, repeat_evidence)
        record["resolved_repeat_evidence"] = str(repeat_evidence.resolve())
    budget_path = Path(record["budget_ledger"])
    if not budget_path.is_absolute():
        budget_path = path.parent / budget_path
    if not budget_path.is_file():
        raise GPUApprovalError(f"GPU budget ledger does not exist: {budget_path}")
    _, budget_summary = _load_gpu_budget(budget_path, record)
    requested_test = os.environ.get("EVALUATE_TEST", "0") == "1"
    if requested_test != record["evaluate_test"]:
        raise GPUApprovalError("EVALUATE_TEST must exactly match the approved run record")
    record["record_path"] = str(path.resolve())
    record["resolved_cohort_artifact"] = str(cohort.resolve())
    record["cohort_sha256"] = cohort_data["cohort_sha256"]
    record["cohort_counts"] = cohort_data["counts"]
    record["resolved_budget_ledger"] = str(budget_path.resolve())
    record["budget_summary"] = budget_summary
    print(f"GPU run approved: {record['run_id']} by {record['approved_by']}")
    return record


def apply_approved_cohort(manifest, record, manifest_path):
    """Restrict a manifest to the exact approved IDs before creating data loaders."""
    if record is None:
        return manifest
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise GPUApprovalError(f"manifest does not exist: {manifest_path}")
    if "song_id" not in manifest.columns or "split" not in manifest.columns:
        raise GPUApprovalError("manifest must contain song_id and split columns")
    if manifest["song_id"].duplicated().any():
        raise GPUApprovalError("manifest contains duplicate song IDs")
    cohort = _load_cohort_artifact(record["resolved_cohort_artifact"])
    digest = hashlib.sha256()
    with manifest_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != cohort["source_manifest"]["sha256"]:
        raise GPUApprovalError("current manifest SHA-256 does not match the frozen cohort source")
    approved_split = {
        song_id: split
        for split, song_ids in cohort["splits"].items()
        for song_id in song_ids
    }
    manifest_split = dict(zip(manifest["song_id"].astype(str), manifest["split"].astype(str)))
    missing = sorted(set(approved_split) - set(manifest_split))
    if missing:
        raise GPUApprovalError(f"approved cohort has {len(missing)} IDs absent from manifest: {missing[:3]}")
    wrong = sorted(
        song_id for song_id, split in approved_split.items()
        if manifest_split[song_id] != split
    )
    if wrong:
        raise GPUApprovalError(f"approved cohort split mismatch for IDs: {wrong[:3]}")
    selected = manifest[manifest["song_id"].astype(str).isin(approved_split)].copy()
    if len(selected) != len(approved_split):
        raise GPUApprovalError("manifest filtering did not produce the exact approved cohort")
    print("Applied approved cohort:", cohort["counts"], cohort["cohort_sha256"])
    return selected.reset_index(drop=True)


def approved_run_limits(record, *, default_epochs, requested_wall_minutes):
    if not isinstance(default_epochs, int) or isinstance(default_epochs, bool) or default_epochs < 1:
        raise GPUApprovalError("default_epochs must be a positive integer")
    if not isinstance(requested_wall_minutes, (int, float)) or isinstance(requested_wall_minutes, bool):
        raise GPUApprovalError("requested_wall_minutes must be numeric")
    if requested_wall_minutes <= 0:
        raise GPUApprovalError("requested_wall_minutes must be positive")
    if record is None:
        return int(default_epochs), float(requested_wall_minutes)
    return (
        min(int(default_epochs), int(record["max_epochs"])),
        min(float(requested_wall_minutes), float(record["max_wall_minutes"])),
    )


def write_gpu_termination_ledger(
    path, *, device, started_at, record, reason, max_epochs, max_wall_minutes
):
    """Persist consumed time before deliberately aborting a capped run."""
    elapsed = time.perf_counter() - started_at
    payload = {
        "schema_version": "gpu_run_termination_v1",
        "status": "terminated_without_result",
        "reason": reason,
        "device": str(device),
        "max_epochs": int(max_epochs),
        "max_wall_minutes": float(max_wall_minutes),
        "gpu_run": record,
        "wall_seconds": elapsed,
        "gpu_wall_hours": elapsed / 3600 if str(device).split(":", 1)[0] == "cuda" else 0.0,
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2))
    print("termination runtime", payload)
    return payload
'''


_namespace: dict = {}
exec(NOTEBOOK_GPU_RUN_CONTRACT, _namespace)
GPUApprovalError = _namespace["GPUApprovalError"]
validate_gpu_run_record = _namespace["validate_gpu_run_record"]
require_gpu_run_approval = _namespace["require_gpu_run_approval"]
apply_approved_cohort = _namespace["apply_approved_cohort"]
approved_run_limits = _namespace["approved_run_limits"]
load_cohort_artifact = _namespace["_load_cohort_artifact"]
load_gpu_budget = _namespace["_load_gpu_budget"]
validate_required_gate_artifacts = _namespace["_validate_required_gate_artifacts"]
validate_repeat_evidence = _namespace["_validate_repeat_evidence"]
write_gpu_termination_ledger = _namespace["write_gpu_termination_ledger"]


def verify_gpu_run_artifacts(record: dict, record_path) -> tuple[dict, dict, dict]:
    """Verify every prerequisite available before reserving or uploading a run."""
    import hashlib
    from pathlib import Path

    path = Path(record_path)
    validated = validate_gpu_run_record(record)

    def resolve(value: str) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else path.parent / candidate

    def verify_file(value: str, expected_hash: str, label: str) -> Path:
        candidate = resolve(value)
        if not candidate.is_file():
            raise GPUApprovalError(f"{label} does not exist: {candidate}")
        digest = hashlib.sha256()
        with candidate.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected_hash:
            raise GPUApprovalError(f"{label} SHA-256 does not match")
        return candidate.resolve()

    cohort_path = resolve(validated["cohort_artifact"])
    if not cohort_path.is_file():
        raise GPUApprovalError(f"cohort artifact does not exist: {cohort_path}")
    cohort = load_cohort_artifact(cohort_path)
    summary = {
        "cohort": str(cohort_path.resolve()),
        "control": None,
        "cheaper_checks": [],
        "reused_artifacts": [],
        "repeat_evidence": None,
    }
    if validated.get("control_artifact"):
        summary["control"] = str(verify_file(
            validated["control_artifact"],
            validated["control_artifact_sha256"],
            "control artifact",
        ))
    for index, check in enumerate(validated["cheaper_checks"]):
        summary["cheaper_checks"].append(str(verify_file(
            check["artifact"], check["sha256"], f"cheaper-check artifact {index}"
        )))
    gate_decision = validate_required_gate_artifacts(validated, summary["cheaper_checks"])
    if gate_decision is not None:
        summary["harmony_cpu_gate"] = {
            "decision": gate_decision["decision"],
            "target_variant": gate_decision["target_variant"],
            "cohort_sha256": gate_decision["cohort_sha256"],
            "screen_dataset_sha256": gate_decision["screen_dataset_sha256"],
        }
    for index, item in enumerate(validated["reused_artifacts"]):
        summary["reused_artifacts"].append(str(verify_file(
            item["path"], item["sha256"], f"reused artifact {index}"
        )))
    if validated["attempt"] == 2:
        repeat_evidence = verify_file(
            validated["repeat_evidence_artifact"],
            validated["repeat_evidence_sha256"],
            "repeat evidence",
        )
        validate_repeat_evidence(validated, repeat_evidence)
        summary["repeat_evidence"] = str(repeat_evidence)
    return validated, cohort, summary


def template_record() -> dict:
    return {
        "schema_version": "gpu_run_request_v1",
        "run_id": "H1_temporal_chroma_seed42",
        "experiment_id": "H1_temporal_chroma",
        "attempt": 1,
        "repeat_of": None,
        "status": "planned",
        "job": "harmony_branch_screening",
        "stage": "branch_screening",
        "question": "Does temporal chroma beat the registered no-harmony control?",
        "single_change": "Enable the 32D temporal chroma branch.",
        "comparison_id": "H1_vs_H0_temporal_chroma_v1",
        "control_run_id": "H0_no_harmony_seed42",
        "control_artifact": "REPLACE_WITH_H0_VALIDATION_PREDICTIONS",
        "control_artifact_sha256": "REPLACE_WITH_LOWERCASE_SHA256",
        "code_commit": "REPLACE_WITH_COMMIT_HASH",
        "cohort_artifact": "REPLACE_WITH_FROZEN_COHORT_JSON",
        "reused_artifacts": [{
            "path": "REPLACE_WITH_CACHED_ENCODER_FEATURES",
            "sha256": "REPLACE_WITH_LOWERCASE_SHA256",
        }],
        "cheaper_checks": [
            {
                "name": "CPU harmony branch gate",
                "status": "passed",
                "evidence": "Preregistered fixed CPU branch screen returned advance_to_gpu_registration.",
                "artifact": "REPLACE_WITH_HARMONY_BRANCH_SCREEN_DECISION",
                "sha256": "REPLACE_WITH_LOWERCASE_SHA256",
            }
        ],
        "seed": 42,
        "max_epochs": 10,
        "max_wall_minutes": 120,
        "estimated_gpu_hours": 2.0,
        "budget_remaining_before_gpu_hours": None,
        "budget_ledger": "REPLACE_WITH_GPU_BUDGET_LEDGER",
        "validation": {
            "split": "validation",
            "metric": "macro_pr_auc",
            "min_effect": 0.001,
            "confidence": 0.95,
            "bootstrap_seed": 42,
        },
        "evaluate_test": False,
        "artifact_dir": "results/runs/H1_temporal_chroma_seed42",
        "approved_by": "REPLACE_AFTER_REVIEW",
        "approved_at": "REPLACE_AFTER_REVIEW",
    }
