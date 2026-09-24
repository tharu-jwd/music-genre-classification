"""Apply a preregistered policy to one CPU temporal-harmony branch screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path


POLICY_SCHEMA = "harmony_branch_screen_policy_v1"
REPORT_SCHEMA = "harmony_branch_screen_v1"
SHA256 = re.compile(r"[0-9a-f]{64}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    result = value.strip()
    if "REPLACE_" in result.upper():
        raise ValueError(f"{field} contains a template placeholder")
    return result


def _finite(value: object, field: str, *, minimum: float = 0.0) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value < minimum
    ):
        raise ValueError(f"{field} must be finite and at least {minimum}")
    return float(value)


def validate_policy(policy: dict) -> dict:
    if not isinstance(policy, dict) or policy.get("schema_version") != POLICY_SCHEMA:
        raise ValueError(f"policy schema_version must be {POLICY_SCHEMA!r}")
    if policy.get("reports_not_seen") is not True:
        raise ValueError("policy must attest reports_not_seen=true")
    registered_at = _text(policy.get("registered_at"), "registered_at")
    approved_by = _text(policy.get("approved_by"), "approved_by")
    try:
        parsed = datetime.fromisoformat(registered_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("registered_at must be ISO-8601") from error
    if parsed.tzinfo is None:
        raise ValueError("registered_at must include a timezone")
    cohort_hash = _text(policy.get("cohort_sha256"), "cohort_sha256")
    dataset_hash = _text(policy.get("screen_dataset_sha256"), "screen_dataset_sha256")
    if SHA256.fullmatch(cohort_hash) is None or SHA256.fullmatch(dataset_hash) is None:
        raise ValueError("policy cohort and dataset hashes must be lowercase SHA-256")
    thresholds = policy.get("thresholds")
    if not isinstance(thresholds, dict):
        raise ValueError("thresholds must be an object")
    normalized = {
        "max_validation_cross_entropy": _finite(
            thresholds.get("max_validation_cross_entropy"),
            "thresholds.max_validation_cross_entropy",
        ),
        "min_validation_cosine_similarity": _finite(
            thresholds.get("min_validation_cosine_similarity"),
            "thresholds.min_validation_cosine_similarity",
        ),
        "min_cross_entropy_improvement_over_training_mean": _finite(
            thresholds.get("min_cross_entropy_improvement_over_training_mean"),
            "thresholds.min_cross_entropy_improvement_over_training_mean",
        ),
        "max_cpu_wall_seconds": _finite(
            thresholds.get("max_cpu_wall_seconds"),
            "thresholds.max_cpu_wall_seconds",
            minimum=1e-12,
        ),
    }
    if normalized["min_validation_cosine_similarity"] > 1:
        raise ValueError("minimum cosine similarity must be in [0, 1]")
    if normalized["min_cross_entropy_improvement_over_training_mean"] <= 0:
        raise ValueError("minimum improvement over training mean must be positive")
    if normalized["max_cpu_wall_seconds"] > 300:
        raise ValueError("policy cannot exceed the 300-second CPU screening cap")
    max_parameters = thresholds.get("max_parameter_count")
    if not isinstance(max_parameters, int) or isinstance(max_parameters, bool) or max_parameters < 1:
        raise ValueError("thresholds.max_parameter_count must be a positive integer")
    normalized["max_parameter_count"] = max_parameters
    return {
        **policy,
        "registered_at": registered_at,
        "approved_by": approved_by,
        "cohort_sha256": cohort_hash,
        "screen_dataset_sha256": dataset_hash,
        "thresholds": normalized,
    }


def decide(policy: dict, report: dict) -> dict:
    policy = validate_policy(policy)
    if not isinstance(report, dict) or report.get("schema_version") != REPORT_SCHEMA:
        raise ValueError(f"report schema_version must be {REPORT_SCHEMA!r}")
    if (
        report.get("status") != "ready"
        or report.get("automatic_selection") is not False
        or report.get("device") != "cpu"
    ):
        raise ValueError("report must be a ready, unselected CPU screen")
    if report.get("cohort_sha256") != policy["cohort_sha256"]:
        raise ValueError("report cohort does not match the preregistered policy")
    if report.get("source_dataset_sha256") != policy["screen_dataset_sha256"]:
        raise ValueError("report dataset does not match the preregistered policy")
    configuration = report.get("configuration")
    expected_configuration = {
        "seed": 42,
        "embedding_dim": 32,
        "hidden_dim": 64,
        "temporal_layers": 2,
        "dropout": 0.1,
        "learning_rate": 1e-3,
        "patience": 3,
    }
    if not isinstance(configuration, dict) or any(
        configuration.get(key) != value for key, value in expected_configuration.items()
    ):
        raise ValueError("report does not use the registered fixed reference configuration")
    validation = report.get("best_validation")
    baselines = report.get("baselines")
    if not isinstance(validation, dict) or not isinstance(baselines, dict):
        raise ValueError("report is missing validation or baseline metrics")
    training_mean = baselines.get("training_mean")
    if not isinstance(training_mean, dict):
        raise ValueError("report is missing the training-mean baseline")
    cross_entropy = _finite(validation.get("cross_entropy"), "validation.cross_entropy")
    cosine = _finite(
        validation.get("mean_cosine_similarity"), "validation.mean_cosine_similarity"
    )
    if cosine > 1:
        raise ValueError("validation cosine similarity must be in [0, 1]")
    baseline_ce = _finite(
        training_mean.get("cross_entropy"), "baselines.training_mean.cross_entropy"
    )
    elapsed = _finite(
        report.get("elapsed_cpu_wall_seconds"), "elapsed_cpu_wall_seconds"
    )
    parameters = report.get("parameter_count")
    if not isinstance(parameters, int) or isinstance(parameters, bool) or parameters < 1:
        raise ValueError("report parameter_count must be positive")
    improvement = baseline_ce - cross_entropy
    thresholds = policy["thresholds"]
    checks = {
        "validation_cross_entropy": cross_entropy <= thresholds["max_validation_cross_entropy"],
        "validation_cosine_similarity": cosine >= thresholds["min_validation_cosine_similarity"],
        "improvement_over_training_mean": (
            improvement >= thresholds["min_cross_entropy_improvement_over_training_mean"]
        ),
        "cpu_wall_time": elapsed <= thresholds["max_cpu_wall_seconds"],
        "parameter_count": parameters <= thresholds["max_parameter_count"],
    }
    advance = all(checks.values())
    return {
        "schema_version": "harmony_branch_screen_decision_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target_variant": "temporal_chroma_v1",
        "cohort_sha256": policy["cohort_sha256"],
        "screen_dataset_sha256": policy["screen_dataset_sha256"],
        "decision": "advance_to_gpu_registration" if advance else "stop_before_gpu",
        "checks": checks,
        "metrics": {
            "validation_cross_entropy": cross_entropy,
            "validation_cosine_similarity": cosine,
            "training_mean_cross_entropy": baseline_ce,
            "cross_entropy_improvement_over_training_mean": improvement,
            "cpu_wall_seconds": elapsed,
            "parameter_count": parameters,
        },
        "criteria": thresholds,
        "next_action": (
            "prepare one preregistered H1 GPU request; this decision does not itself authorize CUDA"
            if advance
            else "stop temporal harmony GPU escalation and report the failed CPU gate"
        ),
    }


def verify_report_files(report: dict, report_path: Path) -> dict:
    dataset_path = Path(str(report.get("source_dataset", "")))
    checkpoint_path = Path(str(report.get("checkpoint", "")))
    if not checkpoint_path.is_absolute():
        checkpoint_path = report_path.parent / checkpoint_path
    expected_dataset = report.get("source_dataset_sha256")
    expected_checkpoint = report.get("checkpoint_sha256")
    if (
        not dataset_path.is_file()
        or SHA256.fullmatch(str(expected_dataset)) is None
        or _sha256_file(dataset_path) != expected_dataset
    ):
        raise ValueError("screen report source dataset is missing or changed")
    if (
        not checkpoint_path.is_file()
        or SHA256.fullmatch(str(expected_checkpoint)) is None
        or _sha256_file(checkpoint_path) != expected_checkpoint
    ):
        raise ValueError("screen report checkpoint is missing or changed")
    return {
        "source_dataset": str(dataset_path.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
    }


def policy_template() -> dict:
    return {
        "schema_version": POLICY_SCHEMA,
        "registered_at": "REPLACE_BEFORE_SCREEN_WITH_TIMEZONE",
        "approved_by": "REPLACE_BEFORE_SCREEN",
        "reports_not_seen": True,
        "cohort_sha256": "REPLACE_WITH_FROZEN_COHORT_SHA256",
        "screen_dataset_sha256": "REPLACE_WITH_SCREEN_DATASET_INDEX_SHA256",
        "thresholds": {
            "max_validation_cross_entropy": None,
            "min_validation_cosine_similarity": None,
            "min_cross_entropy_improvement_over_training_mean": None,
            "max_cpu_wall_seconds": None,
            "max_parameter_count": None,
        },
    }


def policy_template_for_dataset(dataset_path: Path) -> dict:
    try:
        dataset = json.loads(dataset_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read screen dataset index {dataset_path}: {error}") from error
    if (
        not isinstance(dataset, dict)
        or dataset.get("schema_version") != "harmony_screen_dataset_v1"
        or dataset.get("status") != "ready"
    ):
        raise ValueError("policy template requires a ready harmony_screen_dataset_v1 index")
    cohort_hash = dataset.get("cohort_sha256")
    if SHA256.fullmatch(str(cohort_hash)) is None:
        raise ValueError("screen dataset has no valid cohort SHA-256")
    template = policy_template()
    template["cohort_sha256"] = cohort_hash
    template["screen_dataset_sha256"] = _sha256_file(dataset_path)
    return template


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy", nargs="?", type=Path)
    parser.add_argument("report", nargs="?", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--print-policy-template", action="store_true")
    parser.add_argument(
        "--dataset",
        type=Path,
        help="ready screen dataset index used to prefill policy hashes",
    )
    args = parser.parse_args()
    if args.print_policy_template:
        template = (
            policy_template_for_dataset(args.dataset)
            if args.dataset is not None
            else policy_template()
        )
        print(json.dumps(template, indent=2))
        return 0
    if args.dataset is not None:
        parser.error("--dataset is only valid with --print-policy-template")
    if args.policy is None or args.report is None or args.output is None:
        parser.error("policy, report, and --output are required")
    if args.output.exists():
        parser.error("output already exists; branch-screen decisions are immutable")
    policy = json.loads(args.policy.read_text())
    report = json.loads(args.report.read_text())
    verified_files = verify_report_files(report, args.report)
    result = decide(policy, report)
    result["source_policy"] = str(args.policy.resolve())
    result["source_policy_sha256"] = _sha256_file(args.policy)
    result["source_report"] = str(args.report.resolve())
    result["source_report_sha256"] = _sha256_file(args.report)
    result["verified_files"] = verified_files
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
