"""Apply a pre-registered policy to at most two chord-teacher reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path


POLICY_SCHEMA = "chord_teacher_policy_v1"
REPORT_SCHEMA = "chord_teacher_evaluation_v1"
REPORT_PURPOSE = "external_human_annotated_chord_teacher_evaluation"
DECISION_SCHEMA = "chord_teacher_decision_v1"
SOURCE_SCHEMA = "chord_teacher_source_v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _unit(value: object, field: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError(f"{field} must be finite and in [0, 1]")
    return float(value)


def _positive(value: object, field: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{field} must be finite and positive")
    return float(value)


def _positive_unit(value: object, field: str) -> float:
    result = _unit(value, field)
    if result <= 0:
        raise ValueError(f"{field} must be greater than zero")
    return result


def _optional_unit(value: object, field: str) -> float | None:
    return None if value is None else _unit(value, field)


def validate_policy(policy: dict) -> dict:
    if not isinstance(policy, dict) or policy.get("schema_version") != POLICY_SCHEMA:
        raise ValueError(f"policy schema_version must be {POLICY_SCHEMA!r}")
    if policy.get("reports_not_seen") is not True:
        raise ValueError("policy must attest reports_not_seen=true before evaluation")
    registered_at = _text(policy.get("registered_at"), "registered_at")
    approved_by = _text(policy.get("approved_by"), "approved_by")
    if "REPLACE_" in registered_at.upper() or "REPLACE_" in approved_by.upper():
        raise ValueError("policy registration fields still contain template placeholders")
    try:
        parsed_time = datetime.fromisoformat(registered_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("registered_at must be an ISO-8601 timestamp") from error
    if parsed_time.tzinfo is None:
        raise ValueError("registered_at must include a timezone")
    benchmark_name = _text(policy.get("benchmark_name"), "benchmark_name")
    source_manifest_hash = _text(
        policy.get("benchmark_source_manifest_sha256"),
        "benchmark_source_manifest_sha256",
    )
    if re.fullmatch(r"[0-9a-f]{64}", source_manifest_hash) is None:
        raise ValueError("benchmark_source_manifest_sha256 must be a lowercase SHA-256")
    benchmark_tracks = policy.get("benchmark_tracks")
    if not isinstance(benchmark_tracks, list) or not benchmark_tracks:
        raise ValueError("benchmark_tracks must freeze a non-empty benchmark cohort")
    registered_tracks = {}
    for index, item in enumerate(benchmark_tracks):
        if not isinstance(item, dict):
            raise ValueError(f"benchmark_tracks[{index}] must be an object")
        track_id = _text(item.get("track_id"), f"benchmark_tracks[{index}].track_id")
        reference_hash = _text(
            item.get("reference_sha256"), f"benchmark_tracks[{index}].reference_sha256"
        )
        if track_id in registered_tracks or re.fullmatch(r"[0-9a-f]{64}", reference_hash) is None:
            raise ValueError("benchmark_tracks must have unique IDs and lowercase reference hashes")
        registered_tracks[track_id] = reference_hash
    candidates = policy.get("candidates")
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 2:
        raise ValueError("policy must register one or two candidates")
    names = []
    roles = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise ValueError(f"candidates[{index}] must be an object")
        names.append(_text(candidate.get("name"), f"candidates[{index}].name"))
        role = candidate.get("role")
        if role not in {"simple_baseline", "alternative"}:
            raise ValueError(f"candidates[{index}].role is unsupported")
        roles.append(role)
    if len(names) != len(set(names)):
        raise ValueError("candidate names must be unique")
    if roles.count("simple_baseline") != 1 or roles.count("alternative") > 1:
        raise ValueError("policy requires exactly one simple baseline and at most one alternative")

    thresholds = policy.get("thresholds")
    if not isinstance(thresholds, dict):
        raise ValueError("thresholds must be an object")
    normalized_thresholds = {
        "min_weighted_chord_accuracy_majmin": _positive_unit(
            thresholds.get("min_weighted_chord_accuracy_majmin"),
            "thresholds.min_weighted_chord_accuracy_majmin",
        ),
        "min_comparable_duration_fraction": _positive_unit(
            thresholds.get("min_comparable_duration_fraction"),
            "thresholds.min_comparable_duration_fraction",
        ),
        "min_no_chord_f1": _positive_unit(
            thresholds.get("min_no_chord_f1"), "thresholds.min_no_chord_f1"
        ),
        "min_audio_to_runtime_ratio": _positive(
            thresholds.get("min_audio_to_runtime_ratio"),
            "thresholds.min_audio_to_runtime_ratio",
        ),
        "min_wca_improvement_over_simple": _positive_unit(
            thresholds.get("min_wca_improvement_over_simple"),
            "thresholds.min_wca_improvement_over_simple",
        ),
        "max_coverage_drop_vs_simple": _unit(
            thresholds.get("max_coverage_drop_vs_simple"),
            "thresholds.max_coverage_drop_vs_simple",
        ),
        "max_no_chord_f1_drop_vs_simple": _unit(
            thresholds.get("max_no_chord_f1_drop_vs_simple"),
            "thresholds.max_no_chord_f1_drop_vs_simple",
        ),
    }
    require_confidence = thresholds.get("require_confidence")
    if not isinstance(require_confidence, bool):
        raise ValueError("thresholds.require_confidence must be boolean")
    allowed_devices = policy.get("allowed_devices")
    if (
        not isinstance(allowed_devices, list)
        or not allowed_devices
        or any(not isinstance(item, str) or not item.strip() for item in allowed_devices)
    ):
        raise ValueError("allowed_devices must be a non-empty string list")
    return {
        **policy,
        "benchmark_name": benchmark_name,
        "benchmark_source_manifest_sha256": source_manifest_hash,
        "registered_tracks": registered_tracks,
        "candidate_names": names,
        "roles": dict(zip(names, roles)),
        "thresholds": {**normalized_thresholds, "require_confidence": require_confidence},
        "allowed_devices": list(dict.fromkeys(item.strip() for item in allowed_devices)),
    }


def _validate_report(report: dict, expected_name: str, policy: dict) -> dict:
    if not isinstance(report, dict) or report.get("schema_version") != REPORT_SCHEMA:
        raise ValueError(f"report for {expected_name} has the wrong schema")
    if report.get("purpose") != REPORT_PURPOSE or report.get("automatic_selection") is not False:
        raise ValueError(f"report for {expected_name} is not an unselected external evaluation")
    teacher = report.get("teacher")
    benchmark = report.get("benchmark")
    aggregate = report.get("aggregate")
    track_results = report.get("track_results")
    if not all(isinstance(value, dict) for value in (teacher, benchmark, aggregate)):
        raise ValueError(f"report for {expected_name} is missing metadata or aggregates")
    if teacher.get("name") != expected_name:
        raise ValueError(f"report teacher {teacher.get('name')!r} does not match {expected_name!r}")
    if benchmark.get("name") != policy["benchmark_name"]:
        raise ValueError(f"report for {expected_name} uses the wrong benchmark")
    if teacher.get("device") not in policy["allowed_devices"]:
        raise ValueError(f"report for {expected_name} used a non-registered device")
    if not isinstance(track_results, list) or not track_results:
        raise ValueError(f"report for {expected_name} has no track results")
    if report.get("tracks") != len(track_results):
        raise ValueError(f"report for {expected_name} track count is inconsistent")
    confidence = teacher.get("confidence")
    if confidence not in {"none", "frame_probability", "segment_probability"}:
        raise ValueError(f"report for {expected_name} has unsupported confidence metadata")
    tracks = {}
    for item in track_results:
        track_id = _text(item.get("track_id"), "track_results.track_id")
        reference_hash = _text(item.get("reference_sha256"), "reference_sha256")
        if track_id in tracks or re.fullmatch(r"[0-9a-f]{64}", reference_hash) is None:
            raise ValueError(f"report for {expected_name} has invalid track identity/hash")
        tracks[track_id] = (reference_hash, _positive(item.get("duration_seconds"), "duration"))
    reported_reference_hashes = {
        track_id: reference_hash for track_id, (reference_hash, _) in tracks.items()
    }
    if reported_reference_hashes != policy["registered_tracks"]:
        raise ValueError(
            f"report for {expected_name} does not match the preregistered benchmark tracks"
        )
    return {
        "wca": _unit(
            aggregate.get("weighted_chord_accuracy_majmin"),
            "aggregate.weighted_chord_accuracy_majmin",
        ),
        "coverage": _unit(
            aggregate.get("comparable_duration_fraction"),
            "aggregate.comparable_duration_fraction",
        ),
        "no_chord_f1": _optional_unit(
            aggregate.get("no_chord_f1"), "aggregate.no_chord_f1"
        ),
        "speed": _positive(
            teacher.get("audio_to_runtime_ratio"), "teacher.audio_to_runtime_ratio"
        ),
        "confidence": confidence,
        "tracks": tracks,
    }


def decide(policy: dict, reports: dict[str, dict]) -> dict:
    policy = validate_policy(policy)
    if set(reports) != set(policy["candidate_names"]):
        raise ValueError("reports must match exactly the pre-registered candidate names")
    metrics = {
        name: _validate_report(reports[name], name, policy)
        for name in policy["candidate_names"]
    }
    reference_tracks = None
    for name in policy["candidate_names"]:
        tracks = metrics[name]["tracks"]
        if reference_tracks is None:
            reference_tracks = tracks
        elif tracks != reference_tracks:
            raise ValueError("teacher reports do not use identical reference tracks, hashes, and durations")

    thresholds = policy["thresholds"]
    eligible = {}
    reasons = {}
    for name, values in metrics.items():
        failures = []
        if values["wca"] < thresholds["min_weighted_chord_accuracy_majmin"]:
            failures.append("weighted_chord_accuracy_below_minimum")
        if values["coverage"] < thresholds["min_comparable_duration_fraction"]:
            failures.append("comparable_duration_coverage_below_minimum")
        if (
            values["no_chord_f1"] is None
            or values["no_chord_f1"] < thresholds["min_no_chord_f1"]
        ):
            failures.append("no_chord_f1_below_minimum")
        if values["speed"] < thresholds["min_audio_to_runtime_ratio"]:
            failures.append("throughput_below_minimum")
        if thresholds["require_confidence"] and values["confidence"] == "none":
            failures.append("confidence_required_but_unavailable")
        eligible[name] = not failures
        reasons[name] = failures

    simple = next(name for name, role in policy["roles"].items() if role == "simple_baseline")
    alternative = next(
        (name for name, role in policy["roles"].items() if role == "alternative"), None
    )
    selected = None
    rationale = "no_candidate_passed_registered_quality_gates"
    if eligible[simple]:
        selected = simple
        rationale = "simple_baseline_passed_and_is_preferred"
        if alternative is not None and eligible[alternative]:
            improvement = metrics[alternative]["wca"] - metrics[simple]["wca"]
            coverage_drop = metrics[simple]["coverage"] - metrics[alternative]["coverage"]
            no_chord_drop = (
                metrics[simple]["no_chord_f1"] - metrics[alternative]["no_chord_f1"]
            )
            if (
                improvement >= thresholds["min_wca_improvement_over_simple"]
                and coverage_drop <= thresholds["max_coverage_drop_vs_simple"]
                and no_chord_drop <= thresholds["max_no_chord_f1_drop_vs_simple"]
            ):
                selected = alternative
                rationale = "alternative_materially_improved_wca_without_registered_regressions"
    elif alternative is not None and eligible[alternative]:
        selected = alternative
        rationale = "alternative_passed_after_simple_baseline_failed"

    return {
        "schema_version": DECISION_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "decision": "select" if selected is not None else "reject_chord_supervision",
        "selected_teacher": selected,
        "rationale": rationale,
        "eligible": eligible,
        "ineligibility_reasons": reasons,
        "metrics": {
            name: {key: value for key, value in values.items() if key != "tracks"}
            for name, values in metrics.items()
        },
        "criteria": thresholds,
        "next_action": (
            "freeze the selected teacher and confidence policy"
            if selected is not None
            else "ship temporal chroma without chord pseudo-supervision; do not expand teachers"
        ),
    }


def policy_template() -> dict:
    return {
        "schema_version": POLICY_SCHEMA,
        "registered_at": "REPLACE_BEFORE_EVALUATION_WITH_TIMEZONE",
        "approved_by": "REPLACE_BEFORE_EVALUATION",
        "reports_not_seen": True,
        "benchmark_name": "REPLACE_WITH_BENCHMARK_NAME",
        "benchmark_source_manifest_sha256": "REPLACE_WITH_LOWERCASE_SHA256",
        "benchmark_tracks": [{
            "track_id": "REPLACE_WITH_TRACK_ID",
            "reference_sha256": "REPLACE_WITH_LOWERCASE_SHA256",
        }],
        "candidates": [
            {"name": "REPLACE_WITH_SIMPLE_TEACHER", "role": "simple_baseline"},
            {"name": "REPLACE_WITH_ONE_ALTERNATIVE", "role": "alternative"},
        ],
        "allowed_devices": ["cpu"],
        "thresholds": {
            "min_weighted_chord_accuracy_majmin": None,
            "min_comparable_duration_fraction": None,
            "min_no_chord_f1": None,
            "min_audio_to_runtime_ratio": None,
            "min_wca_improvement_over_simple": None,
            "max_coverage_drop_vs_simple": None,
            "max_no_chord_f1_drop_vs_simple": None,
            "require_confidence": True,
        },
    }


def policy_template_for_source(source_path: Path) -> dict:
    try:
        source = json.loads(source_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read chord benchmark source manifest: {error}") from error
    if not isinstance(source, dict) or source.get("schema_version") != SOURCE_SCHEMA:
        raise ValueError(f"source manifest schema_version must be {SOURCE_SCHEMA!r}")
    benchmark = source.get("benchmark")
    tracks = source.get("tracks")
    if not isinstance(benchmark, dict) or not isinstance(tracks, list) or not tracks:
        raise ValueError("source manifest must contain benchmark metadata and tracks")
    frozen_tracks = []
    seen = set()
    for index, item in enumerate(tracks):
        if not isinstance(item, dict):
            raise ValueError(f"source tracks[{index}] must be an object")
        track_id = _text(item.get("track_id"), f"source tracks[{index}].track_id")
        reference_hash = _text(
            item.get("reference_sha256"), f"source tracks[{index}].reference_sha256"
        )
        if track_id in seen or re.fullmatch(r"[0-9a-f]{64}", reference_hash) is None:
            raise ValueError("source tracks must have unique IDs and valid reference hashes")
        seen.add(track_id)
        frozen_tracks.append({"track_id": track_id, "reference_sha256": reference_hash})
    template = policy_template()
    template["benchmark_name"] = _text(benchmark.get("name"), "benchmark.name")
    template["benchmark_source_manifest_sha256"] = _sha256_file(source_path)
    template["benchmark_tracks"] = frozen_tracks
    return template


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy", nargs="?", type=Path)
    parser.add_argument("reports", nargs="*", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--print-policy-template", action="store_true")
    parser.add_argument(
        "--source-manifest",
        type=Path,
        help="frozen chord_teacher_source_v1 used to prefill benchmark identity",
    )
    args = parser.parse_args()
    if args.print_policy_template:
        template = (
            policy_template_for_source(args.source_manifest)
            if args.source_manifest is not None
            else policy_template()
        )
        print(json.dumps(template, indent=2))
        return 0
    if args.source_manifest is not None:
        parser.error("--source-manifest is only valid with --print-policy-template")
    if args.policy is None or not args.reports or args.output is None:
        parser.error("policy, one or two reports, and --output are required")
    if len(args.reports) > 2:
        parser.error("at most two teacher reports are permitted")
    if args.output.exists():
        parser.error("output already exists; teacher decisions are immutable")
    policy = json.loads(args.policy.read_text())
    loaded_reports = []
    for path in args.reports:
        report = json.loads(path.read_text())
        loaded_reports.append((path, report))
    by_name = {report.get("teacher", {}).get("name"): report for _, report in loaded_reports}
    if len(by_name) != len(loaded_reports):
        raise ValueError("teacher report names must be present and unique")
    result = decide(policy, by_name)
    result["source_policy"] = str(args.policy.resolve())
    result["source_policy_sha256"] = _sha256_file(args.policy)
    result["source_reports"] = [
        {"path": str(path.resolve()), "sha256": _sha256_file(path)}
        for path, _ in loaded_reports
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
