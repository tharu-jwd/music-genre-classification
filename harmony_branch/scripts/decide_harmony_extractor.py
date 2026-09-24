"""Apply the pre-registered CPU gate to a harmony extractor comparison report."""

from __future__ import annotations

import _bootstrap  # noqa: F401 - configures direct-script imports

import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_MIN_VALID_FRACTION = 0.05
DEFAULT_MIN_AGREEMENT = 0.75
DEFAULT_MIN_ALIGNMENT_COVERAGE = 0.90
DEFAULT_MAX_CQT_RUNTIME_MULTIPLIER = 2.0
DEFAULT_MAX_CQT_COVERAGE_DEFICIT = 0.10
REGISTERED_SAMPLE_RATE = 22050
HARD_MAX_FILES = 32
HARD_MAX_REGIONS = 384
HARD_MAX_TOTAL_SECONDS = 600.0
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
REPORT_SCHEMA = "harmony_extractor_benchmark_v1"
DECISION_SCHEMA = "harmony_extractor_decision_v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def verify_source_artifact(report: dict) -> Path:
    """Re-hash the exact aligned-region plan before making an immutable decision."""
    value = report.get("source_artifact")
    expected = report.get("source_artifact_sha256")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("registered report must identify its alignment artifact")
    path = Path(value)
    if not path.is_file():
        raise ValueError("source alignment artifact is missing at decision time")
    if SHA256_PATTERN.fullmatch(str(expected)) is None or _sha256_file(path) != expected:
        raise ValueError("source alignment artifact hash changed before decision")
    return path.resolve()


def decide(
    report: dict,
    *,
    min_valid_fraction: float = DEFAULT_MIN_VALID_FRACTION,
    min_agreement: float = DEFAULT_MIN_AGREEMENT,
    min_alignment_coverage: float = DEFAULT_MIN_ALIGNMENT_COVERAGE,
    max_cqt_runtime_multiplier: float = DEFAULT_MAX_CQT_RUNTIME_MULTIPLIER,
    max_cqt_coverage_deficit: float = DEFAULT_MAX_CQT_COVERAGE_DEFICIT,
) -> dict:
    if report.get("schema_version") != REPORT_SCHEMA:
        raise ValueError(f"report schema_version must be {REPORT_SCHEMA!r}")
    if report.get("purpose") != "bounded_cpu_chroma_candidate_comparison":
        raise ValueError("input is not a chroma candidate comparison report")
    if report.get("input_mode") != "aligned_regions":
        raise ValueError("registered decision requires aligned_regions input")
    if report.get("automatic_selection") is not False:
        raise ValueError("comparison report automatic_selection must be false")
    files = report.get("files")
    analysis_units = report.get("analysis_units")
    total_seconds = report.get("total_audio_seconds")
    if report.get("sample_rate") != REGISTERED_SAMPLE_RATE:
        raise ValueError(f"registered report sample_rate must be {REGISTERED_SAMPLE_RATE}")
    limits = report.get("limits")
    if not isinstance(limits, dict):
        raise ValueError("registered report must record its resource limits")
    max_files = limits.get("max_files")
    max_regions = limits.get("max_regions")
    max_seconds = limits.get("max_total_seconds")
    if not isinstance(max_files, int) or not 1 <= max_files <= HARD_MAX_FILES:
        raise ValueError("report max_files exceeds the registered hard cap")
    if not isinstance(max_regions, int) or not 1 <= max_regions <= HARD_MAX_REGIONS:
        raise ValueError("report max_regions exceeds the registered hard cap")
    if not _finite_number(max_seconds) or not 0 < max_seconds <= HARD_MAX_TOTAL_SECONDS:
        raise ValueError("report max_total_seconds exceeds the registered hard cap")
    if not isinstance(files, int) or not 1 <= files <= HARD_MAX_FILES:
        raise ValueError("registered report must contain 1-32 audio files")
    if not isinstance(analysis_units, int) or not 1 <= analysis_units <= HARD_MAX_REGIONS:
        raise ValueError("registered report must contain 1-384 aligned regions")
    if not _finite_number(total_seconds) or not 0 < total_seconds <= HARD_MAX_TOTAL_SECONDS:
        raise ValueError("registered report must contain at most 600 seconds")
    if files > max_files or analysis_units > max_regions or total_seconds > max_seconds:
        raise ValueError("report measurements exceed its declared resource limits")
    items = report.get("items")
    if not isinstance(items, list) or len(items) != analysis_units:
        raise ValueError("report items must match analysis_units")
    if any(item.get("split") not in {"train", "validation"} for item in items):
        raise ValueError("registered report contains an unsupported split")
    artifact_hash = report.get("source_artifact_sha256")
    if not report.get("source_artifact") or not isinstance(artifact_hash, str):
        raise ValueError("registered report must identify and hash its alignment artifact")
    if SHA256_PATTERN.fullmatch(artifact_hash) is None:
        raise ValueError("source alignment artifact hash must be lowercase SHA-256")
    if (
        not 0 <= min_valid_fraction <= 1
        or not 0 <= min_agreement <= 1
        or not 0 <= min_alignment_coverage <= 1
    ):
        raise ValueError("coverage, alignment, and agreement thresholds must be in [0, 1]")
    if max_cqt_runtime_multiplier <= 0 or not 0 <= max_cqt_coverage_deficit <= 1:
        raise ValueError("runtime multiplier must be positive and coverage deficit in [0, 1]")

    aggregate = report.get("aggregate")
    if not isinstance(aggregate, dict) or set(aggregate) != {"cqt", "hpcp_harmonic"}:
        raise ValueError("report must contain exactly the registered CQT and harmonic-HPCP candidates")
    eligible = {}
    reasons = {}
    for name in ("cqt", "hpcp_harmonic"):
        stats = aggregate[name]
        candidate_reasons = []
        if stats.get("failures") != 0:
            candidate_reasons.append("extraction_failures")
        if not isinstance(stats.get("frames"), int) or stats["frames"] <= 0:
            candidate_reasons.append("no_frames")
        valid_frames = stats.get("valid_frames")
        if (
            not isinstance(valid_frames, int)
            or not isinstance(stats.get("frames"), int)
            or not 0 <= valid_frames <= stats["frames"]
        ):
            candidate_reasons.append("invalid_valid_frame_count")
        valid_fraction = stats.get("valid_fraction")
        if not _finite_number(valid_fraction) or not 0 <= valid_fraction <= 1:
            candidate_reasons.append("invalid_valid_frame_coverage")
        elif valid_fraction < min_valid_fraction:
            candidate_reasons.append("insufficient_valid_frame_coverage")
        runtime = stats.get("runtime_seconds")
        if not _finite_number(runtime) or runtime <= 0:
            candidate_reasons.append("invalid_runtime")
        reasons[name] = candidate_reasons
        eligible[name] = not candidate_reasons

    aligned_pairs = report.get("aligned_valid_frame_pairs")
    if not isinstance(aligned_pairs, int) or aligned_pairs < 1:
        raise ValueError("registered report has no aligned valid frame pairs")
    valid_pair_ceiling = min(
        aggregate["cqt"].get("valid_frames", -1),
        aggregate["hpcp_harmonic"].get("valid_frames", -1),
    )
    if valid_pair_ceiling >= 0 and aligned_pairs > valid_pair_ceiling:
        raise ValueError("aligned frame pairs exceed candidate valid-frame counts")
    alignment_coverage = aligned_pairs / valid_pair_ceiling if valid_pair_ceiling > 0 else 0.0
    alignment_gate = alignment_coverage >= min_alignment_coverage
    agreement = report.get("mean_temporal_chroma_cosine_agreement")
    agreement_gate = _finite_number(agreement) and 0 <= agreement <= 1 and agreement >= min_agreement
    decision = "stop"
    selected = None
    rationale = "both_candidates_failed_quality_gate"
    if not alignment_gate:
        rationale = "insufficient_temporal_alignment_requires_method_review"
    elif not agreement_gate:
        rationale = "candidate_disagreement_requires_method_review_not_more_data"
    elif eligible["cqt"]:
        if not eligible["hpcp_harmonic"]:
            decision, selected = "select", "cqt"
            rationale = "cqt_passed_and_hpcp_failed"
        else:
            cqt = aggregate["cqt"]
            hpcp = aggregate["hpcp_harmonic"]
            runtime_ok = cqt["runtime_seconds"] <= (
                max_cqt_runtime_multiplier * hpcp["runtime_seconds"]
            )
            coverage_ok = cqt["valid_fraction"] >= (
                hpcp["valid_fraction"] - max_cqt_coverage_deficit
            )
            if runtime_ok and coverage_ok:
                decision, selected = "select", "cqt"
                rationale = "simpler_cqt_is_not_materially_slower_or_lower_coverage"
            else:
                rationale = "cqt_disadvantage_requires_method_review_not_automatic_hpcp_selection"
    elif eligible["hpcp_harmonic"]:
        decision, selected = "select", "hpcp_harmonic"
        rationale = "hpcp_fallback_passed_after_cqt_failed"

    return {
        "schema_version": DECISION_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "selected_extractor": selected,
        "rationale": rationale,
        "eligible": eligible,
        "ineligibility_reasons": reasons,
        "alignment_coverage": alignment_coverage,
        "alignment_coverage_gate_passed": alignment_gate,
        "agreement_gate_passed": agreement_gate,
        "criteria": {
            "sample_rate": REGISTERED_SAMPLE_RATE,
            "max_files": HARD_MAX_FILES,
            "max_regions": HARD_MAX_REGIONS,
            "max_total_seconds": HARD_MAX_TOTAL_SECONDS,
            "zero_extraction_failures": True,
            "min_valid_fraction": min_valid_fraction,
            "min_temporal_alignment_coverage": min_alignment_coverage,
            "min_mean_temporal_chroma_cosine_agreement": min_agreement,
            "max_cqt_runtime_multiplier": max_cqt_runtime_multiplier,
            "max_cqt_coverage_deficit": max_cqt_coverage_deficit,
            "preference": "CQT for simplicity unless it has a material measured disadvantage",
        },
        "next_action": (
            "freeze the selected extractor contract"
            if decision == "select"
            else "stop extractor expansion and review methods; do not spend GPU"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; gate decisions are immutable")
    report = json.loads(args.report.read_text())
    source_artifact = verify_source_artifact(report)
    decision = decide(report)
    decision["source_report"] = str(args.report.resolve())
    decision["source_report_sha256"] = _sha256_file(args.report)
    decision["verified_source_artifact"] = str(source_artifact)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(decision, indent=2) + "\n")
    print(json.dumps(decision, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
