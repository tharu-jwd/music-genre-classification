"""Evaluate one automatic chord teacher on an existing annotated benchmark."""

from __future__ import annotations

import _bootstrap  # noqa: F401 - configures direct-script imports

import argparse
import hashlib
import importlib.metadata
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


SCHEMA_VERSION = "chord_teacher_benchmark_v1"
PURPOSE = "external_human_annotated_chord_teacher_evaluation"
PITCH_CLASSES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
CANONICAL_CLASSES = ("N",) + tuple(
    f"{root}:{quality}" for root in PITCH_CLASSES for quality in ("maj", "min")
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _finite_positive(value: object, field: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{field} must be finite and positive")
    return float(value)


def _resolve_file(root: Path, value: object, field: str) -> Path:
    path = Path(_required_text(value, field))
    if not path.is_absolute():
        path = root / path
    if not path.is_file():
        raise FileNotFoundError(f"{field} does not exist: {path}")
    return path.resolve()


def _canonical_majmin(labels: list[str]) -> list[str]:
    import mir_eval.chord

    roots, bitmaps, _ = mir_eval.chord.encode_many(labels)
    result = []
    for root, bitmap in zip(roots, bitmaps):
        if root == -1 and np.all(bitmap == 0):
            result.append("N")
        elif root < 0 or np.any(bitmap < 0):
            result.append("X")
        elif bitmap[4] == 1 and bitmap[3] == 0:
            result.append(f"{PITCH_CLASSES[int(root)]}:maj")
        elif bitmap[3] == 1 and bitmap[4] == 0:
            result.append(f"{PITCH_CLASSES[int(root)]}:min")
        else:
            result.append("X")
    return result


def evaluate_manifest(manifest_path: Path) -> dict:
    import mir_eval.chord
    import mir_eval.io
    import mir_eval.util

    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict) or manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"expected schema_version {SCHEMA_VERSION!r}")
    benchmark = manifest.get("benchmark")
    teacher = manifest.get("teacher")
    if not isinstance(benchmark, dict) or not isinstance(teacher, dict):
        raise ValueError("benchmark and teacher must be objects")
    benchmark_name = _required_text(benchmark.get("name"), "benchmark.name")
    _required_text(benchmark.get("source"), "benchmark.source")
    _required_text(benchmark.get("license"), "benchmark.license")
    teacher_name = _required_text(teacher.get("name"), "teacher.name")
    _required_text(teacher.get("version"), "teacher.version")
    settings = teacher.get("settings")
    if not isinstance(settings, dict) or not settings:
        raise ValueError("teacher.settings must be a non-empty object")
    _required_text(teacher.get("license"), "teacher.license")
    runtime_seconds = _finite_positive(teacher.get("runtime_seconds"), "teacher.runtime_seconds")
    _required_text(teacher.get("device"), "teacher.device")
    confidence_kind = teacher.get("confidence")
    if confidence_kind not in {"none", "frame_probability", "segment_probability"}:
        raise ValueError("teacher.confidence must describe whether confidence is available")

    tracks = manifest.get("tracks")
    if not isinstance(tracks, list) or not tracks:
        raise ValueError("tracks must be a non-empty list")
    track_ids = []
    for index, item in enumerate(tracks):
        if not isinstance(item, dict):
            raise ValueError(f"tracks[{index}] must be an object")
        track_ids.append(_required_text(item.get("track_id"), f"tracks[{index}].track_id"))
    if len(track_ids) != len(set(track_ids)):
        raise ValueError("duplicate track_id in benchmark manifest")
    root = manifest_path.parent
    all_comparisons = []
    all_durations = []
    confusion_seconds: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    track_reports = []
    source_files = []
    total_audio_seconds = 0.0
    no_chord_true_positive = 0.0
    no_chord_reference = 0.0
    no_chord_estimated = 0.0

    for index, item in enumerate(tracks):
        track_id = track_ids[index]
        reference_path = _resolve_file(root, item.get("reference"), f"tracks[{index}].reference")
        estimate_path = _resolve_file(root, item.get("estimate"), f"tracks[{index}].estimate")
        if reference_path == estimate_path:
            raise ValueError(f"track {track_id} reference and estimate paths are identical")
        reference_hash = _sha256_file(reference_path)
        estimate_hash = _sha256_file(estimate_path)
        if reference_hash == estimate_hash:
            raise ValueError(
                f"track {track_id} reference and estimate contents are identical; possible leakage"
            )
        ref_intervals, ref_labels = mir_eval.io.load_labeled_intervals(str(reference_path))
        est_intervals, est_labels = mir_eval.io.load_labeled_intervals(str(estimate_path))
        if not len(ref_intervals) or not len(est_intervals):
            raise ValueError(f"track {track_id} has empty reference or estimate intervals")
        start = float(ref_intervals.min())
        end = float(ref_intervals.max())
        if end <= start:
            raise ValueError(f"track {track_id} has non-positive reference duration")
        est_intervals, est_labels = mir_eval.util.adjust_intervals(
            est_intervals, est_labels, start, end, mir_eval.chord.NO_CHORD, mir_eval.chord.NO_CHORD
        )
        intervals, aligned_reference, aligned_estimate = mir_eval.util.merge_labeled_intervals(
            ref_intervals, ref_labels, est_intervals, est_labels
        )
        mir_eval.chord.validate(aligned_reference, aligned_estimate)
        durations = mir_eval.util.intervals_to_durations(intervals).astype(float)
        comparisons = mir_eval.chord.majmin(aligned_reference, aligned_estimate)
        canonical_reference = _canonical_majmin(aligned_reference)
        canonical_estimate = _canonical_majmin(aligned_estimate)
        for ref_class, est_class, duration in zip(
            canonical_reference, canonical_estimate, durations
        ):
            confusion_seconds[ref_class][est_class] += float(duration)
            if ref_class != "X":
                if ref_class == "N":
                    no_chord_reference += float(duration)
                if est_class == "N":
                    no_chord_estimated += float(duration)
                if ref_class == "N" and est_class == "N":
                    no_chord_true_positive += float(duration)
        score = float(mir_eval.chord.weighted_accuracy(comparisons, durations))
        comparable = comparisons >= 0
        duration = float(durations.sum())
        total_audio_seconds += duration
        all_comparisons.append(comparisons)
        all_durations.append(durations)
        track_reports.append({
            "track_id": track_id,
            "duration_seconds": duration,
            "weighted_chord_accuracy_majmin": score,
            "comparable_duration_fraction": (
                float(durations[comparable].sum() / duration) if duration > 0 else 0.0
            ),
            "reference": str(reference_path),
            "reference_sha256": reference_hash,
            "estimate": str(estimate_path),
            "estimate_sha256": estimate_hash,
        })
        source_files.extend((reference_path, estimate_path))

    comparisons = np.concatenate(all_comparisons)
    durations = np.concatenate(all_durations)
    comparable = comparisons >= 0
    per_class = {}
    for class_name in CANONICAL_CLASSES:
        row = confusion_seconds.get(class_name, {})
        support = float(sum(row.values()))
        correct = float(row.get(class_name, 0.0))
        per_class[class_name] = {
            "support_seconds": support,
            "correct_seconds": correct,
            "recall": correct / support if support > 0 else None,
            "confusions_seconds": {
                estimated: seconds
                for estimated, seconds in sorted(row.items())
                if seconds > 0 and estimated != class_name
            },
        }
    precision = (
        no_chord_true_positive / no_chord_estimated if no_chord_estimated > 0 else None
    )
    recall = (
        no_chord_true_positive / no_chord_reference if no_chord_reference > 0 else None
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall > 0
        else None
    )
    unique_files = sorted(set(source_files))
    return {
        "schema_version": "chord_teacher_evaluation_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": PURPOSE,
        "automatic_selection": False,
        "source_manifest": str(manifest_path.resolve()),
        "source_manifest_sha256": _sha256_file(manifest_path),
        "benchmark": {**benchmark, "name": benchmark_name},
        "teacher": {
            **teacher,
            "name": teacher_name,
            "settings": settings,
            "audio_to_runtime_ratio": total_audio_seconds / runtime_seconds,
        },
        "versions": {"mir_eval": importlib.metadata.version("mir_eval")},
        "tracks": len(track_reports),
        "total_audio_seconds": total_audio_seconds,
        "aggregate": {
            "weighted_chord_accuracy_majmin": float(
                mir_eval.chord.weighted_accuracy(comparisons, durations)
            ),
            "comparable_duration_fraction": float(
                durations[comparable].sum() / durations.sum()
            ),
            "no_chord_precision": precision,
            "no_chord_recall": recall,
            "no_chord_f1": f1,
        },
        "per_reference_class": per_class,
        "track_results": track_reports,
        "source_file_count": len(unique_files),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; teacher evaluations are immutable")
    report = evaluate_manifest(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "teacher": report["teacher"]["name"],
        "tracks": report["tracks"],
        "aggregate": report["aggregate"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
