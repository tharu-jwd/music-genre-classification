"""Generate one bounded CPU Essentia chord-teacher baseline for external evaluation."""

from __future__ import annotations

import _bootstrap  # noqa: F401 - configures direct-script imports

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import re
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from harmony_branch.features import extract_hpcp


SOURCE_SCHEMA = "chord_teacher_source_v1"
OUTPUT_SCHEMA = "chord_teacher_benchmark_v1"
RUN_SCHEMA = "essentia_chord_teacher_run_v1"
TEACHER_NAME = "essentia_chords_detection_hpcp"
MAX_TRACKS = 64
MAX_TOTAL_AUDIO_SECONDS = 3600.0
MAX_WALL_SECONDS = 900.0
SAMPLE_RATE = 22050
# Essentia ChordsDetection explicitly assumes frameSize == 2 * hopSize. At
# 22.05 kHz, 2048/1024 preserves useful frequency resolution and ~46 ms steps.
FRAME_SIZE = 2048
HOP_LENGTH = 1024
WINDOW_SECONDS = 2.0
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


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


def _resolve_file(root: Path, value: object, field: str) -> Path:
    path = Path(_required_text(value, field))
    if not path.is_absolute():
        path = root / path
    if not path.is_file():
        raise FileNotFoundError(f"{field} does not exist: {path}")
    return path.resolve()


def _verify_hash(path: Path, expected: object, field: str) -> str:
    if not isinstance(expected, str) or SHA256_PATTERN.fullmatch(expected) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    actual = _sha256_file(path)
    if actual != expected:
        raise ValueError(f"{field} does not match {path}")
    return actual


def canonicalize_essentia_label(label: str) -> str:
    """Map Essentia's fixed major/minor vocabulary to mir_eval labels."""
    flats = {"Bb": "A#", "Eb": "D#", "Ab": "G#"}
    if not isinstance(label, str) or not label:
        return "N"
    minor = label.endswith("m")
    root = label[:-1] if minor else label
    root = flats.get(root, root)
    if root not in {"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"}:
        return "N"
    return f"{root}:{'min' if minor else 'maj'}"


def frame_labels_to_intervals(
    labels: list[str], timestamps: np.ndarray, duration_seconds: float
) -> list[tuple[float, float, str]]:
    """Convert ordered frame labels to merged, gap-free intervals."""
    times = np.asarray(timestamps, dtype=float)
    if len(labels) != len(times) or not labels:
        raise ValueError("labels and timestamps must be non-empty and have equal length")
    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError("duration_seconds must be finite and positive")
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("timestamps must be finite and strictly increasing")

    # The HPCP timestamp is the frame centre. Shifting by the first centre recovers
    # frame starts at 0, hop, 2*hop, ... without extending padded analysis past audio.
    starts = times - times[0]
    intervals: list[tuple[float, float, str]] = []
    for index, label in enumerate(labels):
        start = max(0.0, float(starts[index]))
        end = (
            min(duration_seconds, float(starts[index + 1]))
            if index + 1 < len(starts)
            else duration_seconds
        )
        if end <= start:
            continue
        if intervals and intervals[-1][2] == label and abs(intervals[-1][1] - start) < 1e-7:
            previous = intervals[-1]
            intervals[-1] = (previous[0], end, label)
        else:
            intervals.append((start, end, label))
    if not intervals or intervals[0][0] != 0.0 or abs(intervals[-1][1] - duration_seconds) > 1e-6:
        raise RuntimeError("generated chord intervals do not cover the audio")
    return intervals


def estimate_chords(waveform: np.ndarray, sample_rate: int) -> tuple[list[tuple[float, float, str]], dict]:
    """Run the fixed Essentia baseline; strength remains diagnostic, not confidence."""
    import essentia.standard as es

    audio = np.asarray(waveform, dtype=np.float32)
    duration = len(audio) / sample_rate
    sequence = extract_hpcp(
        audio,
        sample_rate,
        frame_size=FRAME_SIZE,
        hop_length=HOP_LENGTH,
    )
    native_hpcp = np.roll(sequence.chroma, 3, axis=1)  # project C-first -> Essentia A-first
    raw_labels, strengths = es.ChordsDetection(
        hopSize=HOP_LENGTH,
        sampleRate=sample_rate,
        windowSize=WINDOW_SECONDS,
    )(native_hpcp)
    strengths = np.asarray(strengths, dtype=np.float32)
    if len(raw_labels) != len(sequence.valid) or strengths.shape != (len(raw_labels),):
        raise RuntimeError("Essentia returned inconsistent chord outputs")
    labels = [
        canonicalize_essentia_label(str(label)) if valid else "N"
        for label, valid in zip(raw_labels, sequence.valid)
    ]
    intervals = frame_labels_to_intervals(labels, sequence.timestamps, duration)
    valid_strengths = strengths[sequence.valid]
    diagnostics = {
        "frames": len(labels),
        "invalid_frame_fraction": float(1.0 - sequence.valid.mean()),
        "mean_essentia_strength_on_valid_frames": (
            float(valid_strengths.mean()) if len(valid_strengths) else None
        ),
        "essentia_strength_is_confidence": False,
    }
    return intervals, diagnostics


def _write_lab(path: Path, intervals: list[tuple[float, float, str]]) -> None:
    path.write_text("".join(f"{start:.9f}\t{end:.9f}\t{label}\n" for start, end, label in intervals))


def generate(source_manifest: Path, output_dir: Path) -> dict:
    """Generate immutable estimates and the manifest consumed by the evaluator."""
    import librosa
    import soundfile as sf

    source_manifest = Path(source_manifest).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError("output directory already exists; teacher runs are immutable")
    source = json.loads(source_manifest.read_text())
    if not isinstance(source, dict) or source.get("schema_version") != SOURCE_SCHEMA:
        raise ValueError(f"expected schema_version {SOURCE_SCHEMA!r}")
    benchmark = source.get("benchmark")
    if not isinstance(benchmark, dict):
        raise ValueError("benchmark must be an object")
    for field in ("name", "source", "license"):
        _required_text(benchmark.get(field), f"benchmark.{field}")
    tracks = source.get("tracks")
    if not isinstance(tracks, list) or not 1 <= len(tracks) <= MAX_TRACKS:
        raise ValueError(f"tracks must contain 1 to {MAX_TRACKS} pre-registered items")

    root = source_manifest.parent
    prepared = []
    ids = set()
    total_duration = 0.0
    for index, item in enumerate(tracks):
        if not isinstance(item, dict):
            raise ValueError(f"tracks[{index}] must be an object")
        track_id = _required_text(item.get("track_id"), f"tracks[{index}].track_id")
        if track_id in ids:
            raise ValueError(f"duplicate track_id: {track_id}")
        ids.add(track_id)
        audio = _resolve_file(root, item.get("audio"), f"tracks[{index}].audio")
        reference = _resolve_file(root, item.get("reference"), f"tracks[{index}].reference")
        audio_hash = _verify_hash(audio, item.get("audio_sha256"), f"tracks[{index}].audio_sha256")
        reference_hash = _verify_hash(
            reference, item.get("reference_sha256"), f"tracks[{index}].reference_sha256"
        )
        if audio == reference or audio_hash == reference_hash:
            raise ValueError(f"track {track_id} has invalid audio/reference identity")
        info = sf.info(str(audio))
        duration = float(info.duration)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError(f"track {track_id} has invalid audio duration")
        total_duration += duration
        prepared.append((track_id, audio, reference, audio_hash, reference_hash, duration))
    if total_duration > MAX_TOTAL_AUDIO_SECONDS:
        raise ValueError(
            f"pre-registered audio is {total_duration:.1f}s; hard cap is {MAX_TOTAL_AUDIO_SECONDS:.1f}s"
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        estimates_dir = temporary / "teacher-output"
        estimates_dir.mkdir()
        started = time.monotonic()
        generated_tracks = []
        run_tracks = []
        for track_id, audio, reference, audio_hash, reference_hash, duration in prepared:
            if time.monotonic() - started > MAX_WALL_SECONDS:
                raise TimeoutError(
                    f"teacher generation exceeded the {MAX_WALL_SECONDS:.0f}s wall cap"
                )
            waveform, _ = librosa.load(str(audio), sr=SAMPLE_RATE, mono=True)
            intervals, diagnostics = estimate_chords(waveform, SAMPLE_RATE)
            if time.monotonic() - started > MAX_WALL_SECONDS:
                raise TimeoutError(
                    f"teacher generation exceeded the {MAX_WALL_SECONDS:.0f}s wall cap"
                )
            safe_name = hashlib.sha256(track_id.encode()).hexdigest()[:16] + ".lab"
            estimate = estimates_dir / safe_name
            if estimate.exists():
                raise RuntimeError("teacher estimate filename collision")
            _write_lab(estimate, intervals)
            final_estimate = output_dir / "teacher-output" / safe_name
            generated_tracks.append({
                "track_id": track_id,
                "reference": os.path.relpath(reference, output_dir),
                "estimate": os.path.relpath(final_estimate, output_dir),
            })
            run_tracks.append({
                "track_id": track_id,
                "audio": str(audio),
                "audio_sha256": audio_hash,
                "reference": str(reference),
                "reference_sha256": reference_hash,
                "estimate": str(final_estimate),
                "estimate_sha256": _sha256_file(estimate),
                "duration_seconds": duration,
                **diagnostics,
            })
        runtime = time.monotonic() - started
        version = importlib.metadata.version("essentia")
        settings = {
            "hpcp_extractor": "harmony_branch.features.extract_hpcp",
            "sample_rate": SAMPLE_RATE,
            "frame_size": FRAME_SIZE,
            "hop_length": HOP_LENGTH,
            "window_seconds": WINDOW_SECONDS,
            "vocabulary": "12_major_12_minor_no_chord",
            "silence_rule": "invalid_hpcp_frame_to_no_chord",
            "strength_policy": "diagnostic_only_not_probability",
        }
        evaluation_manifest = {
            "schema_version": OUTPUT_SCHEMA,
            "benchmark": benchmark,
            "teacher": {
                "name": TEACHER_NAME,
                "version": version,
                "settings": settings,
                "license": "AGPL-3.0",
                "runtime_seconds": max(runtime, np.finfo(float).eps),
                "device": "cpu",
                "confidence": "none",
            },
            "tracks": generated_tracks,
        }
        temporary_evaluation = temporary / "evaluation-manifest.json"
        temporary_evaluation.write_text(json.dumps(evaluation_manifest, indent=2) + "\n")
        final_evaluation = output_dir / "evaluation-manifest.json"
        report = {
            "schema_version": RUN_SCHEMA,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "ready_for_evaluation",
            "source_manifest": str(source_manifest),
            "source_manifest_sha256": _sha256_file(source_manifest),
            "evaluation_manifest": str(final_evaluation),
            "evaluation_manifest_sha256": _sha256_file(temporary_evaluation),
            "resource_limits": {
                "max_tracks": MAX_TRACKS,
                "max_total_audio_seconds": MAX_TOTAL_AUDIO_SECONDS,
                "max_wall_seconds": MAX_WALL_SECONDS,
                "device": "cpu",
            },
            "runtime_seconds": runtime,
            "total_audio_seconds": total_duration,
            "tracks": run_tracks,
        }
        (temporary / "run-report.json").write_text(json.dumps(report, indent=2) + "\n")
        temporary.rename(output_dir)
        return report
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_manifest", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    report = generate(args.source_manifest, args.output_dir)
    print(json.dumps({
        "status": report["status"],
        "tracks": len(report["tracks"]),
        "total_audio_seconds": report["total_audio_seconds"],
        "runtime_seconds": report["runtime_seconds"],
        "evaluation_manifest": report["evaluation_manifest"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
