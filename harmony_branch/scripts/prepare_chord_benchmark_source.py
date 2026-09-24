"""Freeze an existing chord benchmark mapping for the bounded Essentia baseline."""

from __future__ import annotations

import _bootstrap  # noqa: F401 - configures direct-script imports

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

SOURCE_SCHEMA = "chord_teacher_source_v1"
MAX_TRACKS = 64
MAX_TOTAL_AUDIO_SECONDS = 3600.0
REQUIRED_COLUMNS = ("track_id", "audio", "reference")


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


def _resolve(root: Path, value: object, field: str) -> Path:
    path = Path(_text(value, field))
    if not path.is_absolute():
        path = root / path
    if not path.is_file():
        raise FileNotFoundError(f"{field} does not exist: {path}")
    return path.resolve()


def prepare_source_manifest(
    mapping_path: Path,
    output_path: Path,
    *,
    root: Path,
    benchmark_name: str,
    benchmark_source: str,
    benchmark_license: str,
) -> dict:
    """Hash a preselected mapping; never inspect reference label contents."""
    import soundfile as sf

    mapping_path = Path(mapping_path).resolve()
    output_path = Path(output_path).resolve()
    root = Path(root).resolve()
    if output_path.exists():
        raise FileExistsError("output already exists; benchmark cohorts are immutable")
    with mapping_path.open(newline="", encoding="utf-8", errors="strict") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or any(name not in reader.fieldnames for name in REQUIRED_COLUMNS):
            raise ValueError(f"mapping CSV must contain columns {REQUIRED_COLUMNS}")
        rows = list(reader)
    if not 1 <= len(rows) <= MAX_TRACKS:
        raise ValueError(f"mapping must contain 1 to {MAX_TRACKS} preselected tracks")

    tracks = []
    seen = set()
    total_seconds = 0.0
    for index, row in enumerate(rows):
        track_id = _text(row.get("track_id"), f"row {index + 2} track_id")
        if track_id in seen:
            raise ValueError(f"duplicate track_id: {track_id}")
        seen.add(track_id)
        audio = _resolve(root, row.get("audio"), f"row {index + 2} audio")
        reference = _resolve(root, row.get("reference"), f"row {index + 2} reference")
        if audio == reference:
            raise ValueError(f"track {track_id} audio and reference paths are identical")
        duration = float(sf.info(str(audio)).duration)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError(f"track {track_id} has invalid audio duration")
        total_seconds += duration
        tracks.append({
            "track_id": track_id,
            "audio": str(audio),
            "audio_sha256": _sha256_file(audio),
            "reference": str(reference),
            "reference_sha256": _sha256_file(reference),
        })
    if total_seconds > MAX_TOTAL_AUDIO_SECONDS:
        raise ValueError(
            f"mapping contains {total_seconds:.1f}s audio; hard cap is {MAX_TOTAL_AUDIO_SECONDS:.1f}s"
        )

    result = {
        "schema_version": SOURCE_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "pre_registered_external_chord_teacher_benchmark",
        "reference_usage": "opaque_hash_and_evaluation_only_not_teacher_inference",
        "mapping_csv": str(mapping_path),
        "mapping_csv_sha256": _sha256_file(mapping_path),
        "benchmark": {
            "name": _text(benchmark_name, "benchmark_name"),
            "source": _text(benchmark_source, "benchmark_source"),
            "license": _text(benchmark_license, "benchmark_license"),
        },
        "resource_limits": {
            "max_tracks": MAX_TRACKS,
            "max_total_audio_seconds": MAX_TOTAL_AUDIO_SECONDS,
        },
        "track_count": len(tracks),
        "total_audio_seconds": total_seconds,
        "tracks": tracks,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mapping", type=Path, help="CSV with track_id,audio,reference")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--benchmark-name", required=True)
    parser.add_argument("--benchmark-source", required=True)
    parser.add_argument("--benchmark-license", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = prepare_source_manifest(
        args.mapping,
        args.output,
        root=args.root,
        benchmark_name=args.benchmark_name,
        benchmark_source=args.benchmark_source,
        benchmark_license=args.benchmark_license,
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "tracks": result["track_count"],
        "total_audio_seconds": result["total_audio_seconds"],
        "mapping_csv_sha256": result["mapping_csv_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
