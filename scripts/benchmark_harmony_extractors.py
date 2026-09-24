"""Bounded CPU comparison of the two temporal chroma candidates."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    from scripts.harmony_chroma import extract_cqt_chroma, extract_hpcp
except ModuleNotFoundError:  # Direct execution adds scripts/, not the repository root.
    from harmony_chroma import extract_cqt_chroma, extract_hpcp


DEFAULT_MAX_FILES = 32
DEFAULT_MAX_TOTAL_SECONDS = 600.0
DEFAULT_MAX_REGIONS = DEFAULT_MAX_FILES * 12
REGISTERED_SAMPLE_RATE = 22050
SCHEMA_VERSION = "harmony_extractor_benchmark_v1"


def _validate_caps(*, max_files: int, max_regions: int, max_total_seconds: float) -> None:
    """Allow tighter local limits, never expansion beyond the registered ceilings."""
    if not 1 <= max_files <= DEFAULT_MAX_FILES:
        raise ValueError(f"max_files must be in [1, {DEFAULT_MAX_FILES}]")
    if not 1 <= max_regions <= DEFAULT_MAX_REGIONS:
        raise ValueError(f"max_regions must be in [1, {DEFAULT_MAX_REGIONS}]")
    if not 0 < max_total_seconds <= DEFAULT_MAX_TOTAL_SECONDS:
        raise ValueError(
            f"max_total_seconds must be in (0, {DEFAULT_MAX_TOTAL_SECONDS:g}]"
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _temporal_chroma_agreement(left, right) -> tuple[list[float], int]:
    """Compare nearest valid frames without discarding their temporal order."""
    left_indices = np.flatnonzero(left.valid)
    right_indices = np.flatnonzero(right.valid)
    if not left_indices.size or not right_indices.size:
        return [], 0
    left_times = left.timestamps[left_indices]
    frame_period = max(
        left.hop_length / left.sample_rate,
        right.hop_length / right.sample_rate,
    )
    tolerance = frame_period / 2 + 1e-9
    agreements = []
    for right_index in right_indices:
        timestamp = right.timestamps[right_index]
        insertion = int(np.searchsorted(left_times, timestamp))
        candidates = [
            index for index in (insertion - 1, insertion) if 0 <= index < len(left_times)
        ]
        nearest = min(candidates, key=lambda index: abs(float(left_times[index] - timestamp)))
        if abs(float(left_times[nearest] - timestamp)) > tolerance:
            continue
        left_vector = left.chroma[left_indices[nearest]]
        right_vector = right.chroma[right_index]
        denominator = float(np.linalg.norm(left_vector) * np.linalg.norm(right_vector))
        if denominator > 0:
            cosine = float(np.dot(left_vector, right_vector) / denominator)
            agreements.append(min(1.0, max(0.0, cosine)))
    return agreements, len(agreements)


def _benchmark_decoded(
    decoded: list[dict],
    *,
    sample_rate: int,
    max_files: int,
    max_regions: int,
    max_total_seconds: float,
    input_mode: str,
    source_artifact: Path | None = None,
) -> dict:
    total_seconds = float(sum(unit["duration_seconds"] for unit in decoded))
    unique_paths = sorted({unit["path"] for unit in decoded})
    extractors = {
        "cqt": extract_cqt_chroma,
        "hpcp_harmonic": extract_hpcp,
    }
    aggregate = {
        name: {"runtime_seconds": 0.0, "frames": 0, "valid_frames": 0, "failures": 0}
        for name in extractors
    }
    items = []
    agreements = []
    hashes = {path: _sha256_file(path) for path in unique_paths}
    for unit in decoded:
        path = unit["path"]
        waveform = unit["waveform"]
        actual_rate = unit["sample_rate"]
        duration = unit["duration_seconds"]
        item = {
            "unit_id": unit["unit_id"],
            "song_id": unit.get("song_id"),
            "split": unit.get("split"),
            "path": str(path),
            "sha256": hashes[path],
            "duration_seconds": duration,
            "source_region_seconds": unit["source_region_seconds"],
            "extractors": {},
        }
        sequences = {}
        for name, extractor in extractors.items():
            started = time.perf_counter()
            try:
                sequence = extractor(waveform, actual_rate)
                elapsed = time.perf_counter() - started
                valid_count = int(sequence.valid.sum())
                aggregate[name]["runtime_seconds"] += elapsed
                aggregate[name]["frames"] += len(sequence.timestamps)
                aggregate[name]["valid_frames"] += valid_count
                sequences[name] = sequence
                item["extractors"][name] = {
                    "runtime_seconds": elapsed,
                    "frames": len(sequence.timestamps),
                    "valid_frames": valid_count,
                    "valid_fraction": valid_count / len(sequence.timestamps),
                    "mean_tonal_concentration": (
                        float(sequence.tonal_concentration[sequence.valid].mean())
                        if valid_count else None
                    ),
                    "tuning_value": sequence.tuning_value,
                    "tuning_unit": sequence.tuning_unit,
                }
            except Exception as error:  # Preserve failures in the comparison report.
                elapsed = time.perf_counter() - started
                aggregate[name]["runtime_seconds"] += elapsed
                aggregate[name]["failures"] += 1
                sequences[name] = None
                item["extractors"][name] = {
                    "runtime_seconds": elapsed,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
        left = sequences.get("cqt")
        right = sequences.get("hpcp_harmonic")
        frame_agreements, aligned_pairs = (
            _temporal_chroma_agreement(left, right)
            if left is not None and right is not None
            else ([], 0)
        )
        agreement = float(np.mean(frame_agreements)) if frame_agreements else None
        item["mean_temporal_chroma_cosine_agreement"] = agreement
        item["aligned_valid_frame_pairs"] = aligned_pairs
        agreements.extend(frame_agreements)
        items.append(item)

    for stats in aggregate.values():
        runtime = stats["runtime_seconds"]
        stats["audio_to_runtime_ratio"] = total_seconds / runtime if runtime > 0 else None
        frames = stats["frames"]
        stats["valid_fraction"] = stats["valid_frames"] / frames if frames else None

    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "bounded_cpu_chroma_candidate_comparison",
        "automatic_selection": False,
        "input_mode": input_mode,
        "source_artifact": str(source_artifact.resolve()) if source_artifact else None,
        "source_artifact_sha256": _sha256_file(source_artifact) if source_artifact else None,
        "sample_rate": sample_rate,
        "limits": {
            "max_files": max_files,
            "max_regions": max_regions,
            "max_total_seconds": max_total_seconds,
        },
        "files": len(unique_paths),
        "analysis_units": len(decoded),
        "total_audio_seconds": total_seconds,
        "versions": {
            "numpy": importlib.metadata.version("numpy"),
            "librosa": importlib.metadata.version("librosa"),
            "essentia": importlib.metadata.version("essentia"),
        },
        "aggregate": aggregate,
        "mean_temporal_chroma_cosine_agreement": (
            float(np.mean(agreements)) if agreements else None
        ),
        "aligned_valid_frame_pairs": len(agreements),
        "items": items,
    }


def benchmark_paths(
    paths: list[Path],
    *,
    sample_rate: int = 22050,
    max_files: int = DEFAULT_MAX_FILES,
    max_total_seconds: float = DEFAULT_MAX_TOTAL_SECONDS,
) -> dict:
    """Benchmark whole files; retained for synthetic fixtures and ad-hoc CPU checks."""
    import librosa

    _validate_caps(
        max_files=max_files,
        max_regions=max_files,
        max_total_seconds=max_total_seconds,
    )
    normalized = sorted({Path(path).resolve() for path in paths})
    if not normalized:
        raise ValueError("at least one audio path is required")
    if len(normalized) > max_files:
        raise ValueError(f"requested {len(normalized)} files; cap is {max_files}")
    if sample_rate <= 0:
        raise ValueError("sample rate must be positive")
    missing = [str(path) for path in normalized if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing audio files: {missing[:3]}")

    decoded = []
    total_seconds = 0.0
    for index, path in enumerate(normalized):
        waveform, actual_rate = librosa.load(path, sr=sample_rate, mono=True)
        duration = len(waveform) / actual_rate
        total_seconds += duration
        if total_seconds > max_total_seconds:
            raise ValueError(
                f"development audio totals {total_seconds:.1f}s; cap is {max_total_seconds:.1f}s"
            )
        decoded.append({
            "unit_id": f"file_{index}",
            "path": path,
            "waveform": waveform,
            "sample_rate": actual_rate,
            "duration_seconds": duration,
            "source_region_seconds": [0.0, duration],
        })
    return _benchmark_decoded(
        decoded,
        sample_rate=sample_rate,
        max_files=max_files,
        max_regions=len(decoded),
        max_total_seconds=max_total_seconds,
        input_mode="whole_files",
    )


def benchmark_region_artifact(
    artifact_path: Path,
    *,
    sample_rate: int = 22050,
    max_files: int = DEFAULT_MAX_FILES,
    max_regions: int = DEFAULT_MAX_REGIONS,
    max_total_seconds: float = DEFAULT_MAX_TOTAL_SECONDS,
) -> dict:
    """Benchmark only the frozen model-aligned regions; never inspect test songs."""
    import librosa

    _validate_caps(
        max_files=max_files,
        max_regions=max_regions,
        max_total_seconds=max_total_seconds,
    )
    if sample_rate != REGISTERED_SAMPLE_RATE:
        raise ValueError(
            f"registered region comparison sample_rate must be {REGISTERED_SAMPLE_RATE}"
        )
    artifact_path = Path(artifact_path)
    artifact = json.loads(artifact_path.read_text())
    if artifact.get("schema_version") != "harmony_audio_regions_v1":
        raise ValueError("expected a harmony_audio_regions_v1 artifact")
    songs = artifact.get("songs")
    if not isinstance(songs, list) or not songs:
        raise ValueError("region artifact must contain at least one song")
    if len(songs) > max_files:
        raise ValueError(f"requested {len(songs)} files; cap is {max_files}")
    seen_song_ids = set()
    specifications = []
    requested_seconds = 0.0
    for song in songs:
        song_id = str(song.get("song_id", ""))
        split = song.get("split")
        if not song_id or song_id in seen_song_ids:
            raise ValueError("region artifact has missing or duplicate song IDs")
        seen_song_ids.add(song_id)
        if split not in {"train", "validation"}:
            raise ValueError("region benchmark permits train and validation songs only")
        path = Path(str(song.get("audio_path", ""))).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        regions = song.get("regions")
        if not isinstance(regions, list) or not regions:
            raise ValueError(f"song {song_id} has no aligned regions")
        previous_end = -1.0
        previous_window_index = -1
        for region in regions:
            window_index = region.get("window_index")
            if not isinstance(window_index, int) or window_index <= previous_window_index:
                raise ValueError(f"song {song_id} region window indices are not increasing")
            previous_window_index = window_index
            start = region.get("start_seconds")
            end = region.get("end_seconds")
            if (
                not isinstance(start, (int, float))
                or not isinstance(end, (int, float))
                or start < 0
                or end <= start
                or start < previous_end - 1e-9
            ):
                raise ValueError(f"song {song_id} has invalid or overlapping regions")
            previous_end = float(end)
            requested_seconds += float(end) - float(start)
            specifications.append((song_id, split, path, window_index, float(start), float(end)))
    if len(specifications) > max_regions:
        raise ValueError(f"requested {len(specifications)} regions; cap is {max_regions}")
    if requested_seconds > max_total_seconds:
        raise ValueError(
            f"aligned regions total {requested_seconds:.1f}s; cap is {max_total_seconds:.1f}s"
        )

    decoded = []
    for song_id, split, path, window_index, start, end in specifications:
        requested_duration = end - start
        waveform, actual_rate = librosa.load(
            path, sr=sample_rate, mono=True, offset=start, duration=requested_duration
        )
        duration = len(waveform) / actual_rate
        if duration + 0.1 < requested_duration:
            raise ValueError(
                f"audio ended before aligned region {song_id}:{window_index}: "
                f"requested {requested_duration:.3f}s, decoded {duration:.3f}s"
            )
        decoded.append({
            "unit_id": f"{song_id}:window_{window_index}",
            "song_id": song_id,
            "split": split,
            "path": path,
            "waveform": waveform,
            "sample_rate": actual_rate,
            "duration_seconds": duration,
            "source_region_seconds": [start, end],
        })
    return _benchmark_decoded(
        decoded,
        sample_rate=sample_rate,
        max_files=max_files,
        max_regions=max_regions,
        max_total_seconds=max_total_seconds,
        input_mode="aligned_regions",
        source_artifact=artifact_path,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare CQT and HPCP on a fixed, capped CPU development set."
    )
    parser.add_argument("audio", nargs="*", type=Path)
    parser.add_argument(
        "--regions",
        type=Path,
        help="harmony_audio_regions_v1 artifact; preferred for the registered comparison",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--max-files", type=int, default=DEFAULT_MAX_FILES)
    parser.add_argument("--max-regions", type=int, default=DEFAULT_MAX_REGIONS)
    parser.add_argument("--max-total-seconds", type=float, default=DEFAULT_MAX_TOTAL_SECONDS)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; extractor comparisons are immutable")
    if bool(args.audio) == bool(args.regions):
        parser.error("provide either audio paths or --regions, but not both")
    if args.regions:
        report = benchmark_region_artifact(
            args.regions,
            sample_rate=args.sample_rate,
            max_files=args.max_files,
            max_regions=args.max_regions,
            max_total_seconds=args.max_total_seconds,
        )
    else:
        report = benchmark_paths(
            args.audio,
            sample_rate=args.sample_rate,
            max_files=args.max_files,
            max_total_seconds=args.max_total_seconds,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({key: report[key] for key in ("files", "total_audio_seconds", "aggregate")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
