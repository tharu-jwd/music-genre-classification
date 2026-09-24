"""Materialize a small immutable temporal-chroma pilot after extractor selection."""

from __future__ import annotations

import _bootstrap  # noqa: F401 - configures direct-script imports

import argparse
import hashlib
import importlib.metadata
import json
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from decide_harmony_extractor import decide as decide_extractor
from harmony_branch.features import PITCH_CLASSES, extract_cqt_chroma, extract_hpcp


SCHEMA_VERSION = "harmony_target_pilot_v1"
DECISION_SCHEMA = "harmony_extractor_decision_v1"
REPORT_SCHEMA = "harmony_extractor_benchmark_v1"
REGION_SCHEMA = "harmony_audio_regions_v1"
SAMPLE_RATE = 22050
MAX_FILES = 32
MAX_REGIONS = 384
MAX_TOTAL_SECONDS = 600.0
SHA256 = re.compile(r"[0-9a-f]{64}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_content_sha256(arrays: dict[str, np.ndarray]) -> str:
    """Hash array meaning rather than NPZ container timestamps/compression bytes."""
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(array.dtype.str.encode("ascii") + b"\0")
        digest.update(json.dumps(array.shape).encode("ascii") + b"\0")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _read_json(path: Path, description: str) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {description} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object")
    return value


def _load_selected_extractor(
    decision_path: Path, region_path: Path
) -> tuple[str, dict, Path, str]:
    decision = _read_json(decision_path, "extractor decision")
    if decision.get("schema_version") != DECISION_SCHEMA:
        raise ValueError(f"decision schema_version must be {DECISION_SCHEMA!r}")
    if decision.get("decision") != "select":
        raise ValueError("pilot targets require a successful extractor selection decision")
    selected = decision.get("selected_extractor")
    if selected not in {"cqt", "hpcp_harmonic"}:
        raise ValueError("decision selected_extractor is unsupported")

    report_path_text = decision.get("source_report")
    report_hash = decision.get("source_report_sha256")
    if not isinstance(report_path_text, str) or SHA256.fullmatch(str(report_hash)) is None:
        raise ValueError("decision must identify and hash its source report")
    report_path = Path(report_path_text)
    if not report_path.is_file() or _sha256_file(report_path) != report_hash:
        raise ValueError("decision source report is missing or its hash has changed")
    report = _read_json(report_path, "extractor benchmark report")
    if (
        report.get("schema_version") != REPORT_SCHEMA
        or report.get("purpose") != "bounded_cpu_chroma_candidate_comparison"
        or report.get("input_mode") != "aligned_regions"
        or report.get("automatic_selection") is not False
    ):
        raise ValueError("decision source is not a registered aligned-region comparison")
    if _sha256_file(region_path) != report.get("source_artifact_sha256"):
        raise ValueError("region artifact does not match the selected extractor benchmark")
    reproduced = decide_extractor(report)
    if (
        reproduced.get("decision") != "select"
        or reproduced.get("selected_extractor") != selected
        or decision.get("criteria") != reproduced.get("criteria")
    ):
        raise ValueError("extractor decision does not reproduce from its source report")
    return selected, decision, report_path.resolve(), report_hash


def _region_specifications(region_path: Path) -> tuple[dict, list[dict]]:
    region_artifact = _read_json(region_path, "region artifact")
    if region_artifact.get("schema_version") != REGION_SCHEMA:
        raise ValueError(f"region schema_version must be {REGION_SCHEMA!r}")
    songs = region_artifact.get("songs")
    if not isinstance(songs, list) or not songs or len(songs) > MAX_FILES:
        raise ValueError(f"region artifact must contain 1-{MAX_FILES} songs")

    seen_songs: set[str] = set()
    seen_units: set[str] = set()
    specifications: list[dict] = []
    total_seconds = 0.0
    for song in songs:
        if not isinstance(song, dict):
            raise ValueError("each region song must be an object")
        song_id = str(song.get("song_id", ""))
        split = song.get("split")
        if re.fullmatch(r"\d{7}", song_id) is None or song_id in seen_songs:
            raise ValueError("region artifact has invalid or duplicate seven-digit song IDs")
        seen_songs.add(song_id)
        if split not in {"train", "validation"}:
            raise ValueError("target pilot permits train and validation songs only")
        audio_path = Path(str(song.get("audio_path", ""))).resolve()
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        regions = song.get("regions")
        if not isinstance(regions, list) or not regions:
            raise ValueError(f"song {song_id} has no model-aligned regions")
        previous_index = -1
        previous_end = -1.0
        for region in regions:
            if not isinstance(region, dict):
                raise ValueError(f"song {song_id} has a non-object region")
            window_index = region.get("window_index")
            start = region.get("start_seconds")
            end = region.get("end_seconds")
            if (
                not isinstance(window_index, int)
                or isinstance(window_index, bool)
                or window_index <= previous_index
                or not isinstance(start, (int, float))
                or isinstance(start, bool)
                or not isinstance(end, (int, float))
                or isinstance(end, bool)
                or not np.isfinite(start)
                or not np.isfinite(end)
                or start < 0
                or end <= start
                or start < previous_end - 1e-9
            ):
                raise ValueError(f"song {song_id} has invalid or overlapping regions")
            previous_index = window_index
            previous_end = float(end)
            unit_id = f"{song_id}:window_{window_index}"
            if unit_id in seen_units:
                raise ValueError(f"duplicate target unit {unit_id}")
            seen_units.add(unit_id)
            duration = float(end) - float(start)
            total_seconds += duration
            specifications.append({
                "unit_id": unit_id,
                "song_id": song_id,
                "split": split,
                "window_index": window_index,
                "audio_path": audio_path,
                "start_seconds": float(start),
                "end_seconds": float(end),
                "requested_duration_seconds": duration,
            })
    if len(specifications) > MAX_REGIONS:
        raise ValueError(f"target pilot exceeds the hard cap of {MAX_REGIONS} regions")
    if total_seconds > MAX_TOTAL_SECONDS:
        raise ValueError(f"target pilot exceeds the hard cap of {MAX_TOTAL_SECONDS:g} audio seconds")
    return region_artifact, specifications


def materialize_pilot(region_path: Path, decision_path: Path, output_dir: Path) -> dict:
    """Extract the selected candidate over only the already-capped comparison regions."""
    import librosa

    region_path = Path(region_path).resolve()
    decision_path = Path(decision_path).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError("output directory already exists; pilot targets are immutable")
    selected, decision, source_report_path, source_report_hash = (
        _load_selected_extractor(decision_path, region_path)
    )
    region_artifact, specifications = _region_specifications(region_path)
    extractor = extract_cqt_chroma if selected == "cqt" else extract_hpcp

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    items = []
    failures = []
    audio_hashes: dict[Path, str] = {}
    total_frames = 0
    valid_frames = 0
    try:
        for spec in specifications:
            path = spec["audio_path"]
            try:
                waveform, actual_rate = librosa.load(
                    path,
                    sr=SAMPLE_RATE,
                    mono=True,
                    offset=spec["start_seconds"],
                    duration=spec["requested_duration_seconds"],
                )
                decoded_duration = len(waveform) / actual_rate
                if decoded_duration + 0.1 < spec["requested_duration_seconds"]:
                    raise ValueError(
                        f"audio ended early: requested {spec['requested_duration_seconds']:.3f}s, "
                        f"decoded {decoded_duration:.3f}s"
                    )
                sequence = extractor(waveform, actual_rate)
                sequence.validate()
                keep = sequence.timestamps <= decoded_duration + 1e-6
                if not np.any(keep):
                    raise ValueError("extractor returned no frames inside the decoded region")
                arrays = {
                    "frame_times_seconds": (
                        sequence.timestamps[keep] + spec["start_seconds"]
                    ).astype(np.float32),
                    "chroma": sequence.chroma[keep].astype(np.float32),
                    "chroma_valid": sequence.valid[keep].astype(bool),
                    "tonal_concentration": sequence.tonal_concentration[keep].astype(np.float32),
                }
                filename = f"{spec['song_id']}__window_{spec['window_index']:03d}.npz"
                artifact_path = temporary_dir / filename
                np.savez_compressed(artifact_path, **arrays)
                audio_hashes.setdefault(path, _sha256_file(path))
                frames = len(arrays["frame_times_seconds"])
                valid = int(arrays["chroma_valid"].sum())
                total_frames += frames
                valid_frames += valid
                items.append({
                    **{key: value for key, value in spec.items() if key != "audio_path"},
                    "audio_path": str(path),
                    "audio_sha256": audio_hashes[path],
                    "decoded_duration_seconds": decoded_duration,
                    "artifact": filename,
                    "artifact_sha256": _sha256_file(artifact_path),
                    "array_content_sha256": _array_content_sha256(arrays),
                    "frames": frames,
                    "valid_frames": valid,
                    "tuning_value": sequence.tuning_value,
                    "tuning_unit": sequence.tuning_unit,
                })
            except Exception as error:  # Preserve failures rather than inventing targets.
                failures.append({
                    "unit_id": spec["unit_id"],
                    "song_id": spec["song_id"],
                    "split": spec["split"],
                    "window_index": spec["window_index"],
                    "error_type": type(error).__name__,
                    "error": str(error),
                })

        valid_fraction = valid_frames / total_frames if total_frames else None
        minimum_valid_fraction = decision["criteria"]["min_valid_fraction"]
        quality_failures = []
        if valid_fraction is None or valid_fraction < minimum_valid_fraction:
            quality_failures.append("valid_frame_fraction_below_selected_extractor_gate")
        result = {
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "ready" if not failures and not quality_failures else "failed",
            "target_variant": "temporal_chroma_v1",
            "semantics": "pseudo_label_reference_features_not_human_ground_truth",
            "contains_genre_labels": False,
            "contains_chord_targets": False,
            "selected_extractor": selected,
            "extractor_implementation": (
                "librosa.feature.chroma_cqt"
                if selected == "cqt"
                else "essentia.standard.HPCP+librosa.effects.harmonic"
            ),
            "sample_rate": SAMPLE_RATE,
            "pitch_classes": list(PITCH_CLASSES),
            "frame_times": "absolute seconds from song start on the extractor analysis grid",
            "invalid_frame_policy": "zero chroma plus a separate false chroma_valid mask",
            "tonal_concentration_semantics": (
                "maximum L1-normalized chroma bin; diagnostic, not calibrated confidence"
            ),
            "source_region_artifact": str(region_path),
            "source_region_artifact_sha256": _sha256_file(region_path),
            "source_extractor_decision": str(decision_path),
            "source_extractor_decision_sha256": _sha256_file(decision_path),
            "source_benchmark_report": str(source_report_path),
            "source_benchmark_report_sha256": source_report_hash,
            "source_cohort_sha256": region_artifact.get("cohort_sha256"),
            "limits": {
                "max_files": MAX_FILES,
                "max_regions": MAX_REGIONS,
                "max_total_seconds": MAX_TOTAL_SECONDS,
            },
            "songs": len({item["song_id"] for item in specifications}),
            "requested_regions": len(specifications),
            "materialized_regions": len(items),
            "failure_count": len(failures),
            "frames": total_frames,
            "valid_frames": valid_frames,
            "valid_fraction": valid_fraction,
            "minimum_valid_fraction": minimum_valid_fraction,
            "versions": {
                "numpy": importlib.metadata.version("numpy"),
                "librosa": importlib.metadata.version("librosa"),
                "essentia": importlib.metadata.version("essentia"),
            },
            "items": items,
            "failures": failures,
            "quality_failures": quality_failures,
        }
        (temporary_dir / "index.json").write_text(json.dumps(result, indent=2) + "\n")
        temporary_dir.rename(output_dir)
        return result
    except BaseException:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("regions", type=Path)
    parser.add_argument("decision", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = materialize_pilot(args.regions, args.decision, args.output_dir)
    print(json.dumps({
        "status": result["status"],
        "selected_extractor": result["selected_extractor"],
        "materialized_regions": result["materialized_regions"],
        "failure_count": result["failure_count"],
        "valid_fraction": result["valid_fraction"],
        "output": str(args.output_dir),
    }, indent=2))
    return 0 if result["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
