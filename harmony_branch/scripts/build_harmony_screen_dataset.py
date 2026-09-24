"""Join frozen encoder features with aligned temporal-chroma pilot targets on CPU."""

from __future__ import annotations

import _bootstrap  # noqa: F401 - configures direct-script imports

import argparse
import hashlib
import json
import shutil
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from harmony_branch.alignment import align_chroma_to_intervals


SCHEMA_VERSION = "harmony_screen_dataset_v1"
FEATURE_SCHEMA = "harmony_encoder_cache_pilot_v1"
TARGET_SCHEMA = "harmony_target_pilot_v1"
REGISTERED_MAX_TOKEN_SECONDS = 0.1


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_index(directory: Path, schema: str, description: str) -> tuple[Path, dict]:
    path = Path(directory).resolve() / "index.json"
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {description} index {path}: {error}") from error
    if not isinstance(value, dict) or value.get("schema_version") != schema:
        raise ValueError(f"{description} schema_version must be {schema!r}")
    if value.get("status") != "ready" or value.get("failure_count") != 0:
        raise ValueError(f"{description} must be complete and ready")
    if value.get("contains_genre_labels") is not False:
        raise ValueError(f"{description} must not contain genre labels")
    return path, value


def _verified_artifact(directory: Path, item: dict, description: str) -> Path:
    filename = item.get("artifact")
    expected = item.get("artifact_sha256")
    if not isinstance(filename, str) or not isinstance(expected, str):
        raise ValueError(f"{description} item is missing artifact provenance")
    path = directory / filename
    if not path.is_file() or _sha256_file(path) != expected:
        raise ValueError(f"{description} artifact is missing or changed: {filename}")
    return path


def build_screen_dataset(feature_dir: Path, target_dir: Path, output_dir: Path) -> dict:
    feature_dir = Path(feature_dir).resolve()
    target_dir = Path(target_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError("output directory already exists; screen datasets are immutable")
    feature_index_path, features = _load_index(feature_dir, FEATURE_SCHEMA, "feature cache")
    target_index_path, targets = _load_index(target_dir, TARGET_SCHEMA, "target pilot")
    if features.get("device") != "cpu":
        raise ValueError("pilot feature cache must have been produced on CPU")
    if targets.get("target_variant") != "temporal_chroma_v1":
        raise ValueError("target pilot must use temporal_chroma_v1")
    if features.get("cohort_sha256") != targets.get("source_cohort_sha256"):
        raise ValueError("feature cache and target pilot use different frozen cohorts")
    feature_items = features.get("items")
    target_items = targets.get("items")
    if not isinstance(feature_items, list) or not isinstance(target_items, list):
        raise ValueError("source indexes must contain item lists")
    by_song = {}
    for item in feature_items:
        song_id = item.get("song_id")
        if not isinstance(song_id, str) or song_id in by_song:
            raise ValueError("feature cache has missing or duplicate song IDs")
        if item.get("split") not in {"train", "validation"}:
            raise ValueError("feature cache contains an unsupported split")
        by_song[song_id] = item
    target_by_song = defaultdict(list)
    seen_units = set()
    for item in target_items:
        song_id = item.get("song_id")
        window_index = item.get("window_index")
        unit = (song_id, window_index)
        if song_id not in by_song or not isinstance(window_index, int) or unit in seen_units:
            raise ValueError("target pilot has unknown or duplicate song/window identity")
        seen_units.add(unit)
        target_by_song[song_id].append(item)
    if set(target_by_song) != set(by_song):
        raise ValueError("feature and target pilots must contain the same song IDs")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    output_items = []
    failures = []
    split_counts = {
        split: {"requested_tokens": 0, "valid_target_tokens": 0}
        for split in ("train", "validation")
    }
    try:
        for song_id, feature_item in by_song.items():
            split = feature_item["split"]
            try:
                feature_path = _verified_artifact(feature_dir, feature_item, "feature")
                with np.load(feature_path, allow_pickle=False) as source:
                    required = {
                        "encoded_sequence", "sequence_start_times", "sequence_end_times",
                        "sequence_mask", "sequence_window_index",
                    }
                    if required - set(source.files):
                        raise ValueError("feature artifact is missing temporal arrays")
                    encoded_shape = source["encoded_sequence"].shape
                    starts = source["sequence_start_times"].astype(np.float64)
                    ends = source["sequence_end_times"].astype(np.float64)
                    sequence_mask = source["sequence_mask"].astype(bool)
                    window_index = source["sequence_window_index"].astype(np.int64)
                tokens = len(sequence_mask)
                if len(encoded_shape) != 2 or encoded_shape[0] != tokens:
                    raise ValueError("feature sequence and token metadata lengths differ")
                chroma = np.zeros((tokens, 12), dtype=np.float32)
                chroma_valid = np.zeros(tokens, dtype=bool)
                contributing_frames = np.zeros(tokens, dtype=np.int64)
                screen_mask = np.zeros(tokens, dtype=bool)
                for target_item in sorted(
                    target_by_song[song_id], key=lambda item: item["window_index"]
                ):
                    target_path = _verified_artifact(target_dir, target_item, "target")
                    with np.load(target_path, allow_pickle=False) as source:
                        required = {
                            "frame_times_seconds", "chroma", "chroma_valid",
                        }
                        if required - set(source.files):
                            raise ValueError("target artifact is missing temporal chroma arrays")
                        frame_times = source["frame_times_seconds"]
                        frame_chroma = source["chroma"]
                        frame_valid = source["chroma_valid"]
                    this_window = (
                        sequence_mask & (window_index == target_item["window_index"])
                    )
                    if not np.any(this_window):
                        raise ValueError(
                            f"feature cache lacks target window {target_item['window_index']}"
                        )
                    aligned = align_chroma_to_intervals(
                        frame_times,
                        frame_chroma,
                        frame_valid,
                        starts,
                        ends,
                        this_window,
                        max_token_duration_seconds=REGISTERED_MAX_TOKEN_SECONDS,
                    )
                    overlap = screen_mask & this_window
                    if np.any(overlap):
                        raise ValueError("target windows overlap after token alignment")
                    screen_mask |= this_window
                    chroma[aligned.valid] = aligned.chroma[aligned.valid]
                    chroma_valid |= aligned.valid
                    contributing_frames += aligned.contributing_frames

                arrays = {
                    "chroma_target": chroma,
                    "chroma_valid": chroma_valid,
                    "contributing_source_frames": contributing_frames,
                    "screen_token_mask": screen_mask,
                }
                filename = f"{song_id}.npz"
                output_path = temporary / filename
                np.savez(output_path, **arrays)
                requested = int(screen_mask.sum())
                valid = int(chroma_valid.sum())
                split_counts[split]["requested_tokens"] += requested
                split_counts[split]["valid_target_tokens"] += valid
                output_items.append({
                    "song_id": song_id,
                    "split": split,
                    "feature_artifact": str(feature_path),
                    "feature_artifact_sha256": feature_item["artifact_sha256"],
                    "target_artifact": filename,
                    "target_artifact_sha256": _sha256_file(output_path),
                    "windows": feature_item["windows"],
                    "tokens_per_window": feature_item["tokens_per_window"],
                    "requested_tokens": requested,
                    "valid_target_tokens": valid,
                    "coverage": valid / requested if requested else 0.0,
                })
            except Exception as error:
                failures.append({
                    "song_id": song_id,
                    "split": split,
                    "error_type": type(error).__name__,
                    "error": str(error),
                })

        requested_total = sum(value["requested_tokens"] for value in split_counts.values())
        valid_total = sum(value["valid_target_tokens"] for value in split_counts.values())
        coverage = valid_total / requested_total if requested_total else 0.0
        minimum = targets.get("minimum_valid_fraction")
        quality_failures = []
        if not isinstance(minimum, (int, float)) or coverage < minimum:
            quality_failures.append("aligned_target_coverage_below_extractor_gate")
        for split, value in split_counts.items():
            value["coverage"] = (
                value["valid_target_tokens"] / value["requested_tokens"]
                if value["requested_tokens"]
                else 0.0
            )
            if value["requested_tokens"] == 0 or value["valid_target_tokens"] == 0:
                quality_failures.append(f"{split}_has_no_valid_screening_targets")
        complete = len(output_items) == len(by_song) and not failures and not quality_failures
        result = {
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "ready" if complete else "failed",
            "target_variant": "temporal_chroma_v1",
            "contains_genre_labels": False,
            "registered_max_token_seconds": REGISTERED_MAX_TOKEN_SECONDS,
            "source_feature_index": str(feature_index_path),
            "source_feature_index_sha256": _sha256_file(feature_index_path),
            "source_target_index": str(target_index_path),
            "source_target_index_sha256": _sha256_file(target_index_path),
            "cohort_sha256": features.get("cohort_sha256"),
            "songs": len(output_items),
            "requested_tokens": requested_total,
            "valid_target_tokens": valid_total,
            "coverage": coverage,
            "minimum_valid_fraction": minimum,
            "split_coverage": split_counts,
            "items": output_items,
            "failure_count": len(failures),
            "failures": failures,
            "quality_failures": quality_failures,
        }
        (temporary / "index.json").write_text(json.dumps(result, indent=2) + "\n")
        temporary.rename(output_dir)
        return result
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("feature_cache", type=Path)
    parser.add_argument("target_pilot", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = build_screen_dataset(args.feature_cache, args.target_pilot, args.output_dir)
    print(json.dumps({
        "status": result["status"],
        "songs": result["songs"],
        "coverage": result["coverage"],
        "failure_count": result["failure_count"],
        "output": str(args.output_dir),
    }, indent=2))
    return 0 if result["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
