"""Cache frozen shared-encoder sequences for a small CPU-only harmony cohort."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

try:
    from scripts.gpu_run_contract import load_cohort_artifact
    from scripts.mtg_data_contract import NOTEBOOK_DATA_CONTRACT
    from scripts.shared_audio_encoder import SharedAudioEncoder
except ModuleNotFoundError:  # Direct execution adds scripts/, not the repository root.
    from gpu_run_contract import load_cohort_artifact
    from mtg_data_contract import NOTEBOOK_DATA_CONTRACT
    from shared_audio_encoder import SharedAudioEncoder


SCHEMA_VERSION = "harmony_encoder_cache_pilot_v1"
MAX_SONGS = 32
MAX_CPU_SECONDS = 600.0
MAX_CPU_THREADS = 8

_contract = {"Path": Path, "np": np, "re": __import__("re"), "ANN_DIR": Path(".")}
exec(NOTEBOOK_DATA_CONTRACT, _contract)
segment_logmel_with_metadata = _contract["segment_logmel_with_metadata"]
LOGMEL_SCHEMA_VERSION = _contract["LOGMEL_SCHEMA_VERSION"]
LOGMEL_N_MELS = _contract["LOGMEL_N_MELS"]
LOGMEL_WINDOW_FRAMES = _contract["LOGMEL_WINDOW_FRAMES"]
LOGMEL_MAX_WINDOWS = _contract["LOGMEL_MAX_WINDOWS"]
LOGMEL_SAMPLE_RATE = _contract["LOGMEL_SAMPLE_RATE"]
LOGMEL_HOP_LENGTH = _contract["LOGMEL_HOP_LENGTH"]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_content_sha256(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode() + b"\0")
        digest.update(array.dtype.str.encode() + b"\0")
        digest.update(json.dumps(array.shape).encode() + b"\0")
        digest.update(array.tobytes())
    return digest.hexdigest()


def _resolve_file(value: object, root: Path) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = root / path
    return path.resolve() if path.is_file() else None


def _load_checkpoint(path: Path) -> tuple[dict, int]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError(f"cannot load trusted instrument checkpoint {path}: {error}") from error
    if not isinstance(checkpoint, dict):
        raise ValueError("instrument checkpoint must be a mapping")
    config = checkpoint.get("training_config")
    if not isinstance(config, dict):
        raise ValueError("instrument checkpoint is missing training_config")
    if config.get("input_schema") != LOGMEL_SCHEMA_VERSION:
        raise ValueError("instrument checkpoint uses a different log-Mel input schema")
    if config.get("max_windows") != LOGMEL_MAX_WINDOWS:
        raise ValueError("instrument checkpoint uses a different maximum window count")
    best_metric = checkpoint.get("best_macro_map")
    tags = checkpoint.get("tags")
    if (
        not isinstance(best_metric, (int, float))
        or isinstance(best_metric, bool)
        or not math.isfinite(best_metric)
    ):
        raise ValueError("instrument checkpoint has no finite validation macro mAP")
    if (
        not isinstance(tags, list)
        or not tags
        or any(not isinstance(tag, str) or not tag for tag in tags)
    ):
        raise ValueError("instrument checkpoint has no fixed non-empty tag vocabulary")
    state = checkpoint.get("model")
    if not isinstance(state, dict):
        raise ValueError("instrument checkpoint is missing its model state")
    projection = state.get("proj.weight", state.get("enc.proj.weight"))
    if not isinstance(projection, torch.Tensor) or projection.ndim != 2:
        raise ValueError("instrument checkpoint has no recognizable encoder projection")
    return checkpoint, int(projection.shape[0])


def cache_encoder_pilot(
    manifest_path: Path,
    cohort_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    *,
    root: Path,
    max_songs: int = MAX_SONGS,
    max_cpu_seconds: float = MAX_CPU_SECONDS,
    cpu_threads: int = 4,
) -> dict:
    """Cache one immutable CPU feature set; partial/failed output is never ready."""
    manifest_path = Path(manifest_path).resolve()
    cohort_path = Path(cohort_path).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError("output directory already exists; encoder caches are immutable")
    if not 1 <= max_songs <= MAX_SONGS:
        raise ValueError(f"max_songs must be in [1, {MAX_SONGS}]")
    if not 0 < max_cpu_seconds <= MAX_CPU_SECONDS:
        raise ValueError(f"max_cpu_seconds must be in (0, {MAX_CPU_SECONDS:g}]")
    if not 1 <= cpu_threads <= MAX_CPU_THREADS:
        raise ValueError(f"cpu_threads must be in [1, {MAX_CPU_THREADS}]")
    cohort = load_cohort_artifact(cohort_path)
    selected = {
        song_id: split
        for split, song_ids in cohort["splits"].items()
        for song_id in song_ids
    }
    if any(split == "test" for split in selected.values()):
        raise ValueError("test songs are forbidden in the harmony encoder pilot")
    if len(selected) > max_songs:
        raise ValueError(f"cohort has {len(selected)} songs; cap is {max_songs}")
    if _sha256_file(manifest_path) != cohort["source_manifest"]["sha256"]:
        raise ValueError("manifest SHA-256 does not match the frozen cohort")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    with manifest_path.open(newline="", encoding="utf-8", errors="strict") as handle:
        rows = list(csv.DictReader(handle))
    by_id = {}
    for row in rows:
        song_id = str(row.get("song_id", "")).strip()
        if song_id in by_id:
            raise ValueError(f"duplicate manifest song_id: {song_id}")
        by_id[song_id] = row
    specifications = []
    for song_id, split in selected.items():
        row = by_id.get(song_id)
        if row is None or row.get("split") != split:
            raise ValueError(f"manifest/cohort identity or split mismatch for {song_id}")
        logmel_path = _resolve_file(row.get("logmel_path") or row.get("mel_abs"), Path(root))
        if logmel_path is None:
            raise FileNotFoundError(f"log-Mel path is missing for {song_id}")
        specifications.append((song_id, split, logmel_path))

    checkpoint, output_dim = _load_checkpoint(checkpoint_path)
    model = SharedAudioEncoder(output_dim=output_dim)
    checkpoint_format = model.load_instrument_pretraining(checkpoint)
    model.cpu().eval()
    if next(model.parameters()).device.type != "cpu":
        raise RuntimeError("encoder pilot must remain on CPU")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(cpu_threads)
    started = time.perf_counter()
    items = []
    failures = []
    total_valid_tokens = 0
    total_token_slots = 0
    try:
        for song_id, split, logmel_path in specifications:
            if time.perf_counter() - started >= max_cpu_seconds:
                failures.append({
                    "error_type": "CPUDeadlineReached",
                    "error": "CPU deadline reached before the complete frozen cohort",
                })
                break
            try:
                raw = np.load(logmel_path, mmap_mode="r", allow_pickle=False)
                windows, window_mask, valid_frames, starts = segment_logmel_with_metadata(
                    raw,
                    n_mels=LOGMEL_N_MELS,
                    n_frames=LOGMEL_WINDOW_FRAMES,
                    max_windows=LOGMEL_MAX_WINDOWS,
                )
                real_windows = int(window_mask.sum())
                if real_windows < 1:
                    raise ValueError("song has no real model windows")
                x = torch.from_numpy(windows[:real_windows, None]).unsqueeze(0)
                with torch.inference_mode():
                    encoded = model.encode_temporal(
                        x,
                        torch.from_numpy(window_mask[:real_windows]).unsqueeze(0),
                        torch.from_numpy(valid_frames[:real_windows]).unsqueeze(0),
                        torch.from_numpy(starts[:real_windows]).unsqueeze(0),
                        sample_rate=LOGMEL_SAMPLE_RATE,
                        hop_length=LOGMEL_HOP_LENGTH,
                    )
                arrays = {
                    "encoded_sequence": encoded.encoded_sequence[0].numpy().astype(np.float32),
                    "sequence_times": encoded.sequence_times[0].numpy().astype(np.float32),
                    "sequence_start_times": encoded.sequence_start_times[0].numpy().astype(np.float32),
                    "sequence_end_times": encoded.sequence_end_times[0].numpy().astype(np.float32),
                    "sequence_mask": encoded.sequence_mask[0].numpy().astype(bool),
                    "sequence_window_index": encoded.sequence_window_index[0].numpy().astype(np.int16),
                    "pooled_song": encoded.pooled_song[0].numpy().astype(np.float32),
                }
                filename = f"{song_id}.npz"
                artifact_path = temporary / filename
                np.savez(artifact_path, **arrays)
                valid_tokens = int(arrays["sequence_mask"].sum())
                token_slots = len(arrays["sequence_mask"])
                total_valid_tokens += valid_tokens
                total_token_slots += token_slots
                items.append({
                    "song_id": song_id,
                    "split": split,
                    "logmel_path": str(logmel_path),
                    "logmel_sha256": _sha256_file(logmel_path),
                    "artifact": filename,
                    "artifact_sha256": _sha256_file(artifact_path),
                    "array_content_sha256": _array_content_sha256(arrays),
                    "windows": real_windows,
                    "tokens_per_window": token_slots // real_windows,
                    "token_slots": token_slots,
                    "valid_tokens": valid_tokens,
                })
                if time.perf_counter() - started > max_cpu_seconds:
                    failures.append({
                        "error_type": "CPUDeadlineReached",
                        "error": "CPU deadline exceeded while finishing the current song",
                    })
                    break
            except Exception as error:
                failures.append({
                    "song_id": song_id,
                    "split": split,
                    "error_type": type(error).__name__,
                    "error": str(error),
                })

        elapsed = time.perf_counter() - started
        complete = len(items) == len(specifications) and not failures
        result = {
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "ready" if complete else "failed",
            "purpose": "cpu_only_frozen_encoder_cache_for_harmony_screening",
            "device": "cpu",
            "contains_genre_labels": False,
            "source_manifest": str(manifest_path),
            "source_manifest_sha256": _sha256_file(manifest_path),
            "source_cohort": str(cohort_path),
            "source_cohort_sha256": _sha256_file(cohort_path),
            "cohort_sha256": cohort["cohort_sha256"],
            "source_checkpoint": str(checkpoint_path),
            "source_checkpoint_sha256": _sha256_file(checkpoint_path),
            "checkpoint_format": checkpoint_format,
            "input_schema": LOGMEL_SCHEMA_VERSION,
            "encoder_output_dim": output_dim,
            "limits": {
                "max_songs": MAX_SONGS,
                "max_cpu_seconds": MAX_CPU_SECONDS,
                "max_cpu_threads": MAX_CPU_THREADS,
            },
            "requested_limits": {
                "max_songs": max_songs,
                "max_cpu_seconds": max_cpu_seconds,
                "cpu_threads": cpu_threads,
            },
            "requested_songs": len(specifications),
            "cached_songs": len(items),
            "failure_count": len(failures),
            "token_slots": total_token_slots,
            "valid_tokens": total_valid_tokens,
            "elapsed_cpu_wall_seconds": elapsed,
            "versions": {
                "numpy": importlib.metadata.version("numpy"),
                "torch": importlib.metadata.version("torch"),
            },
            "items": items,
            "failures": failures,
        }
        (temporary / "index.json").write_text(json.dumps(result, indent=2) + "\n")
        temporary.rename(output_dir)
        return result
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    finally:
        torch.set_num_threads(previous_threads)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("cohort", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-songs", type=int, default=MAX_SONGS)
    parser.add_argument("--max-cpu-seconds", type=float, default=MAX_CPU_SECONDS)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    result = cache_encoder_pilot(
        args.manifest,
        args.cohort,
        args.checkpoint,
        args.output_dir,
        root=args.root,
        max_songs=args.max_songs,
        max_cpu_seconds=args.max_cpu_seconds,
        cpu_threads=args.cpu_threads,
    )
    print(json.dumps({
        "status": result["status"],
        "cached_songs": result["cached_songs"],
        "failure_count": result["failure_count"],
        "valid_tokens": result["valid_tokens"],
        "elapsed_cpu_wall_seconds": result["elapsed_cpu_wall_seconds"],
        "output": str(args.output_dir),
    }, indent=2))
    return 0 if result["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
