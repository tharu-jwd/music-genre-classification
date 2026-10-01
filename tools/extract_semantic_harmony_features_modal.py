"""Extract six semantic Harmony descriptors from first-four-minute MP3 audio.

Run from the repository root after ``transfer_audio_drive_to_modal.py`` has
completed:

    modal run --detach tools/extract_semantic_harmony_features_modal.py --workers 4

Inputs
------
* Volume ``mtg-jamendo-audio`` under ``audio/<bucket>/<track>.mp3``.
* Repository ``data/split_csv.csv``. Only its 7,324 listed paths are read.

Outputs (persistent Volume ``music-genre-data``)
-------------------------------------------------
* ``dataset/harmony_semantic_6_df.csv``: TRACK_ID plus the six target columns.
* ``harmony_semantic_6/checkpoint.csv``: resumable working state.
* ``harmony_semantic_6/errors.csv`` and ``config.json``: audit records.

Google Drive credentials are not needed here: all audio is read from the Modal
Volume. The script never changes the existing 45-column harmony_df.csv.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import modal


APP_NAME = "semantic-harmony-feature-extraction"
AUDIO_VOLUME_NAME = "mtg-jamendo-audio"
DATA_VOLUME_NAME = "music-genre-data"

EXPECTED_TRACKS = 7_324
SAMPLE_RATE = 22_050
HOP_LENGTH = 512
MAX_DURATION_SECONDS = 240.0
LOCAL_KEY_SECONDS = 15.0
RMS_DB_BELOW_PEAK = 45.0
CHECKPOINT_EVERY = 50
COMMIT_EVERY = 200

FEATURE_COLUMNS = [
    "global_key_strength",
    "mode_confidence",
    "key_stability",
    "tonal_centroid_radius_mean",
    "tonal_centroid_radius_std",
    "chroma_recurrence_15s",
]
FINAL_COLUMNS = ["TRACK_ID", *FEATURE_COLUMNS]
CHECKPOINT_COLUMNS = ["TRACK_ID", "status", "audio_path", "elapsed_sec", "error", *FEATURE_COLUMNS]

CONTAINER_MANIFEST = "/app/split_csv.csv"
AUDIO_MOUNT = "/audio"
AUDIO_ROOT = "/audio/audio"
DATA_MOUNT = "/data"
WORK_ROOT = "/data/harmony_semantic_6"
FINAL_CSV = "/data/dataset/harmony_semantic_6_df.csv"

# Krumhansl--Kessler key-profile templates.  Values are fixed extraction
# constants, not learned from validation/test tracks.
MAJOR_PROFILE = (6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88)
MINOR_PROFILE = (6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17)

if modal.is_local():
    REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
    MANIFEST_SOURCE = REPOSITORY_ROOT / "data" / "split_csv.csv"
    if not MANIFEST_SOURCE.is_file():
        raise FileNotFoundError(f"Required manifest is missing: {MANIFEST_SOURCE}")

app = modal.App(APP_NAME)
audio_volume = modal.Volume.from_name(AUDIO_VOLUME_NAME)
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install("numpy==1.26.4", "pandas==2.2.3", "librosa==0.10.2.post1")
    # Prevent each worker from creating its own BLAS/OpenMP thread pool.
    .env({
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMBA_NUM_THREADS": "1",
    })
)
if modal.is_local():
    image = image.add_local_file(str(MANIFEST_SOURCE), CONTAINER_MANIFEST, copy=True)


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != EXPECTED_TRACKS:
        raise ValueError(f"expected {EXPECTED_TRACKS:,} manifest rows, found {len(rows):,}")
    if not rows or not {"TRACK_ID", "PATH"}.issubset(rows[0]):
        raise ValueError("split_csv.csv must contain TRACK_ID and PATH")

    selected: list[dict[str, str]] = []
    track_ids: set[str] = set()
    relative_paths: set[str] = set()
    for row in rows:
        track_id = str(row["TRACK_ID"]).strip()
        relative_path = str(row["PATH"]).replace("\\", "/").strip()
        if not re.fullmatch(r"track_\d+", track_id):
            raise ValueError(f"unexpected TRACK_ID: {track_id!r}")
        if not re.fullmatch(r"\d{2}/\d+\.mp3", relative_path):
            raise ValueError(f"unsafe or unexpected PATH: {relative_path!r}")
        if track_id in track_ids or relative_path in relative_paths:
            raise ValueError(f"duplicate track record: {track_id}, {relative_path}")
        if Path(relative_path).stem != str(int(track_id.removeprefix("track_"))):
            raise ValueError(f"TRACK_ID/PATH mismatch: {track_id}, {relative_path}")
        track_ids.add(track_id)
        relative_paths.add(relative_path)
        # Keep container paths POSIX even when the local Modal CLI imports this
        # module on Windows before shipping it to a Linux worker.
        selected.append({"TRACK_ID": track_id, "audio_path": f"{AUDIO_ROOT}/{relative_path}"})
    return selected


def _unit_centered(values: Any):
    """Return an L2-normalized, zero-mean vector; None for a flat vector."""
    import numpy as np

    vector = np.asarray(values, dtype=np.float64)
    vector = vector - vector.mean()
    norm = np.linalg.norm(vector)
    return None if norm <= 1e-12 else vector / norm


def _key_scores(chroma: Any) -> tuple[Any, float, float]:
    """Return best 24-key identity, key strength, and mode confidence."""
    import numpy as np

    profile = _unit_centered(chroma)
    if profile is None:
        raise ValueError("flat chroma profile cannot define a key")
    scores: list[tuple[tuple[str, int], float]] = []
    for mode, template in (("major", MAJOR_PROFILE), ("minor", MINOR_PROFILE)):
        basis = _unit_centered(template)
        assert basis is not None
        for tonic in range(12):
            scores.append(((mode, tonic), float(np.dot(profile, np.roll(basis, tonic)))))
    scores.sort(key=lambda item: item[1], reverse=True)
    best_key, best_score = scores[0]
    best_major = max(score for (mode, _), score in scores if mode == "major")
    best_minor = max(score for (mode, _), score in scores if mode == "minor")
    return best_key, best_score, abs(best_major - best_minor)


def _extract_one(record: dict[str, str], harmonic_only: bool) -> dict[str, Any]:
    """Worker-safe extraction for one track. Returns data rather than writing a Volume."""
    import librosa
    import numpy as np

    started = time.monotonic()
    track_id = record["TRACK_ID"]
    audio_path = record["audio_path"]
    base = {"TRACK_ID": track_id, "audio_path": audio_path}
    try:
        y, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True, duration=MAX_DURATION_SECONDS)
        if y.size < SAMPLE_RATE:
            raise ValueError("decoded audio is shorter than one second")
        signal = librosa.effects.harmonic(y) if harmonic_only else y

        cens = librosa.feature.chroma_cens(
            y=signal,
            sr=SAMPLE_RATE,
            hop_length=HOP_LENGTH,
            n_chroma=12,
            n_octaves=7,
            bins_per_octave=36,
            cqt_mode="hybrid",
        )
        if cens.shape[1] < 4:
            raise ValueError("too few CENS frames")

        # Frame validity derives from harmonic RMS energy, not from a target.
        rms = librosa.feature.rms(y=signal, frame_length=2048, hop_length=HOP_LENGTH)[0]
        frames = min(cens.shape[1], rms.size)
        cens = cens[:, :frames]
        rms = rms[:frames]
        peak = float(np.max(rms))
        if peak <= 1e-12:
            raise ValueError("no harmonic energy")
        valid = rms >= peak * (10.0 ** (-RMS_DB_BELOW_PEAK / 20.0))
        if int(valid.sum()) < 4:
            valid = np.ones(frames, dtype=bool)

        global_chroma = cens[:, valid].mean(axis=1)
        global_key, key_strength, mode_confidence = _key_scores(global_chroma)

        segment_frames = max(1, round(LOCAL_KEY_SECONDS * SAMPLE_RATE / HOP_LENGTH))
        local_keys: list[tuple[str, int]] = []
        for start in range(0, frames, segment_frames):
            stop = min(frames, start + segment_frames)
            segment_valid = valid[start:stop]
            if int(segment_valid.sum()) < 4:
                continue
            local_chroma = cens[:, start:stop][:, segment_valid].mean(axis=1)
            try:
                local_key, _, _ = _key_scores(local_chroma)
            except ValueError:
                continue
            local_keys.append(local_key)
        if not local_keys:
            raise ValueError("no valid local key segments")
        key_stability = float(sum(key == global_key for key in local_keys) / len(local_keys))

        tonnetz = librosa.feature.tonnetz(chroma=cens)
        # ``vector_norm`` is only available in newer NumPy releases.  Keep the
        # pinned NumPy 1.26 image compatible while computing the same L2 radius.
        radius = np.linalg.norm(tonnetz[:, valid], axis=0)
        if radius.size == 0:
            raise ValueError("no valid Tonnetz frames")

        lag = max(1, round(LOCAL_KEY_SECONDS * SAMPLE_RATE / HOP_LENGTH))
        if frames <= lag:
            raise ValueError("track is too short for 15-second chroma recurrence")
        recurrence_valid = valid[lag:] & valid[:-lag]
        if int(recurrence_valid.sum()) < 1:
            raise ValueError("no valid 15-second chroma recurrence pairs")
        # CENS frames are L2 normalized by librosa, so a dot product is cosine similarity.
        recurrence = np.sum(cens[:, lag:] * cens[:, :-lag], axis=0)
        recurrence_15s = float(np.mean(recurrence[recurrence_valid]))

        features = {
            "global_key_strength": float(key_strength),
            "mode_confidence": float(mode_confidence),
            "key_stability": key_stability,
            "tonal_centroid_radius_mean": float(np.mean(radius)),
            "tonal_centroid_radius_std": float(np.std(radius)),
            "chroma_recurrence_15s": recurrence_15s,
        }
        if not all(math.isfinite(value) for value in features.values()):
            raise ValueError("non-finite extracted feature")
        return {**base, "status": "ok", "elapsed_sec": time.monotonic() - started, "error": "", **features}
    except Exception as exc:  # The parent records the track and resumes it later.
        return {
            **base,
            "status": "error",
            "elapsed_sec": time.monotonic() - started,
            "error": f"{type(exc).__name__}: {str(exc)[:600]}",
            **{column: math.nan for column in FEATURE_COLUMNS},
        }


def _load_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    import pandas as pd

    frame = pd.read_csv(path, dtype={"TRACK_ID": str})
    return {str(row["TRACK_ID"]): row.to_dict() for _, row in frame.iterrows()}


def _write_csv_atomically(frame: Any, path: Path) -> None:
    temporary = path.with_suffix(".tmp.csv")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _save_outputs(records: dict[str, dict[str, Any]], selected: list[dict[str, str]], work_root: Path) -> Any:
    import pandas as pd

    ordered = [records[record["TRACK_ID"]] for record in selected if record["TRACK_ID"] in records]
    checkpoint = pd.DataFrame(ordered)
    for column in CHECKPOINT_COLUMNS:
        if column not in checkpoint.columns:
            checkpoint[column] = math.nan
    checkpoint = checkpoint.loc[:, CHECKPOINT_COLUMNS]
    _write_csv_atomically(checkpoint, work_root / "checkpoint.csv")

    errors = checkpoint.loc[checkpoint["status"] != "ok", ["TRACK_ID", "audio_path", "error"]]
    _write_csv_atomically(errors, work_root / "errors.csv")

    final = checkpoint.loc[:, FINAL_COLUMNS]
    _write_csv_atomically(final, Path(FINAL_CSV))
    return checkpoint


@app.function(
    image=image,
    volumes={AUDIO_MOUNT: audio_volume, DATA_MOUNT: data_volume},
    cpu=8.0,
    memory=32_768,
    timeout=24 * 60 * 60,
)
def extract(workers: int = 4, harmonic_only: bool = True) -> dict[str, Any]:
    if not 1 <= workers <= 8:
        raise ValueError("workers must be between 1 and 8")

    selected = _read_manifest(Path(CONTAINER_MANIFEST))
    work_root = Path(WORK_ROOT)
    work_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = work_root / "checkpoint.csv"
    records = _load_checkpoint(checkpoint_path)
    pending = [
        record for record in selected
        if records.get(record["TRACK_ID"], {}).get("status") != "ok"
    ]

    config = {
        "feature_columns": FEATURE_COLUMNS,
        "sample_rate": SAMPLE_RATE,
        "hop_length": HOP_LENGTH,
        "max_duration_seconds": MAX_DURATION_SECONDS,
        "local_key_seconds": LOCAL_KEY_SECONDS,
        "rms_db_below_peak": RMS_DB_BELOW_PEAK,
        "harmonic_only": harmonic_only,
        "workers": workers,
        "input_audio_volume": AUDIO_VOLUME_NAME,
        "output_data_volume": DATA_VOLUME_NAME,
    }
    (work_root / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Selected={len(selected):,}; checkpointed_ok={len(selected)-len(pending):,}; pending={len(pending):,}; workers={workers}", flush=True)

    completed = 0
    started = time.monotonic()
    if pending:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_extract_one, record, harmonic_only) for record in pending]
            for future in as_completed(futures):
                record = future.result()
                records[str(record["TRACK_ID"])] = record
                completed += 1
                if completed % CHECKPOINT_EVERY == 0 or completed == len(pending):
                    checkpoint = _save_outputs(records, selected, work_root)
                    if completed % COMMIT_EVERY == 0 or completed == len(pending):
                        data_volume.commit()
                    ok_count = int((checkpoint["status"] == "ok").sum())
                    error_count = int((checkpoint["status"] != "ok").sum())
                    elapsed_minutes = (time.monotonic() - started) / 60.0
                    print(f"Checkpoint: completed={completed:,}/{len(pending):,}; ok={ok_count:,}; errors={error_count:,}; elapsed_min={elapsed_minutes:.1f}", flush=True)

    checkpoint = _save_outputs(records, selected, work_root)
    data_volume.commit()
    ok_count = int((checkpoint["status"] == "ok").sum())
    error_count = int((checkpoint["status"] != "ok").sum())
    summary = {
        "status": "ok" if ok_count == EXPECTED_TRACKS and error_count == 0 else "incomplete",
        "selected_tracks": len(selected),
        "ok_tracks": ok_count,
        "error_tracks": error_count,
        "final_csv": FINAL_CSV,
        "checkpoint_csv": str(checkpoint_path),
        "errors_csv": str(work_root / "errors.csv"),
        "elapsed_seconds": time.monotonic() - started,
    }
    (work_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    data_volume.commit()
    print(json.dumps(summary, indent=2), flush=True)
    if error_count:
        raise RuntimeError(f"Extraction incomplete: {error_count:,} tracks need retry; rerun the same command.")
    return summary


@app.local_entrypoint()
def main(workers: int = 4, harmonic_only: bool = True) -> None:
    summary = extract.remote(workers=workers, harmonic_only=harmonic_only)
    print(f"PASS: wrote {summary['ok_tracks']:,} rows to {summary['final_csv']}")
