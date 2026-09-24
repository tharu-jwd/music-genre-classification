"""Export the exact waveform regions matching model log-Mel windows for a frozen cohort."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    from scripts.gpu_run_contract import load_cohort_artifact
    from scripts.mtg_data_contract import NOTEBOOK_DATA_CONTRACT
except ModuleNotFoundError:
    from gpu_run_contract import load_cohort_artifact
    from mtg_data_contract import NOTEBOOK_DATA_CONTRACT


_contract = {"Path": Path, "np": np, "re": re, "ANN_DIR": Path(".")}
exec(NOTEBOOK_DATA_CONTRACT, _contract)
logmel_window_plan = _contract["logmel_window_plan"]


SCHEMA_VERSION = "harmony_audio_regions_v1"
MAX_SONGS = 32


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_existing(row: dict, fields: tuple[str, ...], root: Path) -> Path | None:
    for field in fields:
        raw = str(row.get(field, "")).strip()
        if not raw:
            continue
        path = Path(raw)
        if not path.is_absolute():
            path = root / path
        if path.is_file():
            return path.resolve()
    return None


def _strict_true(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1"}


def export_regions(
    manifest_path: Path,
    cohort_path: Path,
    *,
    root: Path,
    max_songs: int = MAX_SONGS,
    regions_per_song: int | None = None,
    allow_test: bool = False,
) -> dict:
    manifest_path = Path(manifest_path)
    cohort = load_cohort_artifact(cohort_path)
    if _sha256_file(manifest_path) != cohort["source_manifest"]["sha256"]:
        raise ValueError("manifest SHA-256 does not match the frozen cohort")
    selected_split = {
        song_id: split
        for split, song_ids in cohort["splits"].items()
        for song_id in song_ids
    }
    if not allow_test and any(split == "test" for split in selected_split.values()):
        raise ValueError("test IDs are forbidden during harmony development")
    if not 1 <= max_songs <= MAX_SONGS:
        raise ValueError(f"max_songs must be in [1, {MAX_SONGS}]")
    if len(selected_split) > max_songs:
        raise ValueError(f"cohort has {len(selected_split)} songs; cap is {max_songs}")
    if regions_per_song is not None and not 1 <= regions_per_song <= 12:
        raise ValueError("regions_per_song must be in [1, 12]")

    with manifest_path.open(newline="", encoding="utf-8", errors="strict") as handle:
        rows = list(csv.DictReader(handle))
    by_id = {}
    for row in rows:
        song_id = str(row.get("song_id", "")).strip()
        if song_id in by_id:
            raise ValueError(f"duplicate manifest song_id: {song_id}")
        by_id[song_id] = row

    songs = []
    for song_id, split in selected_split.items():
        row = by_id.get(song_id)
        if row is None:
            raise ValueError(f"cohort song is absent from manifest: {song_id}")
        if row.get("split") != split:
            raise ValueError(f"split mismatch for {song_id}: {row.get('split')!r} != {split!r}")
        if not _strict_true(row.get("waveform_available", "")):
            raise ValueError(f"waveform is unavailable for cohort song {song_id}")
        audio_path = _resolve_existing(row, ("audio_path",), Path(root))
        logmel_path = _resolve_existing(row, ("logmel_path", "mel_abs"), Path(root))
        if audio_path is None:
            raise FileNotFoundError(f"waveform path is missing for {song_id}")
        if logmel_path is None:
            raise FileNotFoundError(f"log-Mel path is missing for {song_id}")
        raw = np.load(logmel_path, mmap_mode="r", allow_pickle=False)
        regions = logmel_window_plan(raw, n_mels=96, n_frames=1366, max_windows=12)
        total_model_regions = len(regions)
        if regions_per_song is not None and total_model_regions > regions_per_song:
            selected_indices = np.linspace(
                0, total_model_regions - 1, regions_per_song, dtype=int
            ).tolist()
            regions = [regions[index] for index in selected_indices]
        songs.append({
            "song_id": song_id,
            "split": split,
            "audio_path": str(audio_path),
            "logmel_path": str(logmel_path),
            "total_model_regions": total_model_regions,
            "regions": regions,
        })

    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cohort_path": str(Path(cohort_path).resolve()),
        "cohort_sha256": cohort["cohort_sha256"],
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": cohort["source_manifest"]["sha256"],
        "region_policy": "exact_mtg_full_audio_logmel_windows_v1",
        "region_selection": {
            "method": "all" if regions_per_song is None else "evenly_spaced_model_windows_v1",
            "max_regions_per_song": regions_per_song,
        },
        "time_reference": "logmel frame boundaries at 12000 Hz with 256-sample hop",
        "songs": songs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("cohort", type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-songs", type=int, default=MAX_SONGS)
    parser.add_argument("--regions-per-song", type=int)
    parser.add_argument("--allow-test", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; alignment plans are immutable")
    artifact = export_regions(
        args.manifest,
        args.cohort,
        root=args.root,
        max_songs=args.max_songs,
        regions_per_song=args.regions_per_song,
        allow_test=args.allow_test,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "songs": len(artifact["songs"]),
        "regions": sum(len(song["regions"]) for song in artifact["songs"]),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
