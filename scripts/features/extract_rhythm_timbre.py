"""
Member 2 — Rhythm + Timbre feature extraction (librosa, no training).

Computes per 15-second window features and writes song-aligned tables
joinable on the same track/song IDs as the Stage 1 manifest.

Usage (local or Colab after audio paths are available):
  python scripts/features/extract_rhythm_timbre.py \\
      --manifest path/to/song_manifest.csv \\
      --audio-root path/to/wav_or_mp3 \\
      --out-dir /content/drive/MyDrive/MTG_Instrument/features
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import librosa
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install librosa before running this script.") from exc

WINDOW_SEC = 15.0


def _window_slices(y: np.ndarray, sr: int, window_sec: float = WINDOW_SEC):
    hop = int(window_sec * sr)
    for start in range(0, max(len(y) - hop + 1, 1), hop):
        end = start + hop
        if end > len(y):
            break
        yield start, end, y[start:end]


def rhythm_features(y_win: np.ndarray, sr: int) -> dict[str, float]:
    onset_env = librosa.onset.onset_strength(y=y_win, sr=sr)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    tempo_f = float(np.atleast_1d(tempo)[0])
    beat_times = librosa.frames_to_time(beats, sr=sr) if len(beats) else np.array([])
    intervals = np.diff(beat_times) if len(beat_times) > 1 else np.array([np.nan])
    return {
        "tempo": tempo_f,
        "beat_strength_mean": float(np.mean(onset_env)) if len(onset_env) else np.nan,
        "onset_density": float(len(librosa.onset.onset_detect(y=y_win, sr=sr)) / WINDOW_SEC),
        "beat_interval_mean": float(np.nanmean(intervals)),
        "beat_interval_std": float(np.nanstd(intervals)),
    }


def timbre_features(y_win: np.ndarray, sr: int) -> dict[str, float]:
    S = np.abs(librosa.stft(y_win))
    centroid = librosa.feature.spectral_centroid(S=S, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)
    contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
    flatness = librosa.feature.spectral_flatness(S=S)
    rms = librosa.feature.rms(y=y_win)
    # spectral flux via onset strength proxy
    flux = librosa.onset.onset_strength(y=y_win, sr=sr)
    return {
        "spectral_centroid_mean": float(np.mean(centroid)),
        "spectral_bandwidth_mean": float(np.mean(bandwidth)),
        "spectral_contrast_mean": float(np.mean(contrast)),
        "spectral_flatness_mean": float(np.mean(flatness)),
        "rms_mean": float(np.mean(rms)),
        "spectral_flux_mean": float(np.mean(flux)) if len(flux) else np.nan,
    }


def extract_for_track(audio_path: Path, song_id: str) -> list[dict]:
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    rows = []
    for wi, (_, _, y_win) in enumerate(_window_slices(y, sr)):
        row = {"song_id": song_id, "window_idx": wi}
        row.update(rhythm_features(y_win, sr))
        row.update(timbre_features(y_win, sr))
        rows.append(row)
    return rows


def aggregate_song_level(window_df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [c for c in window_df.columns if c not in ("song_id", "window_idx")]
    return window_df.groupby("song_id", as_index=False)[feature_cols].mean()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="CSV with song_id + audio path column")
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--path-col", default="path", help="Manifest column with relative audio path")
    parser.add_argument("--id-col", default="song_id")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None, help="Optional cap for smoke tests")
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    if args.limit:
        manifest = manifest.head(args.limit)

    all_rows: list[dict] = []
    for _, rec in manifest.iterrows():
        song_id = str(rec[args.id_col])
        audio_path = args.audio_root / str(rec[args.path_col])
        if not audio_path.exists():
            print(f"Missing audio: {audio_path}")
            continue
        all_rows.extend(extract_for_track(audio_path, song_id))

    if not all_rows:
        raise SystemExit("No features extracted — check manifest paths.")

    window_df = pd.DataFrame(all_rows)
    song_df = aggregate_song_level(window_df)

    out = args.out_dir
    (out / "rhythm").mkdir(parents=True, exist_ok=True)
    (out / "timbre").mkdir(parents=True, exist_ok=True)

    rhythm_cols = ["song_id"] + [c for c in song_df.columns if c.startswith(("tempo", "beat_", "onset_"))]
    timbre_cols = ["song_id"] + [
        c
        for c in song_df.columns
        if c.startswith(("spectral_", "rms_"))
    ]
    window_df.to_csv(out / "rhythm" / "rhythm_timbre_windows.csv", index=False)
    song_df[rhythm_cols].to_csv(out / "rhythm" / "rhythm_song.csv", index=False)
    song_df[timbre_cols].to_csv(out / "timbre" / "timbre_song.csv", index=False)
    print(f"Wrote rhythm/timbre features under {out}")


if __name__ == "__main__":
    main()
