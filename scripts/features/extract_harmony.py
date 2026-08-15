"""
Member 3 — Harmony feature extraction (chroma + Tonnetz) via librosa.

Per 15-second window, then song-level means aligned to Stage 1 song IDs.

Usage:
  python scripts/features/extract_harmony.py \\
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


def harmony_features(y_win: np.ndarray, sr: int) -> dict[str, float]:
    chroma = librosa.feature.chroma_stft(y=y_win, sr=sr)
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y_win), sr=sr)
    feats: dict[str, float] = {}
    for i in range(chroma.shape[0]):
        feats[f"chroma_{i}_mean"] = float(np.mean(chroma[i]))
    for i in range(tonnetz.shape[0]):
        feats[f"tonnetz_{i}_mean"] = float(np.mean(tonnetz[i]))
    return feats


def extract_for_track(audio_path: Path, song_id: str) -> list[dict]:
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    rows = []
    for wi, (_, _, y_win) in enumerate(_window_slices(y, sr)):
        row = {"song_id": song_id, "window_idx": wi}
        row.update(harmony_features(y_win, sr))
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--path-col", default="path")
    parser.add_argument("--id-col", default="song_id")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
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
    feature_cols = [c for c in window_df.columns if c not in ("song_id", "window_idx")]
    song_df = window_df.groupby("song_id", as_index=False)[feature_cols].mean()

    out = args.out_dir / "harmony"
    out.mkdir(parents=True, exist_ok=True)
    window_df.to_csv(out / "harmony_windows.csv", index=False)
    song_df.to_csv(out / "harmony_song.csv", index=False)
    print(f"Wrote harmony features under {out}")


if __name__ == "__main__":
    main()
