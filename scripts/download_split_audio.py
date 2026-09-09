"""
download_split_audio.py
=======================
Download only the audio files listed in a split CSV from the MTG-Jamendo dataset,
without downloading every file in the full tar shards.

Strategy
--------
MTG-Jamendo audio is published as tar archives:

  Full quality (320 kbps MP3, ~508 GB total):
    https://cdn.freesound.org/mtg-jamendo/raw_30s/audio/raw_30s_audio-{shard:02d}.tar

  Low quality (mono VBR MP3, ~156 GB total):
    https://cdn.freesound.org/mtg-jamendo/raw_30s/audio-low/raw_30s_audio-low-{shard:02d}.tar

Each tar contains files organised as  <folder>/<track_id>.mp3
(e.g. 19/6719.mp3 lives in shard 00, 01, … determined by the leading folder number).

The shard that contains a file is determined by  int(folder) // 100
  folder 00-99   → shard 00
  folder 100-199 → shard 01
  …

We stream each required shard with `requests` and pull out only the members we
need using the `tarfile` streaming API — no temporary full-tar download required.

Usage
-----
    # Full quality (~3-4 GB for the 3,799-track split)
    python scripts/download_split_audio.py --csv data/metadata/stratified_sample.csv --out data/audio

    # Low quality (smaller files, ~1 GB for the split)
    python scripts/download_split_audio.py --csv data/metadata/stratified_sample.csv --out data/audio-low --quality low

    # Limit to specific shards (e.g. shard 0 only, to test)
    python scripts/download_split_audio.py --quality low --shards 0

Dependencies:  pip install requests tqdm pandas
"""

import argparse
import io
import tarfile
from collections import defaultdict
from pathlib import Path

import pandas as pd
import requests
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_URLS = {
    "full": "https://cdn.freesound.org/mtg-jamendo/raw_30s/audio",
    "low":  "https://cdn.freesound.org/mtg-jamendo/raw_30s/audio-low",
}
SHARD_NAMES = {
    "full": "raw_30s_audio-{shard:02d}.tar",
    "low":  "raw_30s_audio-low-{shard:02d}.tar",
}


def shard_for_folder(folder: int) -> int:
    """Map a folder number (the prefix directory in the path) to a shard index."""
    return folder // 100


def build_shard_map(file_paths: list[str]) -> dict[int, dict[str, str]]:
    """
    Returns  {shard_index: {member_path_in_tar: local_relative_path}}
    e.g. {0: {"19/6719.mp3": "19/6719.mp3"}}
    """
    shard_map: dict[int, dict[str, str]] = defaultdict(dict)
    for fp in file_paths:
        fp = fp.strip().lstrip("/")
        parts = fp.split("/")
        if len(parts) != 2:
            print(f"[WARN] Unexpected path format, skipping: {fp}")
            continue
        folder = int(parts[0])
        shard = shard_for_folder(folder)
        shard_map[shard][fp] = fp          # tar member path == relative output path
    return shard_map


def stream_extract_shard(
    shard_idx: int,
    wanted: dict[str, str],   # {tar_member_path: output_relative_path}
    out_dir: Path,
    quality: str = "full",
    chunk_size: int = 1 << 20,  # 1 MB read chunks
) -> int:
    """Stream a shard tar from the CDN and extract only `wanted` members.

    Returns the number of files successfully extracted.
    """
    tar_name = SHARD_NAMES[quality].format(shard=shard_idx)
    url = f"{BASE_URLS[quality]}/{tar_name}"
    print(f"\n[Shard {shard_idx:02d}] Streaming {url}")
    print(f"          Need {len(wanted)} file(s) from this shard")

    remaining = set(wanted.keys())
    extracted = 0

    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()

        # Wrap the streaming response in a file-like object for tarfile
        raw_stream = _StreamWrapper(resp, chunk_size=chunk_size)

        with tarfile.open(fileobj=raw_stream, mode="r|") as tf:
            for member in tf:
                # Normalise the member name (some tars have a leading ./)
                name = member.name.lstrip("./")

                if name not in remaining:
                    # Skip — tarfile streaming: we must still "read through" to advance
                    continue

                dest = out_dir / wanted[name]
                dest.parent.mkdir(parents=True, exist_ok=True)

                if dest.exists():
                    print(f"  [skip] {name}  (already exists)")
                    remaining.discard(name)
                    extracted += 1
                    continue

                fobj = tf.extractfile(member)
                if fobj is None:
                    continue

                dest.write_bytes(fobj.read())
                print(f"  [ok]   {name}")
                remaining.discard(name)
                extracted += 1

                if not remaining:
                    break  # all wanted files for this shard found — stop early

    if remaining:
        print(f"  [WARN] {len(remaining)} file(s) not found in shard {shard_idx:02d}:")
        for r in sorted(remaining):
            print(f"         {r}")

    return extracted


class _StreamWrapper(io.RawIOBase):
    """Thin wrapper that makes a `requests` streaming response look like a readable
    binary file-object (needed by tarfile in streaming mode ``r|``)."""

    def __init__(self, response: requests.Response, chunk_size: int = 1 << 20):
        self._iter = response.iter_content(chunk_size=chunk_size)
        self._buf = b""

    def readable(self) -> bool:
        return True

    def readinto(self, b: bytearray) -> int:
        n = len(b)
        while len(self._buf) < n:
            try:
                self._buf += next(self._iter)
            except StopIteration:
                break
        chunk = self._buf[:n]
        self._buf = self._buf[n:]
        b[: len(chunk)] = chunk
        return len(chunk)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Download MTG-Jamendo split audio (streaming, no full-tar download)")
    parser.add_argument(
        "--csv",
        default="data/metadata/stratified_sample.csv",
        help="Path to the split CSV containing a `file_path` column (default: %(default)s)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory for downloaded .mp3 files. Defaults to data/audio (full) or data/audio-low (low).",
    )
    parser.add_argument(
        "--quality",
        choices=["full", "low"],
        default="full",
        help="Audio quality: 'full' = 320kbps MP3 (~508 GB total), 'low' = mono VBR MP3 (~156 GB total). Default: full",
    )
    parser.add_argument(
        "--shards",
        nargs="*",
        type=int,
        default=None,
        help="Limit to specific shard indices (e.g. --shards 0 1). Default: all required shards.",
    )
    args = parser.parse_args()

    # Resolve paths relative to the project root (script's parent's parent)
    project_root = Path(__file__).resolve().parent.parent
    csv_path = Path(args.csv) if Path(args.csv).is_absolute() else project_root / args.csv

    # Default output dir based on quality
    default_out = "data/audio" if args.quality == "full" else "data/audio-low"
    raw_out = args.out if args.out is not None else default_out
    out_dir = Path(raw_out) if Path(raw_out).is_absolute() else project_root / raw_out

    print(f"CSV:        {csv_path}")
    print(f"Quality:    {args.quality} ({'320kbps MP3' if args.quality == 'full' else 'mono VBR MP3 (smaller)'}")
    print(f"Output dir: {out_dir}")

    # Load split
    df = pd.read_csv(csv_path)
    if "file_path" not in df.columns:
        raise ValueError(f"CSV must have a `file_path` column. Found: {list(df.columns)}")

    file_paths = df["file_path"].dropna().tolist()
    print(f"Tracks in split: {len(file_paths)}")

    # Build shard -> files map
    shard_map = build_shard_map(file_paths)
    all_shards = sorted(shard_map.keys())
    print(f"Shards required: {all_shards}")

    if args.shards is not None:
        all_shards = [s for s in all_shards if s in args.shards]
        print(f"Filtered to:     {all_shards}")

    # Already-downloaded check
    already = sum(1 for fp in file_paths if (out_dir / fp).exists())
    print(f"Already downloaded: {already}/{len(file_paths)}")

    # Stream & extract
    total_extracted = 0
    for shard_idx in tqdm(all_shards, desc="Shards", unit="shard"):
        wanted = shard_map[shard_idx]
        # Skip members that already exist
        wanted_missing = {k: v for k, v in wanted.items() if not (out_dir / v).exists()}
        if not wanted_missing:
            print(f"\n[Shard {shard_idx:02d}] All files already present -- skipping shard entirely")
            total_extracted += len(wanted)
            continue
        total_extracted += stream_extract_shard(shard_idx, wanted_missing, out_dir, quality=args.quality)

    print(f"\nDone. {total_extracted}/{len(file_paths)} files in {out_dir}")


if __name__ == "__main__":
    main()
