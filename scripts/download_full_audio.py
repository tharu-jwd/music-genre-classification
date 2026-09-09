"""
download_full_audio.py
======================
Download the FULL MTG-Jamendo audio dataset (all shards) from the Freesound CDN.

  Full quality  (~508 GB extracted):  --quality full
  Low quality   (~156 GB extracted):  --quality low   ← default

The script:
  1. Tries shards 00, 01, 02, … until the server returns 404 (no more shards).
  2. Streams each tar directly — does NOT save the whole tar to disk first.
  3. Extracts files on the fly into --out, then moves on to the next shard.
  4. Skips already-extracted shards (uses a small marker file per shard).
  5. Is safe to re-run after interruption — already-done shards are skipped.

Storage needed (extracted audio only, no tar files kept):
  Low quality  : ~156 GB
  Full quality : ~508 GB

Usage
-----
    # From the project root:

    pip install requests tqdm

    # Low quality (recommended, ~156 GB)
    python scripts/download_full_audio.py --out data/audio-low --quality low

    # Full quality (~508 GB)
    python scripts/download_full_audio.py --out data/audio --quality full

    # Resume after interruption — just re-run the same command.
    # Already-completed shards are detected via marker files and skipped.
"""

import argparse
import io
import tarfile
from pathlib import Path

import requests
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# CDN config
# ---------------------------------------------------------------------------
BASE_URLS = {
    "full": "https://cdn.freesound.org/mtg-jamendo/raw_30s/audio",
    "low":  "https://cdn.freesound.org/mtg-jamendo/raw_30s/audio-low",
}
SHARD_NAMES = {
    "full": "raw_30s_audio-{shard:02d}.tar",
    "low":  "raw_30s_audio-low-{shard:02d}.tar",
}
# The dataset has been observed to have shards 00–09 (10 shards).
# The script auto-detects the last shard via HTTP 404, so this is just a cap.
MAX_SHARDS = 100


# ---------------------------------------------------------------------------
# Stream wrapper (requests response → file-like object for tarfile)
# ---------------------------------------------------------------------------
class _StreamWrapper(io.RawIOBase):
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
# Shard download
# ---------------------------------------------------------------------------
def download_shard(shard_idx: int, quality: str, out_dir: Path) -> int:
    """Stream-extract one shard. Returns number of files extracted, or -1 if 404."""
    tar_name = SHARD_NAMES[quality].format(shard=shard_idx)
    url = f"{BASE_URLS[quality]}/{tar_name}"
    marker = out_dir / f".shard_{quality}_{shard_idx:02d}_done"

    if marker.exists():
        print(f"[Shard {shard_idx:02d}] Already done — skipping")
        return 0

    print(f"\n[Shard {shard_idx:02d}] Connecting to {url} ...")

    with requests.get(url, stream=True, timeout=60) as resp:
        if resp.status_code == 404:
            print(f"[Shard {shard_idx:02d}] 404 — no more shards.")
            return -1
        resp.raise_for_status()

        # Show content-length if available
        total_bytes = resp.headers.get("Content-Length")
        if total_bytes:
            gb = int(total_bytes) / 1e9
            print(f"[Shard {shard_idx:02d}] Size: {gb:.2f} GB — streaming & extracting...")
        else:
            print(f"[Shard {shard_idx:02d}] Size unknown — streaming & extracting...")

        raw_stream = _StreamWrapper(resp, chunk_size=2 << 20)  # 2 MB chunks
        extracted = 0

        with tarfile.open(fileobj=raw_stream, mode="r|") as tf:
            for member in tqdm(tf, desc=f"  Shard {shard_idx:02d}", unit="file", leave=False):
                if not member.isfile():
                    continue
                name = member.name.lstrip("./")
                dest = out_dir / name
                dest.parent.mkdir(parents=True, exist_ok=True)

                if dest.exists():
                    continue  # already extracted

                fobj = tf.extractfile(member)
                if fobj is None:
                    continue
                dest.write_bytes(fobj.read())
                extracted += 1

    marker.write_text("ok")
    print(f"[Shard {shard_idx:02d}] Done — {extracted} new files extracted")
    return extracted


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Download the full MTG-Jamendo audio dataset (all shards)."
    )
    parser.add_argument(
        "--quality",
        choices=["full", "low"],
        default="low",
        help=(
            "Audio quality to download.\n"
            "  low  = mono VBR MP3 (~156 GB total) [default]\n"
            "  full = 320 kbps stereo MP3 (~508 GB total)"
        ),
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory. Defaults to data/audio-low (low) or data/audio (full).",
    )
    parser.add_argument(
        "--start-shard",
        type=int,
        default=0,
        help="Start from this shard index (default: 0).",
    )
    parser.add_argument(
        "--end-shard",
        type=int,
        default=None,
        help="Stop after this shard index (inclusive). E.g. --end-shard 9 downloads shards 00-09.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    default_out = "data/audio-low" if args.quality == "low" else "data/audio"
    raw_out = args.out if args.out else default_out
    out_dir = Path(raw_out) if Path(raw_out).is_absolute() else project_root / raw_out
    out_dir.mkdir(parents=True, exist_ok=True)

    quality_label = "Low quality mono VBR MP3 (~156 GB)" if args.quality == "low" else "Full quality 320kbps MP3 (~508 GB)"

    print("=" * 60)
    print(f"  MTG-Jamendo Full Dataset Downloader")
    print(f"  Quality    : {args.quality}  ({quality_label})")
    print(f"  Output dir : {out_dir}")
    print("=" * 60)
    print("  Streaming each tar shard — no temp files kept on disk.")
    print("  Re-run at any time to resume; completed shards are skipped.")
    print("=" * 60)

    end = (args.end_shard + 1) if args.end_shard is not None else MAX_SHARDS
    total_files = 0
    for shard_idx in range(args.start_shard, end):
        result = download_shard(shard_idx, args.quality, out_dir)
        if result == -1:
            print(f"\nNo more shards. Total new files: {total_files}")
            break
        total_files += result
    else:
        print(f"\nFinished. Total new files: {total_files}")

    print(f"\nDone! Audio is in: {out_dir}")


if __name__ == "__main__":
    main()
