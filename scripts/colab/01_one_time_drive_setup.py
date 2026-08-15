"""
One-time Colab setup for MTG_Instrument Drive layout + mel shards.

Run in Google Colab (not local) after mounting Drive.
After this succeeds, use 02_session_bootstrap.py every later session.

IMPORTANT: Apply the three Stage 1 patches to the cloned baseline BEFORE
copying into patched_baseline_code (see docs/phase2-verification-checklist.md).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

DRIVE_ROOT = Path("/content/drive/MyDrive/MTG_Instrument")
MEL_DIR = DRIVE_ROOT / "dataset" / "logmel_songs"
SHARD_URLS = [
    f"https://cdn.freesound.org/mtg-jamendo/raw_30s/melspecs/raw_30s_melspecs-{i:02d}.tar"
    for i in (0, 1, 2)
]


def mount_drive() -> None:
    from google.colab import drive  # type: ignore

    drive.mount("/content/drive")


def ensure_dirs() -> None:
    for sub in (
        "dataset/logmel_songs",
        "checkpoints/baseline",
        "checkpoints/stage1",
        "checkpoints/stage2",
        "mlruns",
        "features/instrument",
        "features/rhythm",
        "features/timbre",
        "features/harmony",
        "patched_baseline_code",
    ):
        (DRIVE_ROOT / sub).mkdir(parents=True, exist_ok=True)
    print(f"Drive tree ready under {DRIVE_ROOT}")


def download_and_extract_shards(shard_indices: tuple[int, ...] = (0, 1, 2)) -> None:
    MEL_DIR.mkdir(parents=True, exist_ok=True)
    os.chdir(MEL_DIR)
    for i in shard_indices:
        url = f"https://cdn.freesound.org/mtg-jamendo/raw_30s/melspecs/raw_30s_melspecs-{i:02d}.tar"
        tar_name = Path(url).name
        if not (MEL_DIR / tar_name).exists() and not list(MEL_DIR.glob(f"*{i:02d}*")):
            subprocess.check_call(["wget", "-q", url])
        if (MEL_DIR / tar_name).exists():
            subprocess.check_call(["tar", "-xf", tar_name])
            (MEL_DIR / tar_name).unlink()
            print(f"Extracted and removed {tar_name}")
        else:
            print(f"Skip download for shard {i:02d} (already present or tar missing)")


def clone_baseline_repo(dest: Path = Path("/content/mtg-jamendo-dataset")) -> Path:
    if not dest.exists():
        subprocess.check_call(
            ["git", "clone", "https://github.com/MTG/mtg-jamendo-dataset.git", str(dest)]
        )
    print(
        "Apply Stage 1 patches (best_macro_map, split leakage, resolve_stacked_mel_path) "
        f"under {dest / 'scripts' / 'baseline'}, then call copy_patched_baseline()."
    )
    return dest


def copy_patched_baseline(
    src: Path = Path("/content/mtg-jamendo-dataset/scripts/baseline"),
) -> None:
    dest = DRIVE_ROOT / "patched_baseline_code"
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(["cp", "-r", f"{src}/.", str(dest)])
    print(f"Patched baseline copied to {dest}")


if __name__ == "__main__":
    mount_drive()
    ensure_dirs()
    # Uncomment when ready to pay the one-time download cost:
    # download_and_extract_shards()
    clone_baseline_repo()
    print(
        "Stopped before copy_patched_baseline(). "
        "Patch first, then run copy_patched_baseline() in a follow-up cell."
    )
