"""
Per-session Colab bootstrap: mount Drive, copy mels to local SSD, set paths.

Train against LOCAL_MEL_ROOT; copy checkpoints back to Drive when done.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

DRIVE_ROOT = Path("/content/drive/MyDrive/MTG_Instrument")
DRIVE_MEL_ROOT = DRIVE_ROOT / "dataset" / "logmel_songs"
LOCAL_MEL_ROOT = Path("/content/local_mels")
PATCHED_CODE = DRIVE_ROOT / "patched_baseline_code"
CHECKPOINT_ROOT = DRIVE_ROOT / "checkpoints"


def mount_drive() -> None:
    from google.colab import drive  # type: ignore

    drive.mount("/content/drive")


def sync_mels_to_local(force: bool = False) -> Path:
    LOCAL_MEL_ROOT.mkdir(parents=True, exist_ok=True)
    if force or not any(LOCAL_MEL_ROOT.iterdir()):
        subprocess.check_call(
            ["bash", "-lc", f"cp -r '{DRIVE_MEL_ROOT}/.' '{LOCAL_MEL_ROOT}/'"]
        )
        print(f"Copied mels → {LOCAL_MEL_ROOT}")
    else:
        print(f"Using existing local mels at {LOCAL_MEL_ROOT}")
    return LOCAL_MEL_ROOT


def export_env() -> dict[str, str]:
    env = {
        "DRIVE_ROOT": str(DRIVE_ROOT),
        "DRIVE_MEL_ROOT": str(DRIVE_MEL_ROOT),
        "LOCAL_MEL_ROOT": str(LOCAL_MEL_ROOT),
        "PATCHED_BASELINE": str(PATCHED_CODE),
        "CHECKPOINT_ROOT": str(CHECKPOINT_ROOT),
    }
    for k, v in env.items():
        os.environ[k] = v
        print(f"{k}={v}")
    return env


def push_checkpoints(local_dir: str | Path, stage: str = "stage2") -> None:
    """Copy a local model dir back to Drive checkpoints/<stage>/."""
    dest = CHECKPOINT_ROOT / stage
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(["cp", "-r", str(local_dir), str(dest)])
    print(f"Saved {local_dir} → {dest}")


if __name__ == "__main__":
    mount_drive()
    sync_mels_to_local()
    export_env()
    print("Ready. Point training --audio_path / data loaders at LOCAL_MEL_ROOT.")
