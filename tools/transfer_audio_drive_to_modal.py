"""Copy the 7,324 selected MP3 files from Google Drive to a Modal Volume.

Run from the repository root:

    modal run tools/transfer_audio_drive_to_modal.py

The Modal Secret ``rclone-gdrive`` must expose ``RCLONE_CONFIG_CONTENT``. It
contains the private rclone configuration and is written only to a temporary
file inside the remote container; credentials are never committed to source.
The job is safe to rerun. Existing non-empty MP3 files are skipped and only
missing paths are requested from Google Drive.
"""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import time
from pathlib import Path

import modal


APP_NAME = "mtg-audio-drive-transfer"
VOLUME_NAME = "mtg-jamendo-audio"
SECRET_NAME = "rclone-gdrive"

# This is the ``audio`` directory beneath the user's Drive project folder.
# Override with --remote-root only if the Drive folder has a different name.
REMOTE_ROOT = "gdrive:MTG_Jamendo_7324_full_quality/audio"
EXPECTED_TRACKS = 7_324

DEFAULT_TRANSFERS = 8
DEFAULT_CHECKERS = 16
DEFAULT_TPS_LIMIT = 8
DEFAULT_BUFFER_SIZE = "8M"

CONTAINER_MANIFEST = "/app/split_csv.csv"
VOLUME_ROOT = "/audio"
DESTINATION = "/audio/audio"
RCLONE_CONFIG = "/tmp/rclone.conf"
FILES_FROM = "/tmp/selected-mp3s.txt"

if modal.is_local():
    REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
    MANIFEST_SOURCE = REPOSITORY_ROOT / "data" / "split_csv.csv"
    if not MANIFEST_SOURCE.is_file():
        raise FileNotFoundError(f"Required manifest is missing: {MANIFEST_SOURCE}")

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.11").apt_install("rclone")
if modal.is_local():
    image = image.add_local_file(str(MANIFEST_SOURCE), CONTAINER_MANIFEST, copy=True)


def _read_relative_mp3_paths(manifest_path: Path) -> list[str]:
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if len(rows) != EXPECTED_TRACKS:
        raise ValueError(f"expected {EXPECTED_TRACKS:,} rows, found {len(rows):,}")
    if not rows or not {"TRACK_ID", "PATH"}.issubset(rows[0]):
        raise ValueError("split_csv.csv must contain TRACK_ID and PATH columns")

    paths: list[str] = []
    track_ids: set[str] = set()
    seen_paths: set[str] = set()
    for row in rows:
        track_id = str(row["TRACK_ID"]).strip()
        relative_path = str(row["PATH"]).replace("\\", "/").strip()
        if not re.fullmatch(r"track_\d+", track_id):
            raise ValueError(f"unexpected TRACK_ID: {track_id!r}")
        if not re.fullmatch(r"\d{2}/\d+\.mp3", relative_path):
            raise ValueError(f"unsafe or unexpected audio PATH: {relative_path!r}")
        if track_id in track_ids:
            raise ValueError(f"duplicate TRACK_ID: {track_id}")
        if relative_path in seen_paths:
            raise ValueError(f"duplicate audio PATH: {relative_path}")
        if Path(relative_path).stem != str(int(track_id.removeprefix("track_"))):
            raise ValueError(f"TRACK_ID/PATH mismatch: {track_id} versus {relative_path}")
        track_ids.add(track_id)
        seen_paths.add(relative_path)
        paths.append(relative_path)
    return paths


@app.function(
    image=image,
    volumes={VOLUME_ROOT: volume},
    secrets=[modal.Secret.from_name(SECRET_NAME, required_keys=["RCLONE_CONFIG_CONTENT"])],
    # This is network and storage I/O, not GPU work.
    cpu=4.0,
    memory=4096,
    timeout=24 * 60 * 60,
)
def transfer(
    remote_root: str = REMOTE_ROOT,
    transfers: int = DEFAULT_TRANSFERS,
    checkers: int = DEFAULT_CHECKERS,
    tps_limit: int = DEFAULT_TPS_LIMIT,
    buffer_size: str = DEFAULT_BUFFER_SIZE,
) -> dict[str, object]:
    if not 1 <= transfers <= 32:
        raise ValueError("transfers must be between 1 and 32")
    if not 1 <= checkers <= 64:
        raise ValueError("checkers must be between 1 and 64")
    if not 1 <= tps_limit <= 20:
        raise ValueError("tps_limit must be between 1 and 20")
    if not re.fullmatch(r"\d+[KMG]", buffer_size, flags=re.IGNORECASE):
        raise ValueError("buffer_size must be like 8M, 16M, or 1G")

    selected = _read_relative_mp3_paths(Path(CONTAINER_MANIFEST))
    destination = Path(DESTINATION)
    destination.mkdir(parents=True, exist_ok=True)

    existing_files = 0
    existing_bytes = 0
    pending: list[str] = []
    for relative_path in selected:
        local_path = destination / relative_path
        if local_path.is_file() and local_path.stat().st_size > 0:
            existing_files += 1
            existing_bytes += local_path.stat().st_size
        else:
            pending.append(relative_path)

    rclone_config = Path(RCLONE_CONFIG)
    config_content = os.environ["RCLONE_CONFIG_CONTENT"]
    if "[gdrive]" not in config_content:
        raise ValueError("rclone-gdrive has no [gdrive] remote")
    rclone_config.write_text(config_content, encoding="utf-8")
    rclone_config.chmod(0o600)
    Path(FILES_FROM).write_text("".join(f"{path}\n" for path in pending), encoding="utf-8")

    summary_path = Path(VOLUME_ROOT) / "audio_transfer_summary.json"
    started = time.monotonic()
    print(
        f"Selected={len(selected):,}; already_present={existing_files:,}; "
        f"pending={len(pending):,}; destination={destination}",
        flush=True,
    )
    if pending:
        command = [
            "rclone", "copy", remote_root, str(destination),
            "--config", str(rclone_config),
            "--files-from", FILES_FROM,
            "--transfers", str(transfers),
            "--checkers", str(checkers),
            "--tpslimit", str(tps_limit),
            "--tpslimit-burst", str(tps_limit),
            "--buffer-size", buffer_size,
            "--contimeout", "30s",
            "--timeout", "5m",
            "--retries", "10",
            "--low-level-retries", "20",
            # One quiet status update per minute keeps the Modal UI responsive.
            "--stats", "60s",
            "--stats-one-line",
            "--stats-log-level", "NOTICE",
            "--log-level", "NOTICE",
        ]
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                f"rclone exited with code {completed.returncode}. "
                "Rerun the same command to resume missing files."
            )

    missing: list[str] = []
    zero_length: list[str] = []
    total_bytes = 0
    for relative_path in selected:
        local_path = destination / relative_path
        if not local_path.is_file():
            missing.append(relative_path)
            continue
        size = local_path.stat().st_size
        total_bytes += size
        if size <= 0:
            zero_length.append(relative_path)

    summary: dict[str, object] = {
        "status": "ok" if not missing and not zero_length else "incomplete",
        "selected_files": len(selected),
        "files_present_before_run": existing_files,
        "bytes_present_before_run": existing_bytes,
        "files_requested_from_drive": len(pending),
        "present_files": len(selected) - len(missing),
        "missing_files": len(missing),
        "zero_length_files": len(zero_length),
        "total_bytes": total_bytes,
        "total_gib": total_bytes / 2**30,
        "elapsed_seconds": time.monotonic() - started,
        "remote_root": remote_root,
        "destination": str(destination),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if missing:
        (Path(VOLUME_ROOT) / "missing_audio_paths.txt").write_text(
            "".join(f"{path}\n" for path in missing), encoding="utf-8"
        )
    if zero_length:
        (Path(VOLUME_ROOT) / "zero_length_audio_paths.txt").write_text(
            "".join(f"{path}\n" for path in zero_length), encoding="utf-8"
        )
    volume.commit()
    print(json.dumps(summary, indent=2), flush=True)

    if missing or zero_length:
        raise RuntimeError(
            f"verification failed: missing={len(missing)}, zero_length={len(zero_length)}"
        )
    return summary


@app.local_entrypoint()
def main(
    remote_root: str = REMOTE_ROOT,
    transfers: int = DEFAULT_TRANSFERS,
    checkers: int = DEFAULT_CHECKERS,
    tps_limit: int = DEFAULT_TPS_LIMIT,
    buffer_size: str = DEFAULT_BUFFER_SIZE,
) -> None:
    summary = transfer.remote(
        remote_root=remote_root,
        transfers=transfers,
        checkers=checkers,
        tps_limit=tps_limit,
        buffer_size=buffer_size,
    )
    print(
        f"PASS: {summary['present_files']:,} MP3 files are stored in "
        f"Volume {VOLUME_NAME!r} under audio/."
    )
