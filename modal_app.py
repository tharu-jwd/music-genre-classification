"""Modal runner for joint concept-bottleneck training.

Run from the repository root with, for example:
    modal run --detach modal_app.py --epochs 30 --run-name first-full-run
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import modal


APP_NAME = "music-genre-joint-training"
DATA_VOLUME_NAME = "music-genre-data"
RUNS_VOLUME_NAME = "music-genre-runs"
PROJECT_DIR = Path("/root/project")
DATA_MOUNT = Path("/data")
RUNS_MOUNT = Path("/runs")

app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
runs_volume = modal.Volume.from_name(RUNS_VOLUME_NAME, create_if_missing=True)

# Only ship source and the small split manifest. The 8.5 MB combined table and
# thousands of log-mel arrays are uploaded once to the persistent data Volume.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy==2.2.6",
        "pandas==2.2.3",
        "torch==2.7.1",
    )
    .add_local_dir("concept_fusion", str(PROJECT_DIR / "concept_fusion"))
    .add_local_dir("shared_encoder", str(PROJECT_DIR / "shared_encoder"))
    .add_local_dir("rhythm_branch/src/rhythm_branch", str(PROJECT_DIR / "rhythm_branch/src/rhythm_branch"))
    .add_local_dir("timbre_branch/src/timbre_branch", str(PROJECT_DIR / "timbre_branch/src/timbre_branch"))
    .add_local_dir("harmony_branch/src/harmony_branch", str(PROJECT_DIR / "harmony_branch/src/harmony_branch"))
    .add_local_dir("instrument_branch/docs", str(PROJECT_DIR / "instrument_branch/docs"))
    .add_local_file("scripts/train_joint.py", str(PROJECT_DIR / "scripts/train_joint.py"))
    .add_local_file("scripts/mtg_data_contract.py", str(PROJECT_DIR / "scripts/mtg_data_contract.py"))
)


@app.function(
    image=image,
    gpu="A10",
    cpu=4,
    memory=16_384,
    timeout=24 * 60 * 60,
    volumes={str(DATA_MOUNT): data_volume, str(RUNS_MOUNT): runs_volume},
)
def train_remote(
    run_name: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    num_workers: int,
    max_windows: int,
    quick: bool,
    skip_test: bool,
) -> dict[str, object]:
    """Validate the Volume layout, run training, and persist all outputs."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Modal allocated no CUDA device")

    dataset_dir = DATA_MOUNT / "dataset"
    dataset_csv = dataset_dir / "full_dataset.csv"
    split_csv = dataset_dir / "track_split_assignments.csv"
    logmel_root = DATA_MOUNT / "logmel_songs"
    required = (dataset_csv, split_csv, logmel_root)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "The music-genre-data Volume is incomplete. Missing: " + ", ".join(missing)
        )

    out_dir = RUNS_MOUNT / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(PROJECT_DIR / "scripts/train_joint.py"),
        "--data-dir", str(dataset_dir),
        "--dataset-csv", str(dataset_csv),
        "--split-csv", str(split_csv),
        "--logmel-root", str(logmel_root),
        "--out-dir", str(out_dir),
        "--device", "cuda",
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--lr", str(learning_rate),
        "--num-workers", str(num_workers),
        "--max-windows", str(max_windows),
    ]
    if quick:
        command.append("--quick")
    if skip_test:
        command.append("--skip-test")

    print("Starting:", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_DIR, check=True)
    runs_volume.commit()

    results_path = out_dir / "results.json"
    if not results_path.is_file():
        raise RuntimeError(f"Training completed without {results_path}")
    results = json.loads(results_path.read_text(encoding="utf-8"))
    return {
        "run_name": run_name,
        "results_path": str(results_path),
        "checkpoint_path": str(out_dir / "best.pt"),
        "best_epoch": results["best_epoch"],
        "val_macro_ap": results["val_macro_ap"],
        "test_macro_ap": results["test_macro_ap"],
    }


@app.local_entrypoint()
def main(
    run_name: str = "joint-full-v1",
    epochs: int = 30,
    batch_size: int = 1,
    learning_rate: float = 3e-4,
    num_workers: int = 2,
    max_windows: int = 12,
    gpu: str = "A10",
    quick: bool = False,
    skip_test: bool = False,
) -> None:
    """Submit one GPU training run from any authenticated Modal account."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", run_name):
        raise ValueError("run_name must be 1-64 safe filename characters")
    if epochs < 1 or batch_size < 1 or num_workers < 0 or max_windows < 1:
        raise ValueError("epochs, batch_size, and max_windows must be positive; workers cannot be negative")

    result = train_remote.with_options(gpu=gpu).remote(
        run_name,
        epochs,
        batch_size,
        learning_rate,
        num_workers,
        max_windows,
        quick,
        skip_test,
    )
    print(json.dumps(result, indent=2))
