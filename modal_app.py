"""Modal runner for joint concept-bottleneck training.

Run from the repository root with, for example:
    modal run --detach modal_app.py --epochs 30 --run-name first-full-run
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath

import modal


APP_NAME = "music-genre-joint-training"
DATA_VOLUME_NAME = "music-genre-data"
AUDIO_VOLUME_NAME = "mtg-jamendo"
RUNS_VOLUME_NAME = "music-genre-runs"
DATASET_FILENAME = "vector-dataset-normalized.csv"
# Container paths must retain Linux separators when submitted from Windows.
PROJECT_DIR = PurePosixPath("/root/project")
DATA_MOUNT = PurePosixPath("/data")
AUDIO_MOUNT = PurePosixPath("/mtg")
RUNS_MOUNT = PurePosixPath("/runs")

app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
audio_volume = modal.Volume.from_name(AUDIO_VOLUME_NAME)
runs_volume = modal.Volume.from_name(RUNS_VOLUME_NAME, create_if_missing=True)

# Only ship source and the small split manifest. The 8.5 MB combined table and
# thousands of log-mel arrays are uploaded once to the persistent data Volume.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy==2.2.6",
        "pandas==2.2.3",
        "scikit-learn==1.6.1",
        "torch==2.7.1",
    )
    .add_local_dir("concept_fusion", str(PROJECT_DIR / "concept_fusion"))
    .add_local_dir("shared_encoder", str(PROJECT_DIR / "shared_encoder"))
    .add_local_dir("configs", str(PROJECT_DIR / "configs"))
    .add_local_dir("rhythm_branch/src/rhythm_branch", str(PROJECT_DIR / "rhythm_branch/src/rhythm_branch"))
    .add_local_dir("timbre_branch/src/timbre_branch", str(PROJECT_DIR / "timbre_branch/src/timbre_branch"))
    .add_local_dir("harmony_branch/src/harmony_branch", str(PROJECT_DIR / "harmony_branch/src/harmony_branch"))
    .add_local_dir("instrument_branch/docs", str(PROJECT_DIR / "instrument_branch/docs"))
    .add_local_file("scripts/train_cnn.py", str(PROJECT_DIR / "scripts/train_cnn.py"))
    .add_local_file("scripts/train_joint.py", str(PROJECT_DIR / "scripts/train_joint.py"))
    .add_local_file("scripts/mtg_data_contract.py", str(PROJECT_DIR / "scripts/mtg_data_contract.py"))
    .add_local_file("scripts/paper_artifacts.py", str(PROJECT_DIR / "scripts/paper_artifacts.py"))
    .add_local_file("scripts/paper_experiments.py", str(PROJECT_DIR / "scripts/paper_experiments.py"))
    .add_local_file("scripts/evaluate_checkpoint.py", str(PROJECT_DIR / "scripts/evaluate_checkpoint.py"))
    .add_local_file("scripts/pilot_paper_run.py", str(PROJECT_DIR / "scripts/pilot_paper_run.py"))
    .add_local_file(
        "scripts/build_vector_dataset.py",
        str(PROJECT_DIR / "scripts/build_vector_dataset.py"),
    )
)


@app.function(
    image=image,
    gpu="A10",
    cpu=4,
    memory=16_384,
    timeout=24 * 60 * 60,
    volumes={
        str(DATA_MOUNT): data_volume,
        str(AUDIO_MOUNT): audio_volume,
        str(RUNS_MOUNT): runs_volume,
    },
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
    model: str = "joint",
    branches: str = "",
    seed: int = 42,
    fusion: str = "native_concat",
    genre_predictor: str = "linear",
    genre_hidden_dim: int = 64,
    genre_dropout: float = 0.0,
    concept_dropout: float = 0.0,
    lambda_aux: float = 1.0,
    window_frames: int = 1366,
    weight_decay: float = 1e-4,
    dataset_filename: str = DATASET_FILENAME,
    architecture: dict | None = None,
) -> dict[str, object]:
    """Validate the Volume layout, run training, and persist all outputs."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Modal allocated no CUDA device")
    if dataset_filename not in ("vector-dataset.csv", "vector-dataset-normalized.csv"):
        raise ValueError("Unsupported dataset filename")
    data_volume.reload()
    runs_volume.reload()

    dataset_dir = Path(DATA_MOUNT) / "dataset"
    dataset_csv = dataset_dir / dataset_filename
    split_csv = dataset_dir / "track_split_assignments.csv"
    logmel_root = Path(AUDIO_MOUNT) / "logmel_songs"
    required = (dataset_csv, split_csv, logmel_root)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "The music-genre-data Volume is incomplete. Missing: " + ", ".join(missing)
        )

    out_dir = Path(RUNS_MOUNT) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(PROJECT_DIR / "scripts/train_joint.py"),
        "--model", model,
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
        "--seed", str(seed),
        "--fusion", fusion,
        "--genre-predictor", genre_predictor,
        "--genre-hidden-dim", str(genre_hidden_dim),
        "--genre-dropout", str(genre_dropout),
        "--concept-dropout", str(concept_dropout),
        "--lambda-aux", str(lambda_aux),
        "--window-frames", str(window_frames),
        "--weight-decay", str(weight_decay),
    ]
    if branches:
        command.extend(["--branches", *branches.replace(",", " ").split()])
    if quick:
        command.append("--quick")
    if skip_test:
        command.append("--skip-test")
    if architecture is not None:
        command.extend(["--architecture-json", json.dumps(architecture)])

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
    max_windows: int = 16,
    gpu: str = "A10",
    quick: bool = False,
    skip_test: bool = False,
    background: bool = False,
    model: str = "joint",
    branches: str = "",
    seed: int = 42,
    fusion: str = "native_concat",
    genre_predictor: str = "linear",
    genre_hidden_dim: int = 64,
    genre_dropout: float = 0.0,
    concept_dropout: float = 0.0,
    lambda_aux: float = 1.0,
    window_frames: int = 1366,
    weight_decay: float = 1e-4,
    dataset_filename: str = DATASET_FILENAME,
) -> None:
    """Submit one GPU training run from any authenticated Modal account."""
    if model not in ("joint", "cnn"):
        raise ValueError("model must be joint or cnn")
    if fusion not in ("native_concat", "gated"):
        raise ValueError("fusion must be native_concat or gated")
    if genre_predictor not in ("linear", "mlp") or genre_hidden_dim < 1:
        raise ValueError("genre predictor must be linear or mlp with a positive hidden width")
    if not 0 <= genre_dropout < 1 or not 0 <= concept_dropout < 1:
        raise ValueError("dropout rates must be in [0,1)")
    if genre_predictor == "linear" and genre_dropout != 0:
        raise ValueError("genre dropout requires the MLP predictor")
    if fusion == "gated" and (genre_predictor != "linear" or genre_dropout != 0 or concept_dropout != 0):
        raise ValueError("genre predictor/dropout options require native_concat fusion")
    if not math.isfinite(lambda_aux) or lambda_aux < 0:
        raise ValueError("lambda_aux must be finite and nonnegative")
    selected = branches.replace(",", " ").split()
    allowed = {"instrument", "rhythm", "timbre", "harmony"}
    if selected and selected != ["none"]:
        if set(selected) - allowed or len(set(selected)) != len(selected):
            raise ValueError("branches must be unique concept names, or none")
        if model == "cnn":
            raise ValueError("model cnn cannot enable concept branches")
    if selected and run_name == "joint-full-v1":
        run_name = "cnn-full-v1" if selected == ["none"] else "joint-" + "-".join(selected)
    if model == "cnn" and run_name == "joint-full-v1":
        run_name = "cnn-full-v1"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", run_name):
        raise ValueError("run_name must be 1-64 safe filename characters")
    if epochs < 1 or batch_size < 1 or num_workers < 0 or max_windows < 1:
        raise ValueError("epochs, batch_size, and max_windows must be positive; workers cannot be negative")

    arguments = (
        run_name,
        epochs,
        batch_size,
        learning_rate,
        num_workers,
        max_windows,
        quick,
        skip_test,
        model,
        branches,
        seed,
        fusion,
        genre_predictor,
        genre_hidden_dim,
        genre_dropout,
        concept_dropout,
        lambda_aux,
        window_frames,
        weight_decay,
        dataset_filename,
    )
    remote = train_remote.with_options(gpu=gpu)
    if background:
        call = remote.spawn(*arguments)
        print(json.dumps({
            "status": "submitted",
            "run_name": run_name,
            "function_call_id": call.object_id,
            "results_volume": RUNS_VOLUME_NAME,
        }, indent=2))
        return
    result = remote.remote(*arguments)
    print(json.dumps(result, indent=2))


@app.function(image=image, gpu="A10", cpu=4, memory=16_384, timeout=7200,
              volumes={str(DATA_MOUNT): data_volume, str(AUDIO_MOUNT): audio_volume,
                       str(RUNS_MOUNT): runs_volume})
def evaluate_remote(run_name: str, split: str = "test"):
    """Evaluate the saved checkpoint and persist predictions, never train."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", run_name):
        raise ValueError("Invalid run name")
    data_volume.reload()
    runs_volume.reload()
    command = [sys.executable, str(PROJECT_DIR / "scripts/evaluate_checkpoint.py"),
               "--checkpoint", str(RUNS_MOUNT / run_name / "best.pt"), "--split", split,
               "--device", "cuda", "--num-workers", "2"]
    subprocess.run(command, cwd=PROJECT_DIR, check=True)
    runs_volume.commit()
    return json.loads((Path(RUNS_MOUNT) / run_name / f"evaluation_{split}.json").read_text())


@app.function(image=image, gpu="A10", cpu=4, memory=16_384, timeout=24 * 60 * 60,
              volumes={str(DATA_MOUNT): data_volume, str(AUDIO_MOUNT): audio_volume,
                       str(RUNS_MOUNT): runs_volume})
def train_and_evaluate_remote(training_arguments: dict, train_completed: bool = False):
    """Keep training, validation selection and final test evaluation in one remote job."""
    training = None
    if not train_completed:
        training = train_remote.local(**training_arguments)
    evaluation = evaluate_remote.local(training_arguments["run_name"], "test")
    return dict(run_name=training_arguments["run_name"], training=training, test=evaluation)


@app.function(image=image, cpu=1, memory=2048, timeout=120,
              volumes={str(DATA_MOUNT): data_volume, str(RUNS_MOUNT): runs_volume})
def paper_status_remote(run_names: list[str]):
    """Read completed run provenance on CPU before allocating another GPU."""
    from types import SimpleNamespace
    from scripts.paper_artifacts import dataset_fingerprints, code_fingerprints, sha256
    data_volume.reload()
    runs_volume.reload()
    result = {}
    for name in run_names:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", name):
            raise ValueError("Invalid run name")
        directory = Path(RUNS_MOUNT) / name
        if (directory / "results.json").exists():
            stored = json.loads((directory / "results.json").read_text())
            if stored.get("code_fingerprints") != code_fingerprints():
                raise ValueError(f"Source changed since completed run: {name}")
            cfg = SimpleNamespace(**stored["config"])
            current = dataset_fingerprints(cfg)
            fingerprints = stored.get("dataset_fingerprints", {})
            if set(current) != set(fingerprints) or any(current[key]["sha256"] != fingerprints[key]["sha256"] for key in current):
                raise ValueError(f"Dataset changed since completed run: {name}")
            if not (directory / "best.pt").exists() or not (directory / "validation_predictions.npz").exists():
                raise ValueError(f"Completed run is missing artifacts: {name}")
            evaluations = {}
            for split in ("validation", "test"):
                report = directory / f"evaluation_{split}.json"
                if report.exists() and (directory / f"{split}_predictions.npz").exists():
                    info = json.loads(report.read_text())
                    evaluations[split] = info["checkpoint_sha256"] == sha256(directory / "best.pt")
            result[name] = dict(state="complete", config=stored["config"],
                                metric_version=stored.get("metric_version"), evaluations=evaluations)
        elif directory.exists() and any(directory.iterdir()):
            result[name] = dict(state="partial")
        else:
            result[name] = dict(state="missing")
    return result


@app.function(image=image, gpu="A10", cpu=4, memory=16_384, timeout=1800,
              volumes={str(DATA_MOUNT): data_volume, str(AUDIO_MOUNT): audio_volume,
                       str(RUNS_MOUNT): runs_volume})
def pilot_remote(run_name: str = "paper-pilot-v2", batch_size: int = 1, max_windows: int = 16,
                 dataset_filename: str = "vector-dataset.csv"):
    """Profile the three core configurations on real training tracks."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", run_name):
        raise ValueError("Invalid run name")
    if dataset_filename not in ("vector-dataset.csv", "vector-dataset-normalized.csv"):
        raise ValueError("Unsupported dataset filename")
    data_volume.reload()
    runs_volume.reload()
    report = Path(RUNS_MOUNT) / run_name / "pilot.json"
    command = [sys.executable, str(PROJECT_DIR / "scripts/pilot_paper_run.py"),
               "--data-dir", str(DATA_MOUNT / "dataset"),
               "--dataset-csv", str(DATA_MOUNT / "dataset" / dataset_filename),
               "--split-csv", str(DATA_MOUNT / "dataset" / "track_split_assignments.csv"),
               "--logmel-root", str(AUDIO_MOUNT / "logmel_songs"), "--device", "cuda",
               "--batch-size", str(batch_size), "--max-windows", str(max_windows),
               "--batches", "5", "--warmup", "2", "--output", str(report)]
    subprocess.run(command, cwd=PROJECT_DIR, check=True)
    runs_volume.commit()
    return json.loads(report.read_text())
