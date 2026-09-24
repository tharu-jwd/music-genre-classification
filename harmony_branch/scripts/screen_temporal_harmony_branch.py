"""Run one fixed, capped CPU screen of the reference temporal-chroma branch."""

from __future__ import annotations

import _bootstrap  # noqa: F401 - configures direct-script imports

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from harmony_branch.losses import masked_soft_target_cross_entropy
from harmony_branch.model import TemporalHarmonyBranch


SOURCE_SCHEMA = "harmony_screen_dataset_v1"
SCHEMA_VERSION = "harmony_branch_screen_v1"
SEED = 42
MAX_SONGS = 32
HARD_MAX_EPOCHS = 20
HARD_MAX_CPU_SECONDS = 300.0
HARD_MAX_CPU_THREADS = 8
EMBEDDING_DIM = 32
HIDDEN_DIM = 64
TEMPORAL_LAYERS = 2
DROPOUT = 0.1
LEARNING_RATE = 1e-3
PATIENCE = 3


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_dataset(directory: Path) -> tuple[Path, dict, list[dict]]:
    index_path = directory / "index.json"
    try:
        index = json.loads(index_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read screening dataset index: {error}") from error
    if not isinstance(index, dict) or index.get("schema_version") != SOURCE_SCHEMA:
        raise ValueError(f"screening dataset schema_version must be {SOURCE_SCHEMA!r}")
    if index.get("status") != "ready" or index.get("failure_count") != 0:
        raise ValueError("screening dataset must be complete and ready")
    if index.get("contains_genre_labels") is not False:
        raise ValueError("harmony branch screening must not consume genre labels")
    items = index.get("items")
    if not isinstance(items, list) or not 2 <= len(items) <= MAX_SONGS:
        raise ValueError(f"screening dataset must contain 2-{MAX_SONGS} songs")
    splits = {item.get("split") for item in items}
    if splits != {"train", "validation"}:
        raise ValueError("screening dataset needs non-empty train and validation splits")
    return index_path, index, items


def _load_song(item: dict) -> dict:
    feature_path = Path(str(item.get("feature_artifact", "")))
    target_path = Path(str(item.get("_target_path", "")))
    if (
        not feature_path.is_file()
        or _sha256_file(feature_path) != item.get("feature_artifact_sha256")
    ):
        raise ValueError(f"feature artifact changed for {item.get('song_id')}")
    if (
        not target_path.is_file()
        or _sha256_file(target_path) != item.get("target_artifact_sha256")
    ):
        raise ValueError(f"target artifact changed for {item.get('song_id')}")
    with np.load(feature_path, allow_pickle=False) as feature:
        encoded = feature["encoded_sequence"].astype(np.float32)
        sequence_mask = feature["sequence_mask"].astype(bool)
        window_index = feature["sequence_window_index"].astype(np.int64)
    with np.load(target_path, allow_pickle=False) as target:
        chroma = target["chroma_target"].astype(np.float32)
        chroma_valid = target["chroma_valid"].astype(bool)
        screen_mask = target["screen_token_mask"].astype(bool)
    tokens = len(sequence_mask)
    if (
        encoded.ndim != 2
        or encoded.shape[0] != tokens
        or window_index.shape != (tokens,)
        or chroma.shape != (tokens, 12)
        or chroma_valid.shape != (tokens,)
        or screen_mask.shape != (tokens,)
    ):
        raise ValueError(f"screening arrays have inconsistent shapes for {item.get('song_id')}")
    if np.any(chroma_valid & ~(screen_mask & sequence_mask)):
        raise ValueError("valid chroma exists outside the registered screen/sequence mask")
    windows = item.get("windows")
    tokens_per_window = item.get("tokens_per_window")
    if (
        not isinstance(windows, int)
        or not isinstance(tokens_per_window, int)
        or windows * tokens_per_window != tokens
    ):
        raise ValueError("screening item has an invalid window/token layout")
    return {
        "song_id": item["song_id"],
        "split": item["split"],
        "encoded": torch.from_numpy(encoded).unsqueeze(0),
        "sequence_mask": torch.from_numpy(sequence_mask).unsqueeze(0),
        "window_index": torch.from_numpy(window_index).unsqueeze(0),
        "chroma": torch.from_numpy(chroma).unsqueeze(0),
        "chroma_valid": torch.from_numpy(chroma_valid).unsqueeze(0),
        "windows": windows,
        "tokens_per_window": tokens_per_window,
    }


def _baseline_metrics(train: list[dict], validation: list[dict]) -> dict:
    train_targets = torch.cat([song["chroma"][song["chroma_valid"]] for song in train])
    if len(train_targets) == 0:
        raise ValueError("training split has no valid chroma targets")
    training_mean = train_targets.mean(dim=0)
    training_mean = training_mean / training_mean.sum().clamp_min(1e-12)
    uniform = torch.full((12,), 1 / 12, dtype=torch.float32)

    def evaluate(prediction: torch.Tensor) -> dict:
        targets = torch.cat([song["chroma"][song["chroma_valid"]] for song in validation])
        if len(targets) == 0:
            raise ValueError("validation split has no valid chroma targets")
        repeated = prediction.unsqueeze(0).expand_as(targets)
        cross_entropy = -(targets * repeated.clamp_min(1e-12).log()).sum(dim=-1).mean()
        cosine = F.cosine_similarity(repeated, targets, dim=-1).mean()
        return {
            "cross_entropy": cross_entropy.item(),
            "mean_cosine_similarity": cosine.item(),
        }

    return {
        "uniform": evaluate(uniform),
        "training_mean": evaluate(training_mean),
        "training_mean_distribution": training_mean.tolist(),
    }


@torch.no_grad()
def _evaluate(model: TemporalHarmonyBranch, songs: list[dict]) -> dict:
    model.eval()
    total_loss = 0.0
    total_cosine = 0.0
    total_tokens = 0
    for song in songs:
        output = model(
            song["encoded"],
            song["sequence_mask"],
            song["window_index"],
            windows=song["windows"],
            tokens_per_window=song["tokens_per_window"],
        )
        valid = song["chroma_valid"] & song["sequence_mask"]
        count = int(valid.sum())
        if count == 0:
            continue
        loss = masked_soft_target_cross_entropy(
            output.chroma_logits,
            song["chroma"],
            song["chroma_valid"],
            song["sequence_mask"],
        )
        probability = torch.softmax(output.chroma_logits[valid], dim=-1)
        cosine = F.cosine_similarity(probability, song["chroma"][valid], dim=-1).sum()
        total_loss += loss.item() * count
        total_cosine += cosine.item()
        total_tokens += count
    if total_tokens == 0:
        raise ValueError("evaluation has no valid target tokens")
    return {
        "cross_entropy": total_loss / total_tokens,
        "mean_cosine_similarity": total_cosine / total_tokens,
        "valid_tokens": total_tokens,
    }


def screen_branch(
    dataset_dir: Path,
    output_dir: Path,
    *,
    max_epochs: int = HARD_MAX_EPOCHS,
    max_cpu_seconds: float = HARD_MAX_CPU_SECONDS,
    cpu_threads: int = 4,
) -> dict:
    dataset_dir = Path(dataset_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError("output directory already exists; screening runs are immutable")
    if not 1 <= max_epochs <= HARD_MAX_EPOCHS:
        raise ValueError(f"max_epochs must be in [1, {HARD_MAX_EPOCHS}]")
    if not 0 < max_cpu_seconds <= HARD_MAX_CPU_SECONDS:
        raise ValueError(f"max_cpu_seconds must be in (0, {HARD_MAX_CPU_SECONDS:g}]")
    if not 1 <= cpu_threads <= HARD_MAX_CPU_THREADS:
        raise ValueError(f"cpu_threads must be in [1, {HARD_MAX_CPU_THREADS}]")
    index_path, index, raw_items = _load_dataset(dataset_dir)
    items = []
    for raw in raw_items:
        item = dict(raw)
        item["_target_path"] = str(dataset_dir / item["target_artifact"])
        items.append(_load_song(item))
    train = [item for item in items if item["split"] == "train"]
    validation = [item for item in items if item["split"] == "validation"]
    input_dims = {item["encoded"].shape[-1] for item in items}
    if len(input_dims) != 1:
        raise ValueError("cached encoder feature widths differ across songs")
    baselines = _baseline_metrics(train, validation)

    torch.manual_seed(SEED)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(cpu_threads)
    model = TemporalHarmonyBranch(
        input_dims.pop(),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        temporal_layers=TEMPORAL_LAYERS,
        chord_classes=None,
        dropout=DROPOUT,
    ).cpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    rng = np.random.default_rng(SEED)
    started = time.perf_counter()
    history = []
    best_loss = math.inf
    best_epoch = None
    best_state = None
    epochs_without_improvement = 0
    stop_reason = "max_epochs_reached"
    try:
        for epoch in range(1, max_epochs + 1):
            model.train()
            order = rng.permutation(len(train))
            epoch_loss = 0.0
            epoch_tokens = 0
            deadline = False
            for index_in_split in order:
                if time.perf_counter() - started >= max_cpu_seconds:
                    deadline = True
                    break
                song = train[int(index_in_split)]
                optimizer.zero_grad(set_to_none=True)
                output = model(
                    song["encoded"],
                    song["sequence_mask"],
                    song["window_index"],
                    windows=song["windows"],
                    tokens_per_window=song["tokens_per_window"],
                )
                loss = masked_soft_target_cross_entropy(
                    output.chroma_logits,
                    song["chroma"],
                    song["chroma_valid"],
                    song["sequence_mask"],
                )
                loss.backward()
                optimizer.step()
                count = int(song["chroma_valid"].sum())
                epoch_loss += loss.item() * count
                epoch_tokens += count
                if time.perf_counter() - started >= max_cpu_seconds:
                    deadline = True
                    break
            if deadline:
                stop_reason = "cpu_deadline_reached_between_songs"
                break
            validation_metrics = _evaluate(model, validation)
            history.append({
                "epoch": epoch,
                "train_cross_entropy": epoch_loss / max(epoch_tokens, 1),
                "validation": validation_metrics,
            })
            if validation_metrics["cross_entropy"] < best_loss - 1e-8:
                best_loss = validation_metrics["cross_entropy"]
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= PATIENCE:
                    stop_reason = "early_stopping"
                    break
        elapsed = time.perf_counter() - started
    finally:
        torch.set_num_threads(previous_threads)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        if best_state is None:
            result = {
                "schema_version": SCHEMA_VERSION,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "status": "failed",
                "automatic_selection": False,
                "failure_reason": "no_complete_validation_epoch",
                "stop_reason": stop_reason,
                "elapsed_cpu_wall_seconds": elapsed,
                "history": history,
            }
        else:
            model.load_state_dict(best_state)
            best_metrics = _evaluate(model, validation)
            checkpoint_path = temporary / "best.pt"
            torch.save({
                "model": best_state,
                "configuration": {
                    "seed": SEED,
                    "input_dim": model.input_dim,
                    "embedding_dim": EMBEDDING_DIM,
                    "hidden_dim": HIDDEN_DIM,
                    "temporal_layers": TEMPORAL_LAYERS,
                    "dropout": DROPOUT,
                    "learning_rate": LEARNING_RATE,
                    "patience": PATIENCE,
                },
                "source_dataset_sha256": _sha256_file(index_path),
                "best_epoch": best_epoch,
                "validation": best_metrics,
            }, checkpoint_path)
            result = {
                "schema_version": SCHEMA_VERSION,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "status": "ready",
                "purpose": "cpu_only_reference_temporal_chroma_branch_screen",
                "automatic_selection": False,
                "device": "cpu",
                "source_dataset": str(index_path),
                "source_dataset_sha256": _sha256_file(index_path),
                "cohort_sha256": index.get("cohort_sha256"),
                "configuration": {
                    "seed": SEED,
                    "embedding_dim": EMBEDDING_DIM,
                    "hidden_dim": HIDDEN_DIM,
                    "temporal_layers": TEMPORAL_LAYERS,
                    "dropout": DROPOUT,
                    "learning_rate": LEARNING_RATE,
                    "patience": PATIENCE,
                    "max_epochs": max_epochs,
                    "max_cpu_seconds": max_cpu_seconds,
                    "cpu_threads": cpu_threads,
                },
                "hard_limits": {
                    "max_songs": MAX_SONGS,
                    "max_epochs": HARD_MAX_EPOCHS,
                    "max_cpu_seconds": HARD_MAX_CPU_SECONDS,
                    "max_cpu_threads": HARD_MAX_CPU_THREADS,
                },
                "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                "baselines": baselines,
                "best_epoch": best_epoch,
                "best_validation": best_metrics,
                "checkpoint": "best.pt",
                "checkpoint_sha256": _sha256_file(checkpoint_path),
                "stop_reason": stop_reason,
                "elapsed_cpu_wall_seconds": elapsed,
                "history": history,
                "versions": {
                    "numpy": importlib.metadata.version("numpy"),
                    "torch": importlib.metadata.version("torch"),
                },
            }
        (temporary / "report.json").write_text(json.dumps(result, indent=2) + "\n")
        temporary.rename(output_dir)
        return result
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-epochs", type=int, default=HARD_MAX_EPOCHS)
    parser.add_argument("--max-cpu-seconds", type=float, default=HARD_MAX_CPU_SECONDS)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    result = screen_branch(
        args.dataset,
        args.output_dir,
        max_epochs=args.max_epochs,
        max_cpu_seconds=args.max_cpu_seconds,
        cpu_threads=args.cpu_threads,
    )
    print(json.dumps({
        "status": result["status"],
        "best_epoch": result.get("best_epoch"),
        "best_validation": result.get("best_validation"),
        "automatic_selection": result["automatic_selection"],
        "output": str(args.output_dir),
    }, indent=2))
    return 0 if result["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
