"""Portable prediction artifacts and provenance for the paper experiments."""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np


METRIC_VERSION = "sklearn_average_precision_masked_v2"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_fingerprints(cfg):
    root = Path(cfg.data_dir) if cfg.data_dir is not None else Path(__file__).resolve().parents[1] / "data"
    paths = {"split": Path(cfg.split_csv) if cfg.split_csv else root / "track_split_assignments.csv"}
    if cfg.dataset_csv:
        paths["dataset"] = Path(cfg.dataset_csv)
    else:
        paths.update({name: root / f"{name}.csv" for name in
                      ("logmel_metadata", "genres_df", *(f"{branch}_df" for branch in cfg.branches))})
    for name in ("logmel_config", "logmel_audit"):
        path = root / (name + (".json" if name == "logmel_config" else ".csv"))
        if path.exists():
            paths[name] = path
    return {name: {"path": str(path.resolve()), "sha256": sha256(path)}
            for name, path in paths.items()}


def code_fingerprints():
    root = Path(__file__).resolve().parents[1]
    paths = [root / "scripts" / name for name in ("train_joint.py", "train_cnn.py", "paper_artifacts.py")]
    for directory in ("shared_encoder", "concept_fusion", "rhythm_branch/src/rhythm_branch",
                      "timbre_branch/src/timbre_branch", "harmony_branch/src/harmony_branch"):
        paths.extend((root / directory).glob("*.py"))
    return {path.relative_to(root).as_posix(): sha256(path) for path in sorted(paths)}


def runtime_metadata(device):
    import pandas
    import sklearn
    import torch
    selected = torch.device(device)
    return dict(python=platform.python_version(), platform=platform.platform(),
                numpy=np.__version__, pandas=pandas.__version__, sklearn=sklearn.__version__,
                torch=torch.__version__, torch_cuda=torch.version.cuda, device=str(selected),
                gpu_name=torch.cuda.get_device_name(selected) if selected.type == "cuda" else None)


def export_predictions(path, probabilities, targets, track_ids, genre_names, concepts=None):
    ids, names = np.asarray(track_ids, dtype=str), np.asarray(genre_names, dtype=str)
    if len(set(ids.tolist())) != len(ids):
        raise ValueError("Prediction track IDs must be unique")
    values = dict(probabilities=probabilities, scores=probabilities, targets=targets,
                  track_ids=ids, song_ids=ids, genre_tags=names, label_names=names,
                  metric_version=np.asarray(METRIC_VERSION),
                  prediction_schema=np.asarray("paper_predictions_v2"))
    for name, group in (concepts or {}).items():
        for key, value in group.items():
            values[f"{name}_{key}"] = np.asarray(value)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **values)


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
