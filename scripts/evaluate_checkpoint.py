"""Evaluate a saved paper checkpoint without fitting scalers or training weights."""
from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts import train_joint as j
from scripts import train_cnn as c
from scripts.paper_artifacts import METRIC_VERSION, dataset_fingerprints, code_fingerprints, runtime_metadata, sha256, write_json


def restore(checkpoint, *, device="cpu", data_dir=None, dataset_csv=None,
            split_csv=None, logmel_root=None, batch_size=None, num_workers=None):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("metric_version") != METRIC_VERSION:
        raise ValueError("Checkpoint used an older metric definition; paper runs must be retrained")
    if payload.get("code_fingerprints") != code_fingerprints():
        raise ValueError("Model/evaluation source differs from training; use the frozen source version")
    if payload["genre_tags"] != list(j.GENRE_TAGS):
        raise ValueError("Checkpoint genre vocabulary differs")
    allowed = {field.name for field in fields(j.TrainConfig)}
    cfg = j.TrainConfig(**{key: value for key, value in payload["config"].items() if key in allowed})
    for key in ("data_dir", "dataset_csv", "split_csv", "logmel_root"):
        value = locals()[key]
        setattr(cfg, key, Path(value) if value is not None else
                Path(getattr(cfg, key)) if getattr(cfg, key) is not None else None)
    cfg.branches = j.normalize_branches(payload["branches"])
    cfg.device = device
    if batch_size is not None:
        cfg.batch_size = batch_size
    if num_workers is not None:
        cfg.num_workers = num_workers
    if cfg.batch_size < 1 or cfg.num_workers < 0:
        raise ValueError("batch size must be positive and workers nonnegative")
    recorded = payload.get("dataset_fingerprints")
    if not recorded:
        raise ValueError("Checkpoint has no dataset fingerprints")
    current = dataset_fingerprints(cfg)
    if set(recorded) != set(current) or any(recorded[key]["sha256"] != current[key]["sha256"] for key in recorded):
        raise ValueError("Evaluation dataset or preprocessing metadata differs from training")
    standardizers = {}
    classes = {"timbre": j.TimbreStandardizer, "rhythm": j.RhythmStandardizer,
               "harmony": j.HarmonyStandardizer}
    for name, cls in classes.items():
        if name in cfg.branches:
            state = payload.get(f"{name}_standardizer")
            if state is None:
                raise ValueError(f"Checkpoint is missing the {name} training standardizer")
            standardizers[f"{name}_std"] = cls.from_state_dict(state)
    datasets = j.build_datasets(
        cfg.data_dir or j.ROOT / "data", dataset_csv=cfg.dataset_csv, split_csv=cfg.split_csv,
        logmel_root=cfg.logmel_root, quick=cfg.quick, window_frames=cfg.window_frames,
        max_windows=cfg.max_windows, branches=cfg.branches, **standardizers,
    )[:3]
    for name, dataset in zip(("train", "validation", "test"), datasets):
        if dataset.track_ids != payload["split_track_ids"][name]:
            raise ValueError(f"{name} track IDs/order differ from checkpoint")
    if datasets[0].mel_config != payload["mel_config"]:
        raise ValueError("Audio geometry differs from checkpoint")
    if not cfg.branches:
        model = c.GenreCNN().to(device)
        model.load_state_dict(payload["model_state"])
        modules = {"cnn": model}
    else:
        if payload.get("architecture_version") != "common_concept_branches_v2":
            raise ValueError("Unsupported concept architecture")
        if payload["genre_tags"] != list(j.GENRE_TAGS):
            raise ValueError("Checkpoint genre vocabulary differs")
        configs = payload["branch_configs"]
        factories = {
            "instrument": j.InstrumentHead,
            "timbre": lambda: j.TimbreBranch(j.TimbreBranchConfig(**configs["timbre"])),
            "rhythm": lambda: j.RhythmBranch(j.RhythmBranchConfig(**configs["rhythm"])),
            "harmony": lambda: j.TemporalHarmonyBranch(
                128, embedding_dim=64, descriptor_dim=j.N_HARMONY_DESCRIPTORS),
        }
        modules = {"encoder": j.SharedAudioEncoder().to(device)}
        for name, factory in factories.items():
            modules[f"{name}_head"] = factory().to(device) if name in cfg.branches else None
        if modules["harmony_head"] is not None:
            modules["harmony_head"].chroma_head.requires_grad_(False)
        modules["fusion_model"] = (
            j.NativeConceptFusionModel(cfg.branches, predictor=cfg.genre_predictor,
                hidden_dim=cfg.genre_hidden_dim, hidden_dropout=cfg.genre_dropout,
                dropout_p=cfg.concept_dropout) if cfg.fusion == "native_concat" else
            j.ConceptBottleneckModel(fusion="gated", harmony_embedding_dim=64)
        ).to(device)
        if cfg.fusion == "gated":
            assembler = modules["fusion_model"].assembler
            for name, attribute in {"instrument": "instrument_projection", "rhythm": "rhythm_projection",
                                    "timbre": "timbre_projection", "harmony": "harmony_chroma_projection"}.items():
                getattr(assembler, attribute).requires_grad_(name in cfg.branches)
            assembler.instrument_hidden_projection.requires_grad_(False)
            assembler.harmony_projection.requires_grad_(False)
        for name, module in modules.items():
            if module is not None:
                module.load_state_dict(payload[name])
    count = sum(parameter.numel() for module in modules.values() if module is not None
                for parameter in module.parameters() if parameter.requires_grad)
    if count != payload["parameter_count"]:
        raise ValueError("Restored trainable parameter count differs from checkpoint")
    return payload, cfg, datasets, modules


def evaluate_checkpoint(checkpoint, *, split="test", out_dir=None, overwrite=False, **overrides):
    if split not in ("validation", "test"):
        raise ValueError("split must be validation or test")
    checkpoint = Path(checkpoint)
    destination = Path(out_dir) if out_dir else checkpoint.parent
    prediction_path = destination / f"{split}_predictions.npz"
    report_path = destination / f"evaluation_{split}.json"
    if not overwrite and (prediction_path.exists() or report_path.exists()):
        raise FileExistsError("Evaluation outputs exist; use a new directory or --overwrite")
    payload, cfg, datasets, modules = restore(checkpoint, **overrides)
    dataset = datasets[1 if split == "validation" else 2]
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers, collate_fn=j.collate_fn)
    device = torch.device(cfg.device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    if "cnn" in modules:
        metrics = c.evaluate(modules["cnn"], loader, device, prediction_path)
    else:
        metrics = j.evaluate(loader=loader, device=device, prediction_path=prediction_path, **modules)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    # Recover descriptor units using saved training statistics, without refitting.
    with np.load(prediction_path, allow_pickle=False) as artifact:
        exported = {key: artifact[key] for key in artifact.files}
    for name in ("rhythm", "timbre", "harmony"):
        state = payload.get(f"{name}_standardizer")
        if state is not None:
            mean, scale = np.asarray(state["mean"]), np.asarray(state["scale"])
            exported[f"{name}_predictions_raw"] = exported[f"{name}_predictions"] * scale + mean
            raw_targets = exported[f"{name}_targets"] * scale + mean
            exported[f"{name}_targets_raw"] = np.where(exported[f"{name}_mask"], raw_targets, np.nan)
    np.savez_compressed(prediction_path, **exported)
    report = dict(schema_version="paper_evaluation_v2", metric_version=METRIC_VERSION,
                  checkpoint_sha256=sha256(checkpoint), checkpoint=str(checkpoint.resolve()),
                  prediction_sha256=sha256(prediction_path),
                  split=split, seed=payload["seed"], branches=payload["branches"],
                  environment=runtime_metadata(device), batch_size=cfg.batch_size,
                  parameter_count=payload["parameter_count"], metrics=metrics,
                  evaluation_seconds=elapsed, seconds_per_track=elapsed / len(dataset),
                  peak_gpu_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None)
    write_json(report_path, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    for key in ("data-dir", "dataset-csv", "split-csv", "logmel-root"):
        parser.add_argument(f"--{key}", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = vars(parser.parse_args())
    checkpoint = args.pop("checkpoint")
    print(evaluate_checkpoint(checkpoint, **args))


if __name__ == "__main__":
    main()
