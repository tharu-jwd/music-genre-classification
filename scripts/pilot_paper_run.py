"""Bounded real-cache pilot: checks geometry and profiles training/inference without saving model weights."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from scripts import train_joint as j
from scripts import train_cnn as c
from scripts.paper_artifacts import dataset_fingerprints, write_json


def pilot(*, data_dir, dataset_csv, split_csv=None, logmel_root=None, device="cuda",
          batch_size=1, max_windows=16, window_frames=1366, batches=5, warmup=2,
          num_workers=0, output=None):
    if min(batch_size, max_windows, window_frames, batches) < 1 or warmup < 1 or num_workers < 0:
        raise ValueError("Pilot dimensions/counts must be positive; workers nonnegative")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU available; run this pilot on Modal or explicitly use --device cpu")
    cfg = j.TrainConfig(data_dir=Path(data_dir), dataset_csv=Path(dataset_csv),
                       split_csv=Path(split_csv) if split_csv else None,
                       logmel_root=Path(logmel_root) if logmel_root else None,
                       batch_size=batch_size, max_windows=max_windows, window_frames=window_frames)
    datasets = j.build_datasets(cfg.data_dir, dataset_csv=cfg.dataset_csv, split_csv=cfg.split_csv,
        logmel_root=cfg.logmel_root, max_windows=max_windows, window_frames=window_frames)[:3]
    if any(not len(dataset) for dataset in datasets):
        raise ValueError("Pilot requires nonempty train, validation and test partitions")
    all_ids = [sid for dataset in datasets for sid in dataset.track_ids]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("Track splits overlap")
    if output is not None and Path(output).exists():
        raise FileExistsError(output)
    train = datasets[0]
    # Seeded, repeatable selection. Never read test audio or score test labels.
    j.seed_run(17)
    indices = np.random.default_rng(17).permutation(len(train))[:min(len(train), batch_size * (batches + warmup))]
    loader = DataLoader(Subset(train, indices.tolist()), batch_size=batch_size,
                        shuffle=False, num_workers=num_workers, collate_fn=j.collate_fn)
    device = torch.device(device)
    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    geometries = []
    for idx in indices[:min(len(indices), 8)]:
        example = train[int(idx)]
        mel, valid, starts = example[0]
        seconds = valid.numpy() * train.hop_length / train.sample_rate
        end = starts.numpy() + seconds
        # Centered STFT permits one frame beyond a nominal window boundary.
        if np.max(end) > 240 + 2 * train.hop_length / train.sample_rate:
            raise ValueError("Audio cache exceeds the declared first-four-minute scope")
        geometries.append(dict(track_id=train.track_ids[int(idx)], shape=list(mel.shape),
                               valid_seconds=float(seconds.sum()), end_seconds=float(end.max())))
    report = dict(schema_version="paper_pilot_v2", device=str(device),
                  batch_size=batch_size, warmup_batches=warmup, requested_measured_batches=batches,
                  num_workers=num_workers,
                  gpu_name=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                  torch_version=torch.__version__, sample_rate=train.sample_rate, hop_length=train.hop_length,
                  window_frames=window_frames, max_windows=max_windows,
                  split_sizes={name: len(ds) for name, ds in zip(("train", "validation", "test"), datasets)},
                  dataset_fingerprints=dataset_fingerprints(cfg), sample_geometry=geometries,
                  note="Sampled training tracks; extrapolated times exclude full validation, startup and failures.",
                  configurations={})
    for name, weight in (("cnn", 0), ("full", 1), ("no-supervision", 0)):
        j.seed_run(17)
        if name == "cnn":
            model = c.GenreCNN().to(device)
            modules = [model]
        else:
            encoder = j.SharedAudioEncoder().to(device)
            heads = [j.InstrumentHead().to(device), j.TimbreBranch().to(device), j.RhythmBranch().to(device),
                     j.TemporalHarmonyBranch(128, embedding_dim=64, descriptor_dim=12).to(device)]
            heads[-1].chroma_head.requires_grad_(False)
            fusion = j.NativeConceptFusionModel().to(device)
            modules = [encoder, *heads, fusion]
        for module in modules:
            module.train()
        params = [parameter for module in modules for parameter in module.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        loss_fn = j.JointLossOrchestrator(weights=j.LossWeights(
            instrument=weight, rhythm=weight, timbre=weight, harmony=weight)).to(device)
        durations, inference_times, tracks_seen = [], [], 0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        def forward(batch):
            if name == "cnn":
                logits, targets = c.forward_batch(model, batch, train, device)
                return logits, torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
            moved = [value.to(device) if isinstance(value, torch.Tensor) else value for value in batch]
            mel, wm, vf, ws, genre, instrument, timbre, tm, rhythm, rm, harmony, hm, _ = moved
            encoded = encoder(mel, wm, vf, ws, sample_rate=train.sample_rate, hop_length=train.hop_length)
            bundle, targets = j._build_bundle(encoded, *heads, instr_tgt=instrument, timbre_tgt=timbre,
                timbre_msk=tm, rhythm_tgt=rhythm, rhythm_msk=rm, harmony_tgt=harmony,
                harmony_msk=hm, device=device)
            logits, _ = fusion.from_bundle(bundle)
            return logits, loss_fn(logits, genre, bundle, targets).total
        iterator = iter(loader)
        for step in range(warmup + batches):
            synchronize()
            start = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            optimizer.zero_grad(set_to_none=True)
            logits, loss = forward(batch)
            if not torch.isfinite(loss) or not torch.isfinite(logits).all():
                raise RuntimeError(f"Non-finite pilot output: {name}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5)
            optimizer.step()
            synchronize()
            duration = time.perf_counter() - start
            if step >= warmup:
                durations.append(duration)
                tracks_seen += len(batch[-1])
        peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        for module in modules:
            module.eval()
        with torch.no_grad():
            for batch in loader:
                synchronize()
                start = time.perf_counter()
                forward(batch)
                synchronize()
                inference_times.append((time.perf_counter() - start, len(batch[-1])))
                if len(inference_times) >= batches:
                    break
        seconds_per_track = sum(durations) / tracks_seen
        report["configurations"][name] = dict(parameter_count=sum(parameter.numel() for parameter in params),
            measured_batches=len(durations), training_seconds_per_track=seconds_per_track,
            median_batch_seconds=float(np.median(durations)), max_batch_seconds=max(durations),
            estimated_training_epoch_seconds=seconds_per_track * len(train),
            inference_seconds_per_track=sum(value[0] for value in inference_times) / sum(value[1] for value in inference_times),
            peak_gpu_bytes=peak)
        print(f"Pilot {name}: {report['configurations'][name]}", flush=True)
        del optimizer, modules, params, loss_fn, logits, loss
        if name == "cnn":
            del model
        else:
            del encoder, heads, fusion
        if device.type == "cuda":
            torch.cuda.empty_cache()
    timings = report["configurations"]
    # Planning estimates, not a benchmark of the twelve ablation runs.
    core = sum(timings[name]["estimated_training_epoch_seconds"] for name in timings)
    full = timings["full"]["estimated_training_epoch_seconds"]
    report["planning"] = dict(epochs=cfg.epochs,
        estimated_core_9_training_gpu_hours=3 * cfg.epochs * core / 3600,
        estimated_all_21_training_gpu_hours=3 * cfg.epochs * (core + 4 * full) / 3600,
        assumptions="Each removal is approximated by full-model training time. Excludes validation, startup and retries; sampled tracks may not be representative.")
    if output:
        write_json(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--dataset-csv", type=Path, required=True)
    parser.add_argument("--split-csv", type=Path)
    parser.add_argument("--logmel-root", type=Path)
    parser.add_argument("--device", default="cuda")
    for key, default in (("batch-size", 1), ("max-windows", 16), ("window-frames", 1366),
                         ("batches", 5), ("warmup", 2), ("num-workers", 0)):
        parser.add_argument("--" + key, type=int, default=default)
    parser.add_argument("--output", type=Path, required=True)
    print(pilot(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
