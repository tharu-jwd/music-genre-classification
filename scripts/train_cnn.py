"""Direct CNN baseline using the joint trainer's exact audio and split protocol.

Run via: python scripts/train_joint.py --model cnn [joint data/training options]
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from scripts import train_joint as j


class GenreCNN(nn.Module):
    """Shared three-block audio CNN -> masked song pooling -> MLP -> six logits."""

    def __init__(self):
        super().__init__()
        self.encoder = j.SharedAudioEncoder()
        self.head = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(128, j.N_GENRE_TAGS),
        )

    def forward(self, mel, mask, valid_frames, starts, *, sample_rate, hop_length):
        encoded = self.encoder(mel, mask, valid_frames, starts,
                               sample_rate=sample_rate, hop_length=hop_length)
        return self.head(encoded.pooled_song)


def forward_batch(model, batch, dataset, device):
    # Only audio and its geometry enter the model; auxiliary targets are ignored.
    inputs = [value.to(device) for value in batch[:4]]
    return model(*inputs, sample_rate=dataset.sample_rate,
                 hop_length=dataset.hop_length), batch[4].to(device)


@torch.no_grad()
def evaluate(model, loader, device, prediction_path=None):
    model.eval()
    probabilities, targets, ids, losses = [], [], [], []
    for batch in loader:
        logits, truth = forward_batch(model, batch, loader.dataset, device)
        losses.append(nn.functional.binary_cross_entropy_with_logits(logits, truth).item())
        probabilities.append(logits.sigmoid().cpu())
        targets.append(truth.cpu())
        ids.extend(batch[-1])
    probs, truth = torch.cat(probabilities), torch.cat(targets)
    per_genre = {
        tag: j.macro_average_precision(probs[:, i:i+1], truth[:, i:i+1])
        if truth[:, i].sum() > 0 else None
        for i, tag in enumerate(j.GENRE_TAGS)
    }
    if prediction_path is not None:
        np.savez_compressed(prediction_path, probabilities=probs.numpy(),
                            targets=truth.numpy(), track_ids=np.asarray(ids),
                            genre_tags=np.asarray(j.GENRE_TAGS))
    return {"loss": sum(losses) / len(losses),
            "macro_ap": j.macro_average_precision(probs, truth),
            "per_genre_ap": per_genre}


def train(cfg: j.TrainConfig):
    if cfg.epochs < 1 or cfg.batch_size < 1 or cfg.lr <= 0 or cfg.num_workers < 0:
        raise ValueError("epochs, batch size and learning rate must be positive; workers nonnegative")
    device = torch.device(cfg.device)
    datasets = j.build_datasets(
        cfg.data_dir or j.ROOT / "data", dataset_csv=cfg.dataset_csv,
        split_csv=cfg.split_csv, require_harmony_targets=cfg.require_harmony_targets,
        quick=cfg.quick, logmel_root=cfg.logmel_root,
        window_frames=cfg.window_frames, max_windows=cfg.max_windows,
    )[:3]
    for ds in datasets:
        if not len(ds):
            raise ValueError("All three splits must contain tracks")
        for path in ds.npy_paths:
            if not Path(path).is_file():
                raise FileNotFoundError(path)
    loaders = [DataLoader(ds, batch_size=cfg.batch_size, shuffle=(i == 0),
                          collate_fn=j.collate_fn, num_workers=cfg.num_workers,
                          pin_memory=(device.type == "cuda" and i == 0))
               for i, ds in enumerate(datasets)]
    model = GenreCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=cfg.lr * 0.05)
    out = j.ROOT / cfg.out_dir
    out.mkdir(parents=True, exist_ok=True)
    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in asdict(cfg).items()}
    metadata = {
        "model": "direct_cnn", "config": config,
        "genre_tags": list(j.GENRE_TAGS), "mel_config": datasets[0].mel_config,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "split_track_ids": {name: ds.track_ids for name, ds in
                            zip(("train", "validation", "test"), datasets)},
        "n_train": len(datasets[0]), "n_val": len(datasets[1]), "n_test": len(datasets[2]),
    }
    print(f"Direct CNN: {metadata['parameter_count']} parameters; genres={list(j.GENRE_TAGS)}", flush=True)
    best_ap, best_epoch, history = -1., 0, []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        losses = []
        for batch in loaders[0]:
            logits, truth = forward_batch(model, batch, datasets[0], device)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, truth)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss for {batch[-1]}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()
        val = evaluate(model, loaders[1], device)
        row = {"epoch": epoch, "train_loss": sum(losses) / len(losses),
               "val_loss": val["loss"], "val_macro_ap": val["macro_ap"]}
        history.append(row)
        print(json.dumps(row), flush=True)
        if val["macro_ap"] > best_ap:
            best_ap, best_epoch = val["macro_ap"], epoch
            torch.save({**metadata, "epoch": epoch, "val_macro_ap": best_ap,
                        "model_state": model.state_dict(), "optimizer": optimizer.state_dict()},
                       out / "best.pt")
    best = torch.load(out / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model_state"])
    val = evaluate(model, loaders[1], device, out / "validation_predictions.npz")
    test = evaluate(model, loaders[2], device, out / "test_predictions.npz") if cfg.evaluate_test else {}
    results = {**metadata, "best_epoch": best_epoch, "val_macro_ap": best_ap,
               "val_per_genre_ap": val["per_genre_ap"], "test_evaluated": cfg.evaluate_test,
               "test_macro_ap": test.get("macro_ap"), "test_loss": test.get("loss"),
               "test_per_genre_ap": test.get("per_genre_ap"), "log": history}
    (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Test: {test}\nResults written to {out / 'results.json'}", flush=True)
