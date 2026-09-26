"""
Joint training script for the 6-genre concept-bottleneck DNN.

Data requirements (all in data/):
  logmel_metadata.csv          TRACK_ID -> .npy log-mel path  (mel_bins, T) each
  track_split_assignments.csv  TRACK_ID, split (train/validation/test)
  genres_df.csv                TRACK_ID + 6 genre columns
  instrument_df.csv            TRACK_ID + 41 instrument columns
  timbre_df.csv                TRACK_ID + 35 timbre descriptor columns
  rhythm_df.csv                TRACK_ID + 10 rhythm columns
  harmony_df.csv               TRACK_ID + chroma_*_mean cols (12 used)

Usage:
    python scripts/train_joint.py
    python scripts/train_joint.py --epochs 50 --batch-size 16 --lr 3e-4
    python scripts/train_joint.py --quick          # 3 epochs, 32-track split (smoke test)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
for p in (
    ROOT,
    ROOT / "rhythm_branch" / "src",
    ROOT / "timbre_branch" / "src",
    ROOT / "harmony_branch" / "src",
):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from concept_fusion.contract import (
    GENRE_TAGS,
    INSTRUMENT_TAGS,
    N_GENRE_TAGS,
    N_HARMONY_CHROMA,
    N_INSTRUMENT_TAGS,
    N_RHYTHM_CONCEPTS,
    N_TIMBRE_CONCEPTS,
    ConceptCounts, TIMBRE_FEATURES, RHYTHM_FEATURES,
)
from concept_fusion.harmony_adapter import from_temporal_harmony_branch
from concept_fusion.instrument_adapter import from_instrument_branch
from concept_fusion.joint_loss import SongHarmonyTargets, JointLossOrchestrator, LossWeights
from concept_fusion.model import ConceptBottleneckModel
from concept_fusion.rhythm_adapter import from_rhythm_branch
from concept_fusion.timbre_adapter import from_timbre_branch
from concept_fusion.types import BranchBundle
from harmony_branch.model import TemporalHarmonyBranch
from scripts.mtg_data_contract import NOTEBOOK_DATA_CONTRACT
from rhythm_branch.model import RhythmBranch
from rhythm_branch.preprocessing import RhythmStandardizer
from shared_encoder import SharedAudioEncoder
from timbre_branch.model import TimbreBranch
from timbre_branch.preprocessing import TimbreStandardizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHROMA_COLS: list[str] = [
    "chroma_c_mean", "chroma_csharp_mean", "chroma_d_mean", "chroma_dsharp_mean",
    "chroma_e_mean", "chroma_f_mean", "chroma_fsharp_mean", "chroma_g_mean",
    "chroma_gsharp_mean", "chroma_a_mean", "chroma_asharp_mean", "chroma_b_mean",
]
assert len(CHROMA_COLS) == N_HARMONY_CHROMA, "CHROMA_COLS must have 12 entries"

# Use the same windowing implementation as the preprocessing notebooks.
_data_contract = {"np": np, "Path": Path}
exec(NOTEBOOK_DATA_CONTRACT, _data_contract)
segment_logmel_with_metadata = _data_contract["segment_logmel_with_metadata"]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _norm_id(raw: str) -> str:
    """Normalise track ID to bare zero-padded 7-digit string."""
    s = str(raw).strip().lower().replace("track_", "")
    return f"{int(s):07d}"


class MultiTargetDataset(Dataset):
    """
    Loads log-mel .npy files on demand and returns all branch targets.

    Each .npy is (mel_bins, T_total) or (windows, mel_bins, frames).
    Segmented into bounded ordered windows, preserving valid frames and starts.
    """

    def __init__(
        self,
        track_ids: list[str],
        npy_paths: list[str],
        genre_targets: np.ndarray,       # (N, 6)  float32 binary
        instrument_targets: np.ndarray,  # (N, 41) float32 binary
        timbre_targets: np.ndarray,      # (N, 35) float32 standardized
        timbre_mask: np.ndarray,         # (N, 35) bool
        rhythm_targets: np.ndarray,      # (N, 10) float32 standardized
        rhythm_mask: np.ndarray,         # (N, 10) bool
        harmony_chroma: np.ndarray,      # (N, 12) float32 normalised chroma
        window_frames: int = 1366,
        max_windows: int = 12,
        mel_config: dict | None = None,
        durations: dict[str, float] | None = None,
    ) -> None:
        self.mel_config = mel_config or {}
        self.sample_rate = int(self.mel_config.get("sample_rate", 12000))
        self.hop_length = int(self.mel_config.get("hop_length", 256))
        self.durations = durations or {}
        self.window_frames = window_frames
        self.max_windows = max_windows
        self.track_ids = track_ids
        self.npy_paths = npy_paths
        self.genre = torch.as_tensor(genre_targets, dtype=torch.float32)
        self.instrument = torch.as_tensor(instrument_targets, dtype=torch.float32)
        self.timbre = torch.as_tensor(timbre_targets, dtype=torch.float32)
        self.timbre_mask = torch.as_tensor(timbre_mask, dtype=torch.bool)
        self.rhythm = torch.as_tensor(rhythm_targets, dtype=torch.float32)
        self.rhythm_mask = torch.as_tensor(rhythm_mask, dtype=torch.bool)
        self.harmony = torch.as_tensor(harmony_chroma, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.track_ids)

    def __getitem__(self, idx: int):
        mel = np.load(self.npy_paths[idx], mmap_mode="r")
        if mel.ndim == 3:
            required = {"sample_rate", "hop_length", "window_seconds", "n_mels", "center"}
            if not required.issubset(self.mel_config):
                raise ValueError("Stacked log-mels require data/logmel_config.json with "
                                 "sample_rate, hop_length, window_seconds, n_mels, and center")
            if mel.shape[1] != self.mel_config["n_mels"]:
                raise ValueError(f"Mel-band count differs from config: {mel.shape}")
            if self.track_ids[idx] not in self.durations:
                raise ValueError("Stacked log-mels require track duration in data/logmel_audit.csv")
            if not self.mel_config["center"]:
                raise ValueError("Stacked loader currently requires centered STFT metadata")
            window_seconds = float(self.mel_config["window_seconds"])
            duration = self.durations[self.track_ids[idx]]
            indices = np.arange(len(mel))
            if len(indices) > self.max_windows:
                indices = np.linspace(0, len(mel) - 1, self.max_windows, dtype=int)
            starts = (indices * window_seconds).astype(np.float32)
            seconds = np.clip(duration - starts, 0, window_seconds)
            valid = np.minimum(mel.shape[-1], np.floor(seconds * self.sample_rate / self.hop_length).astype(np.int64) + 1)
            valid[seconds <= 0] = 0
            keep = valid > 0
            windows = np.array(mel[indices[keep]], dtype=np.float32, copy=True)
            starts, valid = starts[keep], valid[keep]
            count = len(windows)
            if not count:
                raise ValueError("Track contains no valid audio windows")
            for window, frames in zip(windows, valid):
                window[:, frames:] = 0
        elif mel.ndim == 2:
            windows, mask, valid, starts = segment_logmel_with_metadata(
                mel, n_mels=int(self.mel_config.get("n_mels", 96)),
                n_frames=self.window_frames, max_windows=self.max_windows
            )
            # The shared window helper uses 12kHz/256; scale to this cache's geometry.
            starts *= (self.hop_length / self.sample_rate) / (256 / 12000)
            count = int(mask.sum())
        else:
            raise ValueError(f"Expected 2D mel or windows/mel/time stack, got {mel.shape}")
        mel_t = (torch.from_numpy(windows[:count]).unsqueeze(1),
                 torch.from_numpy(valid[:count]), torch.from_numpy(starts[:count]))
        return (
            mel_t,
            self.genre[idx],
            self.instrument[idx],
            self.timbre[idx],
            self.timbre_mask[idx],
            self.rhythm[idx],
            self.rhythm_mask[idx],
            self.harmony[idx],
            self.track_ids[idx],
        )


# ---------------------------------------------------------------------------
# Collation: pad variable-length mel within each batch
# ---------------------------------------------------------------------------

def collate_fn(batch):
    (mel_list, genre, instr, timbre, timbre_mask,
     rhythm, rhythm_mask, harmony, ids) = zip(*batch)

    B = len(batch)
    W_max = max(m[0].shape[0] for m in mel_list)
    mel_bins, frames = mel_list[0][0].shape[-2:]
    mel_pad = torch.zeros(B, W_max, 1, mel_bins, frames)
    valid_frames = torch.zeros(B, W_max, dtype=torch.long)
    window_start = torch.zeros(B, W_max)
    for i, (mel, valid, starts) in enumerate(mel_list):
        n = len(mel)
        mel_pad[i, :n] = mel
        valid_frames[i, :n] = valid
        window_start[i, :n] = starts
    window_mask = valid_frames > 0

    return (
        mel_pad,                               # (B, 1, 1, mel, T_max)
        window_mask,                           # (B, 1)
        valid_frames,                          # (B, 1)
        window_start,                          # (B, 1)
        torch.stack(genre),                    # (B, 6)
        torch.stack(instr),                    # (B, 41)
        torch.stack(timbre),                   # (B, 35)
        torch.stack(timbre_mask),              # (B, 35)
        torch.stack(rhythm),                   # (B, 10)
        torch.stack(rhythm_mask),              # (B, 10)
        torch.stack(harmony),                  # (B, 12)
        list(ids),
    )


# ---------------------------------------------------------------------------
# Instrument head (equivalent of the notebook v2 head)
# ---------------------------------------------------------------------------

class InstrumentHead(nn.Module):
    """pooled_song (B, 128) -> 41 probabilities + logits."""

    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.15))
        self.classifier = nn.Linear(128, N_INSTRUMENT_TAGS)

    def forward(self, pooled_song: Tensor) -> dict:
        h = self.hidden(pooled_song)
        logits = self.classifier(h)
        return {
            "concept_values": logits.sigmoid(),
            "logits": logits,
            "supervision_mask": torch.ones_like(logits),
            "fusion_mask": torch.ones(len(logits), 1, device=logits.device),
            "diagnostics": {"hidden": h.detach()},
        }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_csvs(data_dir: Path) -> dict[str, pd.DataFrame]:
    names = (
        "logmel_metadata", "track_split_assignments",
        "genres_df", "instrument_df", "timbre_df", "rhythm_df", "harmony_df",
    )
    out = {}
    for name in names:
        p = data_dir / f"{name}.csv"
        if not p.is_file():
            raise FileNotFoundError(p)
        out[name] = pd.read_csv(p, dtype={"TRACK_ID": str, "track_id": str})
    return out


def _norm_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()
    df[col] = df[col].map(_norm_id)
    return df


def resolve_logmel_path(raw: str, logmel_root: Path | None, data_dir: Path) -> str:
    normalized = str(raw).replace("\\", "/")
    if logmel_root is not None:
        marker = "logmel_songs/"
        if marker not in normalized:
            raise ValueError(f"Cannot relocate log-mel path without {marker}: {raw}")
        relative = normalized.split(marker, 1)[1]
        if ".." in Path(relative).parts:
            raise ValueError(f"Invalid relative log-mel path: {raw}")
        return str(logmel_root / relative)
    path = Path(raw)
    return str(path if path.is_absolute() else data_dir / path)


def build_datasets(
    data_dir: Path,
    *,
    timbre_std: TimbreStandardizer | None = None,
    rhythm_std: RhythmStandardizer | None = None,
    quick: bool = False,
    logmel_root: Path | None = None,
    window_frames: int = 1366,
    max_windows: int = 12,
) -> tuple[
    MultiTargetDataset, MultiTargetDataset, MultiTargetDataset,
    TimbreStandardizer, RhythmStandardizer,
]:
    """Load, align, standardize, and split all data sources."""
    dfs = _load_csvs(data_dir)
    root_config = data_dir / "logmel_root.txt"
    if logmel_root is None and root_config.is_file():
        logmel_root = Path(root_config.read_text(encoding="utf-8-sig").strip())

    config_path = data_dir / "logmel_config.json"
    mel_config = json.loads(config_path.read_text(encoding="utf-8-sig")) if config_path.is_file() else {}
    audit_path = data_dir / "logmel_audit.csv"
    durations = {}
    if audit_path.is_file():
        audit = _norm_col(pd.read_csv(audit_path), "TRACK_ID")
        durations = dict(zip(audit["TRACK_ID"], audit["track_duration_sec"]))

    logmel = _norm_col(dfs["logmel_metadata"], "TRACK_ID")
    splits = _norm_col(
        dfs["track_split_assignments"].rename(columns={"track_id": "TRACK_ID"}), "TRACK_ID"
    )
    genres = _norm_col(dfs["genres_df"], "TRACK_ID")
    instr  = _norm_col(dfs["instrument_df"], "TRACK_ID")
    timbre = _norm_col(dfs["timbre_df"], "TRACK_ID")
    rhythm = _norm_col(dfs["rhythm_df"], "TRACK_ID")
    harmony = _norm_col(dfs["harmony_df"], "TRACK_ID")

    timbre_feat = list(TIMBRE_FEATURES)
    rhythm_feat = list(RHYTHM_FEATURES)

    # Join everything; left-join targets so no audio row is dropped
    master = (
        logmel
        .merge(splits[["TRACK_ID", "split"]], on="TRACK_ID", how="inner")
        .merge(genres[["TRACK_ID"] + list(GENRE_TAGS)], on="TRACK_ID", how="inner")
        .merge(instr[["TRACK_ID"] + list(INSTRUMENT_TAGS)], on="TRACK_ID", how="left")
        .merge(timbre[["TRACK_ID"] + timbre_feat], on="TRACK_ID", how="left")
        .merge(rhythm[["TRACK_ID"] + rhythm_feat], on="TRACK_ID", how="left")
        .merge(harmony[["TRACK_ID"] + CHROMA_COLS], on="TRACK_ID", how="left")
    )

    # Fit standardizers on training rows only (leakage-safe)
    train_mask = master["split"] == "train"
    if timbre_std is None:
        t_vals = master.loc[train_mask, timbre_feat].to_numpy(dtype=np.float64)
        timbre_std = TimbreStandardizer().fit(t_vals, np.isfinite(t_vals))
    if rhythm_std is None:
        r_vals = master.loc[train_mask, rhythm_feat].to_numpy(dtype=np.float64)
        rhythm_std = RhythmStandardizer().fit(r_vals, np.isfinite(r_vals))

    def _make_ds(subset: pd.DataFrame) -> MultiTargetDataset:
        if quick:
            subset = subset.head(32)

        ids   = subset["TRACK_ID"].tolist()
        paths = [resolve_logmel_path(p, logmel_root, data_dir) for p in subset["logmel_path"]]

        g = subset[list(GENRE_TAGS)].to_numpy(dtype=np.float32)
        i = subset[list(INSTRUMENT_TAGS)].to_numpy(dtype=np.float32)

        t_raw = subset[timbre_feat].to_numpy(dtype=np.float64)
        t_msk = np.isfinite(t_raw)
        t_std = timbre_std.transform(np.where(t_msk, t_raw, 0.0))
        t_std[~t_msk] = 0.0

        r_raw = subset[rhythm_feat].to_numpy(dtype=np.float64)
        r_msk = np.isfinite(r_raw)
        r_std = rhythm_std.transform(np.where(r_msk, r_raw, 0.0))
        r_std[~r_msk] = 0.0

        # Normalise chroma to sum-1 distribution
        h_raw = subset[CHROMA_COLS].to_numpy(dtype=np.float32)
        sums = h_raw.sum(axis=1, keepdims=True)
        valid_h = np.isfinite(h_raw).all(axis=1) & (h_raw >= 0).all(axis=1) & (sums[:, 0] > 1e-8)
        h_norm = np.full_like(h_raw, np.nan)
        h_norm[valid_h] = h_raw[valid_h] / sums[valid_h]

        return MultiTargetDataset(ids, paths, g, i, t_std, t_msk, r_std, r_msk, h_norm, window_frames, max_windows, mel_config, durations)

    train_ds = _make_ds(master[master["split"] == "train"])
    val_ds   = _make_ds(master[master["split"] == "validation"])
    test_ds  = _make_ds(master[master["split"] == "test"])

    print(f"  train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")
    return train_ds, val_ds, test_ds, timbre_std, rhythm_std


# ---------------------------------------------------------------------------
# Bundle construction (shared by train and eval)
# ---------------------------------------------------------------------------

def _build_bundle(
    encoded,
    instrument_head: InstrumentHead,
    timbre_head: TimbreBranch,
    rhythm_head: RhythmBranch,
    harmony_head: TemporalHarmonyBranch,
    *,
    instr_tgt: Tensor,
    timbre_tgt: Tensor,
    timbre_msk: Tensor,
    rhythm_tgt: Tensor,
    rhythm_msk: Tensor,
    harmony_chroma: Tensor,
    device: torch.device,
) -> tuple[BranchBundle, dict]:
    # Instrument
    instr_out = instrument_head(encoded.pooled_song)
    instr_out["supervision_mask"] = torch.isfinite(instr_tgt).float()
    instr_br  = from_instrument_branch(instr_out)

    # Timbre
    timbre_raw  = timbre_head(encoded.pooled_song)
    timbre_fmsk = encoded.availability.unsqueeze(1).float()
    timbre_br   = from_timbre_branch(
        timbre_raw,
        supervision_mask=timbre_msk.float(),
        fusion_mask=timbre_fmsk,
    )

    # Rhythm
    rhythm_out = rhythm_head(
        encoded.encoded_sequence,
        encoded.sequence_mask,
        encoded.sequence_window_index,
    )
    rhythm_br = from_rhythm_branch(rhythm_out, supervision_mask=rhythm_msk.float())

    # Predict from audio; CSV song means are used only by the loss.
    windows = encoded.window_repr.shape[1]
    harmony_out = harmony_head(
        encoded.encoded_sequence, encoded.sequence_mask, encoded.sequence_window_index,
        windows=windows, tokens_per_window=encoded.encoded_sequence.shape[1] // windows,
    )
    harmony_br  = from_temporal_harmony_branch(harmony_out)

    bundle = BranchBundle(
        branches={
            "instrument": instr_br,
            "rhythm":     rhythm_br,
            "timbre":     timbre_br,
            "harmony":    harmony_br,
        },
        counts=ConceptCounts(),
    )
    bundle.validate()

    concept_targets = {
        "instrument": instr_tgt.float(),
        "rhythm":     rhythm_tgt.float(),
        "timbre":     timbre_tgt.float(),
        "harmony": SongHarmonyTargets(
            chroma=harmony_chroma,
            chroma_mask=torch.isfinite(harmony_chroma).all(dim=1),
        ),
    }
    return bundle, concept_targets


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    encoder: SharedAudioEncoder,
    instrument_head: InstrumentHead,
    timbre_head: TimbreBranch,
    rhythm_head: RhythmBranch,
    harmony_head: TemporalHarmonyBranch,
    fusion_model: ConceptBottleneckModel,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    for m in (encoder, instrument_head, timbre_head, rhythm_head, harmony_head, fusion_model):
        m.eval()

    genre_loss_fn = JointLossOrchestrator(
        weights=LossWeights(instrument=0, rhythm=0, timbre=0, harmony=0)
    ).to(device)

    all_probs: list[Tensor] = []
    all_targets: list[Tensor] = []
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        (mel, wm, vf, ws, genre_tgt, instr_tgt, timbre_tgt, timbre_msk,
         rhythm_tgt, rhythm_msk, harmony_chroma, _ids) = [
            b.to(device) if isinstance(b, Tensor) else b for b in batch
        ]
        encoded = encoder(mel, wm, vf, ws, sample_rate=loader.dataset.sample_rate, hop_length=loader.dataset.hop_length)
        bundle, concept_targets = _build_bundle(
            encoded, instrument_head, timbre_head, rhythm_head, harmony_head,
            instr_tgt=instr_tgt, timbre_tgt=timbre_tgt, timbre_msk=timbre_msk,
            rhythm_tgt=rhythm_tgt, rhythm_msk=rhythm_msk,
            harmony_chroma=harmony_chroma, device=device,
        )
        logits, _ = fusion_model.from_bundle(bundle, apply_dropout=False)
        br = genre_loss_fn(logits, genre_tgt, bundle, concept_targets)
        total_loss += br.total.item()
        n_batches += 1
        all_probs.append(logits.sigmoid().cpu())
        all_targets.append(genre_tgt.cpu())

    probs   = torch.cat(all_probs)     # (N, 6)
    targets = torch.cat(all_targets)   # (N, 6)

    # Tag-wise average precision → macro AP
    ap_list = []
    for t in range(targets.shape[1]):
        gt = targets[:, t]
        if gt.sum() == 0:
            continue
        p = probs[:, t]
        idx = p.argsort(descending=True)
        tp  = gt[idx].float().cumsum(0)
        pr  = tp / torch.arange(1, len(tp) + 1, dtype=torch.float32)
        ap_list.append((pr * gt[idx].float()).sum() / gt.sum().clamp(min=1))

    macro_ap = float(torch.stack(ap_list).mean()) if ap_list else 0.0
    return {"loss": total_loss / max(n_batches, 1), "macro_ap": macro_ap}


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(cfg: "TrainConfig") -> None:
    device = torch.device(cfg.device)
    print(f"Device: {device}")

    print("Loading data...")
    data_dir = ROOT / "data"
    train_ds, val_ds, test_ds, timbre_std, rhythm_std = build_datasets(
        data_dir, quick=cfg.quick, logmel_root=cfg.logmel_root,
        window_frames=cfg.window_frames, max_windows=cfg.max_windows
    )

    missing = [p for ds in (train_ds, val_ds, test_ds) for p in ds.npy_paths if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} log-mel files unavailable; first: {missing[0]}. Set --logmel-root to the logmel_songs folder.")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=cfg.num_workers,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=cfg.num_workers,
    )

    # Models
    encoder        = SharedAudioEncoder().to(device)
    instrument_head = InstrumentHead().to(device)
    timbre_head    = TimbreBranch().to(device)
    rhythm_head    = RhythmBranch().to(device)
    harmony_head = TemporalHarmonyBranch(128, chord_classes=None).to(device)
    fusion_model   = ConceptBottleneckModel(
        fusion="gated",
        dropout_p=0.15,
        harmony_embedding_dim=harmony_head.embedding_dim,
    ).to(device)

    all_params = (
        list(encoder.parameters())
        + list(instrument_head.parameters())
        + list(timbre_head.parameters())
        + list(rhythm_head.parameters())
        + list(harmony_head.parameters())
        + list(fusion_model.parameters())
    )

    optimizer = torch.optim.AdamW(all_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=cfg.lr * 0.05
    )
    loss_fn = JointLossOrchestrator(
        weights=LossWeights(
            genre=1.0,
            instrument=cfg.lambda_instrument,
            rhythm=cfg.lambda_rhythm,
            timbre=cfg.lambda_timbre,
            harmony=cfg.lambda_harmony,
        )
    ).to(device)

    out_dir = ROOT / cfg.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_ap = -1.0
    best_epoch = 0
    log: list[dict[str, Any]] = []

    for epoch in range(1, cfg.epochs + 1):
        for m in (encoder, instrument_head, timbre_head, rhythm_head, harmony_head, fusion_model):
            m.train()

        epoch_loss = 0.0
        n_batches  = 0
        term_sums: dict[str, float] = {}
        t0 = time.perf_counter()

        for batch in train_loader:
            (mel, wm, vf, ws, genre_tgt, instr_tgt, timbre_tgt, timbre_msk,
             rhythm_tgt, rhythm_msk, harmony_chroma, _ids) = [
                b.to(device) if isinstance(b, Tensor) else b for b in batch
            ]

            encoded = encoder(mel, wm, vf, ws, sample_rate=train_ds.sample_rate, hop_length=train_ds.hop_length)
            bundle, concept_targets = _build_bundle(
                encoded, instrument_head, timbre_head, rhythm_head, harmony_head,
                instr_tgt=instr_tgt, timbre_tgt=timbre_tgt, timbre_msk=timbre_msk,
                rhythm_tgt=rhythm_tgt, rhythm_msk=rhythm_msk,
                harmony_chroma=harmony_chroma, device=device,
            )

            optimizer.zero_grad(set_to_none=True)
            logits, _ = fusion_model.from_bundle(bundle, apply_dropout=True)
            br = loss_fn(logits, genre_tgt, bundle, concept_targets)
            if not torch.isfinite(br.total):
                raise RuntimeError(f"Non-finite training loss for tracks {_ids}")
            br.total.backward()
            nn.utils.clip_grad_norm_(all_params, max_norm=5.0)
            optimizer.step()

            epoch_loss += br.total.item()
            n_batches  += 1
            for name, value in br.terms.items():
                term_sums[name] = term_sums.get(name, 0.0) + value
            if n_batches == 1 or n_batches % 100 == 0:
                elapsed_batch = time.perf_counter() - t0
                print(f"Epoch {epoch} batch {n_batches}/{len(train_loader)} "
                      f"loss={epoch_loss/n_batches:.4f} elapsed={elapsed_batch:.0f}s", flush=True)

        scheduler.step()
        train_loss = epoch_loss / max(n_batches, 1)
        elapsed    = time.perf_counter() - t0

        print("Evaluating validation split...", flush=True)
        val_metrics = evaluate(
            encoder, instrument_head, timbre_head, rhythm_head, harmony_head,
            fusion_model, val_loader, device,
        )
        val_ap = val_metrics["macro_ap"]

        print(
            f"Epoch {epoch:3d}/{cfg.epochs}  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_metrics['loss']:.4f}  "
            f"val_macro_ap={val_ap:.4f}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}  "
            f"({elapsed:.1f}s)"
        )
        log.append({
            "epoch": epoch, "train_loss": train_loss,
            "val_loss": val_metrics["loss"], "val_macro_ap": val_ap,
            "train_terms": {k: v / n_batches for k, v in term_sums.items()},
            "train_seconds": elapsed,
        })

        if val_ap > best_val_ap:
            best_val_ap = val_ap
            best_epoch = epoch
            ckpt = {
                "epoch": epoch,
                "val_macro_ap": val_ap,
                "encoder":          encoder.state_dict(),
                "instrument_head":  instrument_head.state_dict(),
                "timbre_head":      timbre_head.state_dict(),
                "rhythm_head":      rhythm_head.state_dict(),
                "harmony_head":     harmony_head.state_dict(),
                "fusion_model":     fusion_model.state_dict(),
                "optimizer":        optimizer.state_dict(),
                "timbre_standardizer":  timbre_std.state_dict(),
                "rhythm_standardizer":  rhythm_std.state_dict(),
                "genre_tags":       list(GENRE_TAGS),
                "instrument_tags":  list(INSTRUMENT_TAGS),
                "n_genre_tags":     N_GENRE_TAGS,
                "n_instrument_tags": N_INSTRUMENT_TAGS,
                "n_timbre_concepts": N_TIMBRE_CONCEPTS,
                "n_rhythm_concepts": N_RHYTHM_CONCEPTS,
                "n_harmony_chroma":  N_HARMONY_CHROMA,
                "mel_config": train_ds.mel_config,
                "window_frames": cfg.window_frames,
                "max_windows": cfg.max_windows,
                "harmony_target_columns": CHROMA_COLS,
                "harmony_strategy":  "predicted_chroma_song_mean_supervision",
            }
            torch.save(ckpt, out_dir / "best.pt")
            print(f"  -> saved best checkpoint  (val_macro_ap={val_ap:.4f})")

    test_metrics = {"macro_ap": None, "loss": None}
    best = {"epoch": best_epoch, "val_macro_ap": best_val_ap}
    if cfg.evaluate_test:
        # Test evaluation using best checkpoint
        print("\nLoading best checkpoint for test evaluation...")
        best = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
        encoder.load_state_dict(best["encoder"])
        instrument_head.load_state_dict(best["instrument_head"])
        timbre_head.load_state_dict(best["timbre_head"])
        rhythm_head.load_state_dict(best["rhythm_head"])
        harmony_head.load_state_dict(best["harmony_head"])
        fusion_model.load_state_dict(best["fusion_model"])

        test_metrics = evaluate(
            encoder, instrument_head, timbre_head, rhythm_head, harmony_head,
            fusion_model, test_loader, device,
        )
        print(
            f"\nTest: macro_ap={test_metrics['macro_ap']:.4f}  "
            f"loss={test_metrics['loss']:.4f}"
        )

    summary: dict[str, Any] = {
        "mel_config": train_ds.mel_config,
        "test_evaluated": cfg.evaluate_test,
        "best_epoch":    int(best["epoch"]),
        "val_macro_ap":  float(best["val_macro_ap"]),
        "test_macro_ap": test_metrics["macro_ap"],
        "test_loss":     test_metrics["loss"],
        "genre_tags":    list(GENRE_TAGS),
        "n_train":       len(train_ds),
        "n_val":         len(val_ds),
        "n_test":        len(test_ds),
        "log":           log,
    }
    out_path = out_dir / "results.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Results written to {out_path}")


# ---------------------------------------------------------------------------
# Config & entry point
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    epochs:             int   = 30
    batch_size:         int   = 1
    lr:                 float = 3e-4
    weight_decay:       float = 1e-4
    lambda_instrument:  float = 1.0
    lambda_rhythm:      float = 1.0
    lambda_timbre:      float = 1.0
    lambda_harmony:     float = 0.5
    device:             str   = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir:            str   = "results/joint"
    num_workers:        int   = 0
    quick:              bool  = False
    evaluate_test:      bool  = True
    logmel_root:        Path | None = None
    window_frames:     int = 1366
    max_windows:       int = 12


def main() -> None:
    p = argparse.ArgumentParser(description="Joint concept-bottleneck training")
    p.add_argument("--epochs",            type=int,   default=30)
    p.add_argument("--batch-size",        type=int,   default=1)
    p.add_argument("--lr",                type=float, default=3e-4)
    p.add_argument("--weight-decay",      type=float, default=1e-4)
    p.add_argument("--lambda-instrument", type=float, default=1.0)
    p.add_argument("--lambda-rhythm",     type=float, default=1.0)
    p.add_argument("--lambda-timbre",     type=float, default=1.0)
    p.add_argument("--lambda-harmony",    type=float, default=0.5)
    p.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", default="results/joint")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--quick", action="store_true",
                   help="32-track subsets, 3 epochs — smoke test only")
    p.add_argument("--logmel-root", type=Path, help="Local logmel_songs directory; replaces the Colab prefix")
    p.add_argument("--window-frames", type=int, default=1366)
    p.add_argument("--max-windows", type=int, default=12)
    p.add_argument("--skip-test", action="store_true", help="Reserve the test split for final evaluation")
    args = p.parse_args()
    if args.window_frames < 1 or args.max_windows < 1:
        p.error("window-frames and max-windows must be positive")

    cfg = TrainConfig(
        epochs            = 3 if args.quick else args.epochs,
        batch_size        = args.batch_size,
        lr                = args.lr,
        weight_decay      = args.weight_decay,
        lambda_instrument = args.lambda_instrument,
        lambda_rhythm     = args.lambda_rhythm,
        lambda_timbre     = args.lambda_timbre,
        lambda_harmony    = args.lambda_harmony,
        device            = args.device,
        out_dir           = args.out_dir,
        num_workers       = args.num_workers,
        quick             = args.quick,
        evaluate_test     = not args.skip_test,
        logmel_root       = args.logmel_root,
        window_frames     = args.window_frames,
        max_windows       = args.max_windows,
    )

    print("=" * 60)
    print("Joint concept-bottleneck training")
    print(f"  Genres     : {N_GENRE_TAGS} tags -> {list(GENRE_TAGS)}")
    print(f"  Instrument : {N_INSTRUMENT_TAGS} tags")
    print(f"  Timbre     : {N_TIMBRE_CONCEPTS} descriptors")
    print(f"  Rhythm     : {N_RHYTHM_CONCEPTS} AcousticBrainz fields")
    print(f"  Harmony    : {N_HARMONY_CHROMA} predicted chroma bins (CSV song-mean supervision)")
    print(f"  Epochs     : {cfg.epochs}   Batch: {cfg.batch_size}   LR: {cfg.lr}")
    print(f"  Device     : {cfg.device}")
    print("=" * 60)

    train(cfg)


if __name__ == "__main__":
    main()
