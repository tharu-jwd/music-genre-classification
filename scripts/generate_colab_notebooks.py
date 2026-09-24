"""Generate Colab + Google Drive notebooks under notebooks/colab/."""

from __future__ import annotations

import json
from pathlib import Path

from gpu_run_contract import NOTEBOOK_GPU_RUN_CONTRACT
from mtg_data_contract import NOTEBOOK_DATA_CONTRACT

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "colab"
OUT.mkdir(parents=True, exist_ok=True)


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(source)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _lines(source),
    }


def _lines(text: str) -> list[str]:
    text = text.strip("\n")
    if not text.endswith("\n"):
        text += "\n"
    return text.splitlines(keepends=True)


def nb(cells: list[dict]) -> dict:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "cells": cells,
    }


# Shared hosted workflow only. Branch training/eval lives in packages, not 02-09.
KEEP_NOTEBOOKS = {
    "00_download_to_drive.ipynb",
    "01_preprocessing.ipynb",
    "04_rhythm_targets.ipynb",
}


def write(name: str, cells: list[dict]) -> None:
    if name not in KEEP_NOTEBOOKS:
        print(f"skipped retired notebook {name}")
        return
    path = OUT / name
    path.write_text(json.dumps(nb(cells), indent=1), encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)}")


COLAB_SETUP = """\
## Colab + Drive (every notebook)

1. Open in **Google Colab**.
2. Run **Mount Drive** and click **Allow**.
3. Shared folder: `/content/drive/MyDrive/MTG_Instrument`
4. GPU **Off** for 00, 01, and 04. Branch training uses package scripts, not notebooks 02–09.
5. Do **not** re-download mels after notebook 00.
"""

MOUNT = r'''
from pathlib import Path
import os

DRIVE_ROOT = Path("/content/drive/MyDrive/MTG_Instrument")

if not Path("/content/drive/MyDrive").exists():
    from google.colab import drive
    drive.mount("/content/drive")
else:
    print("Drive already mounted")

for sub in ["dataset/logmel_songs", "annotations", "features", "checkpoints", "results/baselines", "results/proposed"]:
    (DRIVE_ROOT / sub).mkdir(parents=True, exist_ok=True)

os.environ["MTG_ROOT"] = str(DRIVE_ROOT)
print("Drive ready:", DRIVE_ROOT)
'''

PATHS = r'''
from pathlib import Path
import os, json, random, re, shutil, socket, time, urllib.request
import numpy as np
import pandas as pd

DRIVE_ROOT = Path(os.environ.get("MTG_ROOT", "/content/drive/MyDrive/MTG_Instrument"))
ROOT = DRIVE_ROOT
MEL_DIR = ROOT / "dataset" / "logmel_songs"
MEL_CACHE = Path("/content/mel_cache")
MEL_CACHE.mkdir(parents=True, exist_ok=True)
ANN_DIR = ROOT / "annotations"
FEAT_DIR = ROOT / "features"
CKPT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"
BASELINE_RESULTS_DIR = RESULTS_DIR / "baselines"
BASELINE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST = ROOT / "dataset" / "song_manifest.csv"
RAW_ANN = "https://raw.githubusercontent.com/MTG/mtg-jamendo-dataset/master/data"
NEEDED_ANN = [
    "splits/split-0/autotagging_genre-train.tsv",
    "splits/split-0/autotagging_genre-validation.tsv",
    "splits/split-0/autotagging_genre-test.tsv",
    "splits/split-0/autotagging_instrument-train.tsv",
    "splits/split-0/autotagging_instrument-validation.tsv",
    "splits/split-0/autotagging_instrument-test.tsv",
]
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def check_internet(host="github.com", port=443, timeout=5) -> bool:
    try:
        socket.create_connection((host, port), timeout=timeout).close()
        return True
    except OSError:
        return False


def ensure_annotations():
    dest_train = ANN_DIR / "splits" / "split-0" / "autotagging_genre-train.tsv"
    if dest_train.exists():
        return
    if not check_internet():
        raise FileNotFoundError("Split TSVs missing and no Internet. Enable Internet and re-run.")
    print("Downloading annotation TSVs to Drive...")
    for rel in NEEDED_ANN:
        dest = ANN_DIR / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"{RAW_ANN}/{rel}", dest)
        print(" ", dest)


def load_mel_npy(mel_abs, retries=5, pause=2.0):
    """Load mel from Drive with retries; cache on Colab disk to avoid FUSE drops."""
    mel_abs = Path(mel_abs)
    sid = normalize_track_id(mel_abs.stem) or mel_abs.stem.replace("/", "_")
    cached = MEL_CACHE / f"{sid}.npy"
    if cached.exists():
        try:
            return np.load(cached)
        except (OSError, ValueError):
            cached.unlink(missing_ok=True)

    last_err = None
    for attempt in range(retries):
        try:
            arr = np.load(mel_abs, mmap_mode=None)
            arr = np.asarray(arr, dtype=np.float32)
            np.save(cached, arr)
            return arr
        except (OSError, ValueError) as e:
            last_err = e
            if attempt + 1 < retries:
                time.sleep(pause * (attempt + 1))
    nbytes = mel_abs.stat().st_size if mel_abs.exists() else "missing"
    raise RuntimeError(
        f"Bad/truncated mel — re-download its shard in notebook 00: {mel_abs} "
        f"({nbytes} bytes on Drive). {last_err}"
    ) from last_err


def scan_bad_mels(df, label="manifest"):
    from tqdm.auto import tqdm

    bad = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"scan {label}"):
        try:
            load_mel_npy(row["mel_abs"])
        except Exception as e:
            bad.append({"song_id": str(row["song_id"]), "mel_abs": row["mel_abs"], "error": str(e)})
    if bad:
        out = RESULTS_DIR / f"bad_mels_{label}.json"
        out.write_text(json.dumps(bad, indent=2))
        print(f"WARNING: {len(bad)} bad mels → {out}")
    else:
        print(f"scan {label}: all {len(df)} mels OK (cache: {MEL_CACHE})")
    return bad


ensure_annotations()
print("ROOT   ", ROOT)
print("MEL_DIR", MEL_DIR, "npy=", len(list(MEL_DIR.rglob("*.npy"))))
print("ANN_DIR", ANN_DIR)
print("MANIFEST", MANIFEST, "exists=", MANIFEST.exists())
'''

PATHS += "\n" + NOTEBOOK_DATA_CONTRACT + "\n" + NOTEBOOK_GPU_RUN_CONTRACT


write(
    "00_download_to_drive.ipynb",
    [
        md("# 00 — Download to Google Drive (Colab)\n\nDownloads split-0 labels and the default **mel shards 00–02** into Drive.\n\nFolder: `/content/drive/MyDrive/MTG_Instrument`\n\nAllow roughly **8 GB free on Drive**. Expand `SHARDS` only when more storage is available. Re-runs skip completed shards."),
        md(COLAB_SETUP),
        md("## Step 0 — Internet"),
        code(
            """import socket
def check_internet(host="github.com", port=443, timeout=5):
    try:
        socket.create_connection((host, port), timeout=timeout).close()
        return True
    except OSError:
        return False
ONLINE = check_internet()
print("Internet:", ONLINE)
if not ONLINE:
    print("Turn on Internet in Colab, then re-run.")
!pip install -q tqdm"""
        ),
        md("## Step 1 — Mount Drive"),
        code(MOUNT),
        md("## Step 2 — Paths"),
        code(PATHS),
        md("## Step 3 — Download the default shards 00–02 onto Drive"),
        code(
            r'''
import subprocess, shutil

SHARDS = [0, 1, 2]  # reliable baseline; use list(range(10)) for the larger subset
BASE_URL = "https://cdn.freesound.org/mtg-jamendo/raw_30s/melspecs"
MEL_DIR.mkdir(parents=True, exist_ok=True)
print("Saving mels to", MEL_DIR)

def free_gb():
    u = shutil.disk_usage(str(MEL_DIR))
    print(f"Drive free: {u.free/1e9:.1f} GB")
    return u.free / 1e9

free_gb()
if not check_internet("cdn.freesound.org") and not check_internet():
    raise RuntimeError("No Internet — cannot download shards.")

for i in SHARDS:
    marker = MEL_DIR / f".shard_{i:02d}_done"
    if marker.exists():
        print(f"shard {i:02d} already on Drive — skip")
        continue
    if free_gb() < 3:
        raise RuntimeError(f"Not enough Drive space for shard {i:02d}")
    tar_name = f"raw_30s_melspecs-{i:02d}.tar"
    tar_path = MEL_DIR / tar_name
    url = f"{BASE_URL}/{tar_name}"
    print("Downloading", url)
    subprocess.check_call(["wget", "-q", "-O", str(tar_path), url])
    print("Extracting", tar_name)
    subprocess.check_call(["tar", "-xf", str(tar_path), "-C", str(MEL_DIR)])
    tar_path.unlink(missing_ok=True)
    marker.write_text("ok")
    print(f"shard {i:02d} saved")

n = len(list(MEL_DIR.rglob("*.npy")))
print("Total .npy on Drive:", n)
assert n > 0
'''
        ),
        md("## Step 4 — Summary"),
        code(
            r'''
summary = {
    "root": str(ROOT),
    "n_npy": len(list(MEL_DIR.rglob("*.npy"))),
    "shards": SHARDS,
}
(RESULTS_DIR / "00_download_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
print("Next: 01_preprocessing.ipynb in Colab, same Drive folder.")
'''
        ),
    ],
)


write(
    "01_preprocessing.ipynb",
    [
        md("# 01 — Preprocessing (Colab + Drive)\n\nJoins Drive mels to official **split-0** and writes `dataset/song_manifest.csv` on Drive."),
        md(COLAB_SETUP),
        md("## Step 0 — Mount Drive"),
        code(MOUNT),
        md("## Step 1 — Paths"),
        code(PATHS),
        md("## Step 2 — Index every `.npy` on Drive"),
        code(
            r'''
rows = []
for p in MEL_DIR.rglob("*.npy"):
    tid = normalize_track_id(p.stem)
    if not tid:
        continue
    rows.append({"song_id": tid, "logmel_path": str(p.relative_to(ROOT)), "mel_abs": str(p), "nbytes": p.stat().st_size})
mel_df = pd.DataFrame(rows)
duplicate_ids = sorted(mel_df.loc[mel_df.duplicated("song_id", keep=False), "song_id"].unique())
if duplicate_ids:
    raise RuntimeError(f"duplicate log-Mels for {len(duplicate_ids)} songs; examples={duplicate_ids[:5]}")
print("Unique songs with mel:", len(mel_df))
if mel_df.empty:
    raise FileNotFoundError(f"No .npy under {MEL_DIR}. Run notebook 00 first.")
mel_df.head()
'''
        ),
        md("## Step 3 — Official split-0 IDs (first-column TSV parse)"),
        code(
            r'''
train_ids = load_split_ids("train")
val_ids = load_split_ids("validation")
test_ids = load_split_ids("test")
assert train_ids.isdisjoint(val_ids) and train_ids.isdisjoint(test_ids) and val_ids.isdisjoint(test_ids)
instrument_train_ids = load_split_ids("train", "instrument")
instrument_val_ids = load_split_ids("validation", "instrument")
instrument_test_ids = load_split_ids("test", "instrument")
assert instrument_train_ids.isdisjoint(instrument_val_ids)
assert instrument_train_ids.isdisjoint(instrument_test_ids)
assert instrument_val_ids.isdisjoint(instrument_test_ids)
instrument_ids = instrument_train_ids | instrument_val_ids | instrument_test_ids
print("Split leakage check: OK")
'''
        ),
        md("## Step 4 — Write manifest to Drive"),
        code(
            r'''
def split_of(sid):
    if sid in train_ids:
        return "train"
    if sid in val_ids:
        return "validation"
    if sid in test_ids:
        return "test"
    return "unused"

mel_df["split"] = mel_df["song_id"].map(split_of)
print(mel_df["split"].value_counts())
manifest = mel_df[mel_df["split"] != "unused"].copy()
if manifest.empty or set(manifest["split"]) != {"train", "validation", "test"}:
    raise RuntimeError("manifest must contain at least one song from every official split")
manifest["audio_path"] = ""
manifest["waveform_available"] = False
manifest["genre_available"] = True
manifest["instrument_available"] = manifest["song_id"].isin(instrument_ids)
manifest["rhythm_available"] = False
manifest["timbre_available"] = False
manifest["harmony_available"] = False
manifest = manifest[[
    "song_id", "split", "audio_path", "logmel_path", "mel_abs", "nbytes", "waveform_available",
    "genre_available", "instrument_available", "rhythm_available",
    "timbre_available", "harmony_available",
]].sort_values("song_id").reset_index(drop=True)
MANIFEST.parent.mkdir(parents=True, exist_ok=True)
manifest.to_csv(MANIFEST, index=False)
print("Wrote", MANIFEST, "rows=", len(manifest))
sample = load_mel_npy(manifest.iloc[0]["mel_abs"])
print("example shape", sample.shape)
(RESULTS_DIR / "01_manifest_summary.json").write_text(json.dumps({
    "n_manifest": int(len(manifest)),
    "split_counts": manifest["split"].value_counts().to_dict(),
    "instrument_available": int(manifest["instrument_available"].sum()),
}, indent=2))
print("Next: 04_rhythm_targets.ipynb, or a published branch package.")
'''
        ),
    ],
)


write(
    "02_direct_cnn_baseline.ipynb",
    [
        md("# 02 — Direct CNN Baseline (Colab + Drive)\n\nDirect multi-label genre prediction from log-Mels. **Runtime → GPU**.\n\nNeeds: notebook 00 + 01 (`song_manifest.csv`).\n\nAlways saves the best checkpoint and validation predictions. Test evaluation is disabled by default and is enabled only for the selected final run with `EVALUATE_TEST=1`."),
        md(COLAB_SETUP),
        code("""!pip install -q scikit-learn tqdm"""),
        md("## Mount Drive"),
        code(MOUNT),
        code(PATHS),
        md("## Load manifest + genre labels"),
        code(
            r'''
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm.auto import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = int(os.environ.get("GPU_BATCH_SIZE", "2"))
if BATCH_SIZE < 1:
    raise ValueError("GPU_BATCH_SIZE must be positive")
GPU_RUN = require_gpu_run_approval(DEVICE, "direct_cnn")
GPU_RUN_STARTED = time.perf_counter()
print("device:", DEVICE)
if not MANIFEST.exists():
    raise FileNotFoundError("Run 01 first — missing song_manifest.csv on Drive")
manifest = pd.read_csv(MANIFEST)
manifest["song_id"] = manifest["song_id"].astype(str).map(lambda s: normalize_track_id(s) or s)
manifest = apply_approved_cohort(manifest, GPU_RUN, MANIFEST)

song_ids = manifest["song_id"].astype(str).tolist()
Y, TAG_NAMES, genre_available = load_split_multihot(song_ids, "genre", "genre")
if not genre_available.all():
    missing = np.asarray(song_ids)[~genre_available]
    raise RuntimeError(f"manifest contains {len(missing)} songs without split-0 genre labels")
print("Y", Y.shape, "pos", float(Y.mean()))
(RESULTS_DIR / "genre_tags.json").write_text(json.dumps(TAG_NAMES, indent=2))
'''
        ),
        md("## Dataset / model / train (best val PR-AUC checkpoint)"),
        code(
            r'''
class MelGenreDataset(Dataset):
    def __init__(self, df, Y, id_to_idx, max_windows=LOGMEL_MAX_WINDOWS,
                 n_mels=LOGMEL_N_MELS, n_frames=LOGMEL_WINDOW_FRAMES):
        self.df = df.reset_index(drop=True)
        self.Y, self.id_to_idx = Y, id_to_idx
        self.max_windows, self.n_mels, self.n_frames = max_windows, n_mels, n_frames

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        windows, mask = segment_logmel(
            load_mel_npy(row["mel_abs"]), n_mels=self.n_mels,
            n_frames=self.n_frames, max_windows=self.max_windows,
        )
        y = self.Y[self.id_to_idx[str(row["song_id"])]]
        return torch.from_numpy(windows[:, None]), torch.from_numpy(mask), torch.from_numpy(y)

id_to_idx = {s: i for i, s in enumerate(song_ids)}

def make_loader(split, bs=BATCH_SIZE, shuffle=False):
    sub = manifest[manifest["split"] == split]
    assert set(sub["split"].unique()) == {split}
    return DataLoader(MelGenreDataset(sub, Y, id_to_idx), batch_size=bs, shuffle=shuffle, num_workers=0)

class BaselineCNN(nn.Module):
    def __init__(self, n_tags):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.window_proj = nn.Sequential(nn.Flatten(), nn.Linear(128*4*4, 256), nn.ReLU(), nn.Dropout(0.3))
        self.head = nn.Linear(256, n_tags)
    def forward(self, x, mask):
        B,W,C,M,T = x.shape
        z = self.window_proj(self.features(x.reshape(B*W,C,M,T))).reshape(B,W,-1)
        weights = mask / mask.sum(1, keepdim=True).clamp_min(1.0)
        return self.head((z * weights.unsqueeze(-1)).sum(1))

model = BaselineCNN(Y.shape[1]).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
crit = nn.BCEWithLogitsLoss()

def nan_safe(y_true, y_prob, kind="roc"):
    scores = []
    for k in range(y_true.shape[1]):
        if y_true[:, k].sum() in (0, len(y_true)):
            continue
        try:
            scores.append(roc_auc_score(y_true[:, k], y_prob[:, k]) if kind == "roc" else average_precision_score(y_true[:, k], y_prob[:, k]))
        except ValueError:
            continue
    return float(np.mean(scores)) if scores else float("nan")

@torch.no_grad()
def evaluate(loader, return_predictions=False):
    model.eval(); ys, ps = [], []
    for x, mask, y in loader:
        ps.append(torch.sigmoid(model(x.to(DEVICE), mask.to(DEVICE))).cpu().numpy()); ys.append(y.numpy())
    yt, yp = np.concatenate(ys), np.concatenate(ps)
    metrics = {"macro_roc_auc": nan_safe(yt, yp, "roc"), "macro_pr_auc": nan_safe(yt, yp, "pr")}
    return (metrics, yt, yp) if return_predictions else metrics

train_loader, val_loader, test_loader = make_loader("train", shuffle=True), make_loader("validation"), make_loader("test")
_x, _mask, _y = next(iter(train_loader))
print("preflight batch", tuple(_x.shape), "expect (bs, 12, 1, 96, 1366)")
assert _x.shape[1:] == (12, 1, 96, 1366), f"re-run this entire cell — got {_x.shape}"
assert torch.all(_mask.sum(1) >= 1), "every song needs at least one real window"
SCAN_MELS = False
if SCAN_MELS:
    bad = scan_bad_mels(manifest, "all")
    if bad:
        raise RuntimeError(f"{len(bad)} bad mels — see {RESULTS_DIR}/bad_mels_all.json")
best_macro_map = float("-inf")
ckpt = CKPT_DIR / "baselines" / "direct_cnn"; ckpt.mkdir(parents=True, exist_ok=True)
hist = []
PATIENCE = 2
MAX_EPOCHS, MAX_WALL_MINUTES = approved_run_limits(
    GPU_RUN, default_epochs=10,
    requested_wall_minutes=float(os.environ.get("MAX_GPU_RUN_MINUTES", "120")),
)
EVALUATE_TEST = os.environ.get("EVALUATE_TEST", "0") == "1"
epochs_without_improvement = 0
GPU_DEADLINE = GPU_RUN_STARTED + MAX_WALL_MINUTES * 60
TRAINING_DEADLINE = GPU_RUN_STARTED + MAX_WALL_MINUTES * 60 * 0.9
wall_cap_reached = False
if DEVICE.type == "cuda":
    model.eval(); opt.zero_grad(set_to_none=True)
    _preflight_loss = crit(model(_x.to(DEVICE), _mask.to(DEVICE)), _y.to(DEVICE))
    _preflight_loss.backward(); opt.zero_grad(set_to_none=True)
    del _preflight_loss
    torch.cuda.empty_cache()
    print("preflight backward: OK")
for epoch in range(1, MAX_EPOCHS + 1):
    model.train(); total = 0
    for x, mask, y in tqdm(train_loader, leave=False):
        if time.perf_counter() >= TRAINING_DEADLINE:
            wall_cap_reached = True
            break
        x, mask, y = x.to(DEVICE), mask.to(DEVICE), y.to(DEVICE)
        opt.zero_grad(); loss = crit(model(x, mask), y); loss.backward(); opt.step()
        total += loss.item() * len(x)
    if wall_cap_reached:
        print(f"training-time reserve reached between batches: {MAX_WALL_MINUTES:.0f} minute total cap")
        break
    vm = evaluate(val_loader)
    if not np.isfinite(vm["macro_pr_auc"]):
        raise RuntimeError("validation PR-AUC is undefined; fix label coverage before spending more GPU time")
    hist.append({"epoch": epoch, "loss": total/len(train_loader.dataset), **vm})
    print(epoch, hist[-1])
    if vm["macro_pr_auc"] > best_macro_map:
        best_macro_map = vm["macro_pr_auc"]
        epochs_without_improvement = 0
        torch.save({
            "model": model.state_dict(), "tags": TAG_NAMES,
            "best_macro_map": best_macro_map, "epoch": epoch,
            "training_config": {"max_epochs": MAX_EPOCHS, "patience": PATIENCE,
                                "max_wall_minutes": MAX_WALL_MINUTES,
                                "input_schema": LOGMEL_SCHEMA_VERSION,
                                "max_windows": LOGMEL_MAX_WINDOWS,
                                "batch_size": BATCH_SIZE,
                                "gpu_run": GPU_RUN},
        }, ckpt/"best.pt")
        print("  ✓ saved", best_macro_map)
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= PATIENCE:
            print(f"early stop: no validation PR-AUC improvement for {PATIENCE} epochs")
            break
    if time.perf_counter() >= TRAINING_DEADLINE:
        print(f"wall-time cap reached: {MAX_WALL_MINUTES:.0f} minutes")
        break

if not (ckpt/"best.pt").is_file():
    write_gpu_termination_ledger(
        BASELINE_RESULTS_DIR/"02_baseline_runtime.json", device=DEVICE,
        started_at=GPU_RUN_STARTED, record=GPU_RUN, reason="no_complete_validation_epoch",
        max_epochs=MAX_EPOCHS, max_wall_minutes=MAX_WALL_MINUTES,
    )
    raise RuntimeError("GPU cap reached before one complete validation epoch; no checkpoint was created")
state = torch.load(ckpt/"best.pt", map_location=DEVICE, weights_only=False)
model.load_state_dict(state["model"])
val_m, val_y, val_p = evaluate(val_loader, return_predictions=True)
pd.DataFrame(hist).to_csv(BASELINE_RESULTS_DIR/"02_baseline_history.csv", index=False)
elapsed_seconds = time.perf_counter() - GPU_RUN_STARTED
runtime = {
    "device": str(DEVICE), "epochs_completed": len(hist),
    "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
    "max_wall_minutes": MAX_WALL_MINUTES,
    "batch_size": BATCH_SIZE, "input_schema": LOGMEL_SCHEMA_VERSION,
    "evaluated_test": EVALUATE_TEST,
    "gpu_run": GPU_RUN,
    "wall_seconds": elapsed_seconds,
    "gpu_wall_hours": elapsed_seconds / 3600 if DEVICE.type == "cuda" else 0.0,
}
(BASELINE_RESULTS_DIR/"02_baseline_runtime.json").write_text(json.dumps(runtime, indent=2))
print("runtime", runtime)
pred_dir = RESULTS_DIR / "predictions"; pred_dir.mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    pred_dir / "02_direct_cnn_validation.npz",
    song_ids=np.asarray(val_loader.dataset.df["song_id"].astype(str).to_numpy(), dtype=str),
    label_names=np.asarray(TAG_NAMES, dtype=str), targets=val_y, scores=val_p,
)
if EVALUATE_TEST:
    test_m, test_y, test_p = evaluate(test_loader, return_predictions=True)
    print("FINAL TEST split-0", test_m)
    (BASELINE_RESULTS_DIR/"02_baseline_test.json").write_text(json.dumps(test_m, indent=2))
    np.savez_compressed(
        pred_dir / "02_direct_cnn_test.npz",
        song_ids=np.asarray(test_loader.dataset.df["song_id"].astype(str).to_numpy(), dtype=str),
        label_names=np.asarray(TAG_NAMES, dtype=str), targets=test_y, scores=test_p,
    )
else:
    print("Test evaluation skipped. Set EVALUATE_TEST=1 only for the selected final run.")
elapsed_seconds = time.perf_counter() - GPU_RUN_STARTED
runtime["wall_seconds"] = elapsed_seconds
runtime["gpu_wall_hours"] = elapsed_seconds / 3600 if DEVICE.type == "cuda" else 0.0
(BASELINE_RESULTS_DIR/"02_baseline_runtime.json").write_text(json.dumps(runtime, indent=2))
print("final runtime", runtime)
'''
        ),
    ],
)


write(
    "03_instrument_pretraining.ipynb",
    [
        md("# 03 — Instrument Pretraining (Colab + Drive)\n\nMIL + attention → 64-d instrument embedding. This can initialize the proposed shared encoder. **GPU On**.\n\nNeeds: 00 + 01. Writes `features/instrument/` and `checkpoints/pretraining/instrument/` on Drive."),
        md(COLAB_SETUP),
        code("""!pip install -q scikit-learn tqdm"""),
        md("## Mount Drive"),
        code(MOUNT),
        code(PATHS),
        md("## Train + export embeddings"),
        code(
            r'''
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import average_precision_score
from tqdm.auto import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = int(os.environ.get("GPU_BATCH_SIZE", "2"))
if BATCH_SIZE < 1:
    raise ValueError("GPU_BATCH_SIZE must be positive")
GPU_RUN = require_gpu_run_approval(DEVICE, "instrument_pretraining")
GPU_RUN_STARTED = time.perf_counter()
if not MANIFEST.exists():
    raise FileNotFoundError("Run 01 first")
manifest = pd.read_csv(MANIFEST)
manifest["song_id"] = manifest["song_id"].astype(str).map(lambda s: normalize_track_id(s) or s)
manifest = apply_approved_cohort(manifest, GPU_RUN, MANIFEST)
EMBED_DIM, MAX_WINDOWS = 64, LOGMEL_MAX_WINDOWS
N_MELS, N_FRAMES = LOGMEL_N_MELS, LOGMEL_WINDOW_FRAMES
song_ids = manifest["song_id"].astype(str).tolist()
id_to_idx = {s: i for i, s in enumerate(song_ids)}

Y, INST_NAMES, instrument_available = load_split_multihot(song_ids, "instrument", "instrument")
manifest["instrument_available"] = instrument_available
if not instrument_available.any():
    raise RuntimeError("manifest has no songs with split-0 instrument annotations")
print("Y_inst", Y.shape, "annotated songs", int(instrument_available.sum()))

class WindowMIL(Dataset):
    def __init__(self, df, max_windows=MAX_WINDOWS, n_mels=N_MELS, n_frames=N_FRAMES):
        self.df = df.reset_index(drop=True)
        self.max_windows, self.n_mels, self.n_frames = max_windows, n_mels, n_frames

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        windows, mask = segment_logmel(
            load_mel_npy(row["mel_abs"]), n_mels=self.n_mels,
            n_frames=self.n_frames, max_windows=self.max_windows,
        )
        y = Y[id_to_idx[str(row["song_id"])]]
        return (
            torch.from_numpy(windows[:, None]),
            torch.from_numpy(mask),
            torch.from_numpy(y),
            str(row["song_id"]),
        )

def make_loader(split, bs=BATCH_SIZE, shuffle=False):
    sub = manifest[(manifest.split==split) & manifest.instrument_available]
    if sub.empty:
        raise RuntimeError(f"no instrument-annotated songs in {split}")
    assert set(sub.split.unique())=={split}
    return DataLoader(WindowMIL(sub), batch_size=bs, shuffle=shuffle, num_workers=0)

class InstrumentMIL(nn.Module):
    def __init__(self, n_tags, emb=EMBED_DIM):
        super().__init__()
        self.cnn = nn.Sequential(nn.Conv2d(1,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                 nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1,1)))
        self.proj = nn.Linear(64, emb)
        self.score = nn.Linear(emb, 1)
        self.head = nn.Linear(emb, n_tags)
    def forward(self, x, mask):
        B,W,C,M,T = x.shape
        H = self.proj(self.cnn(x.reshape(B*W,C,M,T)).flatten(1)).reshape(B,W,-1)
        logits = self.score(H).squeeze(-1).masked_fill(mask<0.5, -1e9)
        w = torch.softmax(logits, -1)
        z = (H * w.unsqueeze(-1)).sum(1)
        return self.head(z), z, w

model = InstrumentMIL(Y.shape[1]).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
crit = nn.BCEWithLogitsLoss()

def macro_map(yt, yp):
    s=[]
    for k in range(yt.shape[1]):
        if yt[:,k].sum() in (0,len(yt)): continue
        try: s.append(average_precision_score(yt[:,k], yp[:,k]))
        except ValueError: pass
    return float(np.mean(s)) if s else float("nan")

@torch.no_grad()
def eval_split(dl):
    model.eval(); ys,ps=[],[]
    for x,mask,y,_ in dl:
        logits,_,_ = model(x.to(DEVICE), mask.to(DEVICE))
        ps.append(torch.sigmoid(logits).cpu().numpy()); ys.append(y.numpy())
    return macro_map(np.concatenate(ys), np.concatenate(ps))

tr, va, te = make_loader("train", shuffle=True), make_loader("validation"), make_loader("test")
_x, _m, _y, _ = next(iter(tr))
print("preflight batch", tuple(_x.shape), "expect (bs, 12, 1, 96, 1366)")
assert _x.shape[2:] == (1, N_MELS, N_FRAMES), f"re-run this entire cell — got {_x.shape}"
SCAN_MELS = False
if SCAN_MELS:
    bad = scan_bad_mels(manifest, "all")
    if bad:
        raise RuntimeError(f"{len(bad)} bad mels — see {RESULTS_DIR}/bad_mels_all.json")
best_macro_map = float("-inf")
ckpt = CKPT_DIR/"pretraining"/"instrument"; ckpt.mkdir(parents=True, exist_ok=True)
PATIENCE = 2
MAX_EPOCHS, MAX_WALL_MINUTES = approved_run_limits(
    GPU_RUN, default_epochs=8,
    requested_wall_minutes=float(os.environ.get("MAX_GPU_RUN_MINUTES", "120")),
)
EVALUATE_TEST = os.environ.get("EVALUATE_TEST", "0") == "1"
epochs_without_improvement = 0
history = []
GPU_DEADLINE = GPU_RUN_STARTED + MAX_WALL_MINUTES * 60
TRAINING_DEADLINE = GPU_RUN_STARTED + MAX_WALL_MINUTES * 60 * 0.9
wall_cap_reached = False
if DEVICE.type == "cuda":
    model.eval(); opt.zero_grad(set_to_none=True)
    _preflight_logits, _, _ = model(_x.to(DEVICE), _m.to(DEVICE))
    _preflight_loss = crit(_preflight_logits, _y.to(DEVICE))
    _preflight_loss.backward(); opt.zero_grad(set_to_none=True)
    del _preflight_logits, _preflight_loss
    torch.cuda.empty_cache()
    print("preflight backward: OK")
for epoch in range(1, MAX_EPOCHS + 1):
    model.train(); total=0
    for x,mask,y,_ in tqdm(tr, leave=False):
        if time.perf_counter() >= TRAINING_DEADLINE:
            wall_cap_reached = True
            break
        x,mask,y = x.to(DEVICE), mask.to(DEVICE), y.to(DEVICE)
        opt.zero_grad(); logits,_,_=model(x,mask); loss=crit(logits,y); loss.backward(); opt.step()
        total += loss.item()*len(x)
    if wall_cap_reached:
        print(f"training-time reserve reached between batches: {MAX_WALL_MINUTES:.0f} minute total cap")
        break
    vm = eval_split(va)
    if not np.isfinite(vm):
        raise RuntimeError("validation macro mAP is undefined; fix label coverage before spending more GPU time")
    history.append({"epoch": epoch, "loss": total/len(tr.dataset), "val_macro_map": vm})
    print(epoch, "val_map", vm)
    if vm > best_macro_map:
        best_macro_map = vm
        epochs_without_improvement = 0
        torch.save({
            "model": model.state_dict(), "best_macro_map": best_macro_map, "tags": INST_NAMES,
            "training_config": {"max_epochs": MAX_EPOCHS, "patience": PATIENCE,
                                "max_wall_minutes": MAX_WALL_MINUTES,
                                "input_schema": LOGMEL_SCHEMA_VERSION,
                                "max_windows": LOGMEL_MAX_WINDOWS,
                                "batch_size": BATCH_SIZE,
                                "gpu_run": GPU_RUN},
        }, ckpt/"best.pt")
        print("  ✓", best_macro_map)
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= PATIENCE:
            print(f"early stop: no validation PR-AUC improvement for {PATIENCE} epochs")
            break
    if time.perf_counter() >= TRAINING_DEADLINE:
        print(f"wall-time cap reached: {MAX_WALL_MINUTES:.0f} minutes")
        break

if not (ckpt/"best.pt").is_file():
    write_gpu_termination_ledger(
        RESULTS_DIR/"pretraining"/"instrument"/"runtime.json", device=DEVICE,
        started_at=GPU_RUN_STARTED, record=GPU_RUN, reason="no_complete_validation_epoch",
        max_epochs=MAX_EPOCHS, max_wall_minutes=MAX_WALL_MINUTES,
    )
    raise RuntimeError("GPU cap reached before one complete validation epoch; no checkpoint was created")
state = torch.load(ckpt/"best.pt", map_location=DEVICE, weights_only=False)
model.load_state_dict(state["model"]); model.eval()
embeds, ids = [], []
export_cap_reached = False
with torch.no_grad():
    for x,mask,y,sid in tqdm(DataLoader(WindowMIL(manifest), batch_size=BATCH_SIZE, num_workers=0)):
        if time.perf_counter() >= GPU_DEADLINE:
            export_cap_reached = True
            break
        _, z, _ = model(x.to(DEVICE), mask.to(DEVICE))
        embeds.append(z.cpu().numpy()); ids.extend(list(sid))
if export_cap_reached:
    write_gpu_termination_ledger(
        RESULTS_DIR/"pretraining"/"instrument"/"runtime.json", device=DEVICE,
        started_at=GPU_RUN_STARTED, record=GPU_RUN, reason="embedding_export_cap_reached",
        max_epochs=MAX_EPOCHS, max_wall_minutes=MAX_WALL_MINUTES,
    )
    raise RuntimeError("GPU wall-time cap reached during embedding export; no partial artifact was saved")
E = np.concatenate(embeds, 0)
out = FEAT_DIR/"instrument"; out.mkdir(parents=True, exist_ok=True)
np.save(out/"instrument_embeddings.npy", E)
(out/"song_ids.json").write_text(json.dumps(ids))
result_dir = RESULTS_DIR/"pretraining"/"instrument"; result_dir.mkdir(parents=True, exist_ok=True)
pd.DataFrame(history).to_csv(result_dir/"history.csv", index=False)
elapsed_seconds = time.perf_counter() - GPU_RUN_STARTED
runtime = {
    "device": str(DEVICE), "epochs_completed": len(history),
    "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
    "max_wall_minutes": MAX_WALL_MINUTES, "batch_size": BATCH_SIZE,
    "input_schema": LOGMEL_SCHEMA_VERSION, "evaluated_test": EVALUATE_TEST,
    "gpu_run": GPU_RUN,
    "wall_seconds_including_export": elapsed_seconds,
    "gpu_wall_hours_including_export": elapsed_seconds/3600 if DEVICE.type == "cuda" else 0.0,
}
(result_dir/"runtime.json").write_text(json.dumps(runtime, indent=2))
print("saved", E.shape, "runtime", runtime)
if EVALUATE_TEST:
    test_metric = {"macro_pr_auc": eval_split(te)}
    (result_dir/"test.json").write_text(json.dumps(test_metric, indent=2))
    print("FINAL instrument TEST", test_metric)
else:
    print("Instrument test skipped. Set EVALUATE_TEST=1 only for the selected final run.")
elapsed_seconds = time.perf_counter() - GPU_RUN_STARTED
runtime["wall_seconds_including_export"] = elapsed_seconds
runtime["gpu_wall_hours_including_export"] = elapsed_seconds/3600 if DEVICE.type == "cuda" else 0.0
(result_dir/"runtime.json").write_text(json.dumps(runtime, indent=2))
print("final runtime", runtime)
'''
        ),
    ],
)


COLAB_04_EXTRACT = r'''
import subprocess
from tqdm.auto import tqdm

if not MANIFEST.exists():
    raise FileNotFoundError("Run 01 first")
manifest = pd.read_csv(MANIFEST)
manifest["song_id"] = manifest["song_id"].astype(str).map(lambda s: normalize_track_id(s) or s)

AB_DIR = ROOT / "dataset" / "acousticbrainz"
AB_DIR.mkdir(parents=True, exist_ok=True)
AB_SHARDS = [0, 1, 2]  # must match the default mel subset from notebook 00
AB_URL = "https://cdn.freesound.org/mtg-jamendo/raw_30s/acousticbrainz"


def download_ab_shards():
    n_json = len(list(AB_DIR.rglob("*.json")))
    if n_json > 0:
        print(f"AcousticBrainz already on Drive: {n_json} JSON under {AB_DIR}")
        return
    if not check_internet("cdn.freesound.org") and not check_internet():
        raise RuntimeError(
            "No AcousticBrainz JSON on Drive and no Internet. "
            "Enable Internet, or run the official MTG script:\n"
            "  python3 scripts/download/download.py --dataset raw_30s "
            "--type acousticbrainz --from mtg-fast --unpack --remove "
            f"{AB_DIR}"
        )
    print("Downloading AcousticBrainz shards 00–02 to", AB_DIR)
    for i in AB_SHARDS:
        marker = AB_DIR / f".ab_shard_{i:02d}_done"
        if marker.exists():
            print(f"AB shard {i:02d} already done — skip")
            continue
        tar_name = f"raw_30s_acousticbrainz-{i:02d}.tar.gz"
        tar_path = AB_DIR / tar_name
        url = f"{AB_URL}/{tar_name}"
        print("Downloading", url)
        subprocess.check_call(["wget", "-q", "-O", str(tar_path), url])
        subprocess.check_call(["tar", "-xzf", str(tar_path), "-C", str(AB_DIR)])
        tar_path.unlink(missing_ok=True)
        marker.write_text("ok")
        print(f"AB shard {i:02d} saved")


def index_ab_json(root: Path) -> dict[str, Path]:
    idx = {}
    for p in root.rglob("*.json"):
        sid = normalize_track_id(p.stem)
        if sid:
            idx[sid] = p
    return idx


def _scalar(v):
    """Unwrap AcousticBrainz scalar or {mean: ...} stats. Never invent a default."""
    if v is None:
        return None
    if isinstance(v, dict):
        if "mean" in v:
            return _scalar(v["mean"])
        return None
    if isinstance(v, (list, tuple)):
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if np.isnan(x):
        return None
    return x


def rhythm_from_ab(doc: dict) -> dict | None:
    block = doc.get("rhythm")
    if not isinstance(block, dict):
        return None
    bpm = _scalar(block.get("bpm"))
    if bpm is None:
        return None
    beats_pos = block.get("beats_position")
    if not isinstance(beats_pos, (list, tuple)):
        beats_pos = []
    beats_count = _scalar(block.get("beats_count"))
    if beats_count is None and beats_pos:
        beats_count = float(len(beats_pos))
    intervals = np.diff(np.asarray(beats_pos, dtype=np.float64)) if len(beats_pos) > 1 else None
    feat = {
        "bpm": bpm,
        "beats_count": beats_count,
        "beats_loudness_mean": _scalar(block.get("beats_loudness")),
        "bpm_histogram_first_peak_bpm": _scalar(block.get("bpm_histogram_first_peak_bpm")),
        "bpm_histogram_first_peak_spread": _scalar(block.get("bpm_histogram_first_peak_spread")),
        "bpm_histogram_first_peak_weight": _scalar(block.get("bpm_histogram_first_peak_weight")),
        "onset_rate": _scalar(block.get("onset_rate")),
        "danceability": _scalar(block.get("danceability")),
        "beat_interval_mean": float(np.mean(intervals)) if intervals is not None else None,
        "beat_interval_std": float(np.std(intervals)) if intervals is not None else None,
    }
    return feat


download_ab_shards()
ab_index = index_ab_json(AB_DIR)
print("AcousticBrainz JSON indexed:", len(ab_index))

rows, missing = [], []
for _, rec in tqdm(manifest.iterrows(), total=len(manifest), desc="rhythm"):
    sid = str(rec["song_id"])
    path = ab_index.get(sid)
    if path is None:
        missing.append({"song_id": sid, "reason": "no_acousticbrainz_json"})
        continue
    try:
        doc = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        feat = rhythm_from_ab(doc)
        if feat is None:
            missing.append({"song_id": sid, "reason": "json_missing_rhythm.bpm", "path": str(path)})
            continue
        feat.update({"song_id": sid, "source": "acousticbrainz", "split": rec["split"]})
        rows.append(feat)
    except Exception as e:
        missing.append({"song_id": sid, "reason": str(e), "path": str(path)})

n_mels = int(len(manifest))
n_ab_disk = int(len(ab_index))
n_written = int(len(rows))
n_excluded = int(len(missing))
print(f"songs with mels (manifest):     {n_mels}")
print(f"AcousticBrainz JSON on disk:    {n_ab_disk}")
print(f"overlap written to CSV:         {n_written}")
print(f"excluded (no/invalid AB JSON):  {n_excluded}")
if n_written == 0:
    raise RuntimeError("No overlapping AcousticBrainz rhythm rows — download AB shards and re-run.")

df = pd.DataFrame(rows)
out = FEAT_DIR / "rhythm"
out.mkdir(parents=True, exist_ok=True)
csv_path = out / "rhythm_song.csv"
df.to_csv(csv_path, index=False)
(RESULTS_DIR / "04_missing_acousticbrainz.json").write_text(json.dumps(missing, indent=2))
summary = {
    "n_manifest_mels": n_mels,
    "n_acousticbrainz_json": n_ab_disk,
    "n_overlap_written": n_written,
    "n_excluded": n_excluded,
    "csv": str(csv_path),
    "source": "acousticbrainz",
}
(RESULTS_DIR / "04_rhythm_summary.json").write_text(json.dumps(summary, indent=2))
print("wrote", csv_path, "rows=", n_written)
print(json.dumps(summary, indent=2))
df.head()
'''


def feature_nb(num, title, kind, extract_fn, out_sub):
    body = '''
import librosa
from tqdm.auto import tqdm
if not MANIFEST.exists():
    raise FileNotFoundError("Run 01 first")
manifest = pd.read_csv(MANIFEST)
manifest["song_id"] = manifest["song_id"].astype(str).map(lambda s: normalize_track_id(s) or s)
WINDOW_SEC, SR = 15.0, 22050
__EXTRACT__

rows = []
for _, rec in tqdm(manifest.iterrows(), total=len(manifest)):
    sid = str(rec["song_id"])
    try:
        S = load_mel_npy(rec["mel_abs"])
        if S.ndim == 3:
            S = S.mean(0)
        feat = mel_proxy_features(S)
        feat.update({"song_id": sid, "source": "mel_proxy", "split": rec["split"]})
        rows.append(feat)
    except Exception as e:
        print("fail", sid, e)
df = pd.DataFrame(rows)
out = FEAT_DIR / "__SUB__"
out.mkdir(parents=True, exist_ok=True)
df.to_csv(out / "__SUB___song.csv", index=False)
print("wrote", out, len(df))
'''.replace("__EXTRACT__", extract_fn).replace("__SUB__", out_sub)
    write(
        f"{num}_{kind}_targets.ipynb",
        [
            md(f"# {num} — {title} (Colab + Drive)\n\nCreates supervision targets for the proposed concept branch and descriptor inputs for the retained baseline. Writes `features/{out_sub}/{out_sub}_song.csv` on Drive. GPU Off. Needs 00+01."),
            md(COLAB_SETUP),
            code("""!pip install -q librosa soundfile tqdm"""),
            md("## Mount Drive"),
            code(MOUNT),
            code(PATHS),
            md("## Extract"),
            code(body),
        ],
    )


write(
    "04_rhythm_targets.ipynb",
    [
        md(
            "# 04 — Rhythm Supervision Targets (Colab + Drive)\n\n"
            "Writes `features/rhythm/rhythm_song.csv` from **AcousticBrainz / Essentia** JSON "
            "(not a mel-proxy). GPU Off. Needs 00+01.\n\n"
            "Downloads AcousticBrainz shards 00–02 by default (matching notebook 00) into "
            "`dataset/acousticbrainz/` if JSON files are not already on Drive."
        ),
        md(COLAB_SETUP),
        code("""!pip install -q tqdm"""),
        md("## Mount Drive"),
        code(MOUNT),
        code(PATHS),
        md("## Download AcousticBrainz JSON (if missing) + extract rhythm fields"),
        code(COLAB_04_EXTRACT),
    ],
)
feature_nb(
    "05",
    "Timbre Supervision Targets",
    "timbre",
    '''
def mel_proxy_features(S):
    freqs = np.linspace(0, 1, S.shape[0])[:, None]
    return dict(
        spectral_centroid_mean=float((S*freqs).sum()/(S.sum()+1e-6)),
        spectral_bandwidth_mean=float(np.std(S.mean(1))),
        spectral_contrast_mean=float(S.max(0).mean()-S.min(0).mean()),
        spectral_flatness_mean=float(np.exp(np.mean(np.log(S+1e-6)))/(np.mean(S)+1e-6)),
        rms_mean=float(np.sqrt(np.mean(S**2))),
        spectral_flux_mean=float(np.mean(np.abs(np.diff(S.mean(0))))),
    )
''',
    "timbre",
)
write(
    "06_harmony_targets.ipynb",
    [
        md(
            """# 06 — Harmony Target Preflight (Colab + Drive)

The former notebook incorrectly split Mel bins into 12 groups and called them
pitch classes. That path has been retired. The default path only audits waveform
availability and writes no harmony targets. Explicit opt-in CPU cells below can run
the reviewed extractor and branch-screening gates from an exact Git commit.

The temporal chroma extractor and automatic chord teacher must pass the quality
gates in `docs/harmony-plan.md` before this notebook becomes a target generator.
The CPU-only CQT candidate has synthetic tests but is not selected until the
real-audio comparison gate passes.
GPU Off. Needs notebook 01 and waveform audio."""
        ),
        md(COLAB_SETUP),
        md("## Mount Drive"),
        code(MOUNT),
        code(PATHS),
        md("## Audit waveform availability (never fall back to Mel pseudo-chroma)"),
        code(
            r'''
from datetime import datetime, timezone

if not MANIFEST.exists():
    raise FileNotFoundError("Run notebook 01 first.")

manifest = pd.read_csv(MANIFEST)
manifest["song_id"] = manifest["song_id"].astype(str).map(lambda s: normalize_track_id(s) or s)
audio_root = ROOT / "dataset" / "audio"
audio_index = {}
if audio_root.exists():
    for path in sorted(audio_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".mp3", ".wav", ".flac", ".ogg"}:
            sid = normalize_track_id(path.stem)
            if sid:
                audio_index.setdefault(sid, []).append(path)

rows = []
for _, rec in manifest.iterrows():
    sid = str(rec["song_id"])
    declared = str(rec.get("audio_path", "")).strip() if pd.notna(rec.get("audio_path")) else ""
    declared_path = Path(declared) if declared else None
    if declared_path is not None and not declared_path.is_absolute():
        declared_path = ROOT / declared_path
    indexed = audio_index.get(sid, [])
    resolved = declared_path if declared_path is not None and declared_path.is_file() else None
    reason = ""
    if resolved is None and len(indexed) == 1:
        resolved = indexed[0]
    elif resolved is None and len(indexed) > 1:
        reason = "duplicate_waveform_candidates"
    elif resolved is None:
        reason = "waveform_not_found"
    rows.append({
        "song_id": sid,
        "split": rec.get("split"),
        "waveform_available": resolved is not None,
        "audio_path": str(resolved) if resolved else "",
        "reason": reason,
    })

availability = pd.DataFrame(rows)
resolved_audio = availability.set_index("song_id")["audio_path"].to_dict()
resolved_available = availability.set_index("song_id")["waveform_available"].to_dict()
manifest["audio_path"] = manifest["song_id"].map(resolved_audio).fillna("")
manifest["waveform_available"] = manifest["song_id"].map(resolved_available).fillna(False).astype(bool)
manifest.to_csv(MANIFEST, index=False)
out = FEAT_DIR / "harmony"
out.mkdir(parents=True, exist_ok=True)
availability.to_csv(out / "harmony_waveform_availability.csv", index=False)
summary = {
    "status": "preflight_only",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "n_manifest": int(len(availability)),
    "n_waveform_available": int(availability["waveform_available"].sum()),
    "n_waveform_missing": int((~availability["waveform_available"]).sum()),
    "n_duplicate_waveform_ids": int((availability["reason"] == "duplicate_waveform_candidates").sum()),
    "creates_targets": False,
    "mel_fallback_allowed": False,
    "next_step": "Use the disabled-by-default CPU gate cells and docs/harmony-plan.md",
}
(out / "harmony_preflight.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
print("No harmony_song.csv was created.")
availability.groupby(["split", "waveform_available"], dropna=False).size()
'''
        ),
        md(
            """## Optional CPU-only harmony gates (disabled by default)

The availability audit above is safe to run normally. The cells below do nothing
unless you explicitly set one of the `RUN_*` environment flags to `1`. They clone
an exact reviewed Git commit, disable CUDA, and write into a new immutable run
directory. Never point them at the held-out test split."""
        ),
        code(
            r'''
import platform, subprocess, sys

RUN_HARMONY_CHROMA_GATE = os.environ.get("RUN_HARMONY_CHROMA_GATE", "0") == "1"
PREPARE_HARMONY_SCREEN = os.environ.get("PREPARE_HARMONY_SCREEN", "0") == "1"
RUN_HARMONY_SCREEN = os.environ.get("RUN_HARMONY_SCREEN", "0") == "1"
PROJECT_CODE = None
HARMONY_RUN_ROOT = None

if any((RUN_HARMONY_CHROMA_GATE, PREPARE_HARMONY_SCREEN, RUN_HARMONY_SCREEN)):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    code_commit = os.environ.get("HARMONY_CODE_COMMIT", "").strip()
    run_name = os.environ.get("HARMONY_RUN_NAME", "").strip()
    if re.fullmatch(r"[0-9a-f]{40}", code_commit) is None:
        raise ValueError("HARMONY_CODE_COMMIT must be the reviewed 40-character Git commit")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{2,63}", run_name) is None:
        raise ValueError("HARMONY_RUN_NAME must be 3-64 safe filename characters")
    PROJECT_CODE = Path("/content/dnn-project")
    if PROJECT_CODE.exists() and not (PROJECT_CODE / ".git").is_dir():
        raise RuntimeError(f"{PROJECT_CODE} exists but is not a Git checkout")
    if not PROJECT_CODE.exists():
        subprocess.check_call([
            "git", "clone", "--filter=blob:none", "--no-checkout",
            "https://github.com/tharu-jwd/dnn-project.git", str(PROJECT_CODE),
        ])
    subprocess.check_call(["git", "-C", str(PROJECT_CODE), "fetch", "origin", code_commit])
    subprocess.check_call(["git", "-C", str(PROJECT_CODE), "checkout", "--detach", code_commit])
    actual_commit = subprocess.check_output(
        ["git", "-C", str(PROJECT_CODE), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_commit != code_commit:
        raise RuntimeError("checked-out code does not match HARMONY_CODE_COMMIT")
    if subprocess.run(["git", "-C", str(PROJECT_CODE), "diff", "--quiet"]).returncode != 0:
        raise RuntimeError("harmony code checkout is dirty")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q", "-r",
        str(PROJECT_CODE / "requirements-harmony.txt"),
    ])
    HARMONY_RUN_ROOT = RESULTS_DIR / "harmony" / run_name
    HARMONY_RUN_ROOT.mkdir(parents=True, exist_ok=True)
    environment_path = HARMONY_RUN_ROOT / "run_environment.json"
    environment = {
        "schema_version": "harmony_hosted_environment_v1",
        "code_commit": actual_commit,
        "repository": "https://github.com/tharu-jwd/dnn-project.git",
        "runtime": "colab",
        "python": sys.version,
        "platform": platform.platform(),
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
    }
    if environment_path.exists():
        previous = json.loads(environment_path.read_text())
        if previous.get("code_commit") != actual_commit:
            raise RuntimeError("run directory was created by a different code commit")
    else:
        environment_path.write_text(json.dumps(environment, indent=2) + "\n")
    print("CPU harmony code:", PROJECT_CODE, actual_commit)
    print("Immutable run root:", HARMONY_RUN_ROOT)
else:
    print("Optional harmony CPU gates disabled; availability audit only.")
'''
        ),
        md("### Stage A — capped real-audio chroma gate"),
        code(
            r'''
if RUN_HARMONY_CHROMA_GATE:
    def run_stage(output, command):
        output = Path(output)
        if output.exists():
            print("Reusing immutable stage:", output)
            return
        print("Running:", " ".join(map(str, command)))
        subprocess.check_call(list(map(str, command)), cwd=PROJECT_CODE)

    py = sys.executable
    scripts = PROJECT_CODE / "scripts"
    cohort = HARMONY_RUN_ROOT / "harmony_extractor_seed42.json"
    regions = HARMONY_RUN_ROOT / "harmony_regions_seed42.json"
    comparison = HARMONY_RUN_ROOT / "harmony_extractor_candidates.json"
    decision = HARMONY_RUN_ROOT / "harmony_extractor_decision.json"
    targets = HARMONY_RUN_ROOT / "harmony-target-pilot"
    run_stage(cohort, [
        py, scripts / "freeze_experiment_cohort.py", MANIFEST,
        "--splits", "train", "validation",
        "--require-available", "waveform_available",
        "--limit", "train=8", "--limit", "validation=2",
        "--seed", "42", "--output", cohort,
    ])
    run_stage(regions, [
        py, scripts / "export_harmony_regions.py", MANIFEST, cohort,
        "--root", ROOT, "--regions-per-song", "2", "--output", regions,
    ])
    run_stage(comparison, [
        py, scripts / "benchmark_harmony_extractors.py",
        "--regions", regions, "--output", comparison,
    ])
    run_stage(decision, [
        py, scripts / "decide_harmony_extractor.py", comparison,
        "--output", decision,
    ])
    run_stage(targets, [
        py, scripts / "materialize_harmony_target_pilot.py", regions, decision,
        "--output-dir", targets,
    ])
    print("Chroma gate artifacts:", HARMONY_RUN_ROOT)
else:
    print("Stage A skipped (RUN_HARMONY_CHROMA_GATE=0).")
'''
        ),
        md("### Stage B — prepare immutable cached screening data"),
        code(
            r'''
if PREPARE_HARMONY_SCREEN:
    checkpoint_text = os.environ.get("HARMONY_INSTRUMENT_CHECKPOINT", "").strip()
    instrument_checkpoint = (
        Path(checkpoint_text) if checkpoint_text
        else CKPT_DIR / "pretraining" / "instrument" / "best.pt"
    )
    if not instrument_checkpoint.is_file():
        raise FileNotFoundError(
            "Validated instrument checkpoint not found. Run notebook 03 or set "
            "HARMONY_INSTRUMENT_CHECKPOINT to its exact best.pt path."
        )
    py = sys.executable
    scripts = PROJECT_CODE / "scripts"
    cohort = HARMONY_RUN_ROOT / "harmony_extractor_seed42.json"
    targets = HARMONY_RUN_ROOT / "harmony-target-pilot"
    feature_cache = HARMONY_RUN_ROOT / "harmony-encoder-cache-pilot"
    screen_dataset = HARMONY_RUN_ROOT / "harmony-screen-dataset-pilot"
    if not feature_cache.exists():
        subprocess.check_call([
            py, scripts / "cache_harmony_encoder_pilot.py",
            MANIFEST, cohort, instrument_checkpoint,
            "--root", ROOT, "--max-songs", "10",
            "--max-cpu-seconds", "600", "--cpu-threads", "4",
            "--output-dir", feature_cache,
        ], cwd=PROJECT_CODE)
    else:
        print("Reusing immutable stage:", feature_cache)
    if not screen_dataset.exists():
        subprocess.check_call([
            py, scripts / "build_harmony_screen_dataset.py",
            feature_cache, targets, "--output-dir", screen_dataset,
        ], cwd=PROJECT_CODE)
    else:
        print("Reusing immutable stage:", screen_dataset)
    template = HARMONY_RUN_ROOT / "harmony-branch-screen-policy.template.json"
    if not template.exists():
        with template.open("w") as handle:
            subprocess.check_call([
                py, scripts / "decide_harmony_branch_screen.py",
                "--print-policy-template", "--dataset", screen_dataset / "index.json",
            ], stdout=handle, cwd=PROJECT_CODE)
    print("Review and fill policy before Stage C:", template)
else:
    print("Stage B skipped (PREPARE_HARMONY_SCREEN=0).")
'''
        ),
        md("### Stage C — one fixed CPU screen and preregistered decision"),
        code(
            r'''
if RUN_HARMONY_SCREEN:
    policy_text = os.environ.get("HARMONY_SCREEN_POLICY", "").strip()
    if not policy_text:
        raise ValueError("Set HARMONY_SCREEN_POLICY to the completed preregistered policy JSON")
    policy = Path(policy_text)
    if not policy.is_file():
        raise FileNotFoundError(policy)
    sys.path.insert(0, str(PROJECT_CODE))
    from scripts.decide_harmony_branch_screen import validate_policy
    validate_policy(json.loads(policy.read_text()))
    screen_dataset = HARMONY_RUN_ROOT / "harmony-screen-dataset-pilot"
    screen_output = HARMONY_RUN_ROOT / "harmony-branch-screen"
    screen_decision = HARMONY_RUN_ROOT / "harmony-branch-screen-decision.json"
    py = sys.executable
    scripts = PROJECT_CODE / "scripts"
    if not screen_output.exists():
        subprocess.check_call([
            py, scripts / "screen_temporal_harmony_branch.py", screen_dataset,
            "--max-epochs", "20", "--max-cpu-seconds", "300",
            "--cpu-threads", "4", "--output-dir", screen_output,
        ], cwd=PROJECT_CODE)
    else:
        print("Reusing immutable stage:", screen_output)
    if not screen_decision.exists():
        subprocess.check_call([
            py, scripts / "decide_harmony_branch_screen.py",
            policy, screen_output / "report.json", "--output", screen_decision,
        ], cwd=PROJECT_CODE)
    else:
        print("Reusing immutable decision:", screen_decision)
    print(json.loads(screen_decision.read_text())["decision"])
else:
    print("Stage C skipped (RUN_HARMONY_SCREEN=0).")
'''
        ),
    ],
)


write(
    "07_descriptor_fusion_baseline.ipynb",
    [
        md("# 07 — Descriptor-Fusion Baseline (Colab + Drive)\n\nCombines the learned instrument embedding with rhythm, timbre, and a reviewed global harmony summary. This is a comparison baseline, not the proposed temporal four-branch model.\n\n**Blocked by design** until the harmony quality gates produce `global_tonal_summary_v1`; notebook 06 currently performs preflight only. Also needs 01 + 03 + 04 + 05. **GPU On**.\n\nAlways saves the best checkpoint and validation predictions. Test evaluation is disabled by default and is enabled only for the selected final run with `EVALUATE_TEST=1`."),
        md(COLAB_SETUP),
        code("""!pip install -q scikit-learn tqdm"""),
        md("## Mount Drive"),
        code(MOUNT),
        code(PATHS),
        md("## Load features + train"),
        code(
            r'''
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm.auto import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GPU_RUN = require_gpu_run_approval(DEVICE, "descriptor_fusion")
GPU_RUN_STARTED = time.perf_counter()
if not MANIFEST.exists():
    raise FileNotFoundError("Run 01 first")
manifest = pd.read_csv(MANIFEST)
manifest["song_id"] = manifest["song_id"].astype(str).map(lambda s: normalize_track_id(s) or s)
manifest = apply_approved_cohort(manifest, GPU_RUN, MANIFEST)
E = np.load(FEAT_DIR/"instrument"/"instrument_embeddings.npy")
inst_ids = json.loads((FEAT_DIR/"instrument"/"song_ids.json").read_text())
inst_map = {normalize_track_id(s) or str(s): E[i] for i,s in enumerate(inst_ids)}

def load_feat(sub):
    p = FEAT_DIR/sub/f"{sub}_song.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p)
    df["song_id"] = df["song_id"].astype(str).map(lambda s: normalize_track_id(s) or s)
    return df.set_index("song_id")

rhythm, timbre, harmony = load_feat("rhythm"), load_feat("timbre"), load_feat("harmony")
if "source" in rhythm.columns and (rhythm["source"] == "mel_proxy").any():
    raise RuntimeError("rhythm_song.csv still has mel_proxy placeholders — re-run notebook 04 (AcousticBrainz)")
required_harmony_meta = {"source", "schema_version", "target_variant"}
missing_harmony_meta = required_harmony_meta - set(harmony.columns)
if missing_harmony_meta:
    raise RuntimeError(
        "Harmony descriptors are stale or unvalidated; missing metadata columns: "
        f"{sorted(missing_harmony_meta)}. Complete the harmony-plan quality gates."
    )
if harmony["source"].astype(str).str.contains("mel_proxy", case=False).any():
    raise RuntimeError("Mel-proxy harmony is invalid and cannot enter descriptor fusion.")
if not harmony["target_variant"].astype(str).eq("global_tonal_summary_v1").all():
    raise RuntimeError("Descriptor fusion requires the reviewed global_tonal_summary_v1 artifact.")

def ncols(df):
    metadata = {"source", "split", "schema_version", "target_variant", "extractor_version", "teacher_name", "teacher_version"}
    return [c for c in df.columns if c not in metadata and pd.api.types.is_numeric_dtype(df[c])]
r_cols, t_cols, h_cols = ncols(rhythm), ncols(timbre), ncols(harmony)
n_before = len(manifest)
have = set(inst_map) & set(rhythm.index) & set(timbre.index) & set(harmony.index)
manifest = manifest[manifest["song_id"].isin(have)].copy()
print(f"Descriptor baseline overlap: {len(manifest)} / {n_before} songs have all four concepts")
if manifest.empty:
    raise RuntimeError("No overlapping songs with validated descriptors — complete notebooks 03–06 and their quality gates.")
ids = manifest["song_id"].astype(str).tolist()
id_to_idx = {s:i for i,s in enumerate(ids)}

Y, TAG_NAMES, genre_available = load_split_multihot(ids, "genre", "genre")
if not genre_available.all():
    raise RuntimeError("descriptor cohort contains songs without split-0 genre labels")

class DS(Dataset):
    def __init__(self, df): self.df = df.reset_index(drop=True)
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        sid = str(self.df.iloc[i]["song_id"])
        inst = inst_map[sid].astype(np.float32)
        r = rhythm.loc[sid, r_cols].astype(np.float32).fillna(0).values if sid in rhythm.index else np.zeros(len(r_cols), np.float32)
        t = timbre.loc[sid, t_cols].astype(np.float32).fillna(0).values if sid in timbre.index else np.zeros(len(t_cols), np.float32)
        h = harmony.loc[sid, h_cols].astype(np.float32).fillna(0).values if sid in harmony.index else np.zeros(len(h_cols), np.float32)
        return torch.tensor(inst), torch.tensor(r), torch.tensor(t), torch.tensor(h), torch.tensor(Y[id_to_idx[sid]])

def loader(split, bs=32, shuffle=False):
    sub = manifest[manifest.split==split]
    return DataLoader(DS(sub), batch_size=bs, shuffle=shuffle, num_workers=0)

class AttentionFusion(nn.Module):
    def __init__(self, d_i,d_r,d_t,d_h, token=64, fused=128, n_tags=87):
        super().__init__()
        self.p_i,self.p_r,self.p_t,self.p_h = nn.Linear(d_i,token),nn.Linear(d_r,token),nn.Linear(d_t,token),nn.Linear(d_h,token)
        self.attn = nn.MultiheadAttention(token,1,batch_first=True)
        self.out = nn.Sequential(nn.Linear(token,fused), nn.ReLU(), nn.Dropout(0.2))
        self.head = nn.Linear(fused, n_tags)
    def forward(self, inst,r,t,h):
        tok = torch.stack([self.p_i(inst),self.p_r(r),self.p_t(t),self.p_h(h)],1)
        o,w = self.attn(tok,tok,tok,need_weights=True)
        return self.head(self.out(o.mean(1))), w

FUSION = "attention"
model = AttentionFusion(64,len(r_cols),len(t_cols),len(h_cols), n_tags=Y.shape[1]).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
crit = nn.BCEWithLogitsLoss()

def nan_safe(yt,yp,kind="roc"):
    s=[]
    for k in range(yt.shape[1]):
        if yt[:,k].sum() in (0,len(yt)): continue
        try: s.append(roc_auc_score(yt[:,k],yp[:,k]) if kind=="roc" else average_precision_score(yt[:,k],yp[:,k]))
        except ValueError: pass
    return float(np.mean(s)) if s else float("nan")

@torch.no_grad()
def evaluate(dl, return_predictions=False):
    model.eval(); ys,ps=[],[]
    for inst,r,t,h,y in dl:
        logits,_=model(inst.to(DEVICE),r.to(DEVICE),t.to(DEVICE),h.to(DEVICE))
        ps.append(torch.sigmoid(logits).cpu().numpy()); ys.append(y.numpy())
    yt,yp=np.concatenate(ys),np.concatenate(ps)
    metrics = {"macro_roc_auc": nan_safe(yt,yp,"roc"), "macro_pr_auc": nan_safe(yt,yp,"pr")}
    return (metrics, yt, yp) if return_predictions else metrics

tr,va,te = loader("train", shuffle=True), loader("validation"), loader("test")
best_macro_map=float("-inf")
ckpt=CKPT_DIR/"baselines"/"descriptor_fusion"; ckpt.mkdir(parents=True, exist_ok=True)
hist=[]
PATIENCE = 3
MAX_EPOCHS, MAX_WALL_MINUTES = approved_run_limits(
    GPU_RUN, default_epochs=15,
    requested_wall_minutes=float(os.environ.get("MAX_GPU_RUN_MINUTES", "120")),
)
EVALUATE_TEST = os.environ.get("EVALUATE_TEST", "0") == "1"
epochs_without_improvement = 0
GPU_DEADLINE = GPU_RUN_STARTED + MAX_WALL_MINUTES * 60
TRAINING_DEADLINE = GPU_RUN_STARTED + MAX_WALL_MINUTES * 60 * 0.9
wall_cap_reached = False
if DEVICE.type == "cuda":
    _pi, _pr, _pt, _ph, _py = next(iter(tr))
    model.eval(); opt.zero_grad(set_to_none=True)
    _preflight_logits, _ = model(
        _pi.to(DEVICE), _pr.to(DEVICE), _pt.to(DEVICE), _ph.to(DEVICE)
    )
    _preflight_loss = crit(_preflight_logits, _py.to(DEVICE))
    _preflight_loss.backward(); opt.zero_grad(set_to_none=True)
    del _pi, _pr, _pt, _ph, _py, _preflight_logits, _preflight_loss
    torch.cuda.empty_cache()
    print("preflight backward: OK")
for epoch in range(1, MAX_EPOCHS + 1):
    model.train(); total=0
    for inst,r,t,h,y in tqdm(tr, leave=False):
        if time.perf_counter() >= TRAINING_DEADLINE:
            wall_cap_reached = True
            break
        inst,r,t,h,y=[a.to(DEVICE) for a in (inst,r,t,h,y)]
        opt.zero_grad(); logits,_=model(inst,r,t,h); loss=crit(logits,y); loss.backward(); opt.step()
        total += loss.item()*len(y)
    if wall_cap_reached:
        print(f"training-time reserve reached between batches: {MAX_WALL_MINUTES:.0f} minute total cap")
        break
    vm=evaluate(va)
    if not np.isfinite(vm["macro_pr_auc"]):
        raise RuntimeError("validation PR-AUC is undefined; fix label coverage before spending more GPU time")
    hist.append({"epoch":epoch,"loss":total/len(tr.dataset),**vm}); print(epoch, hist[-1])
    if vm["macro_pr_auc"]>best_macro_map:
        best_macro_map=vm["macro_pr_auc"]
        epochs_without_improvement = 0
        torch.save({
            "model":model.state_dict(),"fusion":FUSION,
            "best_macro_map":best_macro_map,"tags":TAG_NAMES,
            "training_config":{"max_epochs":MAX_EPOCHS,"patience":PATIENCE,
                               "max_wall_minutes":MAX_WALL_MINUTES,
                               "gpu_run":GPU_RUN},
        }, ckpt/f"best_{FUSION}.pt")
        print("  ✓", best_macro_map)
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= PATIENCE:
            print(f"early stop: no validation PR-AUC improvement for {PATIENCE} epochs")
            break
    if time.perf_counter() >= TRAINING_DEADLINE:
        print(f"wall-time cap reached: {MAX_WALL_MINUTES:.0f} minutes")
        break
if not (ckpt/f"best_{FUSION}.pt").is_file():
    write_gpu_termination_ledger(
        BASELINE_RESULTS_DIR/f"07_descriptor_fusion_{FUSION}_runtime.json", device=DEVICE,
        started_at=GPU_RUN_STARTED, record=GPU_RUN, reason="no_complete_validation_epoch",
        max_epochs=MAX_EPOCHS, max_wall_minutes=MAX_WALL_MINUTES,
    )
    raise RuntimeError("GPU cap reached before one complete validation epoch; no checkpoint was created")
state=torch.load(ckpt/f"best_{FUSION}.pt", map_location=DEVICE, weights_only=False)
model.load_state_dict(state["model"])
val_m,val_y,val_p=evaluate(va, return_predictions=True)
pd.DataFrame(hist).to_csv(BASELINE_RESULTS_DIR/f"07_descriptor_fusion_{FUSION}_history.csv", index=False)
elapsed_seconds = time.perf_counter() - GPU_RUN_STARTED
runtime = {
    "device":str(DEVICE),"epochs_completed":len(hist),
    "max_epochs":MAX_EPOCHS,"patience":PATIENCE,
    "max_wall_minutes":MAX_WALL_MINUTES,
    "evaluated_test":EVALUATE_TEST,
    "gpu_run":GPU_RUN,
    "wall_seconds":elapsed_seconds,
    "gpu_wall_hours":elapsed_seconds/3600 if DEVICE.type=="cuda" else 0.0,
}
(BASELINE_RESULTS_DIR/f"07_descriptor_fusion_{FUSION}_runtime.json").write_text(json.dumps(runtime, indent=2))
print("runtime", runtime)
pred_dir = RESULTS_DIR / "predictions"; pred_dir.mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    pred_dir / f"07_descriptor_fusion_{FUSION}_validation.npz",
    song_ids=np.asarray(va.dataset.df["song_id"].astype(str).to_numpy(), dtype=str),
    label_names=np.asarray(TAG_NAMES, dtype=str), targets=val_y, scores=val_p,
)
if EVALUATE_TEST:
    test_m,test_y,test_p=evaluate(te, return_predictions=True)
    print("FINAL TEST split-0", test_m)
    (BASELINE_RESULTS_DIR/f"07_descriptor_fusion_{FUSION}_test.json").write_text(json.dumps(test_m, indent=2))
    np.savez_compressed(
        pred_dir / f"07_descriptor_fusion_{FUSION}_test.npz",
        song_ids=np.asarray(te.dataset.df["song_id"].astype(str).to_numpy(), dtype=str),
        label_names=np.asarray(TAG_NAMES, dtype=str), targets=test_y, scores=test_p,
    )
else:
    print("Test evaluation skipped. Set EVALUATE_TEST=1 only for the selected final run.")
elapsed_seconds = time.perf_counter() - GPU_RUN_STARTED
runtime["wall_seconds"] = elapsed_seconds
runtime["gpu_wall_hours"] = elapsed_seconds/3600 if DEVICE.type=="cuda" else 0.0
(BASELINE_RESULTS_DIR/f"07_descriptor_fusion_{FUSION}_runtime.json").write_text(json.dumps(runtime, indent=2))
print("final runtime", runtime)
'''
        ),
    ],
)


write(
    "08_baseline_evaluation.ipynb",
    [
        md("# 08 — Baseline Evaluation (Colab + Drive)\n\nCompares recorded test metrics from the direct CNN and descriptor-fusion baselines. It does not claim to evaluate the proposed model."),
        md(COLAB_SETUP),
        md("## Mount Drive"),
        code(MOUNT),
        code(PATHS),
        code(
            r'''
result_files = sorted(BASELINE_RESULTS_DIR.glob("02_baseline_test.json")) + sorted(BASELINE_RESULTS_DIR.glob("07_descriptor_fusion_*_test.json"))
if not result_files:
    raise FileNotFoundError("Run notebooks 02 and 07 before comparing baseline metrics.")
rows = []
for f in result_files:
    rows.append({"model": f.stem, **json.loads(f.read_text())})
tbl = pd.DataFrame(rows)
tbl.to_csv(BASELINE_RESULTS_DIR/"08_core_comparison.csv", index=False)
print("Wrote", BASELINE_RESULTS_DIR/"08_core_comparison.csv")
tbl
'''
        ),
    ],
)


COLAB_09_EVAL = r'''
import matplotlib.pyplot as plt
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset

if not MANIFEST.exists():
    raise FileNotFoundError("Run 01 first")

CKPT_PATH = CKPT_DIR / "baselines" / "descriptor_fusion" / "best_attention.pt"
if not CKPT_PATH.exists() or CKPT_PATH.stat().st_size == 0:
    raise FileNotFoundError(
        "Notebook 09 refuses placeholder attention. Train notebook 07 first so this exists and is non-empty:\n"
        f"  {CKPT_PATH}\n"
        "07 must be trained AFTER notebook 04 writes AcousticBrainz rhythm features."
    )

rhythm_csv = FEAT_DIR / "rhythm" / "rhythm_song.csv"
if not rhythm_csv.exists():
    raise FileNotFoundError("Run notebook 04 first (AcousticBrainz rhythm).")
rhythm_src = pd.read_csv(rhythm_csv)
if "source" in rhythm_src.columns and (rhythm_src["source"] == "mel_proxy").any():
    raise RuntimeError("rhythm_song.csv still has mel_proxy rows — re-run notebook 04, then re-train 07.")

manifest = pd.read_csv(MANIFEST)
manifest["song_id"] = manifest["song_id"].astype(str).map(lambda s: normalize_track_id(s) or s)
E = np.load(FEAT_DIR / "instrument" / "instrument_embeddings.npy")
inst_ids = json.loads((FEAT_DIR / "instrument" / "song_ids.json").read_text())
inst_map = {normalize_track_id(s) or str(s): E[i] for i, s in enumerate(inst_ids)}


def load_feat(sub):
    p = FEAT_DIR / sub / f"{sub}_song.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p)
    df["song_id"] = df["song_id"].astype(str).map(lambda s: normalize_track_id(s) or s)
    return df.set_index("song_id")


rhythm, timbre, harmony = load_feat("rhythm"), load_feat("timbre"), load_feat("harmony")

required_harmony_meta = {"source", "schema_version", "target_variant"}
if required_harmony_meta - set(harmony.columns):
    raise RuntimeError("Harmony descriptor artifact predates the reviewed harmony schema; retrain notebook 07 after replacement.")
if harmony["source"].astype(str).str.contains("mel_proxy", case=False).any():
    raise RuntimeError("Mel-proxy harmony is invalid; notebook 09 refuses this checkpoint lineage.")

def ncols(df):
    metadata = {"source", "split", "schema_version", "target_variant", "extractor_version", "teacher_name", "teacher_version"}
    return [c for c in df.columns if c not in metadata and pd.api.types.is_numeric_dtype(df[c])]

r_cols, t_cols, h_cols = ncols(rhythm), ncols(timbre), ncols(harmony)
have = set(inst_map) & set(rhythm.index) & set(timbre.index) & set(harmony.index)
manifest = manifest[manifest["song_id"].isin(have)].copy()
test_ids = manifest.loc[manifest.split == "test", "song_id"].astype(str).head(12).tolist()
if not test_ids:
    raise RuntimeError("No overlapping test songs with all four concepts — check 03–06 then 07.")
print("test sample", test_ids)
print("using checkpoint", CKPT_PATH, "bytes=", CKPT_PATH.stat().st_size)

ids = manifest["song_id"].astype(str).tolist()
id_to_idx = {s: i for i, s in enumerate(ids)}
Y, TAG_NAMES, genre_available = load_split_multihot(ids, "genre", "genre")
if not genre_available.all():
    raise RuntimeError("explainability cohort contains songs without split-0 genre labels")
n_tags = len(TAG_NAMES)


class DS(Dataset):
    def __init__(self, song_ids):
        self.ids = list(song_ids)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        sid = self.ids[i]
        inst = inst_map[sid].astype(np.float32)
        r = rhythm.loc[sid, r_cols].astype(np.float32).fillna(0).values
        t = timbre.loc[sid, t_cols].astype(np.float32).fillna(0).values
        h = harmony.loc[sid, h_cols].astype(np.float32).fillna(0).values
        return torch.tensor(inst), torch.tensor(r), torch.tensor(t), torch.tensor(h), sid


class AttentionFusion(nn.Module):
    def __init__(self, d_i, d_r, d_t, d_h, token=64, fused=128, n_tags=87):
        super().__init__()
        self.p_i, self.p_r, self.p_t, self.p_h = (
            nn.Linear(d_i, token), nn.Linear(d_r, token), nn.Linear(d_t, token), nn.Linear(d_h, token)
        )
        self.attn = nn.MultiheadAttention(token, 1, batch_first=True)
        self.out = nn.Sequential(nn.Linear(token, fused), nn.ReLU(), nn.Dropout(0.2))
        self.head = nn.Linear(fused, n_tags)

    def forward(self, inst, r, t, h):
        tok = torch.stack([self.p_i(inst), self.p_r(r), self.p_t(t), self.p_h(h)], 1)
        o, w = self.attn(tok, tok, tok, need_weights=True)
        return self.head(self.out(o.mean(1))), w


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
n_tags = len(state.get("tags") or []) or n_tags
model = AttentionFusion(64, len(r_cols), len(t_cols), len(h_cols), n_tags=n_tags).to(DEVICE)
model.load_state_dict(state["model"])
model.eval()

concept_names = ["instrument", "rhythm", "timbre", "harmony"]
attn_rows = []
with torch.no_grad():
    for inst, r, t, h, sid in DataLoader(DS(test_ids), batch_size=8, num_workers=0):
        _, w = model(inst.to(DEVICE), r.to(DEVICE), t.to(DEVICE), h.to(DEVICE))
        # w: (B, 4, 4) — mean over query tokens → per-concept weight
        weights = w.mean(dim=1).cpu().numpy()
        for i, song in enumerate(sid):
            row = {"song_id": str(song)}
            row.update({c: float(weights[i, j]) for j, c in enumerate(concept_names)})
            attn_rows.append(row)

attn_df = pd.DataFrame(attn_rows)
attn_df.to_csv(BASELINE_RESULTS_DIR / "09_attention_weights_sample.csv", index=False)
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(concept_names, attn_df[concept_names].mean(0).values)
ax.set_title("Mean concept attention (descriptor-fusion baseline)")
ax.set_ylabel("mean attention")
fig.tight_layout()
fig.savefig(BASELINE_RESULTS_DIR / "09_mean_attention.png", dpi=150)
plt.show()
qual = []
for sid in test_ids[:5]:
    top = concept_names[int(attn_df.loc[attn_df.song_id == sid, concept_names].values.argmax())]
    qual.append({"song_id": sid, "audible_dominant_concept": "", "model_top_concept": top, "agree": "", "comment": ""})
pd.DataFrame(qual).to_csv(BASELINE_RESULTS_DIR / "09_qualitative_listening.csv", index=False)
print("Wrote real attention under", BASELINE_RESULTS_DIR)
attn_df
'''


write(
    "09_baseline_explainability.ipynb",
    [
        md("# 09 — Descriptor-Fusion Baseline Analysis (Colab + Drive)\n\nNeeds 01 + **trained notebook 07** (`checkpoints/baselines/descriptor_fusion/best_attention.pt`). GPU Off.\n\nExports the baseline's per-concept attention for comparison with the future proposed model."),
        md(COLAB_SETUP),
        code("""!pip install -q matplotlib tqdm"""),
        md("## Mount Drive"),
        code(MOUNT),
        code(PATHS),
        md("## Guard + descriptor-fusion baseline attention"),
        code(COLAB_09_EVAL),
    ],
)

print("Done", OUT)
