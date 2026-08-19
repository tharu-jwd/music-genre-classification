"""Generate the full staged Kaggle pipeline notebooks under notebooks/."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks"
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


def write(name: str, cells: list[dict]) -> None:
    path = OUT / name
    path.write_text(json.dumps(nb(cells), indent=1), encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)}")


KAGGLE_SETUP = """\
## Kaggle setup (every notebook)

1. **Settings → Internet → On** (right sidebar). Without this you get `Could not resolve host: github.com`.
2. **Add Data** (if this is not notebook `00` in the same session):
   - Attach the dataset you saved from notebook `00` (`mtg-instrument-cache`), **or**
   - Keep running inside the **same** Kaggle notebook after `00` (same `/kaggle/working`).
3. Each Kaggle notebook starts with an **empty** `/kaggle/working`. Files from a previous notebook are gone unless you attached them under `/kaggle/input`.
4. This bootstrap cell **auto-finds** files in `/kaggle/working` and `/kaggle/input`, copies a cache into working if needed, and **re-downloads annotations** if split TSVs are missing.
"""

SHARED_BOOTSTRAP = r'''
from pathlib import Path
import os, json, random, re, shutil, socket, urllib.request
import numpy as np
import pandas as pd

WORKING_ROOT = Path("/kaggle/working/MTG_Instrument")
INPUT_BASE = Path("/kaggle/input")
RAW_ANN = "https://raw.githubusercontent.com/MTG/mtg-jamendo-dataset/master/data"
NEEDED_ANN = [
    "splits/split-0/autotagging_genre-train.tsv",
    "splits/split-0/autotagging_genre-validation.tsv",
    "splits/split-0/autotagging_genre-test.tsv",
    "splits/split-0/autotagging_instrument-train.tsv",
    "splits/split-0/autotagging_instrument-validation.tsv",
    "splits/split-0/autotagging_instrument-test.tsv",
    "autotagging_genre.tsv",
    "autotagging_instrument.tsv",
]
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def check_internet(host: str = "github.com", port: int = 443, timeout: float = 5) -> bool:
    try:
        socket.create_connection((host, port), timeout=timeout).close()
        return True
    except OSError:
        return False


def normalize_track_id(raw) -> str | None:
    """MTG ids are 7-digit zero-padded (track_0000948 → 0000948)."""
    m = re.search(r"(\d+)", str(raw))
    if not m:
        return None
    return f"{int(m.group(1)):07d}"


def _find_file(name: str, bases: list[Path]) -> Path | None:
    for base in bases:
        if not base.exists():
            continue
        hits = list(base.rglob(name))
        if hits:
            return hits[0]
    return None


def discover_input_root() -> Path | None:
    """Find a previous notebook-00 output or MTG data folder under /kaggle/input."""
    if not INPUT_BASE.exists():
        return None
    for marker in [
        "song_manifest.csv",
        "autotagging_genre-train.tsv",
        "autotagging_genre.tsv",
        ".shard_00_done",
    ]:
        hit = _find_file(marker, [INPUT_BASE])
        if hit is None:
            continue
        if marker == "song_manifest.csv":
            return hit.parents[1]  # .../MTG_Instrument/dataset/song_manifest.csv
        if marker == "autotagging_genre-train.tsv":
            # .../annotations/splits/split-0/file  OR  .../data/splits/split-0/file
            p = hit
            for _ in range(6):
                if (p / "dataset").exists() or p.name in {"MTG_Instrument", "data"}:
                    return p if p.name != "data" else p
                p = p.parent
            return hit.parents[2]
        if marker == "autotagging_genre.tsv":
            parent = hit.parent
            if parent.name == "annotations":
                return parent.parent
            return parent  # MTG data/
        if marker == ".shard_00_done":
            return hit.parents[2]  # .../MTG_Instrument/dataset/logmel_songs/.shard
    for p in INPUT_BASE.rglob("MTG_Instrument"):
        if p.is_dir():
            return p
    return None


def copy_cache_into_working(src: Path) -> None:
    """/kaggle/input is read-only — copy into working so later cells can write."""
    WORKING_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Copying cache {src} → {WORKING_ROOT} (may take a few minutes)...")
    for item in src.iterdir():
        dest = WORKING_ROOT / item.name
        if dest.exists():
            continue
        if item.is_dir():
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)
    print("Copy done.")


def ensure_annotations(ann_dir: Path) -> Path:
    """Make sure split-0 TSVs exist; wget them if this is a fresh Kaggle session."""
    train = ann_dir / "splits" / "split-0" / "autotagging_genre-train.tsv"
    if train.exists():
        return ann_dir

    # maybe files are flat, or under /kaggle/input with a different layout
    hit = _find_file("autotagging_genre-train.tsv", [ann_dir, INPUT_BASE, Path("/kaggle/working")])
    if hit is not None:
        dest = ann_dir / "splits" / "split-0" / hit.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if hit.resolve() != dest.resolve():
            shutil.copy2(hit, dest)
        # copy sibling split files from the same folder
        for name in [
            "autotagging_genre-validation.tsv",
            "autotagging_genre-test.tsv",
            "autotagging_instrument-train.tsv",
            "autotagging_instrument-validation.tsv",
            "autotagging_instrument-test.tsv",
        ]:
            sib = hit.parent / name
            if sib.exists():
                shutil.copy2(sib, dest.parent / name)
        genre_full = _find_file("autotagging_genre.tsv", [hit.parents[2] if len(hit.parents) > 2 else hit.parent, INPUT_BASE])
        if genre_full:
            shutil.copy2(genre_full, ann_dir / "autotagging_genre.tsv")
        inst_full = _find_file("autotagging_instrument.tsv", [hit.parents[2] if len(hit.parents) > 2 else hit.parent, INPUT_BASE])
        if inst_full:
            shutil.copy2(inst_full, ann_dir / "autotagging_instrument.tsv")
        print("Recovered split files from", hit.parent)
        return ann_dir

    if not check_internet():
        raise FileNotFoundError(
            "Split TSVs not found and Internet is OFF.\n"
            "Do ONE of:\n"
            "  A) Settings → Internet → On, re-run this cell (auto-download)\n"
            "  B) Add Data → attach notebook-00 output dataset (mtg-instrument-cache)\n"
            "  C) Stay in the SAME Kaggle session after running notebook 00"
        )

    print("Split TSVs missing — downloading official MTG annotations...")
    n = 0
    for rel in NEEDED_ANN:
        dest = ann_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = f"{RAW_ANN}/{rel}"
        print("  wget", url)
        urllib.request.urlretrieve(url, dest)
        n += 1
    print(f"Downloaded {n} annotation files into {ann_dir}")
    return ann_dir


def load_split_ids(split: str, subset: str = "genre") -> set[str]:
    candidates = [
        ANN_DIR / "splits" / "split-0" / f"autotagging_{subset}-{split}.tsv",
        ANN_DIR / f"autotagging_{subset}-{split}.tsv",
        ANN_DIR / "splits" / "split-0" / f"{split}.tsv",
        ANN_DIR / f"{split}.tsv",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        found = _find_file(f"autotagging_{subset}-{split}.tsv", [ANN_DIR, INPUT_BASE, Path("/kaggle/working")])
        path = found
    if path is None:
        raise FileNotFoundError(
            f"No split file for {subset}/{split}.\n"
            "Re-run the bootstrap cell after enabling Internet, or attach notebook-00 output."
        )
    df = pd.read_csv(path, sep="\t")
    col = "TRACK_ID" if "TRACK_ID" in df.columns else df.columns[0]
    ids = set()
    for v in df[col].astype(str):
        tid = normalize_track_id(v)
        if tid:
            ids.add(tid)
    print(f"{split:12s}  {len(ids):6d} ids   ← {path}")
    return ids


ONLINE = check_internet()
print("Internet reachable:", ONLINE)

input_root = discover_input_root()
print("Discovered /kaggle/input cache:", input_root)

if input_root is not None and not (WORKING_ROOT / "annotations").exists() and not (WORKING_ROOT / "dataset" / "song_manifest.csv").exists():
    # If input looks like MTG_Instrument, copy it; if it looks like MTG data/, copy into annotations
    if (input_root / "dataset").exists() or (input_root / "annotations").exists():
        copy_cache_into_working(input_root)
    elif (input_root / "splits").exists() or (input_root / "autotagging_genre.tsv").exists():
        dest = WORKING_ROOT / "annotations"
        dest.mkdir(parents=True, exist_ok=True)
        for rel in NEEDED_ANN:
            s = input_root / rel
            if not s.exists():
                s = input_root / Path(rel).name
            if s.exists():
                d = dest / rel
                d.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(s, d)
                print("copied", d)

ROOT = WORKING_ROOT
ROOT.mkdir(parents=True, exist_ok=True)
MEL_DIR = ROOT / "dataset" / "logmel_songs"
ANN_DIR = ROOT / "annotations"
FEAT_DIR = ROOT / "features"
CKPT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"
MANIFEST = ROOT / "dataset" / "song_manifest.csv"

for p in [MEL_DIR, ANN_DIR, FEAT_DIR, CKPT_DIR, RESULTS_DIR, ROOT / "dataset"]:
    p.mkdir(parents=True, exist_ok=True)

ANN_DIR = ensure_annotations(ANN_DIR)

print("ROOT     =", ROOT)
print("MEL_DIR  =", MEL_DIR, "npy=", len(list(MEL_DIR.rglob('*.npy'))))
print("ANN_DIR  =", ANN_DIR)
print("split-0 train exists:", (ANN_DIR / "splits/split-0/autotagging_genre-train.tsv").exists())
print("MANIFEST =", MANIFEST, "exists=", MANIFEST.exists())
'''

INTRO_00 = """\
# 00 — Kaggle Data Download (MTG-Jamendo)

**Goal:** Download official split-0 annotations + log-mel shards into `/kaggle/working/MTG_Instrument`.

**After this notebook:** *Save Version → Save output* and turn it into a private dataset (`mtg-instrument-cache`). Attach that dataset in notebooks 01–09.
"""


write(
    "00_kaggle_data_download.ipynb",
    [
        md(INTRO_00),
        md(KAGGLE_SETUP),
        md("## Step 0 — Check Internet and install packages\n\nRun this first. If it prints `Internet reachable: False`, stop and enable Internet, then re-run."),
        code(
            """import socket

def check_internet(host="github.com", port=443, timeout=5):
    try:
        socket.create_connection((host, port), timeout=timeout).close()
        return True
    except OSError:
        return False

ONLINE = check_internet()
print("Internet reachable:", ONLINE)
if not ONLINE:
    print("\\n❌ Turn ON Internet: Settings (right sidebar) → Internet → On, then re-run.")
else:
    print("✓ Network OK")

!pip install -q tqdm"""
        ),
        md("## Step 1 — Shared paths\n\nCreates `/kaggle/working/MTG_Instrument/...`. If you already attached a cache dataset, it is copied into working automatically."),
        code(SHARED_BOOTSTRAP),
        md(
            """## Step 2 — Download / copy annotation TSVs

Official files (not `train.tsv`):

- `splits/split-0/autotagging_genre-{train,validation,test}.tsv`
- `splits/split-0/autotagging_instrument-{train,validation,test}.tsv`
- `autotagging_genre.tsv`, `autotagging_instrument.tsv`

The bootstrap cell already tries to wget these if they are missing. This cell **verifies** they exist and lists them."""
        ),
        code(
            r'''
print("Annotation files on disk:")
found = list(ANN_DIR.rglob("*.tsv"))
for p in sorted(found):
    print(f"  {p.relative_to(ANN_DIR)}  ({p.stat().st_size:,} bytes)")
assert (ANN_DIR / "splits" / "split-0" / "autotagging_genre-train.tsv").exists(), (
    "genre-train TSV still missing — enable Internet and re-run the bootstrap cell"
)
print("\\n✓ split-0 genre/instrument TSVs are present")
'''
        ),
        md(
            """## Step 3 — Download mel-spectrogram shards 00–02

These tars are large. Needs Internet to `cdn.freesound.org`.

If you already attached a cache that contains `dataset/logmel_songs/*.npy`, this cell skips shards that have a `.shard_XX_done` marker."""
        ),
        code(
            r'''
import subprocess

if not check_internet("cdn.freesound.org") and not check_internet():
    npy = list(MEL_DIR.rglob("*.npy"))
    if not npy:
        raise RuntimeError(
            "Internet is OFF and no .npy mels were found.\\n"
            "Enable Internet, or Add Data with extracted mels under dataset/logmel_songs."
        )
    print(f"Offline: using {len(npy)} existing .npy files")
else:
    SHARDS = [0, 1, 2]
    BASE_URL = "https://cdn.freesound.org/mtg-jamendo/raw_30s/melspecs"
    MEL_DIR.mkdir(parents=True, exist_ok=True)

    def download_shard(i: int):
        tar_name = f"raw_30s_melspecs-{i:02d}.tar"
        tar_path = MEL_DIR / tar_name
        url = f"{BASE_URL}/{tar_name}"
        marker = MEL_DIR / f".shard_{i:02d}_done"
        if marker.exists() and any(MEL_DIR.rglob("*.npy")):
            print(f"shard {i:02d} already done — skip")
            return
        if not tar_path.exists():
            print(f"Downloading {url} ...")
            subprocess.check_call(["wget", "-q", "-O", str(tar_path), url])
        print(f"Extracting {tar_name} ...")
        subprocess.check_call(["tar", "-xf", str(tar_path), "-C", str(MEL_DIR)])
        tar_path.unlink(missing_ok=True)
        marker.write_text("ok")
        print(f"shard {i:02d} ready")

    for i in SHARDS:
        download_shard(i)

npy_count = len(list(MEL_DIR.rglob("*.npy")))
print(f"Total .npy files under MEL_DIR: {npy_count}")
assert npy_count > 0, "No mel .npy found — check download / internet"
'''
        ),
        md("## Step 4 — Save a summary\n\n**Save Version → Save output** after this, then Add that dataset in notebook 01."),
        code(
            r'''
SHARDS = [0, 1, 2]
summary = {
    "root": str(ROOT),
    "mel_dir": str(MEL_DIR),
    "ann_dir": str(ANN_DIR),
    "n_npy": len(list(MEL_DIR.rglob("*.npy"))),
    "shards": SHARDS,
    "genre_train_exists": (ANN_DIR / "splits/split-0/autotagging_genre-train.tsv").exists(),
}
(RESULTS_DIR / "00_download_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
print("\\nNext: Save Version (with output) → then run 01_preprocessing.ipynb with that dataset attached.")
'''
        ),
    ],
)


write(
    "01_preprocessing.ipynb",
    [
        md(
            """# 01 — Preprocessing & Song Manifest

Build `dataset/song_manifest.csv`: every available mel file joined to official **split-0** train/val/test.

**This is the notebook that failed** if you started a *new* Kaggle session: `/kaggle/working` is empty, so `autotagging_genre-train.tsv` is gone. The bootstrap cell now **re-downloads** those TSVs (Internet ON) or copies them from **Add Data**."""
        ),
        md(KAGGLE_SETUP),
        md("## Step 0 — Bootstrap paths + recover split files\n\nIf `split-0 train exists: True` at the end, you are good. If False, enable Internet and re-run."),
        code(SHARED_BOOTSTRAP),
        md(
            """## Step 1 — Index every mel `.npy` on disk

MTG mels are nested (often `00/track.npy`). We keep a 7-digit `song_id` so it matches `TRACK_ID` in the TSV files."""
        ),
        code(
            r'''
def track_id_from_path(p: Path) -> str | None:
    return normalize_track_id(p.stem)

rows = []
for p in MEL_DIR.rglob("*.npy"):
    tid = track_id_from_path(p)
    if tid is None:
        continue
    rel = str(p.relative_to(ROOT)) if str(p).startswith(str(ROOT)) else str(p)
    rows.append({"song_id": tid, "mel_path": rel, "mel_abs": str(p), "nbytes": p.stat().st_size})

mel_df = pd.DataFrame(rows).drop_duplicates("song_id")
print("Unique songs with mel:", len(mel_df))
if mel_df.empty:
    raise FileNotFoundError(
        f"No .npy files under {MEL_DIR}.\\n"
        "Run notebook 00 in this session, or Add Data → attach mtg-instrument-cache."
    )
mel_df.head()
'''
        ),
        md(
            """## Step 2 — Load official split-0 IDs (genre subset)

Files used:

`annotations/splits/split-0/autotagging_genre-{train,validation,test}.tsv`

We also assert **no leakage**: train ∩ val ∩ test must all be empty."""
        ),
        code(
            r'''
train_ids = load_split_ids("train")
val_ids = load_split_ids("validation")
test_ids = load_split_ids("test")

assert train_ids.isdisjoint(val_ids), "train overlaps validation"
assert train_ids.isdisjoint(test_ids), "train overlaps test"
assert val_ids.isdisjoint(test_ids), "VALIDATION must not intersect TEST"
print("Split leakage check: OK")
'''
        ),
        md("## Step 3 — Join mels to splits and write the manifest\n\nSongs not in split-0 (because we only downloaded shards 00–02) are dropped as `unused`."),
        code(
            r'''
def split_of(sid: str) -> str:
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
MANIFEST.parent.mkdir(parents=True, exist_ok=True)
manifest.to_csv(MANIFEST, index=False)
print("Wrote", MANIFEST, "rows=", len(manifest))
manifest.head()
'''
        ),
        md("## Step 4 — Sanity-check one spectrogram shape"),
        code(
            r'''
sample = np.load(manifest.iloc[0]["mel_abs"])
print("example shape:", sample.shape, "dtype:", sample.dtype)
(RESULTS_DIR / "01_manifest_summary.json").write_text(json.dumps({
    "n_manifest": int(len(manifest)),
    "split_counts": manifest["split"].value_counts().to_dict(),
    "example_shape": list(sample.shape),
}, indent=2))
print("Next: 02_cnn_baseline.ipynb")
'''
        ),
    ],
)


write(
    "02_cnn_baseline.ipynb",
    [
        md(
            """# 02 — CNN Baseline (Genre Multi-label)

Train a compact CNN on log-mel for **multi-label genre**.

- Split: official **split-0** only (never a random split)
- Metrics: NaN-safe macro ROC-AUC and PR-AUC (undefined tags are **excluded**, not zeroed)
- Checkpoint: **best validation** macro PR-AUC (`best_macro_map` is updated inside the save branch)

Paper reference (full set): **0.7260 ROC-AUC / 0.1592 PR-AUC** — shard subset will differ."""
        ),
        md(KAGGLE_SETUP),
        md("## Step 0 — Packages + GPU\n\nEnable **GPU** (T4) in Settings for this notebook."),
        code("""!pip install -q scikit-learn tqdm"""),
        md("## Step 1 — Bootstrap paths (find manifest + annotations)"),
        code(SHARED_BOOTSTRAP),
        md("## Step 2 — Load manifest and genre multi-hot labels\n\nParses `autotagging_genre.tsv` (and split files) into a binary matrix `Y`."),
        code(
            r'''
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm.auto import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", DEVICE)

if not MANIFEST.exists():
    raise FileNotFoundError("song_manifest.csv missing — run notebook 01 first (same session or attach its output).")
manifest = pd.read_csv(MANIFEST)
manifest["song_id"] = manifest["song_id"].astype(str).map(lambda s: normalize_track_id(s) or s)
assert set(manifest["split"]) <= {"train", "validation", "test"}

def load_genre_multihot(song_ids: list[str]):
    candidates = [
        ANN_DIR / "autotagging_genre.tsv",
        ANN_DIR / "splits" / "split-0" / "autotagging_genre-train.tsv",
    ]
    candidates += list(ANN_DIR.rglob("*genre*.tsv"))
    tag_to_idx, rows = {}, {sid: set() for sid in song_ids}
    for path in candidates:
        if not path.exists():
            continue
        df = pd.read_csv(path, sep="\t")
        id_col = "TRACK_ID" if "TRACK_ID" in df.columns else df.columns[0]
        tag_col = "TAGS" if "TAGS" in df.columns else df.columns[-1]
        for _, r in df.iterrows():
            sid = normalize_track_id(r[id_col])
            if sid not in rows:
                continue
            raw = r[tag_col]
            if pd.isna(raw):
                continue
            for tag in str(raw).replace("|", "\t").split("\t"):
                leaf = tag.strip().split("/")[-1].split("---")[-1]
                if not leaf or leaf.lower() in {"nan", "none", "tags"}:
                    continue
                if leaf not in tag_to_idx:
                    tag_to_idx[leaf] = len(tag_to_idx)
                rows[sid].add(leaf)
        if tag_to_idx:
            print("Parsed genre tags from", path, "n_tags=", len(tag_to_idx))
            break
    if not tag_to_idx:
        raise RuntimeError("Could not parse genre TSV — re-run bootstrap / notebook 00")
    names = [None] * len(tag_to_idx)
    for t, i in tag_to_idx.items():
        names[i] = t
    Y = np.zeros((len(song_ids), len(names)), dtype=np.float32)
    id_to_row = {s: i for i, s in enumerate(song_ids)}
    for sid, tags in rows.items():
        i = id_to_row[sid]
        for t in tags:
            Y[i, tag_to_idx[t]] = 1.0
    return Y, names

song_ids = manifest["song_id"].astype(str).tolist()
Y, TAG_NAMES = load_genre_multihot(song_ids)
print("Y shape:", Y.shape, "positive rate:", float(Y.mean()))
(RESULTS_DIR / "genre_tags.json").write_text(json.dumps(TAG_NAMES, indent=2))
'''
        ),
        md("## Step 3 — Dataset / loaders (split-0 only)\n\nValidation is **never** merged with test."),
        code(
            r'''
class MelGenreDataset(Dataset):
    def __init__(self, df: pd.DataFrame, Y: np.ndarray, id_to_idx: dict, max_windows: int = 12):
        self.df = df.reset_index(drop=True)
        self.Y = Y
        self.id_to_idx = id_to_idx
        self.max_windows = max_windows

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        x = np.load(row["mel_abs"])
        if x.ndim == 2:
            x = x[None, ...]
        W = x.shape[0]
        if W >= self.max_windows:
            x = x[: self.max_windows]
        else:
            pad = np.zeros((self.max_windows - W, *x.shape[1:]), dtype=x.dtype)
            x = np.concatenate([x, pad], axis=0)
        x_mean = x.mean(axis=0, keepdims=True)
        y = self.Y[self.id_to_idx[str(row["song_id"])]]
        return torch.tensor(x_mean, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

id_to_idx = {s: i for i, s in enumerate(song_ids)}

def make_loader(split: str, bs: int = 16, shuffle=False):
    sub = manifest[manifest["split"] == split]
    assert set(sub["split"].unique()) == {split}, "split leakage"
    ds = MelGenreDataset(sub, Y, id_to_idx)
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=2, pin_memory=torch.cuda.is_available())

train_loader = make_loader("train", shuffle=True)
val_loader = make_loader("validation")
test_loader = make_loader("test")
print({s: int((manifest.split == s).sum()) for s in ["train", "validation", "test"]})
'''
        ),
        md("## Step 4 — Model"),
        code(
            r'''
class BaselineCNN(nn.Module):
    def __init__(self, n_tags: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, n_tags),
        )

    def forward(self, x):
        return self.head(self.features(x))

model = BaselineCNN(n_tags=Y.shape[1]).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()
print(model)
'''
        ),
        md("## Step 5 — Train; save **best val** checkpoint\n\n`best_macro_map` is assigned **inside** the `if val improves` branch (Stage 1 / baseline bug-fix)."),
        code(
            r'''
def nan_safe_macro_auc(y_true, y_prob, kind="roc"):
    scores = []
    for k in range(y_true.shape[1]):
        if y_true[:, k].sum() in (0, len(y_true)):
            continue
        try:
            if kind == "roc":
                scores.append(roc_auc_score(y_true[:, k], y_prob[:, k]))
            else:
                scores.append(average_precision_score(y_true[:, k], y_prob[:, k]))
        except ValueError:
            continue
    return float(np.mean(scores)) if scores else float("nan")


@torch.no_grad()
def evaluate(loader):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x = x.to(DEVICE)
        prob = torch.sigmoid(model(x)).cpu().numpy()
        ys.append(y.numpy())
        ps.append(prob)
    y_true, y_prob = np.concatenate(ys), np.concatenate(ps)
    return {"macro_roc_auc": nan_safe_macro_auc(y_true, y_prob, "roc"),
            "macro_pr_auc": nan_safe_macro_auc(y_true, y_prob, "pr")}


def train_one_epoch(loader):
    model.train()
    total = 0.0
    for x, y in tqdm(loader, leave=False):
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        opt.step()
        total += loss.item() * len(x)
    return total / len(loader.dataset)

EPOCHS = 10
best_macro_map = 0.0
ckpt_dir = CKPT_DIR / "baseline"
ckpt_dir.mkdir(parents=True, exist_ok=True)
history = []

for epoch in range(1, EPOCHS + 1):
    tr_loss = train_one_epoch(train_loader)
    val_m = evaluate(val_loader)
    history.append({"epoch": epoch, "train_loss": tr_loss, **val_m})
    print(f"epoch {epoch}: loss={tr_loss:.4f} val_roc={val_m['macro_roc_auc']:.4f} val_pr={val_m['macro_pr_auc']:.4f}")
    if val_m["macro_pr_auc"] > best_macro_map:
        best_macro_map = val_m["macro_pr_auc"]
        torch.save({"model": model.state_dict(), "tags": TAG_NAMES, "best_macro_map": best_macro_map, "epoch": epoch},
                   ckpt_dir / "best.pt")
        print("  ✓ saved best checkpoint @", best_macro_map)

state = torch.load(ckpt_dir / "best.pt", map_location=DEVICE, weights_only=False)
model.load_state_dict(state["model"])
test_m = evaluate(test_loader)
print("TEST (split-0 only):", test_m)
pd.DataFrame(history).to_csv(RESULTS_DIR / "02_baseline_history.csv", index=False)
(RESULTS_DIR / "02_baseline_test.json").write_text(json.dumps(test_m, indent=2))
'''
        ),
    ],
)


write(
    "03_instrument_embedding.ipynb",
    [
        md(
            """# 03 — Stage 1: Instrument Embedding (MIL + Attention)

Learn a **64-d song-level instrument embedding** with attention pooling over 15s windows.

**Bug-fix checklist (must hold here):**
1. `best_macro_map` updated inside the checkpoint-save branch
2. Val loader uses validation rows only (test never unioned in)
3. Mel paths resolve under `MEL_DIR` / `ROOT` only"""
        ),
        md(KAGGLE_SETUP),
        md("## Step 0 — Packages"),
        code("""!pip install -q scikit-learn tqdm"""),
        md("## Step 1 — Bootstrap paths"),
        code(SHARED_BOOTSTRAP),
        md("## Step 2 — Instrument multi-hot labels + manifest"),
        code(
            r'''
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import average_precision_score
from tqdm.auto import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not MANIFEST.exists():
    raise FileNotFoundError("Run notebook 01 first.")
manifest = pd.read_csv(MANIFEST)
manifest["song_id"] = manifest["song_id"].astype(str).map(lambda s: normalize_track_id(s) or s)
EMBED_DIM, MAX_WINDOWS = 64, 12
song_ids = manifest["song_id"].astype(str).tolist()

tag_to_idx, rows = {}, {s: set() for s in song_ids}
for path in [ANN_DIR / "autotagging_instrument.tsv", *ANN_DIR.rglob("*instrument*.tsv")]:
    if not Path(path).exists():
        continue
    df = pd.read_csv(path, sep="\t")
    id_col = "TRACK_ID" if "TRACK_ID" in df.columns else df.columns[0]
    tag_col = "TAGS" if "TAGS" in df.columns else df.columns[-1]
    for _, r in df.iterrows():
        sid = normalize_track_id(r[id_col])
        if sid not in rows:
            continue
        raw = r[tag_col]
        if pd.isna(raw):
            continue
        for tag in str(raw).replace("|", "\t").split("\t"):
            leaf = tag.strip().split("/")[-1].split("---")[-1]
            if not leaf or leaf.lower() in {"nan", "none", "tags"}:
                continue
            if leaf not in tag_to_idx:
                tag_to_idx[leaf] = len(tag_to_idx)
            rows[sid].add(leaf)
    if tag_to_idx:
        print("instruments from", path, "n=", len(tag_to_idx))
        break
assert tag_to_idx, "No instrument tags — re-run bootstrap / notebook 00"
INST_NAMES = [None] * len(tag_to_idx)
for t, i in tag_to_idx.items():
    INST_NAMES[i] = t
Y = np.zeros((len(song_ids), len(INST_NAMES)), np.float32)
id_to_idx = {s: i for i, s in enumerate(song_ids)}
for sid, tags in rows.items():
    i = id_to_idx[sid]
    for t in tags:
        Y[i, tag_to_idx[t]] = 1.0
print("Y_inst", Y.shape, "pos rate", float(Y.mean()))
'''
        ),
        md("## Step 3 — Window MIL dataset (path fallback stays under MEL_DIR)"),
        code(
            r'''
def resolve_stacked_mel_path(mel_abs: str) -> Path:
    p = Path(mel_abs)
    if p.exists():
        return p
    tid = normalize_track_id(p.stem)
    hits = list(MEL_DIR.rglob(f"*{tid}*.npy")) if tid else []
    if not hits:
        raise FileNotFoundError(f"mel not under MEL_DIR for {mel_abs}")
    return hits[0]


class WindowMILDataset(Dataset):
    def __init__(self, df, Y, id_to_idx):
        self.df = df.reset_index(drop=True)
        self.Y, self.id_to_idx = Y, id_to_idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        x = np.load(resolve_stacked_mel_path(row["mel_abs"]))
        if x.ndim == 2:
            x = x[None, ...]
        W = x.shape[0]
        if W >= MAX_WINDOWS:
            x = x[:MAX_WINDOWS]
            mask = np.ones(MAX_WINDOWS, np.float32)
        else:
            pad = np.zeros((MAX_WINDOWS - W, *x.shape[1:]), x.dtype)
            x = np.concatenate([x, pad], 0)
            mask = np.array([1] * W + [0] * (MAX_WINDOWS - W), np.float32)
        y = self.Y[self.id_to_idx[str(row["song_id"])]]
        return (
            torch.tensor(x[:, None, :, :], dtype=torch.float32),
            torch.tensor(mask),
            torch.tensor(y, dtype=torch.float32),
            str(row["song_id"]),
        )


def make_loader(split, bs=8, shuffle=False):
    sub = manifest[manifest["split"] == split]
    assert set(sub["split"].unique()) == {split}
    return DataLoader(WindowMILDataset(sub, Y, id_to_idx), batch_size=bs, shuffle=shuffle, num_workers=2)
'''
        ),
        md("## Step 4 — Encoder + attention pool + instrument head"),
        code(
            r'''
class WindowEncoder(nn.Module):
    def __init__(self, emb=EMBED_DIM):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(64, emb)

    def forward(self, x):
        B, W, C, M, T = x.shape
        h = self.cnn(x.reshape(B * W, C, M, T)).flatten(1)
        return self.proj(h).reshape(B, W, -1)


class AttnPool(nn.Module):
    def __init__(self, emb=EMBED_DIM):
        super().__init__()
        self.score = nn.Linear(emb, 1)

    def forward(self, H, mask):
        logits = self.score(H).squeeze(-1).masked_fill(mask < 0.5, -1e9)
        w = torch.softmax(logits, dim=-1)
        z = torch.sum(H * w.unsqueeze(-1), dim=1)
        return z, w


class Stage1Model(nn.Module):
    def __init__(self, n_tags, emb=EMBED_DIM):
        super().__init__()
        self.enc = WindowEncoder(emb)
        self.pool = AttnPool(emb)
        self.head = nn.Linear(emb, n_tags)

    def forward(self, x, mask):
        H = self.enc(x)
        z, attn = self.pool(H, mask)
        return self.head(z), z, attn

model = Stage1Model(n_tags=Y.shape[1]).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()
'''
        ),
        md("## Step 5 — Train + export 64-d embeddings for every manifest song"),
        code(
            r'''
def macro_map(y_true, y_prob):
    scores = []
    for k in range(y_true.shape[1]):
        if y_true[:, k].sum() in (0, len(y_true)):
            continue
        try:
            scores.append(average_precision_score(y_true[:, k], y_prob[:, k]))
        except ValueError:
            continue
    return float(np.mean(scores)) if scores else float("nan")


@torch.no_grad()
def eval_split(loader):
    model.eval()
    ys, ps = [], []
    for x, mask, y, _ in loader:
        logits, _, _ = model(x.to(DEVICE), mask.to(DEVICE))
        ps.append(torch.sigmoid(logits).cpu().numpy())
        ys.append(y.numpy())
    return macro_map(np.concatenate(ys), np.concatenate(ps))

train_loader, val_loader, test_loader = make_loader("train", shuffle=True), make_loader("validation"), make_loader("test")
EPOCHS = 8
best_macro_map = 0.0
ckpt_dir = CKPT_DIR / "stage1"
ckpt_dir.mkdir(parents=True, exist_ok=True)

for epoch in range(1, EPOCHS + 1):
    model.train()
    total = 0.0
    for x, mask, y, _ in tqdm(train_loader, leave=False):
        x, mask, y = x.to(DEVICE), mask.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        logits, _, _ = model(x, mask)
        loss = criterion(logits, y)
        loss.backward()
        opt.step()
        total += loss.item() * len(x)
    val_map = eval_split(val_loader)
    print(f"epoch {epoch}: loss={total/len(train_loader.dataset):.4f} val_macro_map={val_map:.4f}")
    if val_map > best_macro_map:
        best_macro_map = val_map
        torch.save({"model": model.state_dict(), "best_macro_map": best_macro_map, "epoch": epoch, "tags": INST_NAMES},
                   ckpt_dir / "best.pt")
        print("  ✓ checkpoint", best_macro_map)

state = torch.load(ckpt_dir / "best.pt", map_location=DEVICE, weights_only=False)
model.load_state_dict(state["model"])
model.eval()
all_loader = DataLoader(WindowMILDataset(manifest, Y, id_to_idx), batch_size=8)
embeds, ids = [], []
with torch.no_grad():
    for x, mask, y, sid in tqdm(all_loader):
        _, z, _ = model(x.to(DEVICE), mask.to(DEVICE))
        embeds.append(z.cpu().numpy())
        ids.extend(list(sid))
E = np.concatenate(embeds, 0)
out = FEAT_DIR / "instrument"
out.mkdir(parents=True, exist_ok=True)
np.save(out / "instrument_embeddings.npy", E)
(out / "song_ids.json").write_text(json.dumps(ids))
print("saved", E.shape, "test macro_map", eval_split(test_loader))
'''
        ),
    ],
)


def feature_nb(num: str, title: str, kind: str, extract_fn: str, out_sub: str) -> None:
    body = '''
import librosa
from tqdm.auto import tqdm

if not MANIFEST.exists():
    raise FileNotFoundError("Run notebook 01 first.")
manifest = pd.read_csv(MANIFEST)
manifest["song_id"] = manifest["song_id"].astype(str).map(lambda s: normalize_track_id(s) or s)
AUDIO_ROOT = Path("/kaggle/input/mtg-jamendo-audio")
WINDOW_SEC = 15.0
SR = 22050

__EXTRACT_FN__

def features_from_mel(mel_path: Path) -> dict:
    S = np.load(mel_path)
    if S.ndim == 3:
        S = S.mean(0)
    return mel_proxy_features(S)

rows = []
for _, rec in tqdm(manifest.iterrows(), total=len(manifest)):
    sid = str(rec["song_id"])
    audio_candidate = None
    if "audio_path" in manifest.columns and pd.notna(rec.get("audio_path", None)):
        audio_candidate = Path(rec["audio_path"])
    elif AUDIO_ROOT.exists():
        hits = list(AUDIO_ROOT.rglob(f"*{sid}*.mp3"))
        audio_candidate = hits[0] if hits else None
    try:
        if audio_candidate and Path(audio_candidate).exists():
            feat = extract_from_audio(Path(audio_candidate))
            src = "audio"
        else:
            feat = features_from_mel(Path(rec["mel_abs"]))
            src = "mel_proxy"
    except Exception as e:
        print("fail", sid, e)
        continue
    feat.update({"song_id": sid, "source": src, "split": rec["split"]})
    rows.append(feat)

df = pd.DataFrame(rows)
out = FEAT_DIR / "__OUT_SUB__"
out.mkdir(parents=True, exist_ok=True)
df.to_csv(out / "__OUT_SUB___song.csv", index=False)
print(df.head())
print("wrote", out, "n=", len(df))
'''.replace("__EXTRACT_FN__", extract_fn).replace("__OUT_SUB__", out_sub)

    write(
        f"{num}_{kind}_features.ipynb",
        [
            md(
                f"""# {num} — {title}

Extract **{kind}** features per song (`song_id` aligned with the Stage 1 manifest).

On Kaggle, raw MP3s are optional. Default path: **approximate from log-mel** already on disk (Phase 2 practical path). Attach an audio dataset under `/kaggle/input/mtg-jamendo-audio` to use real librosa audio features."""
            ),
            md(KAGGLE_SETUP),
            md("## Step 0 — Packages (`librosa` is slow on CPU; that is OK for this notebook)"),
            code("""!pip install -q librosa soundfile tqdm"""),
            md("## Step 1 — Bootstrap paths"),
            code(SHARED_BOOTSTRAP),
            md(f"## Step 2 — Extract {kind} and write `features/{out_sub}/{out_sub}_song.csv`"),
            code(body),
        ],
    )


extract_rhythm = r'''
def extract_from_audio(path: Path) -> dict:
    y, sr = librosa.load(path, sr=SR, mono=True, duration=60)
    hop = int(WINDOW_SEC * sr)
    vals = []
    for start in range(0, max(len(y) - hop, 0) + 1, hop):
        yw = y[start:start + hop]
        if len(yw) < hop // 2:
            break
        onset_env = librosa.onset.onset_strength(y=yw, sr=sr)
        tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        bt = librosa.frames_to_time(beats, sr=sr) if len(beats) else np.array([])
        intervals = np.diff(bt) if len(bt) > 1 else np.array([np.nan])
        vals.append(dict(
            tempo=float(np.atleast_1d(tempo)[0]),
            beat_strength_mean=float(np.mean(onset_env)),
            onset_density=float(len(librosa.onset.onset_detect(y=yw, sr=sr)) / WINDOW_SEC),
            beat_interval_mean=float(np.nanmean(intervals)),
            beat_interval_std=float(np.nanstd(intervals)),
        ))
    return pd.DataFrame(vals).mean(numeric_only=True).to_dict() if vals else {}

def mel_proxy_features(S: np.ndarray) -> dict:
    env = S.mean(axis=0)
    env = (env - env.mean()) / (env.std() + 1e-6)
    return dict(
        tempo=float(60.0),
        beat_strength_mean=float(np.mean(np.abs(env))),
        onset_density=float(np.mean(env > 1.0)),
        beat_interval_mean=float("nan"),
        beat_interval_std=float("nan"),
    )
'''

extract_timbre = r'''
def extract_from_audio(path: Path) -> dict:
    y, sr = librosa.load(path, sr=SR, mono=True, duration=60)
    hop = int(WINDOW_SEC * sr)
    vals = []
    for start in range(0, max(len(y) - hop, 0) + 1, hop):
        yw = y[start:start + hop]
        if len(yw) < hop // 2:
            break
        st = np.abs(librosa.stft(yw))
        vals.append(dict(
            spectral_centroid_mean=float(np.mean(librosa.feature.spectral_centroid(S=st, sr=sr))),
            spectral_bandwidth_mean=float(np.mean(librosa.feature.spectral_bandwidth(S=st, sr=sr))),
            spectral_contrast_mean=float(np.mean(librosa.feature.spectral_contrast(S=st, sr=sr))),
            spectral_flatness_mean=float(np.mean(librosa.feature.spectral_flatness(S=st))),
            rms_mean=float(np.mean(librosa.feature.rms(y=yw))),
            spectral_flux_mean=float(np.mean(librosa.onset.onset_strength(y=yw, sr=sr))),
        ))
    return pd.DataFrame(vals).mean(numeric_only=True).to_dict() if vals else {}

def mel_proxy_features(S: np.ndarray) -> dict:
    freqs = np.linspace(0, 1, S.shape[0])[:, None]
    centroid = float(((S * freqs).sum() / (S.sum() + 1e-6)))
    return dict(
        spectral_centroid_mean=centroid,
        spectral_bandwidth_mean=float(np.std(S.mean(1))),
        spectral_contrast_mean=float(S.max(0).mean() - S.min(0).mean()),
        spectral_flatness_mean=float(np.exp(np.mean(np.log(S + 1e-6))) / (np.mean(S) + 1e-6)),
        rms_mean=float(np.sqrt(np.mean(S ** 2))),
        spectral_flux_mean=float(np.mean(np.abs(np.diff(S.mean(0))))),
    )
'''

extract_harmony = r'''
def extract_from_audio(path: Path) -> dict:
    y, sr = librosa.load(path, sr=SR, mono=True, duration=60)
    hop = int(WINDOW_SEC * sr)
    vals = []
    for start in range(0, max(len(y) - hop, 0) + 1, hop):
        yw = y[start:start + hop]
        if len(yw) < hop // 2:
            break
        chroma = librosa.feature.chroma_stft(y=yw, sr=sr)
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(yw), sr=sr)
        row = {f"chroma_{i}_mean": float(np.mean(chroma[i])) for i in range(chroma.shape[0])}
        row.update({f"tonnetz_{i}_mean": float(np.mean(tonnetz[i])) for i in range(tonnetz.shape[0])})
        vals.append(row)
    return pd.DataFrame(vals).mean(numeric_only=True).to_dict() if vals else {}

def mel_proxy_features(S: np.ndarray) -> dict:
    bands = np.array_split(S, 12, axis=0)
    chroma = np.stack([b.mean() for b in bands])
    chroma = chroma / (chroma.sum() + 1e-6)
    row = {f"chroma_{i}_mean": float(chroma[i]) for i in range(12)}
    for i in range(6):
        row[f"tonnetz_{i}_mean"] = float(np.dot(chroma, np.cos(2 * np.pi * (i + 1) * np.arange(12) / 12)))
    return row
'''

feature_nb("04", "Rhythm Feature Extraction", "rhythm", extract_rhythm, "rhythm")
feature_nb("05", "Timbre Feature Extraction", "timbre", extract_timbre, "timbre")
feature_nb("06", "Harmony Feature Extraction", "harmony", extract_harmony, "harmony")


write(
    "07_fusion_genre_classifier.ipynb",
    [
        md(
            """# 07 — Stage 2: Fusion + Multi-label Genre Classifier

Needs: Stage 1 embeddings + rhythm/timbre/harmony CSVs.

- Fusion A: concat → linear
- Fusion B: single-head attention over the 4 concept tokens
- Head: 87-ish genre tags, BCE with logits
- Report **split-0 test** only"""
        ),
        md(KAGGLE_SETUP),
        md("## Step 0 — Packages (enable GPU)"),
        code("""!pip install -q scikit-learn tqdm"""),
        md("## Step 1 — Bootstrap paths"),
        code(SHARED_BOOTSTRAP),
        md("## Step 2 — Load concept features + genre labels"),
        code(
            r'''
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm.auto import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not MANIFEST.exists():
    raise FileNotFoundError("Run notebooks 01 then 03–06 first.")
manifest = pd.read_csv(MANIFEST)
manifest["song_id"] = manifest["song_id"].astype(str).map(lambda s: normalize_track_id(s) or s)

inst_dir = FEAT_DIR / "instrument"
if not (inst_dir / "instrument_embeddings.npy").exists():
    raise FileNotFoundError("Missing Stage 1 embeddings — run notebook 03.")
E = np.load(inst_dir / "instrument_embeddings.npy")
inst_ids = json.loads((inst_dir / "song_ids.json").read_text())
inst_map = {normalize_track_id(s) or str(s): E[i] for i, s in enumerate(inst_ids)}

def load_feat(sub):
    p = FEAT_DIR / sub / f"{sub}_song.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p} — run the matching 04/05/06 notebook.")
    df = pd.read_csv(p)
    df["song_id"] = df["song_id"].astype(str).map(lambda s: normalize_track_id(s) or s)
    return df.set_index("song_id")

rhythm, timbre, harmony = load_feat("rhythm"), load_feat("timbre"), load_feat("harmony")

def num_cols(df):
    return [c for c in df.columns if c not in ("song_id", "source", "split") and pd.api.types.is_numeric_dtype(df[c])]

r_cols, t_cols, h_cols = num_cols(rhythm), num_cols(timbre), num_cols(harmony)
ids = manifest["song_id"].astype(str).tolist()
id_to_idx = {s: i for i, s in enumerate(ids)}

# genre Y
candidates = [ANN_DIR / "autotagging_genre.tsv", *ANN_DIR.rglob("*genre*.tsv")]
tag_to_idx, rows = {}, {s: set() for s in ids}
for path in candidates:
    if not Path(path).exists():
        continue
    df = pd.read_csv(path, sep="\t")
    id_col = "TRACK_ID" if "TRACK_ID" in df.columns else df.columns[0]
    tag_col = "TAGS" if "TAGS" in df.columns else df.columns[-1]
    for _, r in df.iterrows():
        sid = normalize_track_id(r[id_col])
        if sid not in rows:
            continue
        raw = r[tag_col]
        if pd.isna(raw):
            continue
        for tag in str(raw).replace("|", "\t").split("\t"):
            leaf = tag.strip().split("/")[-1].split("---")[-1]
            if not leaf or leaf.lower() in {"nan", "tags"}:
                continue
            tag_to_idx.setdefault(leaf, len(tag_to_idx))
            rows[sid].add(leaf)
    if tag_to_idx:
        print("genre tags from", path, len(tag_to_idx))
        break
TAG_NAMES = [None] * len(tag_to_idx)
for t, i in tag_to_idx.items():
    TAG_NAMES[i] = t
Y = np.zeros((len(ids), len(TAG_NAMES)), np.float32)
for i, sid in enumerate(ids):
    for t in rows[sid]:
        Y[i, tag_to_idx[t]] = 1.0
print("Y", Y.shape, "inst/rhythm/timbre/harmony dims", 64, len(r_cols), len(t_cols), len(h_cols))
'''
        ),
        md("## Step 3 — Fusion models (set `FUSION = \"attention\"` or `\"linear\"`)"),
        code(
            r'''
class ConceptDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        sid = str(self.df.iloc[i]["song_id"])
        inst = inst_map[sid].astype(np.float32)
        r = rhythm.loc[sid, r_cols].astype(np.float32).fillna(0).values if sid in rhythm.index else np.zeros(len(r_cols), np.float32)
        t = timbre.loc[sid, t_cols].astype(np.float32).fillna(0).values if sid in timbre.index else np.zeros(len(t_cols), np.float32)
        h = harmony.loc[sid, h_cols].astype(np.float32).fillna(0).values if sid in harmony.index else np.zeros(len(h_cols), np.float32)
        y = Y[id_to_idx[sid]]
        return torch.tensor(inst), torch.tensor(r), torch.tensor(t), torch.tensor(h), torch.tensor(y)

def loader(split, bs=32, shuffle=False):
    sub = manifest[manifest["split"] == split]
    assert set(sub["split"].unique()) == {split}
    return DataLoader(ConceptDataset(sub), batch_size=bs, shuffle=shuffle)

class LinearFusion(nn.Module):
    def __init__(self, d_inst, d_r, d_t, d_h, fused=128, n_tags=87):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(d_inst + d_r + d_t + d_h, fused), nn.ReLU(), nn.Dropout(0.2))
        self.head = nn.Linear(fused, n_tags)

    def forward(self, inst, r, t, h):
        return self.head(self.proj(torch.cat([inst, r, t, h], -1))), None

class AttentionFusion(nn.Module):
    def __init__(self, d_inst, d_r, d_t, d_h, token=64, fused=128, n_tags=87):
        super().__init__()
        self.p_i, self.p_r, self.p_t, self.p_h = nn.Linear(d_inst, token), nn.Linear(d_r, token), nn.Linear(d_t, token), nn.Linear(d_h, token)
        self.attn = nn.MultiheadAttention(token, 1, batch_first=True)
        self.out = nn.Sequential(nn.Linear(token, fused), nn.ReLU(), nn.Dropout(0.2))
        self.head = nn.Linear(fused, n_tags)

    def forward(self, inst, r, t, h):
        tokens = torch.stack([self.p_i(inst), self.p_r(r), self.p_t(t), self.p_h(h)], 1)
        attn_out, w = self.attn(tokens, tokens, tokens, need_weights=True)
        return self.head(self.out(attn_out.mean(1))), w

FUSION = "attention"
dims = (64, len(r_cols), len(t_cols), len(h_cols))
Model = AttentionFusion if FUSION == "attention" else LinearFusion
model = Model(*dims, n_tags=Y.shape[1]).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
crit = nn.BCEWithLogitsLoss()
print("fusion", FUSION, "dims", dims)
'''
        ),
        md("## Step 4 — Train; keep best val PR-AUC; evaluate split-0 **test**"),
        code(
            r'''
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
def evaluate(dl):
    model.eval()
    ys, ps = [], []
    for inst, r, t, h, y in dl:
        inst, r, t, h = inst.to(DEVICE), r.to(DEVICE), t.to(DEVICE), h.to(DEVICE)
        logits, _ = model(inst, r, t, h)
        ps.append(torch.sigmoid(logits).cpu().numpy())
        ys.append(y.numpy())
    yt, yp = np.concatenate(ys), np.concatenate(ps)
    return {"macro_roc_auc": nan_safe(yt, yp, "roc"), "macro_pr_auc": nan_safe(yt, yp, "pr")}

train_dl, val_dl, test_dl = loader("train", shuffle=True), loader("validation"), loader("test")
best_macro_map = 0.0
ckpt = CKPT_DIR / "stage2"
ckpt.mkdir(parents=True, exist_ok=True)
hist = []
for epoch in range(1, 16):
    model.train()
    total = 0
    for inst, r, t, h, y in tqdm(train_dl, leave=False):
        inst, r, t, h, y = inst.to(DEVICE), r.to(DEVICE), t.to(DEVICE), h.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        logits, _ = model(inst, r, t, h)
        loss = crit(logits, y)
        loss.backward()
        opt.step()
        total += loss.item() * len(y)
    vm = evaluate(val_dl)
    hist.append({"epoch": epoch, "loss": total / len(train_dl.dataset), **vm})
    print(epoch, hist[-1])
    if vm["macro_pr_auc"] > best_macro_map:
        best_macro_map = vm["macro_pr_auc"]
        torch.save({"model": model.state_dict(), "fusion": FUSION, "best_macro_map": best_macro_map, "tags": TAG_NAMES}, ckpt / f"best_{FUSION}.pt")
        print("  ✓ saved", best_macro_map)

state = torch.load(ckpt / f"best_{FUSION}.pt", map_location=DEVICE, weights_only=False)
model.load_state_dict(state["model"])
test_m = evaluate(test_dl)
print("TEST split-0", test_m)
pd.DataFrame(hist).to_csv(RESULTS_DIR / f"07_stage2_{FUSION}_history.csv", index=False)
(RESULTS_DIR / f"07_stage2_{FUSION}_test.json").write_text(json.dumps(test_m, indent=2))
'''
        ),
    ],
)


write(
    "08_ablations_and_tuning.ipynb",
    [
        md(
            """# 08 — Ablations, Tuning & Computational Analysis

1. Stage 2 vs CNN baseline (0.7260 / 0.1592 paper ref)
2. Concept-count: instrument → +rhythm → +timbre → full four
3. Fusion-type: linear vs attention
4. LR × batch-size sweep plan
5. Params / latency"""
        ),
        md(KAGGLE_SETUP),
        md("## Step 0 — Packages"),
        code("""!pip install -q scikit-learn tqdm"""),
        md("## Step 1 — Bootstrap paths"),
        code(SHARED_BOOTSTRAP),
        md("## Step 2 — Collect metrics already written by notebooks 02 and 07"),
        code(
            r'''
baseline_ref = {"macro_roc_auc": 0.7260, "macro_pr_auc": 0.1592}
rows = [{"model": "CNN baseline (paper ref, full set)", **baseline_ref}]
for f in sorted(RESULTS_DIR.glob("02_baseline_test.json")) + sorted(RESULTS_DIR.glob("07_stage2_*_test.json")):
    rows.append({"model": f.stem, **json.loads(f.read_text())})
ablation_table = pd.DataFrame(rows)
ablation_table.to_csv(RESULTS_DIR / "08_core_comparison.csv", index=False)
ablation_table
'''
        ),
        md("## Step 3 — Parameter count + dummy latency (replace with real Stage 2 model when wired)"),
        code(
            r'''
import time, torch, torch.nn as nn

class Tiny(nn.Module):
    def __init__(self, d, n=87):
        super().__init__()
        self.fc = nn.Linear(d, n)

    def forward(self, x):
        return self.fc(x)

param_rows = []
for name, dims in [("full_concat", 64 + 5 + 6 + 18), ("inst64", 64)]:
    m = Tiny(dims)
    param_rows.append({"config": name, "params": sum(p.numel() for p in m.parameters())})

m = Tiny(64 + 5 + 6 + 18)
x = torch.randn(32, 64 + 5 + 6 + 18)
t0 = time.time()
with torch.no_grad():
    for _ in range(50):
        _ = m(x)
latency_ms = (time.time() - t0) / 50 * 1000
pd.DataFrame(param_rows).assign(batch_infer_ms=latency_ms).to_csv(RESULTS_DIR / "08_compute.csv", index=False)
print("wrote 08_compute.csv infer_ms", latency_ms)
'''
        ),
        md("## Step 4 — Hyperparameter sweep **plan** (fill after plugging the Stage 2 train loop)"),
        code(
            r'''
sweep = [{"lr": lr, "batch_size": bs, "status": "todo — reuse notebook 07 train loop"} for lr in [1e-3, 3e-4] for bs in [16, 32, 64]]
pd.DataFrame(sweep).to_csv(RESULTS_DIR / "08_sweep_plan.csv", index=False)
pd.DataFrame(sweep)
'''
        ),
    ],
)


write(
    "09_explainability_eval.ipynb",
    [
        md(
            """# 09 — Explainability Evaluation

On held-out **split-0 test** tracks:

1. Attention over {Instrument, Rhythm, Timbre, Harmony}
2. Occlusion Δ macro PR when a concept is zeroed
3. Qualitative listening table for the paper"""
        ),
        md(KAGGLE_SETUP),
        md("## Step 0 — Packages"),
        code("""!pip install -q matplotlib tqdm"""),
        md("## Step 1 — Bootstrap paths"),
        code(SHARED_BOOTSTRAP),
        md("## Step 2 — Sample test song IDs and locate Stage 2 checkpoints"),
        code(
            r'''
if not MANIFEST.exists():
    raise FileNotFoundError("Run notebook 01 first.")
manifest = pd.read_csv(MANIFEST)
manifest["song_id"] = manifest["song_id"].astype(str).map(lambda s: normalize_track_id(s) or s)
test_ids = manifest.loc[manifest["split"] == "test", "song_id"].astype(str).head(12).tolist()
print("sample test songs:", test_ids)
print("stage2 ckpts:", list((CKPT_DIR / "stage2").glob("best_*.pt")))
'''
        ),
        md("## Step 3 — Mean concept attention plot (replace placeholder with real `attn_weights` from notebook 07)"),
        code(
            r'''
import matplotlib.pyplot as plt

concept_names = ["instrument", "rhythm", "timbre", "harmony"]
rng = np.random.default_rng(0)
attn = rng.dirichlet(np.ones(4), size=max(len(test_ids), 1))
attn_df = pd.DataFrame(attn, columns=concept_names)
attn_df.insert(0, "song_id", test_ids + [""] * (len(attn_df) - len(test_ids)))
attn_df = attn_df.head(len(test_ids))
attn_df.to_csv(RESULTS_DIR / "09_attention_weights_sample.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(concept_names, attn[: len(test_ids)].mean(0) if test_ids else attn.mean(0))
ax.set_ylabel("mean attention")
ax.set_title("Concept attention (sample test subset)")
fig.tight_layout()
fig.savefig(RESULTS_DIR / "09_mean_attention.png", dpi=150)
plt.show()
attn_df.head()
'''
        ),
        md("## Step 4 — Occlusion + qualitative listening templates (fill by hand / model hooks)"),
        code(
            r'''
concept_names = ["instrument", "rhythm", "timbre", "harmony"]
rows = [{"occluded_concept": c, "delta_macro_pr_auc": None, "notes": "wire Stage2 checkpoint"} for c in concept_names]
qual = []
for sid in test_ids[:5]:
    top = concept_names[int(attn_df.loc[attn_df.song_id == sid, concept_names].values.argmax())] if sid in set(attn_df.song_id) else ""
    qual.append({"song_id": sid, "audible_dominant_concept": "", "model_top_concept": top, "agree": "", "comment": ""})
pd.DataFrame(rows).to_csv(RESULTS_DIR / "09_occlusion_template.csv", index=False)
pd.DataFrame(qual).to_csv(RESULTS_DIR / "09_qualitative_listening.csv", index=False)
print("Wrote templates under", RESULTS_DIR)
'''
        ),
    ],
)

print("Done. Notebooks in", OUT)
