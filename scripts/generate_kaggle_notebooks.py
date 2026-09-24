"""Generate the full staged Kaggle pipeline notebooks under notebooks/."""

from __future__ import annotations

import json
from pathlib import Path

from gpu_run_contract import NOTEBOOK_GPU_RUN_CONTRACT
from mtg_data_contract import NOTEBOOK_DATA_CONTRACT

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "kaggle"
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

### A. Settings
1. Right sidebar → **Internet → On** (required for downloads).
2. **GPU**: Off for `00`/`01`/`04`–`06`. **GPU (T4)** on for `02`/`03`/`07`.

### B. How data moves (do not skip)
Kaggle **does not** keep `/kaggle/working` when you open a *new* notebook.

**After notebook 00 finishes:**
1. **Save Version** (top-right) → **Save & Run All** (or Quick Save if already finished).
2. Open **Advanced** → tick **Always save output**.
3. Wait until the version is **Success**.
4. Note the short kernel slug assigned to notebook `00`.

**In the next notebook (01, then 02, …):**
1. **Add Input** (right sidebar) → **Your notebooks** / **Notebook Output**.
2. Select the latest successful output from notebook `00`.
3. Files appear below `/kaggle/input/<kernel-slug>/` (**read-only**).
4. This bootstrap **reads mels from that input** instead of duplicating them in writable storage.
5. It **writes** new files (manifest, features, checkpoints) to `/kaggle/working/MTG_Instrument`.
6. **Save Version + save output** again so the *next* notebook can **Add Input** *this* notebook too (chain: 00 → 01 → 02 …).

### C. CLI (laptop only — not needed on Kaggle)
```bash
kaggle kernels output <account>/<kernel-slug> -p ./from_00
```
On Kaggle you **Add Input** instead of this command.

### D. GitHub
Commit code and notebook changes to a feature branch and merge them into `main` after review. Do **not** commit `.npy` shards or checkpoints.
"""

SHARED_BOOTSTRAP = r'''
from pathlib import Path
import os, json, random, re, shutil, socket, time, urllib.request
import numpy as np
import pandas as pd

KERNEL_SLUG = os.environ.get("KAGGLE_KERNEL_SLUG", "dnn-download-data-1")
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
        if marker == ".shard_00_done":
            return hit.parents[2]  # .../MTG_Instrument/dataset/logmel_songs/.shard
    for p in INPUT_BASE.rglob("MTG_Instrument"):
        if p.is_dir():
            return p
    return None


def find_mel_dir() -> Path:
    """Prefer an attached notebook output and leave large mel files read-only."""
    bases = [
        Path(f"/kaggle/input/{KERNEL_SLUG}") / "MTG_Instrument" / "dataset" / "logmel_songs",
        Path(f"/kaggle/input/{KERNEL_SLUG}") / "dataset" / "logmel_songs",
        WORKING_ROOT / "dataset" / "logmel_songs",
    ]
    kernel = Path(f"/kaggle/input/{KERNEL_SLUG}")
    extra = []
    if INPUT_BASE.exists():
        extra.append(INPUT_BASE)
    if kernel.exists():
        extra.append(kernel)
    for b in bases:
        if b.exists() and next(b.rglob("*.npy"), None) is not None:
            return b
    for b in extra:
        if not b.exists():
            continue
        for candidate in b.rglob("logmel_songs"):
            if candidate.is_dir() and next(candidate.rglob("*.npy"), None) is not None:
                return candidate
    return WORKING_ROOT / "dataset" / "logmel_songs"


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


MEL_CACHE = Path("/kaggle/working/mel_cache")
MEL_CACHE.mkdir(parents=True, exist_ok=True)


def load_mel_npy(mel_abs, retries=5, pause=1.0):
    """Load mel with retries; cache under /kaggle/working for stable re-reads."""
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
        f"Bad/truncated mel — re-run notebook 00 for this shard: {mel_abs} "
        f"({nbytes} bytes). {last_err}"
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


ONLINE = check_internet()
print("Internet reachable:", ONLINE)
print("KERNEL_SLUG =", KERNEL_SLUG)
print("/kaggle/input folders:", list(INPUT_BASE.iterdir()) if INPUT_BASE.exists() else "n/a")

ROOT = WORKING_ROOT
ROOT.mkdir(parents=True, exist_ok=True)
MEL_DIR = find_mel_dir()
ANN_DIR = ROOT / "annotations"
# if annotations only exist on the attached kernel, point there (read-only is OK)
for cand in [
    Path(f"/kaggle/input/{KERNEL_SLUG}") / "MTG_Instrument" / "annotations",
    Path(f"/kaggle/input/{KERNEL_SLUG}") / "annotations",
]:
    if (cand / "splits" / "split-0" / "autotagging_genre-train.tsv").exists():
        ANN_DIR = cand
        break
FEAT_DIR = ROOT / "features"
CKPT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"
BASELINE_RESULTS_DIR = RESULTS_DIR / "baselines"
MANIFEST = ROOT / "dataset" / "song_manifest.csv"
att_manifest = _find_file("song_manifest.csv", [INPUT_BASE, Path("/kaggle/working")])
if not MANIFEST.exists() and att_manifest is not None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(att_manifest, MANIFEST)
        print("Copied song_manifest.csv from", att_manifest)
    except OSError:
        MANIFEST = att_manifest

for p in [ROOT / "dataset", ROOT / "annotations", FEAT_DIR, CKPT_DIR, BASELINE_RESULTS_DIR, RESULTS_DIR / "proposed"]:
    p.mkdir(parents=True, exist_ok=True)

# Restore the small artifacts produced by attached earlier stages. Large mel
# arrays stay read-only under /kaggle/input and are never duplicated here.
attached_artifacts = {
    "instrument_embeddings.npy": FEAT_DIR / "instrument" / "instrument_embeddings.npy",
    "song_ids.json": FEAT_DIR / "instrument" / "song_ids.json",
    "rhythm_song.csv": FEAT_DIR / "rhythm" / "rhythm_song.csv",
    "timbre_song.csv": FEAT_DIR / "timbre" / "timbre_song.csv",
    "harmony_song.csv": FEAT_DIR / "harmony" / "harmony_song.csv",
    "best_attention.pt": CKPT_DIR / "baselines" / "descriptor_fusion" / "best_attention.pt",
    "best_linear.pt": CKPT_DIR / "baselines" / "descriptor_fusion" / "best_linear.pt",
}
for name, destination in attached_artifacts.items():
    if destination.exists():
        continue
    source = _find_file(name, [INPUT_BASE])
    if source is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        print("Recovered", destination.relative_to(ROOT), "from", source)

if INPUT_BASE.exists():
    for pattern in ("02_baseline_*.json", "07_descriptor_fusion_*.json", "07_descriptor_fusion_*_history.csv"):
        for source in INPUT_BASE.rglob(pattern):
            destination = BASELINE_RESULTS_DIR / source.name
            if not destination.exists():
                shutil.copy2(source, destination)

# Small TSVs: copy/wget into working. Large mels stay on /kaggle/input.
ANN_DIR = ensure_annotations(ROOT / "annotations")

print("ROOT     =", ROOT)
print("MEL_DIR  =", MEL_DIR, "npy=", len(list(MEL_DIR.rglob('*.npy'))))
print("ANN_DIR  =", ANN_DIR)
print("split-0 train exists:", (ANN_DIR / "splits/split-0/autotagging_genre-train.tsv").exists())
print("MANIFEST =", MANIFEST, "exists=", MANIFEST.exists())
'''

SHARED_BOOTSTRAP += "\n" + NOTEBOOK_DATA_CONTRACT + "\n" + NOTEBOOK_GPU_RUN_CONTRACT

INTRO_00 = """\
# 00 — Kaggle Data Download (MTG-Jamendo)

**Goal:** Download official split-0 annotations and the default **mel shards 00–02** into `/kaggle/working/MTG_Instrument`.

This is **not** the full MTG-Jamendo MP3 set. It is the configured log-Mel subset plus labels.

**After this notebook (required):** Save Version with **output** so notebook `01` can attach it through **Add Input**.
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

Only the six split-specific files are downloaded; the larger unsplit label files
are intentionally excluded so they cannot introduce extra classes.

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
            """## Step 3 — Download the default mel shards **00–02**

Each tar is extracted then **deleted** to save disk. `/kaggle/working` is ~20GB — if a later shard fails with “No space left”, stop, Save Version with what you have, or split remaining shards into a second download kernel.

Needs Internet to `cdn.freesound.org`. Already-done shards (`.shard_XX_done`) are skipped."""
        ),
        code(
            r'''
import subprocess, shutil

def free_gb(path="/kaggle/working"):
    u = shutil.disk_usage(path)
    print(f"disk free: {u.free/1e9:.1f} GB  used: {u.used/1e9:.1f} GB")
    return u.free / 1e9

free_gb()
SHARDS = [0, 1, 2]  # reliable baseline; use list(range(10)) for the larger subset
print("Will download shards:", [f"{i:02d}" for i in SHARDS])

if not check_internet("cdn.freesound.org") and not check_internet():
    npy = list(MEL_DIR.rglob("*.npy"))
    if not npy:
        raise RuntimeError(
            "Internet is OFF and no .npy mels were found.\\n"
            "Enable Internet, or Add Input with extracted mels."
        )
    print(f"Offline: using {len(npy)} existing .npy files")
else:
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
        if free_gb() < 2.5:
            raise RuntimeError(f"Less than 2.5 GB free — cannot download shard {i:02d}. Save Version now.")
        if not tar_path.exists():
            print(f"Downloading {url} ...")
            subprocess.check_call(["wget", "-q", "-O", str(tar_path), url])
        print(f"Extracting {tar_name} ...")
        subprocess.check_call(["tar", "-xf", str(tar_path), "-C", str(MEL_DIR)])
        tar_path.unlink(missing_ok=True)
        marker.write_text("ok")
        print(f"shard {i:02d} ready")
        free_gb()

    for i in SHARDS:
        download_shard(i)

npy_count = len(list(MEL_DIR.rglob("*.npy")))
print(f"Total .npy files under MEL_DIR: {npy_count}")
assert npy_count > 0, "No mel .npy found — check download / internet"
'''
        ),
        md("""## Step 4 — Summary, then persist output

1. Confirm `n_npy` > 0 and the configured shards are listed.
2. **Save Version** → enable **Always save output**.
3. Open **01_preprocessing** as a **new** Kaggle notebook.
4. Attach notebook `00` using **Add Input → Notebook Output**.
5. Run 01. Repeat Save+Add Input for 02, 03, …"""),
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
print("\\nNEXT: Save Version (save output) → new notebook 01 → attach this output with Add Input")
'''
        ),
    ],
)


write(
    "01_preprocessing.ipynb",
    [
        md(
            """# 01 — Preprocessing & Song Manifest

Join every downloaded mel `.npy` to official **split-0**.

**Before Run All:** attach the saved output from notebook `00` with **Add Input → Notebook Output**.

Mels stay below `/kaggle/input` (read-only). This notebook writes `song_manifest.csv` to `/kaggle/working`."""
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
    rows.append({"song_id": tid, "logmel_path": rel, "mel_abs": str(p), "nbytes": p.stat().st_size})

mel_df = pd.DataFrame(rows)
duplicate_ids = sorted(mel_df.loc[mel_df.duplicated("song_id", keep=False), "song_id"].unique())
if duplicate_ids:
    raise RuntimeError(f"duplicate log-Mels for {len(duplicate_ids)} songs; examples={duplicate_ids[:5]}")
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
instrument_train_ids = load_split_ids("train", "instrument")
instrument_val_ids = load_split_ids("validation", "instrument")
instrument_test_ids = load_split_ids("test", "instrument")

assert train_ids.isdisjoint(val_ids), "train overlaps validation"
assert train_ids.isdisjoint(test_ids), "train overlaps test"
assert val_ids.isdisjoint(test_ids), "VALIDATION must not intersect TEST"
assert instrument_train_ids.isdisjoint(instrument_val_ids)
assert instrument_train_ids.isdisjoint(instrument_test_ids)
assert instrument_val_ids.isdisjoint(instrument_test_ids)
instrument_ids = instrument_train_ids | instrument_val_ids | instrument_test_ids
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
manifest.head()
'''
        ),
        md("## Step 4 — Sanity-check one spectrogram shape"),
        code(
            r'''
sample = load_mel_npy(manifest.iloc[0]["mel_abs"])
print("example shape:", sample.shape, "dtype:", sample.dtype)
(RESULTS_DIR / "01_manifest_summary.json").write_text(json.dumps({
    "n_manifest": int(len(manifest)),
    "split_counts": manifest["split"].value_counts().to_dict(),
    "instrument_available": int(manifest["instrument_available"].sum()),
    "example_shape": list(sample.shape),
}, indent=2))
print("Next: 02_direct_cnn_baseline.ipynb")
'''
        ),
    ],
)


write(
    "02_direct_cnn_baseline.ipynb",
    [
        md(
            """# 02 — Direct CNN Baseline

Train a compact CNN on log-mel for **multi-label genre**.

- Split: official **split-0** only (never a random split)
- Metrics: NaN-safe macro ROC-AUC and PR-AUC (undefined tags are **excluded**, not zeroed)
- Checkpoint: **best validation** macro PR-AUC (`best_macro_map` is updated inside the save branch)
- Always save validation predictions; evaluate test only for the selected final run
  after setting `EVALUATE_TEST=1`

Paper reference (full set): **0.7260 ROC-AUC / 0.1592 PR-AUC** — shard subset will differ."""
        ),
        md(KAGGLE_SETUP),
        md("## Step 0 — Packages + GPU\n\nEnable **GPU** (T4) in Settings for this notebook."),
        code("""!pip install -q scikit-learn tqdm"""),
        md("## Step 1 — Bootstrap paths (find manifest + annotations)"),
        code(SHARED_BOOTSTRAP),
        md("## Step 2 — Load manifest and genre multi-hot labels\n\nParses only the three official split-0 genre files and freezes their shared 87-label ordering."),
        code(
            r'''
import torch
import torch.nn as nn
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
    raise FileNotFoundError("song_manifest.csv missing — run notebook 01 first (same session or attach its output).")
manifest = pd.read_csv(MANIFEST)
manifest["song_id"] = manifest["song_id"].astype(str).map(lambda s: normalize_track_id(s) or s)
manifest = apply_approved_cohort(manifest, GPU_RUN, MANIFEST)
assert set(manifest["split"]) <= {"train", "validation", "test"}

song_ids = manifest["song_id"].astype(str).tolist()
Y, TAG_NAMES, genre_available = load_split_multihot(song_ids, "genre", "genre")
if not genre_available.all():
    missing = np.asarray(song_ids)[~genre_available]
    raise RuntimeError(f"manifest contains {len(missing)} songs without split-0 genre labels")
print("Y shape:", Y.shape, "positive rate:", float(Y.mean()))
(RESULTS_DIR / "genre_tags.json").write_text(json.dumps(TAG_NAMES, indent=2))
'''
        ),
        md("## Step 3 — Dataset / loaders (split-0 only)\n\nValidation is **never** merged with test."),
        code(
            r'''
class MelGenreDataset(Dataset):
    def __init__(self, df: pd.DataFrame, Y: np.ndarray, id_to_idx: dict,
                 max_windows: int = LOGMEL_MAX_WINDOWS, n_mels: int = LOGMEL_N_MELS,
                 n_frames: int = LOGMEL_WINDOW_FRAMES):
        self.df = df.reset_index(drop=True)
        self.Y = Y
        self.id_to_idx = id_to_idx
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

def make_loader(split: str, bs: int = BATCH_SIZE, shuffle=False):
    sub = manifest[manifest["split"] == split]
    assert set(sub["split"].unique()) == {split}, "split leakage"
    ds = MelGenreDataset(sub, Y, id_to_idx)
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=0, pin_memory=torch.cuda.is_available())

train_loader = make_loader("train", shuffle=True)
val_loader = make_loader("validation")
test_loader = make_loader("test")
_x, _mask, _y = next(iter(train_loader))
print("preflight batch", tuple(_x.shape), "expect (bs, 12, 1, 96, 1366)")
assert _x.shape[1:] == (12, 1, 96, 1366), f"re-run this entire cell — got {_x.shape}"
assert torch.all(_mask.sum(1) >= 1), "every song needs at least one real window"
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
        self.window_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Dropout(0.3),
        )
        self.head = nn.Linear(256, n_tags)

    def forward(self, x, mask):
        B, W, C, M, T = x.shape
        z = self.window_proj(self.features(x.reshape(B * W, C, M, T))).reshape(B, W, -1)
        weights = mask / mask.sum(1, keepdim=True).clamp_min(1.0)
        return self.head((z * weights.unsqueeze(-1)).sum(1))

model = BaselineCNN(n_tags=Y.shape[1]).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()
print(model)
'''
        ),
        md("## Step 5 — Train; save the **best validation** checkpoint"),
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
def evaluate(loader, return_predictions=False):
    model.eval()
    ys, ps = [], []
    for x, mask, y in loader:
        x, mask = x.to(DEVICE), mask.to(DEVICE)
        prob = torch.sigmoid(model(x, mask)).cpu().numpy()
        ys.append(y.numpy())
        ps.append(prob)
    y_true, y_prob = np.concatenate(ys), np.concatenate(ps)
    metrics = {"macro_roc_auc": nan_safe_macro_auc(y_true, y_prob, "roc"),
               "macro_pr_auc": nan_safe_macro_auc(y_true, y_prob, "pr")}
    return (metrics, y_true, y_prob) if return_predictions else metrics


def train_one_epoch(loader, deadline):
    model.train()
    total = 0.0
    for x, mask, y in tqdm(loader, leave=False):
        if time.perf_counter() >= deadline:
            return None, True
        x, mask, y = x.to(DEVICE), mask.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        loss = criterion(model(x, mask), y)
        loss.backward()
        opt.step()
        total += loss.item() * len(x)
    return total / len(loader.dataset), False

PATIENCE = 2
EPOCHS, MAX_WALL_MINUTES = approved_run_limits(
    GPU_RUN, default_epochs=10,
    requested_wall_minutes=float(os.environ.get("MAX_GPU_RUN_MINUTES", "120")),
)
EVALUATE_TEST = os.environ.get("EVALUATE_TEST", "0") == "1"
best_macro_map = float("-inf")
ckpt_dir = CKPT_DIR / "baselines" / "direct_cnn"
ckpt_dir.mkdir(parents=True, exist_ok=True)
history = []
epochs_without_improvement = 0
GPU_DEADLINE = GPU_RUN_STARTED + MAX_WALL_MINUTES * 60
TRAINING_DEADLINE = GPU_RUN_STARTED + MAX_WALL_MINUTES * 60 * 0.9
if DEVICE.type == "cuda":
    model.eval(); opt.zero_grad(set_to_none=True)
    _preflight_loss = criterion(model(_x.to(DEVICE), _mask.to(DEVICE)), _y.to(DEVICE))
    _preflight_loss.backward(); opt.zero_grad(set_to_none=True)
    del _preflight_loss
    torch.cuda.empty_cache()
    print("preflight backward: OK")

for epoch in range(1, EPOCHS + 1):
    tr_loss, wall_cap_reached = train_one_epoch(train_loader, TRAINING_DEADLINE)
    if wall_cap_reached:
        print(f"training-time reserve reached between batches: {MAX_WALL_MINUTES:.0f} minute total cap")
        break
    val_m = evaluate(val_loader)
    if not np.isfinite(val_m["macro_pr_auc"]):
        raise RuntimeError("validation PR-AUC is undefined; fix label coverage before spending more GPU time")
    history.append({"epoch": epoch, "train_loss": tr_loss, **val_m})
    print(f"epoch {epoch}: loss={tr_loss:.4f} val_roc={val_m['macro_roc_auc']:.4f} val_pr={val_m['macro_pr_auc']:.4f}")
    if val_m["macro_pr_auc"] > best_macro_map:
        best_macro_map = val_m["macro_pr_auc"]
        epochs_without_improvement = 0
        torch.save({"model": model.state_dict(), "tags": TAG_NAMES, "best_macro_map": best_macro_map, "epoch": epoch,
                    "training_config": {"max_epochs": EPOCHS, "patience": PATIENCE,
                                        "max_wall_minutes": MAX_WALL_MINUTES,
                                        "input_schema": LOGMEL_SCHEMA_VERSION,
                                        "max_windows": LOGMEL_MAX_WINDOWS,
                                        "batch_size": BATCH_SIZE,
                                        "gpu_run": GPU_RUN}},
                   ckpt_dir / "best.pt")
        print("  ✓ saved best checkpoint @", best_macro_map)
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= PATIENCE:
            print(f"early stop: no validation PR-AUC improvement for {PATIENCE} epochs")
            break
    if time.perf_counter() >= TRAINING_DEADLINE:
        print(f"wall-time cap reached: {MAX_WALL_MINUTES:.0f} minutes")
        break

if not (ckpt_dir / "best.pt").is_file():
    write_gpu_termination_ledger(
        BASELINE_RESULTS_DIR / "02_baseline_runtime.json", device=DEVICE,
        started_at=GPU_RUN_STARTED, record=GPU_RUN, reason="no_complete_validation_epoch",
        max_epochs=EPOCHS, max_wall_minutes=MAX_WALL_MINUTES,
    )
    raise RuntimeError("GPU cap reached before one complete validation epoch; no checkpoint was created")
state = torch.load(ckpt_dir / "best.pt", map_location=DEVICE, weights_only=False)
model.load_state_dict(state["model"])
val_m, val_y, val_p = evaluate(val_loader, return_predictions=True)
pd.DataFrame(history).to_csv(BASELINE_RESULTS_DIR / "02_baseline_history.csv", index=False)
elapsed_seconds = time.perf_counter() - GPU_RUN_STARTED
runtime = {
    "device": str(DEVICE), "epochs_completed": len(history),
    "max_epochs": EPOCHS, "patience": PATIENCE,
    "max_wall_minutes": MAX_WALL_MINUTES,
    "batch_size": BATCH_SIZE, "input_schema": LOGMEL_SCHEMA_VERSION,
    "evaluated_test": EVALUATE_TEST,
    "gpu_run": GPU_RUN,
    "wall_seconds": elapsed_seconds,
    "gpu_wall_hours": elapsed_seconds / 3600 if DEVICE.type == "cuda" else 0.0,
}
(BASELINE_RESULTS_DIR / "02_baseline_runtime.json").write_text(json.dumps(runtime, indent=2))
print("runtime", runtime)
pred_dir = RESULTS_DIR / "predictions"
pred_dir.mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    pred_dir / "02_direct_cnn_validation.npz",
    song_ids=np.asarray(val_loader.dataset.df["song_id"].astype(str).to_numpy(), dtype=str),
    label_names=np.asarray(TAG_NAMES, dtype=str), targets=val_y, scores=val_p,
)
if EVALUATE_TEST:
    test_m, test_y, test_p = evaluate(test_loader, return_predictions=True)
    print("FINAL TEST (split-0 only):", test_m)
    (BASELINE_RESULTS_DIR / "02_baseline_test.json").write_text(json.dumps(test_m, indent=2))
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
(BASELINE_RESULTS_DIR / "02_baseline_runtime.json").write_text(json.dumps(runtime, indent=2))
print("final runtime", runtime)
'''
        ),
    ],
)


write(
    "03_instrument_pretraining.ipynb",
    [
        md(
            """# 03 — Instrument Pretraining (MIL + Attention)

Learn a **64-d song-level instrument embedding** with attention pooling over ordered
29.1-second log-Mel windows. The trained CNN can initialize the proposed shared encoder.

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
BATCH_SIZE = int(os.environ.get("GPU_BATCH_SIZE", "2"))
if BATCH_SIZE < 1:
    raise ValueError("GPU_BATCH_SIZE must be positive")
GPU_RUN = require_gpu_run_approval(DEVICE, "instrument_pretraining")
GPU_RUN_STARTED = time.perf_counter()
if not MANIFEST.exists():
    raise FileNotFoundError("Run notebook 01 first.")
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
    def __init__(self, df, Y, id_to_idx, max_windows=MAX_WINDOWS, n_mels=N_MELS, n_frames=N_FRAMES):
        self.df = df.reset_index(drop=True)
        self.Y, self.id_to_idx = Y, id_to_idx
        self.max_windows, self.n_mels, self.n_frames = max_windows, n_mels, n_frames

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        windows, mask = segment_logmel(
            load_mel_npy(resolve_stacked_mel_path(row["mel_abs"])), n_mels=self.n_mels,
            n_frames=self.n_frames, max_windows=self.max_windows,
        )
        y = self.Y[self.id_to_idx[str(row["song_id"])]]
        return (
            torch.from_numpy(windows[:, None, :, :]),
            torch.from_numpy(mask),
            torch.from_numpy(y),
            str(row["song_id"]),
        )


def make_loader(split, bs=BATCH_SIZE, shuffle=False):
    sub = manifest[(manifest["split"] == split) & manifest["instrument_available"]]
    if sub.empty:
        raise RuntimeError(f"no instrument-annotated songs in {split}")
    assert set(sub["split"].unique()) == {split}
    return DataLoader(WindowMILDataset(sub, Y, id_to_idx), batch_size=bs, shuffle=shuffle, num_workers=0)
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


class InstrumentMILModel(nn.Module):
    def __init__(self, n_tags, emb=EMBED_DIM):
        super().__init__()
        self.enc = WindowEncoder(emb)
        self.pool = AttnPool(emb)
        self.head = nn.Linear(emb, n_tags)

    def forward(self, x, mask):
        H = self.enc(x)
        z, attn = self.pool(H, mask)
        return self.head(z), z, attn

model = InstrumentMILModel(n_tags=Y.shape[1]).to(DEVICE)
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
_x, _m, _y, _ = next(iter(train_loader))
print("preflight batch", tuple(_x.shape), "expect (bs, 12, 1, 96, 1366)")
assert _x.shape[2:] == (1, N_MELS, N_FRAMES), f"re-run this entire cell — got {_x.shape}"
SCAN_MELS = False
if SCAN_MELS:
    bad = scan_bad_mels(manifest, "all")
    if bad:
        raise RuntimeError(f"{len(bad)} bad mels — see {RESULTS_DIR}/bad_mels_all.json")
PATIENCE = 2
EPOCHS, MAX_WALL_MINUTES = approved_run_limits(
    GPU_RUN, default_epochs=8,
    requested_wall_minutes=float(os.environ.get("MAX_GPU_RUN_MINUTES", "120")),
)
EVALUATE_TEST = os.environ.get("EVALUATE_TEST", "0") == "1"
best_macro_map = float("-inf")
ckpt_dir = CKPT_DIR / "pretraining" / "instrument"
ckpt_dir.mkdir(parents=True, exist_ok=True)
epochs_without_improvement = 0
history = []
GPU_DEADLINE = GPU_RUN_STARTED + MAX_WALL_MINUTES * 60
TRAINING_DEADLINE = GPU_RUN_STARTED + MAX_WALL_MINUTES * 60 * 0.9
wall_cap_reached = False
if DEVICE.type == "cuda":
    model.eval(); opt.zero_grad(set_to_none=True)
    _preflight_logits, _, _ = model(_x.to(DEVICE), _m.to(DEVICE))
    _preflight_loss = criterion(_preflight_logits, _y.to(DEVICE))
    _preflight_loss.backward(); opt.zero_grad(set_to_none=True)
    del _preflight_logits, _preflight_loss
    torch.cuda.empty_cache()
    print("preflight backward: OK")

for epoch in range(1, EPOCHS + 1):
    model.train()
    total = 0.0
    for x, mask, y, _ in tqdm(train_loader, leave=False):
        if time.perf_counter() >= TRAINING_DEADLINE:
            wall_cap_reached = True
            break
        x, mask, y = x.to(DEVICE), mask.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        logits, _, _ = model(x, mask)
        loss = criterion(logits, y)
        loss.backward()
        opt.step()
        total += loss.item() * len(x)
    if wall_cap_reached:
        print(f"training-time reserve reached between batches: {MAX_WALL_MINUTES:.0f} minute total cap")
        break
    val_map = eval_split(val_loader)
    if not np.isfinite(val_map):
        raise RuntimeError("validation macro mAP is undefined; fix label coverage before spending more GPU time")
    history.append({"epoch": epoch, "loss": total/len(train_loader.dataset), "val_macro_map": val_map})
    print(f"epoch {epoch}: loss={total/len(train_loader.dataset):.4f} val_macro_map={val_map:.4f}")
    if val_map > best_macro_map:
        best_macro_map = val_map
        epochs_without_improvement = 0
        torch.save({
            "model": model.state_dict(), "best_macro_map": best_macro_map, "epoch": epoch, "tags": INST_NAMES,
            "training_config": {"max_epochs": EPOCHS, "patience": PATIENCE,
                                "max_wall_minutes": MAX_WALL_MINUTES,
                                "input_schema": LOGMEL_SCHEMA_VERSION,
                                "max_windows": LOGMEL_MAX_WINDOWS,
                                "batch_size": BATCH_SIZE,
                                "gpu_run": GPU_RUN},
        },
                   ckpt_dir / "best.pt")
        print("  ✓ checkpoint", best_macro_map)
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= PATIENCE:
            print(f"early stop: no validation PR-AUC improvement for {PATIENCE} epochs")
            break
    if time.perf_counter() >= TRAINING_DEADLINE:
        print(f"wall-time cap reached: {MAX_WALL_MINUTES:.0f} minutes")
        break

if not (ckpt_dir / "best.pt").is_file():
    write_gpu_termination_ledger(
        RESULTS_DIR / "pretraining" / "instrument" / "runtime.json", device=DEVICE,
        started_at=GPU_RUN_STARTED, record=GPU_RUN, reason="no_complete_validation_epoch",
        max_epochs=EPOCHS, max_wall_minutes=MAX_WALL_MINUTES,
    )
    raise RuntimeError("GPU cap reached before one complete validation epoch; no checkpoint was created")
state = torch.load(ckpt_dir / "best.pt", map_location=DEVICE, weights_only=False)
model.load_state_dict(state["model"])
model.eval()
all_loader = DataLoader(WindowMILDataset(manifest, Y, id_to_idx), batch_size=BATCH_SIZE, num_workers=0)
embeds, ids = [], []
export_cap_reached = False
with torch.no_grad():
    for x, mask, y, sid in tqdm(all_loader):
        if time.perf_counter() >= GPU_DEADLINE:
            export_cap_reached = True
            break
        _, z, _ = model(x.to(DEVICE), mask.to(DEVICE))
        embeds.append(z.cpu().numpy())
        ids.extend(list(sid))
if export_cap_reached:
    write_gpu_termination_ledger(
        RESULTS_DIR / "pretraining" / "instrument" / "runtime.json", device=DEVICE,
        started_at=GPU_RUN_STARTED, record=GPU_RUN, reason="embedding_export_cap_reached",
        max_epochs=EPOCHS, max_wall_minutes=MAX_WALL_MINUTES,
    )
    raise RuntimeError("GPU wall-time cap reached during embedding export; no partial artifact was saved")
E = np.concatenate(embeds, 0)
out = FEAT_DIR / "instrument"
out.mkdir(parents=True, exist_ok=True)
np.save(out / "instrument_embeddings.npy", E)
(out / "song_ids.json").write_text(json.dumps(ids))
result_dir = RESULTS_DIR / "pretraining" / "instrument"
result_dir.mkdir(parents=True, exist_ok=True)
pd.DataFrame(history).to_csv(result_dir / "history.csv", index=False)
elapsed_seconds = time.perf_counter() - GPU_RUN_STARTED
runtime = {
    "device": str(DEVICE), "epochs_completed": len(history),
    "max_epochs": EPOCHS, "patience": PATIENCE,
    "max_wall_minutes": MAX_WALL_MINUTES, "batch_size": BATCH_SIZE,
    "input_schema": LOGMEL_SCHEMA_VERSION, "evaluated_test": EVALUATE_TEST,
    "gpu_run": GPU_RUN,
    "wall_seconds_including_export": elapsed_seconds,
    "gpu_wall_hours_including_export": elapsed_seconds / 3600 if DEVICE.type == "cuda" else 0.0,
}
(result_dir / "runtime.json").write_text(json.dumps(runtime, indent=2))
print("saved", E.shape, "runtime", runtime)
if EVALUATE_TEST:
    test_metric = {"macro_pr_auc": eval_split(test_loader)}
    (result_dir / "test.json").write_text(json.dumps(test_metric, indent=2))
    print("FINAL instrument TEST", test_metric)
else:
    print("Instrument test skipped. Set EVALUATE_TEST=1 only for the selected final run.")
elapsed_seconds = time.perf_counter() - GPU_RUN_STARTED
runtime["wall_seconds_including_export"] = elapsed_seconds
runtime["gpu_wall_hours_including_export"] = elapsed_seconds / 3600 if DEVICE.type == "cuda" else 0.0
(result_dir / "runtime.json").write_text(json.dumps(runtime, indent=2))
print("final runtime", runtime)
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
    S = load_mel_npy(mel_path)
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
        f"{num}_{kind}_targets.ipynb",
        [
            md(
                f"""# {num} — {title}

Extract **{kind}** supervision targets per song (`song_id` aligned with the shared manifest).

On Kaggle, raw MP3s are optional. By default these descriptors are approximated from the available log-Mels. Attach audio under `/kaggle/input/mtg-jamendo-audio` to use waveform-based librosa features."""
            ),
            md(KAGGLE_SETUP),
            md("## Step 0 — Packages (`librosa` is slow on CPU; that is OK for this notebook)"),
            code("""!pip install -q librosa soundfile tqdm"""),
            md("## Step 1 — Bootstrap paths"),
            code(SHARED_BOOTSTRAP),
            md(f"## Step 2 — Create {kind} targets and write `features/{out_sub}/{out_sub}_song.csv`"),
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

KAGGLE_04_EXTRACT = r'''
import subprocess
from tqdm.auto import tqdm

if not MANIFEST.exists():
    raise FileNotFoundError("Run notebook 01 first.")
manifest = pd.read_csv(MANIFEST)
manifest["song_id"] = manifest["song_id"].astype(str).map(lambda s: normalize_track_id(s) or s)

AB_DIR = ROOT / "dataset" / "acousticbrainz"
AB_DIR.mkdir(parents=True, exist_ok=True)
# Prefer already-unpacked JSON attached via Add Input
for cand in [
    Path(f"/kaggle/input/{KERNEL_SLUG}") / "MTG_Instrument" / "dataset" / "acousticbrainz",
    Path(f"/kaggle/input/{KERNEL_SLUG}") / "dataset" / "acousticbrainz",
]:
    if cand.exists() and next(cand.rglob("*.json"), None) is not None:
        AB_DIR = cand
        print("Using attached AcousticBrainz at", AB_DIR)
        break

AB_SHARDS = [0, 1, 2]  # match the default mel subset
AB_URL = "https://cdn.freesound.org/mtg-jamendo/raw_30s/acousticbrainz"


def download_ab_shards():
    n_json = len(list(AB_DIR.rglob("*.json")))
    if n_json > 0:
        print(f"AcousticBrainz already present: {n_json} JSON under {AB_DIR}")
        return
    if not check_internet("cdn.freesound.org") and not check_internet():
        raise RuntimeError(
            "No AcousticBrainz JSON and Internet is OFF. Enable Internet, or attach unpacked "
            "raw_30s acousticbrainz shards under dataset/acousticbrainz."
        )
    dest = ROOT / "dataset" / "acousticbrainz"
    dest.mkdir(parents=True, exist_ok=True)
    print("Downloading AcousticBrainz shards 00–02 to", dest)
    for i in AB_SHARDS:
        marker = dest / f".ab_shard_{i:02d}_done"
        if marker.exists():
            print(f"AB shard {i:02d} already done — skip")
            continue
        tar_name = f"raw_30s_acousticbrainz-{i:02d}.tar.gz"
        tar_path = dest / tar_name
        url = f"{AB_URL}/{tar_name}"
        print("Downloading", url)
        subprocess.check_call(["wget", "-q", "-O", str(tar_path), url])
        subprocess.check_call(["tar", "-xzf", str(tar_path), "-C", str(dest)])
        tar_path.unlink(missing_ok=True)
        marker.write_text("ok")
        print(f"AB shard {i:02d} saved")
    return dest


def index_ab_json(root: Path) -> dict:
    idx = {}
    for p in root.rglob("*.json"):
        sid = normalize_track_id(p.stem)
        if sid:
            idx[sid] = p
    return idx


def _scalar(v):
    if v is None:
        return None
    if isinstance(v, dict):
        return _scalar(v["mean"]) if "mean" in v else None
    if isinstance(v, (list, tuple)):
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(x) else x


def rhythm_from_ab(doc: dict):
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
    return {
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


maybe = download_ab_shards()
if maybe is not None:
    AB_DIR = maybe
ab_index = index_ab_json(AB_DIR)
print("AcousticBrainz JSON indexed:", len(ab_index), "at", AB_DIR)

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
    raise RuntimeError("No overlapping AcousticBrainz rhythm rows.")

df = pd.DataFrame(rows)
out = FEAT_DIR / "rhythm"
out.mkdir(parents=True, exist_ok=True)
df.to_csv(out / "rhythm_song.csv", index=False)
(RESULTS_DIR / "04_missing_acousticbrainz.json").write_text(json.dumps(missing, indent=2))
summary = {
    "n_manifest_mels": n_mels,
    "n_acousticbrainz_json": n_ab_disk,
    "n_overlap_written": n_written,
    "n_excluded": n_excluded,
    "source": "acousticbrainz",
}
(RESULTS_DIR / "04_rhythm_summary.json").write_text(json.dumps(summary, indent=2))
print("wrote", out / "rhythm_song.csv", "rows=", n_written)
print(json.dumps(summary, indent=2))
df.head()
'''

write(
    "04_rhythm_targets.ipynb",
    [
        md(
            """# 04 — Rhythm Supervision Targets

Writes `features/rhythm/rhythm_song.csv` from **AcousticBrainz / Essentia** JSON
(`rhythm.bpm`, beat stats, histogram peaks). Mel-proxy tempo is **not** used.

Needs notebook 01. Downloads AcousticBrainz shards **00–02** by default if JSON is not already attached."""
        ),
        md(KAGGLE_SETUP),
        md("## Step 0 — Packages"),
        code("""!pip install -q tqdm"""),
        md("## Step 1 — Bootstrap paths"),
        code(SHARED_BOOTSTRAP),
        md("## Step 2 — AcousticBrainz rhythm fields"),
        code(KAGGLE_04_EXTRACT),
    ],
)
feature_nb("05", "Timbre Supervision Targets", "timbre", extract_timbre, "timbre")

write(
    "06_harmony_targets.ipynb",
    [
        md(
            """# 06 — Harmony Target Preflight

The former notebook used a song-wide chroma/Tonnetz average and silently created
fake pitch classes by splitting Mel bins when waveform audio was absent. Both paths
are retired from target generation.

The default path audits waveform availability only and writes no harmony targets.
Explicit opt-in CPU cells below can run the reviewed extractor and branch-screening
gates from an exact Git commit. The CPU-only CQT candidate has synthetic tests but
has not passed the real-audio comparison gate."""
        ),
        md(KAGGLE_SETUP),
        md("## Step 0 — Bootstrap paths"),
        code(SHARED_BOOTSTRAP),
        md("## Step 1 — Audit waveform availability (no Mel fallback)"),
        code(
            r'''
from datetime import datetime, timezone

if not MANIFEST.exists():
    raise FileNotFoundError("Run notebook 01 first.")

manifest = pd.read_csv(MANIFEST)
manifest["song_id"] = manifest["song_id"].astype(str).map(lambda s: normalize_track_id(s) or s)
audio_roots = [
    Path("/kaggle/input/mtg-jamendo-audio"),
    Path(f"/kaggle/input/{KERNEL_SLUG}") / "MTG_Instrument" / "dataset" / "audio",
    ROOT / "dataset" / "audio",
]
audio_index = {}
for audio_root in audio_roots:
    if not audio_root.exists():
        continue
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
unless one of the `RUN_*` flags is explicitly set to `1`. They use an exact reviewed
Git commit, disable CUDA, and save into a new immutable run directory. For Stage B,
attach notebook 03's output and point `HARMONY_INSTRUMENT_CHECKPOINT` to its exact
instrument `best.pt` file."""
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
    PROJECT_CODE = Path("/kaggle/working/dnn-project")
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
        "runtime": "kaggle",
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
    if not checkpoint_text:
        raise ValueError(
            "Set HARMONY_INSTRUMENT_CHECKPOINT to notebook 03's attached instrument best.pt"
        )
    instrument_checkpoint = Path(checkpoint_text)
    if not instrument_checkpoint.is_file():
        raise FileNotFoundError(instrument_checkpoint)
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
        md(
            """# 07 — Descriptor-Fusion Baseline

Combines the learned instrument embedding with rhythm/timbre/harmony descriptors. This is a comparison baseline, not the proposed four-branch model.

Blocked until the harmony quality gates produce a reviewed
`global_tonal_summary_v1`; notebook 06 currently performs preflight only.

- Fusion A: concat → linear
- Fusion B: single-head attention over the 4 concept tokens
- Head: 87-ish genre tags, BCE with logits
- Always save validation predictions; evaluate **split-0 test** only for the selected
  final run after setting `EVALUATE_TEST=1`"""
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
GPU_RUN = require_gpu_run_approval(DEVICE, "descriptor_fusion")
GPU_RUN_STARTED = time.perf_counter()
if not MANIFEST.exists():
    raise FileNotFoundError("Run notebooks 01 then 03–06 first.")
manifest = pd.read_csv(MANIFEST)
manifest["song_id"] = manifest["song_id"].astype(str).map(lambda s: normalize_track_id(s) or s)
manifest = apply_approved_cohort(manifest, GPU_RUN, MANIFEST)

inst_dir = FEAT_DIR / "instrument"
if not (inst_dir / "instrument_embeddings.npy").exists():
    raise FileNotFoundError("Missing instrument embeddings — run notebook 03.")
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

def num_cols(df):
    metadata = {"song_id", "source", "split", "schema_version", "target_variant", "extractor_version", "teacher_name", "teacher_version"}
    return [c for c in df.columns if c not in metadata and pd.api.types.is_numeric_dtype(df[c])]

r_cols, t_cols, h_cols = num_cols(rhythm), num_cols(timbre), num_cols(harmony)
n_before = len(manifest)
have = set(inst_map) & set(rhythm.index) & set(timbre.index) & set(harmony.index)
manifest = manifest[manifest["song_id"].isin(have)].copy()
print(f"Descriptor baseline overlap: {len(manifest)} / {n_before} songs have all four concepts")
if manifest.empty:
    raise RuntimeError("No overlapping songs with validated descriptors — complete notebooks 03–06 and their quality gates.")
ids = manifest["song_id"].astype(str).tolist()
id_to_idx = {s: i for i, s in enumerate(ids)}

Y, TAG_NAMES, genre_available = load_split_multihot(ids, "genre", "genre")
if not genre_available.all():
    raise RuntimeError("descriptor cohort contains songs without split-0 genre labels")
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
    return DataLoader(ConceptDataset(sub), batch_size=bs, shuffle=shuffle, num_workers=0)

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
        md("## Step 4 — Train; keep best validation PR-AUC; test is final-run opt-in"),
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
def evaluate(dl, return_predictions=False):
    model.eval()
    ys, ps = [], []
    for inst, r, t, h, y in dl:
        inst, r, t, h = inst.to(DEVICE), r.to(DEVICE), t.to(DEVICE), h.to(DEVICE)
        logits, _ = model(inst, r, t, h)
        ps.append(torch.sigmoid(logits).cpu().numpy())
        ys.append(y.numpy())
    yt, yp = np.concatenate(ys), np.concatenate(ps)
    metrics = {"macro_roc_auc": nan_safe(yt, yp, "roc"), "macro_pr_auc": nan_safe(yt, yp, "pr")}
    return (metrics, yt, yp) if return_predictions else metrics

train_dl, val_dl, test_dl = loader("train", shuffle=True), loader("validation"), loader("test")
best_macro_map = float("-inf")
ckpt = CKPT_DIR / "baselines" / "descriptor_fusion"
ckpt.mkdir(parents=True, exist_ok=True)
hist = []
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
    _pi, _pr, _pt, _ph, _py = next(iter(train_dl))
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
    model.train()
    total = 0
    for inst, r, t, h, y in tqdm(train_dl, leave=False):
        if time.perf_counter() >= TRAINING_DEADLINE:
            wall_cap_reached = True
            break
        inst, r, t, h, y = inst.to(DEVICE), r.to(DEVICE), t.to(DEVICE), h.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        logits, _ = model(inst, r, t, h)
        loss = crit(logits, y)
        loss.backward()
        opt.step()
        total += loss.item() * len(y)
    if wall_cap_reached:
        print(f"training-time reserve reached between batches: {MAX_WALL_MINUTES:.0f} minute total cap")
        break
    vm = evaluate(val_dl)
    if not np.isfinite(vm["macro_pr_auc"]):
        raise RuntimeError("validation PR-AUC is undefined; fix label coverage before spending more GPU time")
    hist.append({"epoch": epoch, "loss": total / len(train_dl.dataset), **vm})
    print(epoch, hist[-1])
    if vm["macro_pr_auc"] > best_macro_map:
        best_macro_map = vm["macro_pr_auc"]
        epochs_without_improvement = 0
        torch.save({
            "model": model.state_dict(), "fusion": FUSION,
            "best_macro_map": best_macro_map, "tags": TAG_NAMES,
            "training_config": {"max_epochs": MAX_EPOCHS, "patience": PATIENCE,
                                "max_wall_minutes": MAX_WALL_MINUTES,
                                "gpu_run": GPU_RUN},
        }, ckpt / f"best_{FUSION}.pt")
        print("  ✓ saved", best_macro_map)
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= PATIENCE:
            print(f"early stop: no validation PR-AUC improvement for {PATIENCE} epochs")
            break

    if time.perf_counter() >= TRAINING_DEADLINE:
        print(f"wall-time cap reached: {MAX_WALL_MINUTES:.0f} minutes")
        break

if not (ckpt / f"best_{FUSION}.pt").is_file():
    write_gpu_termination_ledger(
        BASELINE_RESULTS_DIR / f"07_descriptor_fusion_{FUSION}_runtime.json", device=DEVICE,
        started_at=GPU_RUN_STARTED, record=GPU_RUN, reason="no_complete_validation_epoch",
        max_epochs=MAX_EPOCHS, max_wall_minutes=MAX_WALL_MINUTES,
    )
    raise RuntimeError("GPU cap reached before one complete validation epoch; no checkpoint was created")
state = torch.load(ckpt / f"best_{FUSION}.pt", map_location=DEVICE, weights_only=False)
model.load_state_dict(state["model"])
val_m, val_y, val_p = evaluate(val_dl, return_predictions=True)
pd.DataFrame(hist).to_csv(BASELINE_RESULTS_DIR / f"07_descriptor_fusion_{FUSION}_history.csv", index=False)
elapsed_seconds = time.perf_counter() - GPU_RUN_STARTED
runtime = {
    "device": str(DEVICE), "epochs_completed": len(hist),
    "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
    "max_wall_minutes": MAX_WALL_MINUTES,
    "evaluated_test": EVALUATE_TEST,
    "gpu_run": GPU_RUN,
    "wall_seconds": elapsed_seconds,
    "gpu_wall_hours": elapsed_seconds / 3600 if DEVICE.type == "cuda" else 0.0,
}
(BASELINE_RESULTS_DIR / f"07_descriptor_fusion_{FUSION}_runtime.json").write_text(json.dumps(runtime, indent=2))
print("runtime", runtime)
pred_dir = RESULTS_DIR / "predictions"
pred_dir.mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    pred_dir / f"07_descriptor_fusion_{FUSION}_validation.npz",
    song_ids=np.asarray(val_dl.dataset.df["song_id"].astype(str).to_numpy(), dtype=str),
    label_names=np.asarray(TAG_NAMES, dtype=str), targets=val_y, scores=val_p,
)
if EVALUATE_TEST:
    test_m, test_y, test_p = evaluate(test_dl, return_predictions=True)
    print("FINAL TEST split-0", test_m)
    (BASELINE_RESULTS_DIR / f"07_descriptor_fusion_{FUSION}_test.json").write_text(json.dumps(test_m, indent=2))
    np.savez_compressed(
        pred_dir / f"07_descriptor_fusion_{FUSION}_test.npz",
        song_ids=np.asarray(test_dl.dataset.df["song_id"].astype(str).to_numpy(), dtype=str),
        label_names=np.asarray(TAG_NAMES, dtype=str), targets=test_y, scores=test_p,
    )
else:
    print("Test evaluation skipped. Set EVALUATE_TEST=1 only for the selected final run.")
elapsed_seconds = time.perf_counter() - GPU_RUN_STARTED
runtime["wall_seconds"] = elapsed_seconds
runtime["gpu_wall_hours"] = elapsed_seconds / 3600 if DEVICE.type == "cuda" else 0.0
(BASELINE_RESULTS_DIR / f"07_descriptor_fusion_{FUSION}_runtime.json").write_text(json.dumps(runtime, indent=2))
print("final runtime", runtime)
'''
        ),
    ],
)


write(
    "08_baseline_evaluation.ipynb",
    [
        md(
            """# 08 — Baseline Evaluation

Compare recorded test metrics from the direct CNN and descriptor-fusion baselines. This notebook does not claim to evaluate the proposed model."""
        ),
        md(KAGGLE_SETUP),
        md("## Step 0 — Packages"),
        code("""!pip install -q scikit-learn tqdm"""),
        md("## Step 1 — Bootstrap paths"),
        code(SHARED_BOOTSTRAP),
        md("## Step 2 — Collect metrics already written by notebooks 02 and 07"),
        code(
            r'''
result_files = sorted(BASELINE_RESULTS_DIR.glob("02_baseline_test.json")) + sorted(BASELINE_RESULTS_DIR.glob("07_descriptor_fusion_*_test.json"))
if not result_files:
    raise FileNotFoundError("Run notebooks 02 and 07 before comparing baseline metrics.")
rows = []
for f in result_files:
    rows.append({"model": f.stem, **json.loads(f.read_text())})
ablation_table = pd.DataFrame(rows)
ablation_table.to_csv(BASELINE_RESULTS_DIR / "08_core_comparison.csv", index=False)
ablation_table
'''
        ),
    ],
)


write(
    "09_baseline_explainability.ipynb",
    [
        md(
            """# 09 — Descriptor-Fusion Baseline Analysis

On held-out **split-0 test** tracks:

1. Attention over {Instrument, Rhythm, Timbre, Harmony}
2. Qualitative listening table for baseline inspection

Attention is recorded for comparison only; it is not treated as a validated explanation."""
        ),
        md(KAGGLE_SETUP),
        md("## Step 0 — Packages"),
        code("""!pip install -q matplotlib tqdm"""),
        md("## Step 1 — Bootstrap paths"),
        code(SHARED_BOOTSTRAP),
        md("## Step 2 — Guard: require a trained descriptor-fusion checkpoint"),
        code(
            r'''
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
_rcheck = pd.read_csv(rhythm_csv)
if "source" in _rcheck.columns and (_rcheck["source"] == "mel_proxy").any():
    raise RuntimeError("rhythm_song.csv still has mel_proxy rows — re-run notebook 04, then re-train 07.")
print("checkpoint OK", CKPT_PATH, "bytes=", CKPT_PATH.stat().st_size)
'''
        ),
        md("## Step 3 — Descriptor-fusion attention on held-out test songs"),
        code(
            r'''
import matplotlib.pyplot as plt
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset

if not MANIFEST.exists():
    raise FileNotFoundError("Run notebook 01 first.")
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

def num_cols(df):
    metadata = {"song_id", "source", "split", "schema_version", "target_variant", "extractor_version", "teacher_name", "teacher_version"}
    return [c for c in df.columns if c not in metadata and pd.api.types.is_numeric_dtype(df[c])]

r_cols, t_cols, h_cols = num_cols(rhythm), num_cols(timbre), num_cols(harmony)
have = set(inst_map) & set(rhythm.index) & set(timbre.index) & set(harmony.index)
manifest = manifest[manifest["song_id"].isin(have)].copy()
test_ids = manifest.loc[manifest["split"] == "test", "song_id"].astype(str).head(12).tolist()
if not test_ids:
    raise RuntimeError("No overlapping test songs with all four concepts.")
print("sample test songs:", test_ids)

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
        self.p_i = nn.Linear(d_i, token)
        self.p_r = nn.Linear(d_r, token)
        self.p_t = nn.Linear(d_t, token)
        self.p_h = nn.Linear(d_h, token)
        self.attn = nn.MultiheadAttention(token, 1, batch_first=True)
        self.out = nn.Sequential(nn.Linear(token, fused), nn.ReLU(), nn.Dropout(0.2))
        self.head = nn.Linear(fused, n_tags)
    def forward(self, inst, r, t, h):
        tok = torch.stack([self.p_i(inst), self.p_r(r), self.p_t(t), self.p_h(h)], 1)
        o, w = self.attn(tok, tok, tok, need_weights=True)
        return self.head(self.out(o.mean(1))), w

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
n_tags = len(state.get("tags") or []) or 87
model = AttentionFusion(64, len(r_cols), len(t_cols), len(h_cols), n_tags=n_tags).to(DEVICE)
model.load_state_dict(state["model"])
model.eval()

concept_names = ["instrument", "rhythm", "timbre", "harmony"]
attn_rows = []
with torch.no_grad():
    for inst, r, t, h, sid in DataLoader(DS(test_ids), batch_size=8, num_workers=0):
        _, w = model(inst.to(DEVICE), r.to(DEVICE), t.to(DEVICE), h.to(DEVICE))
        weights = w.mean(dim=1).cpu().numpy()
        for i, song in enumerate(sid):
            row = {"song_id": str(song)}
            row.update({c: float(weights[i, j]) for j, c in enumerate(concept_names)})
            attn_rows.append(row)

attn_df = pd.DataFrame(attn_rows)
attn_df.to_csv(BASELINE_RESULTS_DIR / "09_attention_weights_sample.csv", index=False)
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(concept_names, attn_df[concept_names].mean(0).values)
ax.set_ylabel("mean attention")
ax.set_title("Mean concept attention (descriptor-fusion baseline)")
fig.tight_layout()
fig.savefig(BASELINE_RESULTS_DIR / "09_mean_attention.png", dpi=150)
plt.show()
attn_df.head()
'''
        ),
        md("## Step 4 — Qualitative listening templates (human labels only — model side is real)"),
        code(
            r'''
qual = []
for sid in test_ids[:5]:
    top = concept_names[int(attn_df.loc[attn_df.song_id == sid, concept_names].values.argmax())]
    qual.append({"song_id": sid, "audible_dominant_concept": "", "model_top_concept": top, "agree": "", "comment": ""})
pd.DataFrame(qual).to_csv(BASELINE_RESULTS_DIR / "09_qualitative_listening.csv", index=False)
print("Wrote real attention + listening template under", BASELINE_RESULTS_DIR)
'''
        ),
    ],
)

print("Done. Notebooks in", OUT)
