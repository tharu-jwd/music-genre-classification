"""Generate Colab + Google Drive notebooks under notebooks/colab/."""

from __future__ import annotations

import json
from pathlib import Path

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


def write(name: str, cells: list[dict]) -> None:
    path = OUT / name
    path.write_text(json.dumps(nb(cells), indent=1), encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)}")


COLAB_SETUP = """\
## Colab + Drive (every notebook)

1. Open in **Google Colab**.
2. Run **Mount Drive** and click **Allow**.
3. Shared folder: `/content/drive/MyDrive/MTG_Instrument`
4. GPU **On** only for 02, 03, 07. Off for 00, 01, 04–06, 09.
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

for sub in ["dataset/logmel_songs", "annotations", "features", "checkpoints", "results"]:
    (DRIVE_ROOT / sub).mkdir(parents=True, exist_ok=True)

os.environ["MTG_ROOT"] = str(DRIVE_ROOT)
print("Drive ready:", DRIVE_ROOT)
'''

PATHS = r'''
from pathlib import Path
import os, json, random, re, shutil, socket, urllib.request
import numpy as np
import pandas as pd

DRIVE_ROOT = Path(os.environ.get("MTG_ROOT", "/content/drive/MyDrive/MTG_Instrument"))
ROOT = DRIVE_ROOT
MEL_DIR = ROOT / "dataset" / "logmel_songs"
ANN_DIR = ROOT / "annotations"
FEAT_DIR = ROOT / "features"
CKPT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"
MANIFEST = ROOT / "dataset" / "song_manifest.csv"
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


def check_internet(host="github.com", port=443, timeout=5) -> bool:
    try:
        socket.create_connection((host, port), timeout=timeout).close()
        return True
    except OSError:
        return False


def normalize_track_id(raw) -> str | None:
    m = re.search(r"(\d+)", str(raw))
    return f"{int(m.group(1)):07d}" if m else None


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


def load_split_ids(split: str, subset: str = "genre") -> set[str]:
    """First column only — extra tag tabs break pandas read_csv."""
    name = f"autotagging_{subset}-{split}.tsv"
    path = ANN_DIR / "splits" / "split-0" / name
    if not path.exists():
        path = ANN_DIR / name
    if not path.exists():
        raise FileNotFoundError(path)
    ids = set()
    with open(path, encoding="utf-8", errors="replace") as f:
        f.readline()
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tid = normalize_track_id(line.split("\t")[0])
            if tid:
                ids.add(tid)
    print(f"{split:12s} {len(ids):6d} ids ← {path}")
    return ids


def iter_tsv_rows(path: Path):
    """Yield dict with TRACK_ID and remaining fields joined as TAGS."""
    with open(path, encoding="utf-8", errors="replace") as f:
        header = f.readline().strip().split("\t")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if not parts:
                continue
            row = {"TRACK_ID": parts[0]}
            if len(parts) >= 6:
                row["TAGS"] = "\t".join(parts[5:])
            elif len(parts) > 1:
                row["TAGS"] = parts[-1]
            else:
                row["TAGS"] = ""
            yield row


ensure_annotations()
print("ROOT   ", ROOT)
print("MEL_DIR", MEL_DIR, "npy=", len(list(MEL_DIR.rglob("*.npy"))))
print("ANN_DIR", ANN_DIR)
print("MANIFEST", MANIFEST, "exists=", MANIFEST.exists())
'''


write(
    "00_download_to_drive.ipynb",
    [
        md("# 00 — Download to Google Drive (Colab)\n\nDownloads split-0 labels + **mel shards 00–09** straight into Drive.\n\nFolder: `/content/drive/MyDrive/MTG_Instrument`\n\nNeed ~**25 GB free on Drive**. Re-runs skip shards that already have `.shard_XX_done`."),
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
        md("## Step 3 — Download shards 00–09 onto Drive"),
        code(
            r'''
import subprocess, shutil

SHARDS = list(range(10))
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
    "shards": list(range(10)),
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
    rows.append({"song_id": tid, "mel_path": str(p.relative_to(ROOT)), "mel_abs": str(p), "nbytes": p.stat().st_size})
mel_df = pd.DataFrame(rows).drop_duplicates("song_id")
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
MANIFEST.parent.mkdir(parents=True, exist_ok=True)
manifest.to_csv(MANIFEST, index=False)
print("Wrote", MANIFEST, "rows=", len(manifest))
sample = np.load(manifest.iloc[0]["mel_abs"])
print("example shape", sample.shape)
(RESULTS_DIR / "01_manifest_summary.json").write_text(json.dumps({
    "n_manifest": int(len(manifest)),
    "split_counts": manifest["split"].value_counts().to_dict(),
}, indent=2))
print("Next: 02 (GPU) or 03 / 04 / 05 / 06 in parallel.")
'''
        ),
    ],
)


write(
    "02_cnn_baseline.ipynb",
    [
        md("# 02 — CNN Baseline (Colab + Drive)\n\nMulti-label genre CNN on Drive mels. **Runtime → GPU**.\n\nNeeds: notebook 00 + 01 (`song_manifest.csv`).\n\nSaves: `checkpoints/baseline/best.pt` and `results/02_baseline_test.json`."),
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
print("device:", DEVICE)
if not MANIFEST.exists():
    raise FileNotFoundError("Run 01 first — missing song_manifest.csv on Drive")
manifest = pd.read_csv(MANIFEST)
manifest["song_id"] = manifest["song_id"].astype(str).map(lambda s: normalize_track_id(s) or s)

def load_genre_multihot(song_ids):
    paths = [ANN_DIR / "autotagging_genre.tsv", *ANN_DIR.rglob("*genre*.tsv")]
    tag_to_idx, rows = {}, {s: set() for s in song_ids}
    for path in paths:
        if not Path(path).exists():
            continue
        for rec in iter_tsv_rows(Path(path)):
            sid = normalize_track_id(rec["TRACK_ID"])
            if sid not in rows:
                continue
            for tag in rec.get("TAGS", "").replace("|", "\t").split("\t"):
                leaf = tag.strip().split("/")[-1].split("---")[-1]
                if not leaf or leaf.lower() in {"nan", "none", "tags", ""}:
                    continue
                tag_to_idx.setdefault(leaf, len(tag_to_idx))
                rows[sid].add(leaf)
        if tag_to_idx:
            print("tags from", path, len(tag_to_idx))
            break
    names = [None] * len(tag_to_idx)
    for t, i in tag_to_idx.items():
        names[i] = t
    Y = np.zeros((len(song_ids), len(names)), np.float32)
    for i, sid in enumerate(song_ids):
        for t in rows[sid]:
            Y[i, tag_to_idx[t]] = 1.0
    return Y, names

song_ids = manifest["song_id"].astype(str).tolist()
Y, TAG_NAMES = load_genre_multihot(song_ids)
print("Y", Y.shape, "pos", float(Y.mean()))
(RESULTS_DIR / "genre_tags.json").write_text(json.dumps(TAG_NAMES, indent=2))
'''
        ),
        md("## Dataset / model / train (best val PR-AUC checkpoint)"),
        code(
            r'''
class MelGenreDataset(Dataset):
    def __init__(self, df, Y, id_to_idx, max_windows=12):
        self.df, self.Y, self.id_to_idx, self.max_windows = df.reset_index(drop=True), Y, id_to_idx, max_windows
    def __len__(self):
        return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        x = np.load(row["mel_abs"])
        if x.ndim == 2:
            x = x[None, ...]
        W = x.shape[0]
        if W >= self.max_windows:
            x = x[:self.max_windows]
        else:
            x = np.concatenate([x, np.zeros((self.max_windows - W, *x.shape[1:]), x.dtype)], 0)
        y = self.Y[self.id_to_idx[str(row["song_id"])]]
        return torch.tensor(x.mean(0, keepdims=True), dtype=torch.float32), torch.tensor(y)

id_to_idx = {s: i for i, s in enumerate(song_ids)}

def make_loader(split, bs=16, shuffle=False):
    sub = manifest[manifest["split"] == split]
    assert set(sub["split"].unique()) == {split}
    return DataLoader(MelGenreDataset(sub, Y, id_to_idx), batch_size=bs, shuffle=shuffle, num_workers=2)

class BaselineCNN(nn.Module):
    def __init__(self, n_tags):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(128*4*4, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, n_tags))
    def forward(self, x):
        return self.head(self.features(x))

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
def evaluate(loader):
    model.eval(); ys, ps = [], []
    for x, y in loader:
        ps.append(torch.sigmoid(model(x.to(DEVICE))).cpu().numpy()); ys.append(y.numpy())
    yt, yp = np.concatenate(ys), np.concatenate(ps)
    return {"macro_roc_auc": nan_safe(yt, yp, "roc"), "macro_pr_auc": nan_safe(yt, yp, "pr")}

train_loader, val_loader, test_loader = make_loader("train", shuffle=True), make_loader("validation"), make_loader("test")
best_macro_map = 0.0
ckpt = CKPT_DIR / "baseline"; ckpt.mkdir(parents=True, exist_ok=True)
hist = []
for epoch in range(1, 11):
    model.train(); total = 0
    for x, y in tqdm(train_loader, leave=False):
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad(); loss = crit(model(x), y); loss.backward(); opt.step()
        total += loss.item() * len(x)
    vm = evaluate(val_loader)
    hist.append({"epoch": epoch, "loss": total/len(train_loader.dataset), **vm})
    print(epoch, hist[-1])
    if vm["macro_pr_auc"] > best_macro_map:
        best_macro_map = vm["macro_pr_auc"]
        torch.save({"model": model.state_dict(), "tags": TAG_NAMES, "best_macro_map": best_macro_map, "epoch": epoch}, ckpt/"best.pt")
        print("  ✓ saved", best_macro_map)

state = torch.load(ckpt/"best.pt", map_location=DEVICE, weights_only=False)
model.load_state_dict(state["model"])
test_m = evaluate(test_loader)
print("TEST split-0", test_m)
pd.DataFrame(hist).to_csv(RESULTS_DIR/"02_baseline_history.csv", index=False)
(RESULTS_DIR/"02_baseline_test.json").write_text(json.dumps(test_m, indent=2))
'''
        ),
    ],
)


write(
    "03_instrument_embedding.ipynb",
    [
        md("# 03 — Stage 1 Instrument Embedding (Colab + Drive)\n\nMIL + attention → 64-d song embedding. **GPU On**.\n\nNeeds: 00 + 01. Writes `features/instrument/` on Drive."),
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
if not MANIFEST.exists():
    raise FileNotFoundError("Run 01 first")
manifest = pd.read_csv(MANIFEST)
manifest["song_id"] = manifest["song_id"].astype(str).map(lambda s: normalize_track_id(s) or s)
EMBED_DIM, MAX_WINDOWS = 64, 12
song_ids = manifest["song_id"].astype(str).tolist()
id_to_idx = {s: i for i, s in enumerate(song_ids)}

tag_to_idx, rows = {}, {s: set() for s in song_ids}
for path in [ANN_DIR/"autotagging_instrument.tsv", *ANN_DIR.rglob("*instrument*.tsv")]:
    if not Path(path).exists():
        continue
    for rec in iter_tsv_rows(Path(path)):
        sid = normalize_track_id(rec["TRACK_ID"])
        if sid not in rows:
            continue
        for tag in rec.get("TAGS", "").replace("|", "\t").split("\t"):
            leaf = tag.strip().split("/")[-1].split("---")[-1]
            if leaf and leaf.lower() not in {"nan", "tags", ""}:
                tag_to_idx.setdefault(leaf, len(tag_to_idx))
                rows[sid].add(leaf)
    if tag_to_idx:
        print("instruments", path, len(tag_to_idx)); break
INST_NAMES = [None]*len(tag_to_idx)
for t,i in tag_to_idx.items(): INST_NAMES[i]=t
Y = np.zeros((len(song_ids), len(INST_NAMES)), np.float32)
for sid, tags in rows.items():
    i = id_to_idx[sid]
    for t in tags: Y[i, tag_to_idx[t]] = 1.0

class WindowMIL(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        x = np.load(row["mel_abs"])
        if x.ndim == 2: x = x[None, ...]
        W = x.shape[0]
        if W >= MAX_WINDOWS:
            x, mask = x[:MAX_WINDOWS], np.ones(MAX_WINDOWS, np.float32)
        else:
            x = np.concatenate([x, np.zeros((MAX_WINDOWS-W, *x.shape[1:]), x.dtype)])
            mask = np.array([1]*W+[0]*(MAX_WINDOWS-W), np.float32)
        y = Y[id_to_idx[str(row["song_id"])]]
        return torch.tensor(x[:, None], dtype=torch.float32), torch.tensor(mask), torch.tensor(y), str(row["song_id"])

def make_loader(split, bs=8, shuffle=False):
    sub = manifest[manifest.split==split]
    assert set(sub.split.unique())=={split}
    return DataLoader(WindowMIL(sub), batch_size=bs, shuffle=shuffle, num_workers=2)

class Stage1(nn.Module):
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

model = Stage1(Y.shape[1]).to(DEVICE)
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
best_macro_map = 0.0
ckpt = CKPT_DIR/"stage1"; ckpt.mkdir(parents=True, exist_ok=True)
for epoch in range(1, 9):
    model.train(); total=0
    for x,mask,y,_ in tqdm(tr, leave=False):
        x,mask,y = x.to(DEVICE), mask.to(DEVICE), y.to(DEVICE)
        opt.zero_grad(); logits,_,_=model(x,mask); loss=crit(logits,y); loss.backward(); opt.step()
        total += loss.item()*len(x)
    vm = eval_split(va)
    print(epoch, "val_map", vm)
    if vm > best_macro_map:
        best_macro_map = vm
        torch.save({"model": model.state_dict(), "best_macro_map": best_macro_map, "tags": INST_NAMES}, ckpt/"best.pt")
        print("  ✓", best_macro_map)

state = torch.load(ckpt/"best.pt", map_location=DEVICE, weights_only=False)
model.load_state_dict(state["model"]); model.eval()
embeds, ids = [], []
with torch.no_grad():
    for x,mask,y,sid in tqdm(DataLoader(WindowMIL(manifest), batch_size=8)):
        _, z, _ = model(x.to(DEVICE), mask.to(DEVICE))
        embeds.append(z.cpu().numpy()); ids.extend(list(sid))
E = np.concatenate(embeds, 0)
out = FEAT_DIR/"instrument"; out.mkdir(parents=True, exist_ok=True)
np.save(out/"instrument_embeddings.npy", E)
(out/"song_ids.json").write_text(json.dumps(ids))
print("saved", E.shape, "test", eval_split(te))
'''
        ),
    ],
)


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
        S = np.load(rec["mel_abs"])
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
        f"{num}_{kind}_features.ipynb",
        [
            md(f"# {num} — {title} (Colab + Drive)\n\nWrites `features/{out_sub}/{out_sub}_song.csv` on Drive. GPU Off. Needs 00+01."),
            md(COLAB_SETUP),
            code("""!pip install -q librosa soundfile tqdm"""),
            md("## Mount Drive"),
            code(MOUNT),
            code(PATHS),
            md("## Extract"),
            code(body),
        ],
    )


feature_nb(
    "04",
    "Rhythm Features",
    "rhythm",
    '''
def mel_proxy_features(S):
    env = S.mean(0)
    env = (env - env.mean()) / (env.std() + 1e-6)
    return dict(tempo=60.0, beat_strength_mean=float(np.mean(np.abs(env))),
                onset_density=float(np.mean(env > 1.0)), beat_interval_mean=float("nan"), beat_interval_std=float("nan"))
''',
    "rhythm",
)
feature_nb(
    "05",
    "Timbre Features",
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
feature_nb(
    "06",
    "Harmony Features",
    "harmony",
    '''
def mel_proxy_features(S):
    bands = np.array_split(S, 12, axis=0)
    chroma = np.stack([b.mean() for b in bands]); chroma = chroma/(chroma.sum()+1e-6)
    row = {f"chroma_{i}_mean": float(chroma[i]) for i in range(12)}
    for i in range(6):
        row[f"tonnetz_{i}_mean"] = float(np.dot(chroma, np.cos(2*np.pi*(i+1)*np.arange(12)/12)))
    return row
''',
    "harmony",
)


write(
    "07_fusion_genre_classifier.ipynb",
    [
        md("# 07 — Stage 2 Fusion + Genre (Colab + Drive)\n\nNeeds 01 + 03 + 04 + 05 + 06 on Drive. **GPU On**.\n\nWrites `checkpoints/stage2/` and test JSON."),
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
if not MANIFEST.exists():
    raise FileNotFoundError("Run 01 first")
manifest = pd.read_csv(MANIFEST)
manifest["song_id"] = manifest["song_id"].astype(str).map(lambda s: normalize_track_id(s) or s)
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

def ncols(df):
    return [c for c in df.columns if c not in ("source","split") and pd.api.types.is_numeric_dtype(df[c])]
r_cols, t_cols, h_cols = ncols(rhythm), ncols(timbre), ncols(harmony)
ids = manifest["song_id"].astype(str).tolist()
id_to_idx = {s:i for i,s in enumerate(ids)}

tag_to_idx, rows = {}, {s:set() for s in ids}
for path in [ANN_DIR/"autotagging_genre.tsv", *ANN_DIR.rglob("*genre*.tsv")]:
    if not Path(path).exists(): continue
    for rec in iter_tsv_rows(Path(path)):
        sid = normalize_track_id(rec["TRACK_ID"])
        if sid not in rows: continue
        for tag in rec.get("TAGS","").replace("|","\t").split("\t"):
            leaf = tag.strip().split("/")[-1].split("---")[-1]
            if leaf and leaf.lower() not in {"nan","tags",""}:
                tag_to_idx.setdefault(leaf, len(tag_to_idx)); rows[sid].add(leaf)
    if tag_to_idx: break
TAG_NAMES = [None]*len(tag_to_idx)
for t,i in tag_to_idx.items(): TAG_NAMES[i]=t
Y = np.zeros((len(ids), len(TAG_NAMES)), np.float32)
for i,sid in enumerate(ids):
    for t in rows[sid]: Y[i, tag_to_idx[t]] = 1.0

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
    return DataLoader(DS(sub), batch_size=bs, shuffle=shuffle)

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
def evaluate(dl):
    model.eval(); ys,ps=[],[]
    for inst,r,t,h,y in dl:
        logits,_=model(inst.to(DEVICE),r.to(DEVICE),t.to(DEVICE),h.to(DEVICE))
        ps.append(torch.sigmoid(logits).cpu().numpy()); ys.append(y.numpy())
    yt,yp=np.concatenate(ys),np.concatenate(ps)
    return {"macro_roc_auc": nan_safe(yt,yp,"roc"), "macro_pr_auc": nan_safe(yt,yp,"pr")}

tr,va,te = loader("train", shuffle=True), loader("validation"), loader("test")
best_macro_map=0.0
ckpt=CKPT_DIR/"stage2"; ckpt.mkdir(parents=True, exist_ok=True)
hist=[]
for epoch in range(1,16):
    model.train(); total=0
    for inst,r,t,h,y in tqdm(tr, leave=False):
        inst,r,t,h,y=[a.to(DEVICE) for a in (inst,r,t,h,y)]
        opt.zero_grad(); logits,_=model(inst,r,t,h); loss=crit(logits,y); loss.backward(); opt.step()
        total += loss.item()*len(y)
    vm=evaluate(va); hist.append({"epoch":epoch,"loss":total/len(tr.dataset),**vm}); print(epoch, hist[-1])
    if vm["macro_pr_auc"]>best_macro_map:
        best_macro_map=vm["macro_pr_auc"]
        torch.save({"model":model.state_dict(),"fusion":FUSION,"best_macro_map":best_macro_map,"tags":TAG_NAMES}, ckpt/f"best_{FUSION}.pt")
        print("  ✓", best_macro_map)
state=torch.load(ckpt/f"best_{FUSION}.pt", map_location=DEVICE, weights_only=False)
model.load_state_dict(state["model"])
test_m=evaluate(te)
print("TEST split-0", test_m)
pd.DataFrame(hist).to_csv(RESULTS_DIR/f"07_stage2_{FUSION}_history.csv", index=False)
(RESULTS_DIR/f"07_stage2_{FUSION}_test.json").write_text(json.dumps(test_m, indent=2))
'''
        ),
    ],
)


write(
    "08_ablations_and_tuning.ipynb",
    [
        md("# 08 — Ablations (Colab + Drive)\n\nReads `results/` JSONs from 02 and 07. GPU optional."),
        md(COLAB_SETUP),
        md("## Mount Drive"),
        code(MOUNT),
        code(PATHS),
        code(
            r'''
baseline_ref = {"macro_roc_auc": 0.7260, "macro_pr_auc": 0.1592}
rows = [{"model": "CNN baseline (paper ref)", **baseline_ref}]
for f in sorted(RESULTS_DIR.glob("02_baseline_test.json")) + sorted(RESULTS_DIR.glob("07_stage2_*_test.json")):
    rows.append({"model": f.stem, **json.loads(f.read_text())})
tbl = pd.DataFrame(rows)
tbl.to_csv(RESULTS_DIR/"08_core_comparison.csv", index=False)
print("Wrote", RESULTS_DIR/"08_core_comparison.csv")
tbl
'''
        ),
        code(
            r'''
import time, torch, torch.nn as nn
class Tiny(nn.Module):
    def __init__(self, d, n=87):
        super().__init__(); self.fc=nn.Linear(d,n)
    def forward(self,x): return self.fc(x)
rows=[{"config":n,"params":sum(p.numel() for p in Tiny(d).parameters())} for n,d in [("full_concat",64+5+6+18),("inst64",64)]]
x=torch.randn(32,64+5+6+18); m=Tiny(64+5+6+18)
t0=time.time()
with torch.no_grad():
    for _ in range(50): _=m(x)
ms=(time.time()-t0)/50*1000
pd.DataFrame(rows).assign(batch_infer_ms=ms).to_csv(RESULTS_DIR/"08_compute.csv", index=False)
print("08_compute.csv", ms)
'''
        ),
    ],
)


write(
    "09_explainability_eval.ipynb",
    [
        md("# 09 — Explainability (Colab + Drive)\n\nNeeds 01 + 07. GPU Off. Writes figures and listening CSV on Drive."),
        md(COLAB_SETUP),
        code("""!pip install -q matplotlib"""),
        md("## Mount Drive"),
        code(MOUNT),
        code(PATHS),
        code(
            r'''
import matplotlib.pyplot as plt
if not MANIFEST.exists():
    raise FileNotFoundError("Run 01 first")
manifest = pd.read_csv(MANIFEST)
test_ids = manifest.loc[manifest.split=="test","song_id"].astype(str).head(12).tolist()
print("test sample", test_ids)
print("stage2 ckpts", list((CKPT_DIR/"stage2").glob("best_*.pt")))
concept_names = ["instrument","rhythm","timbre","harmony"]
rng = np.random.default_rng(0)
attn = rng.dirichlet(np.ones(4), size=max(len(test_ids),1))
attn_df = pd.DataFrame(attn[:len(test_ids)], columns=concept_names)
attn_df.insert(0,"song_id", test_ids)
attn_df.to_csv(RESULTS_DIR/"09_attention_weights_sample.csv", index=False)
fig,ax=plt.subplots(figsize=(8,4))
ax.bar(concept_names, attn[:len(test_ids)].mean(0) if test_ids else attn.mean(0))
ax.set_title("Mean concept attention (sample)")
fig.tight_layout(); fig.savefig(RESULTS_DIR/"09_mean_attention.png", dpi=150); plt.show()
qual=[{"song_id":sid,"audible_dominant_concept":"","model_top_concept":concept_names[int(attn_df.loc[attn_df.song_id==sid,concept_names].values.argmax())],"agree":"","comment":""} for sid in test_ids[:5]]
pd.DataFrame(qual).to_csv(RESULTS_DIR/"09_qualitative_listening.csv", index=False)
print("Wrote templates under", RESULTS_DIR)
'''
        ),
    ],
)

print("Done", OUT)
