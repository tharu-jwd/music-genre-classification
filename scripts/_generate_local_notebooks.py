"""Generate Windows-local versions of the three-shard experiment notebooks.

The source notebooks remain unchanged under ``notebooks/colab``. Generated local
notebooks use ``data/MTG_Instrument`` by default and can be pointed elsewhere
with the ``MTG_ROOT`` environment variable.
"""

from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = REPO_ROOT / "notebooks" / "colab"
OUTPUT_DIR = REPO_ROOT / "notebooks" / "local"

LOCAL_SETUP = '''from pathlib import Path
import os

# Change this one value if you want the data on another local drive.
ROOT = Path(os.environ.get("MTG_ROOT", Path.cwd() / "data" / "MTG_Instrument")).resolve()
for sub in ["dataset/logmel_songs", "annotations", "features", "checkpoints", "results"]:
    (ROOT / sub).mkdir(parents=True, exist_ok=True)

os.environ["MTG_ROOT"] = str(ROOT)
print("Local data root:", ROOT)
'''

LOCAL_GUIDE = '''## Local Windows setup (every notebook)

1. Run this cell. It creates/uses `data/MTG_Instrument` in the repository.
2. To use a different drive, set the `MTG_ROOT` environment variable before
   opening Jupyter.
3. Confirm that PyTorch reports `cuda_available: True` before training in
   notebooks 02, 03, and 07.
4. For the RTX 2050 (4 GB), the baseline uses batch size 4 and Stage 1 uses 2.
'''


def text(cell: dict) -> str:
    return "".join(cell.get("source", []))


def lines(value: str) -> list[str]:
    if not value.endswith("\n"):
        value += "\n"
    return value.splitlines(keepends=True)


def adapt_cell(value: str) -> str:
    if 'DRIVE_ROOT = Path("/content/drive/MyDrive/MTG_Instrument")' in value:
        return LOCAL_SETUP
    if value.startswith("## Colab + Drive"):
        return LOCAL_GUIDE

    if value.startswith("# 00 — Download to Google Drive"):
        value = value.replace("# 00 — Download to Google Drive (Colab)", "# 00 — Download locally (Windows)")
        value = value.replace("straight into Drive", "into the local data directory")
        value = value.replace("Folder: `/content/drive/MyDrive/MTG_Instrument`", "Default folder: `data/MTG_Instrument`")
        value = value.replace("free on Drive", "free on the selected disk")
    value = value.replace("onto Drive", "locally")
    value = value.replace("Drive free:", "Disk free:")
    value = value.replace("already on Drive", "already downloaded")
    value = value.replace("on Drive:", "locally:")
    value = value.replace("npy on Drive:", "local npy files:")
    value = value.replace("Next: 01_preprocessing.ipynb in Colab, same Drive folder.", "Next: 01_preprocessing.ipynb in this local folder.")
    value = value.replace('"shards": list(range(10)),', '"shards": SHARDS,')

    # The Colab source uses shell wget/tar. urllib and tarfile work on a stock
    # Windows Python installation, without requiring extra command-line tools.
    value = value.replace("import subprocess, shutil", "import shutil, tarfile")
    value = value.replace(
        'subprocess.check_call(["wget", "-q", "-O", str(tar_path), url])',
        "urllib.request.urlretrieve(url, tar_path)",
    )
    value = value.replace(
        'subprocess.check_call(["tar", "-xf", str(tar_path), "-C", str(MEL_DIR)])',
        "with tarfile.open(tar_path) as archive:\n        archive.extractall(MEL_DIR)",
    )

    # Local storage is reliable enough to load directly. This avoids making a
    # second multi-gigabyte copy of the Mel files as the Drive cache would.
    value = value.replace('MEL_CACHE = Path("/content/mel_cache")', 'MEL_CACHE = ROOT / ".mel_cache"\nLOCAL_MODE = True')
    value = value.replace(
        "    mel_abs = Path(mel_abs)\n    sid = normalize_track_id(mel_abs.stem) or mel_abs.stem.replace(\"/\", \"_\")",
        "    mel_abs = Path(mel_abs)\n    if LOCAL_MODE:\n        return np.asarray(np.load(mel_abs), dtype=np.float32)\n    sid = normalize_track_id(mel_abs.stem) or mel_abs.stem.replace(\"/\", \"_\")",
    )

    # Conservative defaults for the 4 GB RTX 2050. Stage 2 consumes only
    # compact feature vectors and therefore keeps its original batch size.
    value = value.replace("def make_loader(split, bs=16, shuffle=False):", "def make_loader(split, bs=4, shuffle=False):")
    value = value.replace("def make_loader(split, bs=8, shuffle=False):", "def make_loader(split, bs=2, shuffle=False):")
    value = value.replace("DataLoader(WindowMIL(manifest), batch_size=8, num_workers=0)", "DataLoader(WindowMIL(manifest), batch_size=2, num_workers=0)")
    return value


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for source_path in sorted(SOURCE_DIR.glob("*.ipynb")):
        notebook = json.loads(source_path.read_text(encoding="utf-8"))
        for cell in notebook["cells"]:
            cell["source"] = lines(adapt_cell(text(cell)))
        target = OUTPUT_DIR / source_path.name.replace("_to_drive", "_local")
        target.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
        print(f"wrote {target.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
