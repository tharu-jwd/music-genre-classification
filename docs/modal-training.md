# Training the joint architecture on Modal

The Modal runner trains `scripts/train_joint.py` against `data/full_dataset.csv`.
Source code is built into a reproducible image, while large data and run outputs
remain in persistent Modal Volumes. Each collaborator creates the same two Volume
names in their own Modal workspace, so no credentials or workspace IDs belong in
Git.

## 1. Expected data

`full_dataset.csv` contains the log-mel path, 41 instrument labels, 10 rhythm
targets, 35 timbre targets, and 6 genre labels. It does not contain the 12-bin
chroma distribution used by the harmony auxiliary loss. The trainer therefore
still predicts harmony and feeds it into fusion, but masks the harmony auxiliary
loss for this dataset. This is intentional; do not treat the existing tonal
summary columns as chroma bins.

The log-mel arrays are not stored in Git. The Modal data Volume must look like:

```text
music-genre-data/
├── dataset/
│   ├── full_dataset.csv
│   ├── track_split_assignments.csv
│   ├── logmel_config.json        # optional, required for stacked 3-D arrays
│   └── logmel_audit.csv          # optional, required for stacked 3-D arrays
└── logmel_songs/
    └── <suffix>/<track-id>.npy
```

The `logmel_songs` relative path is taken from each CSV `logmel_path`; its old
Colab prefix is replaced by the Volume mount automatically.

## 2. One-time setup per collaborator

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-modal.txt
modal setup
modal volume create music-genre-data
modal volume create music-genre-runs
```

Upload the combined table, the tracked split assignments, and your local log-mel
cache:

```bash
modal volume put music-genre-data data/full_dataset.csv dataset/full_dataset.csv
modal volume put music-genre-data data/track_split_assignments.csv dataset/track_split_assignments.csv
modal volume put music-genre-data /absolute/path/to/logmel_songs logmel_songs
```

If `data/logmel_config.json` and `data/logmel_audit.csv` exist, upload them into
`dataset/` too. They are mandatory when each `.npy` has shape
`(windows, mel_bins, frames)` and optional for a continuous `(mel_bins, frames)`
array.

Confirm the upload:

```bash
modal volume ls music-genre-data dataset
modal volume ls music-genre-data logmel_songs
```

## 3. Smoke test and full training

Run the bounded smoke test first. `--quick` uses at most 32 tracks per split and
three epochs:

```bash
modal run modal_app.py --quick --run-name smoke-v1
```

Then submit the full run. `--detach` lets the job continue if the local terminal
disconnects:

```bash
modal run --detach modal_app.py \
  --run-name joint-full-v1 \
  --epochs 30 \
  --batch-size 1 \
  --gpu A10
```

The GPU can be changed at submission time, for example `--gpu L40S` or
`--gpu A100-40GB`. Start with batch size 1 because the model processes up to 12
long windows per track; raise it only after observing GPU memory use.

## 4. Retrieve results

Every run writes `best.pt` and `results.json` under its run name:

```bash
modal volume ls music-genre-runs joint-full-v1
modal volume get music-genre-runs joint-full-v1 ./modal-results/joint-full-v1
```

Use a new `--run-name` for every experiment. Runs with the same name share an
output directory and can overwrite artifacts.

## Git hand-off

The combined CSV and split assignments are already tracked, so a collaborator can
upload both immediately after cloning. Do not commit `.npy` arrays, Modal
credentials, checkpoints, or results; the existing `.gitignore` keeps those large
or private artifacts out of Git.
