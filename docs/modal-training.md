# Training the joint architecture on Modal

The Modal runner trains `scripts/train_joint.py` against the future
`data/dataset.csv` contract.
Source code is built into a reproducible image, while large data and run outputs
remain in persistent Modal Volumes. Each collaborator creates the same two Volume
names in their own Modal workspace, so no credentials or workspace IDs belong in
Git.

## 1. Expected data

Each CSV row represents one track and logically contains six vectors:

| Field | Shape | Role |
|---|---:|---|
| `logmel_path` | path to `(mel, time)` or `(windows, mel, time)` | input to the shared CNN encoder |
| instrument columns | 41 | supervision for the instrument branch |
| rhythm columns | 10 | supervision for the rhythm branch |
| timbre columns | 35 | supervision for the timbre branch |
| `chroma_*_mean` columns | 12 | supervision for the harmony branch |
| genre columns | 6 | final multi-label target after concept fusion |

The vectors are stored as explicit numeric columns rather than JSON strings in a
single CSV cell. The exact vector dimensions and column names are defined by
`concept_fusion/contract.py` and `scripts/train_joint.py`. The four branch vectors
are training targets: the fusion layer receives the four **predicted** concept
vectors, preventing ground-truth concepts from leaking into genre inference.

Print the machine-readable contract at any time with:

```bash
python scripts/train_joint.py --print-dataset-schema
```

The existing `full_dataset.csv` can still be used locally as a transitional
dataset, but it has no 12-bin chroma vector. In that compatibility mode the
harmony auxiliary loss is masked. Modal deliberately enables
`--require-harmony-targets` for the future `dataset.csv`, so an incomplete file
fails validation immediately.

The log-mel arrays are not stored in Git. The Modal data Volume must look like:

```text
music-genre-data/
├── dataset/
│   ├── dataset.csv
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
modal volume put music-genre-data data/dataset.csv dataset/dataset.csv
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

The split assignments are already tracked. Once prepared, distribute
`data/dataset.csv` through the project-approved dataset channel, then upload it to
each collaborator's Modal Volume. Do not commit `.npy` arrays, Modal credentials,
checkpoints, or results; the existing `.gitignore` keeps those large or private
artifacts out of Git.
