# Training the joint architecture on Modal

The Modal runner trains `scripts/train_joint.py` against
`data/vector-dataset-normalized.csv`.
Source code is built into a reproducible image, while large data and run outputs
remain in persistent Modal Volumes. Each collaborator creates the same two Volume
names in their own Modal workspace, so no credentials or workspace IDs belong in
Git.

## 1. Expected data

Each CSV row represents one track with this compact seven-column schema:

```text
track_id,path,instrument_vector,rhythm_vector,timbre_vector,harmony_vector,genre
```

| Field | Shape | Role |
|---|---:|---|
| `path` | path to `(mel, time)` or `(windows, mel, time)` | input to the shared CNN encoder |
| `instrument_vector` | 41 | supervision for the instrument branch |
| `rhythm_vector` | 10 | supervision for the rhythm branch |
| `timbre_vector` | 35 | supervision for the timbre branch |
| `harmony_vector` | 12 | tonal descriptor vector from the source dataset |
| `genre` | 6 | final multi-label target after concept fusion |

Each vector cell is a compact JSON array such as `[0,1,0]`. The exact vector
dimensions and member order are defined by `scripts/build_vector_dataset.py` and
can be printed by the trainer. The branch vectors are training targets: the fusion
layer receives **predicted** concepts, preventing ground-truth concepts from
leaking into genre inference.

Print the machine-readable contract at any time with:

```bash
python scripts/train_joint.py --print-dataset-schema
```

Regenerate the compact dataset deterministically with:

```bash
python scripts/build_vector_dataset.py --overwrite
```

The generated harmony vector contains the 12 tonal summary descriptors present in
`full_dataset.csv`; it is not a chroma distribution. The current temporal harmony
head expects 12 pitch-class probabilities, so its auxiliary loss remains masked
until that head is changed to descriptor regression. The vector is preserved in
the CSV without misinterpreting or normalizing it.

The log-mel arrays are not stored in Git. The Modal data Volume must look like:

```text
music-genre-data/
├── dataset/
│   ├── vector-dataset-normalized.csv
│   ├── track_split_assignments.csv
│   ├── logmel_config.json        # optional, required for stacked 3-D arrays
│   └── logmel_audit.csv          # optional, required for stacked 3-D arrays
└── logmel_songs/
    └── <suffix>/<track-id>.npy
```

The `logmel_songs` relative path is taken from each CSV `path`; its old
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
modal volume put music-genre-data \
  data/vector-dataset-normalized.csv \
  dataset/vector-dataset-normalized.csv
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

The normalized vector dataset and split assignments are tracked, so collaborators
can upload both after cloning. Do not commit `.npy` arrays, Modal credentials,
checkpoints, or results; the existing `.gitignore` keeps those large or private
artifacts out of Git.
