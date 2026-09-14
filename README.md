# Concept-Guided Music Genre Classification

This repository contains the current research pipeline for explainable, multi-label music genre classification on the MTG-Jamendo dataset.

The system represents a song using four musical concepts:

- **Instrumentation** — a learned 64-dimensional song embedding from an attention-based multiple-instance learning model.
- **Rhythm** — tempo and beat descriptors from AcousticBrainz/Essentia metadata.
- **Timbre** — spectral descriptors extracted from audio when available, with a log-Mel approximation for the notebook-only workflow.
- **Harmony** — chroma and Tonnetz descriptors extracted from audio when available, with a log-Mel approximation for the notebook-only workflow.

These representations are joined by either a linear projection or single-head attention and passed to a multi-label genre classifier.

```text
MTG-Jamendo log-Mel songs
          │
          ├── CNN baseline ────────────────────────> genre scores
          │
          └── instrument MIL embedding (64-d)
                    + rhythm descriptors
                    + timbre descriptors
                    + harmony descriptors
                              │
                         concept fusion
                              │
                         genre scores
```

## Start here

Choose exactly one hosted workflow:

- **Google Colab:** follow [notebooks/colab/README.md](notebooks/colab/README.md).
- **Kaggle:** follow [notebooks/kaggle/README.md](notebooks/kaggle/README.md).

Both workflows use the same numbered stages. Do not mix their intermediate paths or storage mechanisms.

| Notebook | Purpose | Main output |
|---|---|---|
| `00` | Download annotations and log-Mel shards | Dataset cache |
| `01` | Build the manifest with official `split-0` | `song_manifest.csv` |
| `02` | Train the direct CNN genre baseline | Baseline checkpoint and metrics |
| `03` | Train instrument MIL and export embeddings | 64-d instrument vectors |
| `04` | Extract rhythm descriptors | `rhythm_song.csv` |
| `05` | Extract timbre descriptors | `timbre_song.csv` |
| `06` | Extract harmony descriptors | `harmony_song.csv` |
| `07` | Train concept fusion and genre classifier | Stage 2 checkpoint and metrics |
| `08` | Collect comparisons and experiment plans | Comparison tables |
| `09` | Export attention summaries for test songs | Attention figure and review table |

The dependency order is `00 → 01 → 02/03 → 04/05/06 → 07 → 08/09`. Notebooks `04`, `05`, and `06` can run independently after `01`.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── data/
│   └── .gitkeep                 # local data is ignored
├── docs/
│   ├── architecture.md
│   ├── current-status.md
│   ├── data-layout.md
│   ├── development.md
│   ├── kaggle-how-to.md
│   └── project-guidelines.md
├── notebooks/
│   ├── README.md
│   ├── colab/                   # Google Drive workflow, 00–09
│   └── kaggle/                  # Kaggle output workflow, 00–09
└── scripts/
    ├── generate_colab_notebooks.py
    └── generate_kaggle_notebooks.py
```

The generator scripts are the source of truth for notebook code. When changing pipeline logic, edit a generator and regenerate its notebooks; do not maintain a notebook-only fork.

## Data and evaluation rules

- Report metrics only on the official MTG-Jamendo `split-0` test set.
- Select checkpoints using validation data only. Never use test data for model selection.
- Exclude undefined per-tag values from macro metrics instead of converting them to zero.
- Join all feature sources using the normalized `song_id`.
- Keep downloaded data, generated features, checkpoints, and notebook outputs outside Git.
- The default baseline downloads shards `00–02`. Increase the `SHARDS` lists only when the selected platform has enough storage, and use the same shard set for log-Mels and AcousticBrainz.

See [docs/architecture.md](docs/architecture.md) for the model and data contracts, [docs/data-layout.md](docs/data-layout.md) for artifact locations, and [docs/current-status.md](docs/current-status.md) for an honest implementation status.

## Development

Generate both notebook sets with Python 3.11 or newer:

```bash
python3 scripts/generate_colab_notebooks.py
python3 scripts/generate_kaggle_notebooks.py
```

The generated notebooks intentionally contain no committed execution output. Training happens on Colab or Kaggle, and resulting artifacts must be persisted using that platform’s storage workflow.

Dependencies used by the notebooks are listed in `requirements.txt`. PyTorch installation differs by CPU/CUDA platform, so hosted notebooks rely on the runtime-provided PyTorch build.

## Project status

The pipeline through Stage 2 training is implemented. It has not been validated end-to-end from the committed repository because datasets, checkpoints, and run outputs are not versioned here. Full ablation training and concept-occlusion evaluation remain planned work; they are not presented as completed results.

This repository is for academic research. Use MTG-Jamendo data according to its own license and terms.
