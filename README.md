# Concept-Guided Music Genre Classification

This project is developing an explainable, multi-label music genre classifier for the MTG-Jamendo dataset.

The research target is a neural network that learns four musical concepts from the same song representation:

- instruments — what is playing;
- rhythm — the beat and tempo;
- timbre — the character of the sound;
- harmony — how notes and chords relate.

The model will learn how much each concept matters, combine them, and predict every genre that fits the song.

<p align="center">
  <img src="docs/diagrams/proposed-concept-guided-architecture.png" alt="Proposed concept-guided architecture" width="1100">
</p>

## What exists today

The proposed architecture is the target, not a completed implementation. The repository currently provides the data pipeline and two comparison baselines:

1. **Direct CNN baseline** — predicts genres directly from log-Mel spectrograms.
2. **Descriptor-fusion baseline** — combines a learned instrument embedding with rhythm, timbre, and harmony descriptors.

The existing concept extraction work is still useful: instrument pretraining can initialize the shared encoder, while rhythm, timbre, and harmony descriptors become supervision targets for the proposed concept branches.

See [current status](docs/current-status.md), [baseline architecture](docs/baseline-architecture.md), [proposed architecture](docs/proposed-architecture.md), and the [implementation roadmap](docs/roadmap.md).

## Notebook pipeline

Choose one runtime and stay with it:

- [Google Colab](notebooks/colab/README.md) stores artifacts in Google Drive.
- [Kaggle](notebooks/kaggle/README.md) passes saved notebook outputs between stages.

| Notebook | Role |
|---|---|
| `00_download_*` | Download annotations and log-Mel shards |
| `01_preprocessing` | Build the official `split-0` manifest |
| `02_direct_cnn_baseline` | Train baseline A |
| `03_instrument_pretraining` | Learn the instrument representation and reusable encoder |
| `04_rhythm_targets` | Prepare rhythm supervision targets |
| `05_timbre_targets` | Prepare timbre supervision targets |
| `06_harmony_targets` | Prepare harmony supervision targets |
| `07_descriptor_fusion_baseline` | Train baseline B |
| `08_baseline_evaluation` | Compare recorded baseline metrics |
| `09_baseline_explainability` | Inspect descriptor-fusion attention |

The proposed-model training and evaluation notebooks will be added only when their implementation exists; the repository does not contain empty placeholder notebooks.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── data/                              # ignored local datasets
├── docs/
│   ├── baseline-architecture.md
│   ├── proposed-architecture.md
│   ├── current-status.md
│   ├── roadmap.md
│   ├── data-layout.md
│   ├── development.md
│   ├── project-guidelines.md
│   ├── kaggle-how-to.md
│   └── diagrams/
│       └── proposed-concept-guided-architecture.png
├── notebooks/
│   ├── colab/                         # generated Colab workflow
│   └── kaggle/                        # generated Kaggle workflow
└── scripts/
    ├── generate_colab_notebooks.py
    └── generate_kaggle_notebooks.py
```

The generator scripts are the source of truth for notebook code. Change a generator and regenerate its notebook set; do not maintain notebook-only forks.

## Non-negotiable evaluation rules

- Use the official MTG-Jamendo `split-0` partitions.
- Select models using validation data only.
- Evaluate the test partition only after model selection.
- Exclude undefined per-tag values from macro metrics rather than replacing them with zero.
- Record the tag order and normalization statistics in checkpoints.
- Keep datasets, extracted targets, checkpoints, and results outside Git.

The default development subset uses shards `00–02`. Expand log-Mel and AcousticBrainz shards together when more storage is available.
