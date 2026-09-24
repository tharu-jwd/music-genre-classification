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

The [standalone instrument branch notebook](instrument_branch/notebooks/03_instrument_branch.ipynb)
implements the proposed shared-input instrument head, masked supervision and
40-concept bottleneck. See its [run and integration guide](instrument_branch/README.md)
for required encoder exports and the completed official annotation audit. It is
separate from the legacy Stage 1 baseline; real training and joint-model
experiments still require the shared encoder outputs.

The [timbre branch](timbre_branch/README.md) is a strict 128-to-35 concept bottleneck:
`h_audio (B,128) → … → z_timbre (B,35)` standardized named descriptors. Fusion
consumes only those 35 values, never the raw encoder vector.

The [concept fusion package](concept_fusion/) on `thevindu-concept-fusion` owns
`Linear(40,64)` for instrument probabilities and `Linear(35,64)` for timbre
concepts, then gated/concat/attention fusion to 87 genre logits.
Run `python scripts/run_all_fusion.py --quick`. See [the fusion runbook](docs/concept-fusion-runbook.md)
and [the architecture history](docs/architecture-from-plan-to-implementation.md).

## Repository structure

```text
.
├── README.md
├── concept_fusion/                    # gated fusion, genre head, eval (this branch)
├── instrument_branch/                 # Anupama instrument v2
├── timbre_branch/                     # Senindu 35-D timbre bottleneck
├── docs/
├── notebooks/dataset_split/           # EDA + official split table
└── scripts/run_all_fusion.py
```

## Non-negotiable evaluation rules

- Use the official MTG-Jamendo `split-0` partitions.
- Select models using validation data only.
- Evaluate the test partition only after model selection.
- Exclude undefined per-tag values from macro metrics rather than replacing them with zero.
- Record the tag order and normalization statistics in checkpoints.
- Keep datasets, extracted targets, checkpoints, and results outside Git.

The default development subset uses shards `00–02`. Expand log-Mel and AcousticBrainz shards together when more storage is available.
