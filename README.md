# Concept-Guided Music Genre Classification

This project is developing an inspectable, multi-label music genre classifier for
the MTG-Jamendo dataset. The proposed model learns four musical concepts from one
shared audio representation:

- instruments — what is playing;
- rhythm — the beat and tempo;
- timbre — the character of the sound;
- harmony — pitch classes, tonal movement, and chord changes over time.

It learns how much each concept contributes, combines the four representations, and
predicts every genre that applies to a song.

<p align="center">
  <img src="docs/diagrams/proposed-concept-guided-architecture.svg" alt="Proposed concept-guided architecture" width="1100">
</p>

## Current state

The diagram is the target architecture, not a completed implementation. The
repository contains generated workflows for data preparation, two comparison
baselines, instrument pretraining, concept-target preparation, standalone instrument,
timbre, and rhythm branches, and harmony preflight.
A full clean hosted run has not been proven. The split parser, fixed vocabularies,
song windowing, cohort consistency, instrument-label availability, bounded harmony
extractor/branch screening, and GPU runtime gates have CPU-tested implementations.
The resource-capped harmony CPU ladder now lives under `harmony_branch/scripts/`;
the retired notebook 06 is not part of the hosted workflow.
A bounded CPU Essentia chord-baseline generator and evaluator are ready, but their
external benchmark has not run. The timbre implementation and its synthetic smoke
test are present, but its real target table and shared-encoder inputs are not tracked.
The rhythm branch and fusion adapter pass synthetic CPU contract tests, but the real
AcousticBrainz coverage/interval audit and shared-encoder cache are not tracked.
Real-audio harmony selection, chord-teacher acceptance, real-branch data loading,
and a clean end-to-end hosted run remain unresolved before joint-training outputs are
trustworthy. The temporal harmony/fusion adapter and its CPU integration tests are
implemented.

The current `main` branch also includes the instrument and timbre workstreams. The
fixture-tested fusion prototype from `origin/thevindu-concept-fusion` is integrated
here with a revised harmony contract: temporal predictions remain auxiliary outputs,
and fusion projects the configurable song embedding to 64D. See the project plan
before starting real training.

See the [project status and remaining work](docs/project-plan.md) for the exact
blockers and implementation sequence.

## Documentation

Each document has one purpose:

| Document | Purpose |
|---|---|
| [Architecture](docs/architecture.md) | Existing baselines and the proposed model design |
| [Shared CNN encoder](docs/shared-cnn-encoder-architecture.md) | Implemented CNN, tensor contract, temporal geometry, masks, and checkpoint compatibility |
| [Project plan](docs/project-plan.md) | Current status, blockers, ownership boundaries, and remaining work |
| [Harmony branch](harmony_branch/README.md) | Owned code, tests, documentation, and integration boundary |
| [Harmony plan](harmony_branch/docs/plan.md) | Step-by-step work owned by the harmony branch |
| [Harmony integration hand-off](harmony_branch/docs/integration-handoff.md) | Current interface, required artifacts, and executable next commands |
| [Team standards](docs/team-standards.md) | Shared data, model, artifact, evaluation, and development contracts |

## Notebook workflows

Choose one runtime for the shared data path:

- [Google Colab workflow](notebooks/colab/README.md) persists artifacts in Drive.
- [Kaggle workflow](notebooks/kaggle/README.md) passes saved outputs between notebooks.

Only `00` download, `01` preprocessing, and `04` rhythm-target extraction remain.
The old numbered notebooks `02`–`03` and `05`–`09` are retired; instrument, timbre,
harmony, and fusion now live in their packages. Generator scripts write only those
three essential notebooks.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── data/                         # ignored local datasets
├── docs/
│   ├── architecture.md
│   ├── shared-cnn-encoder-architecture.md
│   ├── project-plan.md
│   ├── team-standards.md
│   └── diagrams/
│       └── proposed-concept-guided-architecture.svg
├── notebooks/
│   ├── colab/
│   ├── kaggle/
│   └── dataset_split/
├── instrument_branch/             # 40 instrument concepts from a 128D song input
├── timbre_branch/                  # 35 standardized timbre concepts from a 128D input
├── rhythm_branch/                  # 10 learned temporal rhythm concepts
├── harmony_branch/                 # temporal harmony package, scripts, tests, and docs
├── shared_encoder/                 # modular shared CNN, geometry, validation, and types
├── concept_fusion/                # shared contracts, projections, losses, and fusion
├── scripts/
│   ├── generate_colab_notebooks.py
│   ├── generate_kaggle_notebooks.py
│   ├── shared_audio_encoder.py    # compatibility import for shared_encoder/
│   ├── freeze_experiment_cohort.py
│   ├── manage_gpu_budget.py
│   └── paired_bootstrap_compare.py
└── tests/
    └── test_*.py
```

Large datasets, extracted targets, checkpoints, predictions, and results remain
outside Git. Their agreed layout and metadata are defined in the team standards.
