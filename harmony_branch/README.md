# Harmony branch

## Current data contract

The completed baseline uses **45 interpretable track-level harmony descriptors**
extracted from the first `min(track duration, 240 seconds)` of all 7,324 selected
tracks. The clean table is `../data/harmony_df.csv`; the detailed auditable table
is `data/harmony_features_raw.csv`.

See [docs/feature-contract.md](docs/feature-contract.md) for the exact feature
names, definitions, extraction settings, provenance, and completed-data audit.

The temporal chroma/chord code and the remainder of this README describe a legacy
experimental route. They remain available for later comparison, but they must not
be mistaken for the schema of the completed 45-D descriptor dataset.

This directory owns the temporal harmony workstream. It follows the same ownership
boundary as `instrument_branch/`, `timbre_branch/`, and `rhythm_branch/`.

```text
harmony_branch/
├── ARCHITECTURE.md
├── README.md
├── requirements.txt
├── src/harmony_branch/
│   ├── model.py
│   ├── losses.py
│   ├── features.py
│   └── alignment.py
├── scripts/
├── tests/
└── docs/
```

## Contract

For current extracted features, use the
[45-D feature contract](docs/feature-contract.md). See
[ARCHITECTURE.md](ARCHITECTURE.md) for both the current descriptor boundary and the
legacy temporal network, tensor, alignment, masking, loss, and fusion design.

The branch consumes fine-grained ordered shared-encoder features with masks and
window identity. It returns:

- a configurable song embedding (32D in the initial screen);
- temporal chroma logits `(B,T,12)`;
- optional temporal chord logits `(B,T,25)`;
- prediction masks and branch availability.

It does not own a 64D fusion token. Fusion owns the
`Linear(D_harmony,64)` projection. Temporal predictions remain intact for masked
auxiliary losses and are never replaced by the obsolete 18-value summary.

The only harmony-specific file that belongs in the shared fusion package is
`concept_fusion/harmony_adapter.py`. Necessary shared contract, projection, and
joint-loss support remain team-owned and are covered by integration tests.

## Tests

From the repository root:

```bash
uv run --with-requirements requirements.txt \
  python -m pytest harmony_branch/tests -q
```

The tests use synthetic audio and fixture tensors. Their scores are engineering
checks, not research results.

## Legacy temporal-path gates

The extractor comparison, chord-teacher benchmark, target pilot, encoder cache, and
branch screen are CPU-only and resource-capped. See the [implementation
plan](docs/plan.md) and [integration hand-off](docs/integration-handoff.md) for exact
commands and missing external artifacts.

Full-corpus **descriptor extraction is complete** and validated for all 7,324
selected tracks. Temporal chroma/chord screening is still a separate, unfinished
research path because its aligned targets, shared-encoder checkpoint, and approved
annotated chord benchmark have not been supplied. Completion of the descriptor
dataset must not be interpreted as completion of that temporal experiment.
