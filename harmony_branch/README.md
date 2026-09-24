# Harmony branch

This directory owns the temporal harmony workstream. It follows the same ownership
boundary as `instrument_branch/`, `timbre_branch/`, and `rhythm_branch/`.

```text
harmony_branch/
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

## Real-data gates

The extractor comparison, chord-teacher benchmark, target pilot, encoder cache, and
branch screen are CPU-only and resource-capped. See the [implementation
plan](docs/plan.md) and [integration hand-off](docs/integration-handoff.md) for exact
commands and missing external artifacts.

Real-data screening is not complete because the repository does not contain the
canonical waveform manifest/audio, validated shared-encoder checkpoint, real timbre
target table, or approved annotated chord-benchmark mapping. Joint GPU training is
not approved.
