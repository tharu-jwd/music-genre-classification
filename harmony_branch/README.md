# Harmony branch

## Current data contract

The completed baseline uses **45 interpretable track-level harmony descriptors**
extracted from the first `min(track duration, 240 seconds)` of all 7,324 selected
tracks. The clean table is `../data/harmony_df.csv`; the detailed auditable table
is `data/harmony_features_raw.csv`.

See [docs/feature-contract.md](docs/feature-contract.md) for the exact feature
names, definitions, extraction settings, provenance, and completed-data audit.

The completed v3 joint run selected 12 descriptors for prediction and genre
fusion. The new v4 training configuration uses all 45 and is not yet a trained
result. See the [audited handoff](docs/integration-handoff.md) for the exact v3
targets, scores, checkpoint checks, and decision. Temporal chroma/chord
supervision remains a separate experimental route; its 12 pitch classes must
not be confused with the 12 selected song descriptors.

Dehan's detailed, diagrammed architecture record is the
[version register](docs/architecture-versions/README.md), with the current
[Harmony v3 snapshot](docs/architecture-versions/v3/README.md) and the
[v4 implementation record](docs/architecture-versions/v4/README.md).

Read by purpose:

| Need | Document | Status |
|---|---|---|
| Understand the current 45-target implementation | [v4 record](docs/architecture-versions/v4/README.md) | Implemented; untrained |
| Review the completed 12-target model and diagrams | [v3 snapshot](docs/architecture-versions/v3/README.md) | Historical trained run |
| Know the 45 extracted values and their provenance | [Feature contract](docs/feature-contract.md) | Current data contract |
| Hand off targets, fusion interface, results, and limits | [Integration handoff](docs/integration-handoff.md) | Audited run |
| Review the supplied files | [Evidence index](docs/evidence/README.md) | Source evidence, not instructions |
| Evaluate redundancy and potential feature sets | [Feature evaluation](docs/feature-selection-evaluation.md) | Findings; all-45 option now implemented, not yet evaluated |
| Explore musical information beyond the original 45 | [52-entry candidate catalog](docs/harmony-feature-candidate-catalog.md) | Broad theoretical search with musical rationale |
| Choose candidates we can extract automatically | [Feasibility shortlist](docs/pseudo-label-feasibility-shortlist.md) | Tool review, proposed ten-value audio pilot, and teacher gates |
| Consider future aligned chroma/chord work | [Temporal proposal](TEMPORAL_CHROMA_PROPOSAL.md) and [research plan](docs/temporal-chroma-research-plan.md) | Historical/proposed; not v3 |

The repository-root [evaluation plan](../plan.md) records the completed audit;
it is not a second architecture specification.

This directory owns the temporal harmony workstream. It follows the same ownership
boundary as `instrument_branch/`, `timbre_branch/`, and `rhythm_branch/`.

```text
harmony_branch/
├── ARCHITECTURE.md
├── TEMPORAL_CHROMA_PROPOSAL.md
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

For the extracted table, use the [45-D feature contract](docs/feature-contract.md).
For the completed v3 trained model and fusion boundary, use the
[audited handoff](docs/integration-handoff.md). See the [current architecture
entry point](ARCHITECTURE.md), [v3 snapshot](docs/architecture-versions/v3/README.md),
and [v4 record](docs/architecture-versions/v4/README.md) for implemented network,
tensor, masking, and fusion details. The older
[temporal proposal](TEMPORAL_CHROMA_PROPOSAL.md) is separate.

The branch consumes fine-grained ordered shared-encoder features with masks and
window identity. The completed v3 run returned:

- a configurable song embedding (32D in the initial screen);
- 12 predicted standardized song-level descriptors for primary fusion;
- temporal chroma logits `(B,T,12)`;
- optional temporal chord logits `(B,T,25)` (disabled in the current run);
- prediction masks and branch availability.

It does not own a 64D fusion token. Primary fusion owns the `Linear(12,64)`
projection of predicted descriptors; `Linear(D_harmony,64)` over the embedding
is an ablation route. Temporal predictions remain available for separate
experiments but are not supervised in the current descriptor run.
In the untrained v4 option, the descriptor output is `(B,45)` and the
fusion-owned projection is `Linear(45,64)`; the other shapes are unchanged.

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

## Optional temporal-path research

The extractor comparison, chord-teacher benchmark, target pilot, encoder cache, and
branch screen are CPU-only and resource-capped. See the [implementation
plan](docs/temporal-chroma-research-plan.md) for the proposed gates and commands.
They are not prerequisites for using the current descriptor-based v3 branch.

Full-corpus **descriptor extraction is complete** and validated for all 7,324
selected tracks. Temporal chroma/chord screening is still a separate, unfinished
research path because its aligned targets, shared-encoder checkpoint, and approved
annotated chord benchmark have not been supplied. Completion of the descriptor
dataset must not be interpreted as completion of that temporal experiment.
