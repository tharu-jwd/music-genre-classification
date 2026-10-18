# Harmony integration hand-off

## Current extracted-feature hand-off

The current baseline now has complete real data for all 7,324 selected tracks:

- detailed provenance and diagnostics:
  `harmony_branch/data/harmony_features_raw.csv`;
- clean model table: `data/harmony_df.csv`; and
- schema and semantics: [feature-contract.md](feature-contract.md).

The clean table contains `TRACK_ID` plus 45 finite harmony descriptors. A harmony
prediction head may accept the shared encoder's 128-D track embedding and predict
these 45 standardized concepts. The 45 predicted values are the explainable
concept-bottleneck output supplied to fusion (or to a fusion-owned projection).
Normalization parameters must be fitted using training rows only.

The temporal interface documented below remains a legacy experimental alternative;
its pooled 12-D chroma output is not the completed 45-D dataset contract.

This is the short operational hand-off for teammates. Detailed research decisions
remain in the [harmony plan](plan.md).

## Interface frozen in code

Harmony returns a configurable song embedding (32D for the ablation), temporal
chroma logits `(B,T,12)`, optional chord logits `(B,T,25)`, prediction masks, and
availability. It does not return a 64D fusion token. Primary `concept_fusion`
softmaxes the chroma logits per token, masked-means valid tokens, owns
`Linear(12,64)`, and applies the fusion mask after projection. `Linear(32,64)` over
the song embedding is retained as `embedding_fusion` only.

Temporal chroma and chords keep separate target masks. Missing pseudo-supervision
masks only the corresponding auxiliary loss; it does not remove an available
predicted harmony branch from genre fusion. A missing/unusable audio branch sets the
fusion mask to zero. The pooled 12-bin chroma in `BranchOutput.concept_values` is a
prediction used by primary fusion and is never substituted for the temporal target.

Run the contract and fixture checks with:

```bash
uv run --with-requirements requirements.txt \
  python -m pytest harmony_branch/tests/test_harmony_fusion_contract.py tests/test_acceptance.py -q
uv run --with-requirements requirements.txt \
  python scripts/run_all_fusion.py --quick
```

Fixture scores are discarded engineering checks, not research results.

## Compatibility audit

| Component | Compatible boundary | Remaining live-input issue |
|---|---|---|
| Instrument v2 | Adapter accepts 40 probabilities/logits and fusion owns 40-to-64 projection | Implementation is notebook-contained and needs real 128D song representations |
| Timbre v2 | Adapter accepts the fixed 35 standardized values and fusion owns 35-to-64 projection | Real target table and real 128D song representations are absent |
| Shared encoder | `SharedAudioEncoder` can emit configurable-width ordered tokens and pooled output; harmony consumes the ordered tokens | The validated checkpoint must use the team-approved configuration; instrument/timbre require 128D pooled input |
| Harmony descriptor baseline | 128D shared track embedding predicts 45 interpretable descriptors; detailed and clean real tables cover all 7,324 tracks | Prediction-head training and fusion integration remain |
| Harmony temporal v2 (legacy experiment) | Adapter preserves temporal logits and projects masked-pooled 12D predicted chroma; 32D embedding route is an ablation | Aligned temporal targets and cached ordered encoder features are absent |
| Rhythm v2 | Adapter sends ten predicted descriptors through fusion-owned 10→64; 64D embedding route is an ablation | Real target coverage audit and cached `(B,T,128)` encoder features are absent |
| Fusion | Four 64D tokens, masks, joint losses, removal, gradients, and restore are fixture-tested | A real `BranchBundle` data loader remains project-level work |

## External inputs still required

| Provider | Required artifact | Expected location or hand-off |
|---|---|---|
| Data-pipeline owner | Canonical selected-track metadata and frozen train/validation/test assignment | `data/split_csv.csv` and `data/track_split_assignments.csv` |
| Shared-encoder/instrument owner | Validated checkpoint, vocabulary, validation metric, encoder configuration, and provenance | `checkpoints/pretraining/instrument/best.pt` and its metadata |
| Chord-benchmark owner/team | Legally usable existing annotated benchmark and immutable audio/reference mapping; no new manual labels | mapping CSV passed to `prepare_chord_benchmark_source.py` |
| Timbre owner | Ignored 7,324-row real target table when real joint rows are assembled | `timbre_branch/data/timbre_features_raw.csv` |
| Fusion owner/team | Ratification of predicted-concept contract v0.3 | [ADR 0001](../../docs/adr/0001-concept-fusion-architecture.md) |

The harmony descriptor tables are present and audited. The remaining external
inputs in this section are required only for the legacy temporal/chord experiment
or later joint-model training. Therefore the following temporal real-data commands
are intentionally not claimed as completed.

## Commands once the artifacts arrive

Freeze the registered 8/2 train/validation harmony cohort and exact audio regions:

```bash
python scripts/freeze_experiment_cohort.py dataset/song_manifest.csv \
  --splits train validation --require-available waveform_available \
  --limit train=8 --limit validation=2 --seed 42 \
  --output results/cohorts/harmony_extractor_seed42.json

python harmony_branch/scripts/export_harmony_regions.py dataset/song_manifest.csv \
  results/cohorts/harmony_extractor_seed42.json \
  --root <MTG-root> --regions-per-song 2 \
  --output results/cohorts/harmony_regions_seed42.json
```

Run and decide the bounded CPU extractor comparison:

```bash
uv run --with-requirements harmony_branch/requirements.txt \
  python harmony_branch/scripts/benchmark_harmony_extractors.py \
  --regions results/cohorts/harmony_regions_seed42.json \
  --output results/evaluations/harmony_extractor_candidates.json

python harmony_branch/scripts/decide_harmony_extractor.py \
  results/evaluations/harmony_extractor_candidates.json \
  --output results/evaluations/harmony_extractor_decision.json
```

Prepare the existing annotated chord benchmark before inference:

```bash
python harmony_branch/scripts/prepare_chord_benchmark_source.py <mapping.csv> \
  --root <benchmark-root> --benchmark-name <name> \
  --benchmark-source <source> --benchmark-license <license> \
  --output results/chord-teacher/source.json
```

Then follow Steps 4, 6, 7, and 9 of the harmony plan for policy registration,
teacher inference/evaluation, target materialization, encoder caching, screening
dataset assembly, and the pre-registered branch decision. Every command is CPU-only
until a passing branch decision and a separate GPU approval exist.

## Readiness meaning

- **Fixture integration:** ready and CPU-tested.
- **45-D descriptor extraction:** complete and audited for 7,324 tracks.
- **Temporal real-data screening:** blocked on the temporal artifacts above.
- **Joint GPU training:** not approved; it additionally requires all four real
  branch rows, the frozen comparison cohort, team-ratified v0.3, and a passing CPU
  branch decision.
