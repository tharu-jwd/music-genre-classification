# Project status and remaining work

This is the single project-wide progress document. It records what code exists, what
is trustworthy, what is blocked, and what remains. Architecture details belong in
[architecture.md](architecture.md); team-wide implementation contracts belong in
[team-standards.md](team-standards.md).

## Status meanings

| Status | Meaning |
|---|---|
| Not started | The agreed component has not been implemented |
| Code present | An implementation exists, but a clean successful run is not proven |
| Blocked | A known issue prevents trustworthy use or comparison |
| Ready | Tests and a representative smoke run pass |
| Integrated | The component works inside the proposed model |

Committed notebooks intentionally contain no outputs, so their presence alone is
not evidence that a stage works end to end.

## Current status

| Area | Status | Main issue or next action |
|---|---|---|
| Download workflow | Code present | Verify a clean hosted-runtime run |
| Official split manifest | Code present | Strict variable-width parser and canonical fields pass CPU tests; verify a hosted run |
| Genre vocabulary | Code present | Official split files yield a fixed sorted 87-label vocabulary; verify generated artifacts |
| Direct CNN baseline | Code present | Ordered masked windows are implemented; do not spend GPU until the comparison cohort is frozen |
| Instrument pretraining | Code present | Uses only annotated songs for supervision; verify a capped hosted run after data preflight |
| Rhythm targets | Code present | Define and validate the versioned target contract |
| Timbre targets | Blocked | Replace invalid log-Mel proxies with trustworthy extraction |
| Harmony preflight | Code present | Reports waveform availability and exports exact model-window regions; clean hosted decode checks remain |
| Harmony target design | Code present | CQT and harmonic-HPCP pass synthetic CPU tests; aligned-region runner and immutable decision gate are ready, bounded real-audio comparison remains |
| Harmony pseudo-labels | Code present | A provenance-checked temporal-chroma pilot writer and bounded CPU Essentia chord baseline are tested; full generation still requires real extractor evidence and an accepted teacher/confidence policy |
| Descriptor-fusion baseline | Blocked | Normalize descriptors and use the same cohort as baseline A |
| Baseline evaluation | Blocked | Recompute comparable results after upstream fixes |
| Attention inspection | Code present | Treat it as inspection, not proof of explanation |
| Shared encoder module | Code present | Reusable interface/cache/alignment code is wired into opt-in notebook 06 CPU stages; cache a real hosted cohort and later integrate the selected branch into the joint model |
| Learned concept branches | Code present | Harmony reference branch, fixed CPU screen, baselines, and preregistered advance/stop gate pass tests; real-target execution, other branches, and integration remain |
| Joint loss and gated fusion | Not started | Implement after branch interfaces are agreed |
| Proposed-model evaluation | Not started | Requires the integrated model and fair baselines |

Because joint training is not implemented, it is deliberately absent from the GPU
job allowlist; no run record can reserve accelerator time for it yet.

GPU training is now fail-closed: every CUDA notebook requires a validated approval
record and a schema-checked frozen cohort, verifies the current manifest hash,
filters to those exact IDs, clamps the approved epoch/wall-time limits, and stores
the approval record with its checkpoint and runtime ledger. Each run must also have
a matching reservation in the shared `gpu_budget_v1` ledger; completed runs release
their reservation and debit measured hours. Harmony GPU registration additionally
requires a semantically valid passing CPU-branch decision; a second attempt requires
either an actually ambiguous paired comparison or a recorded failed-run termination,
not an arbitrary hashed note. Cohorts are created by
`scripts/freeze_experiment_cohort.py`, and the budget is managed by
`scripts/manage_gpu_budget.py`. No GPU run is approved in the repository.

## Work sequence

### Phase 1 — Stabilize the data pipeline

**Goal:** Ensure every team member trains and evaluates on the same correctly parsed
songs and labels.

- parse the official variable-width annotation files correctly;
- build one canonical manifest with normalized song IDs and official `split-0`;
- use split-specific genre and instrument vocabularies in a fixed order;
- segment full-song log-Mels into real windows;
- validate window masks, padding, and truncation;
- add a tiny shared fixture and data-pipeline smoke test.

No model score should be treated as a baseline result until this phase is complete.

Current CPU evidence: the shared parser preserves variable-width tag columns,
rejects split vocabulary mismatches, and was checked against the six official
split-0 files. It recovered 55,094 genre songs with 87 labels and 24,976 instrument
songs with 40 labels. Synthetic tests cover song-ID normalization, missing-label
masks, ordered window selection, short-song padding, and window masks. A clean
hosted manifest run is still required, so Phase 1 is not yet marked ready.

### Phase 2 — Produce trustworthy concept supervision

**Goal:** Create aligned concept targets that mean what their names claim.

- freeze versioned schemas for instrument, rhythm, and timbre;
- freeze the temporal harmony schema after comparing chroma extractors and chord teachers;
- validate rhythm sources and values;
- replace unsupported timbre proxies and permanently reject Mel-band harmony proxies;
- preserve temporal chroma, timestamps, teacher confidence, and pseudo-label masks;
- record extraction failures rather than silently writing zeros;
- fit continuous-target normalization on training rows only;
- save per-target supervision masks;
- align all concept targets to the canonical manifest.

Detailed harmony work is tracked in [harmony-plan.md](harmony-plan.md).
Its pseudo-label teacher selection is a quality gate: failure to find an acceptable
teacher reduces scope to temporal chroma rather than silently accepting poor chords.
The complete cheap harmony artifact chain also has a two-song CPU integration test;
this proves that the stages connect but does not replace the registered real-audio
quality gates. A separate three-track CPU integration test proves the external chord
benchmark chain from immutable source mapping and pre-result policy registration
through Essentia inference, `mir_eval`, and the accept/reject decision.

### Phase 3 — Establish fair baselines

**Goal:** Produce honest comparison numbers before evaluating the proposed model.

- rerun the direct CNN after windowing and label fixes;
- rerun instrument pretraining with correct missing-label handling;
- rerun descriptor fusion with normalized, validated descriptors;
- evaluate both genre baselines on the same official test-song cohort;
- save configurations, checkpoints, predictions, tag order, and evaluated IDs.

### Phase 4 — Extract reusable model code

**Goal:** Stop duplicating neural-network logic inside generated notebooks.

- move the window CNN and masked song pooling into importable modules;
- expose fine within-window encoded features, interval boundaries, window identity,
  and masks before time pooling; one vector per 29.1-second window is insufficient;
- add shape, padding-mask, and gradient tests;
- support loading the validated instrument-pretraining weights;
- make Colab and Kaggle use the same core implementation.

### Phase 5 — Implement the proposed network

**Goal:** Build the architecture shown in the project diagram.

- add learned instrument, rhythm, and timbre branches;
- add a temporal harmony branch after its extractor/teacher quality gates pass;
- expose concept embeddings and prediction heads through one common interface;
- implement masked auxiliary losses;
- implement gated fusion and the 128-dimensional fused representation;
- add the multi-label genre head and joint training loop;
- verify gradients, checkpoint restoration, and partially missing supervision.

The harmony implementation follows its own ordered sequence: waveform/alignment
preflight, temporal chroma comparison, chord-teacher benchmark, immutable raw
pseudo-labels, masks/schema, temporal model interface, standalone training, then
joint integration. See [harmony-plan.md](harmony-plan.md) for exit criteria.

### Phase 6 — Evaluate the contribution

**Goal:** Determine whether concept supervision improves genre classification and
whether the model's concept behaviour is meaningful.

GPU experiments follow successive gates. Reuse compatible checkpoints and cached
encoder outputs; do not retrain a baseline merely to rename or repackage it.

The full-data runs are sequential:

1. Reuse or complete one direct-CNN and one validated descriptor-fusion baseline.
2. Train one no-harmony control and one full model with temporal chroma.
3. Stop harmony experiments if temporal chroma does not improve the agreed
   validation criterion.
4. If it improves, train one no-concept-supervision control to separate architectural
   capacity from the effect of concept supervision.
5. Train one chroma-plus-chords model only if its teacher and frozen-encoder screening
   pass and the temporal-chroma result justifies continuing.

Use one seed, early stopping, and a fixed short cap during screening. Repeat only the
selected finalist if the remaining GPU budget permits. Select with validation data
and evaluate the final configuration on test once.

Before any repeat, use cached validation predictions for a paired song-level
bootstrap of the pre-registered primary metric. A clearly inferior or negligible
variant stops. An ambiguous comparison permits one targeted repeat of that pair;
continued ambiguity is reported rather than expanded into a sweep.

The following are conditional diagnostics, not default experiments: removing every
other concept separately, concatenation versus gated fusion, initialization changes,
embedding-width sweeps, broad loss-weight sweeps, key-relative chords, shuffled
labels, and repeated seeds for rejected variants. Run one only when the primary
results are ambiguous and it answers a named question that cheaper analysis cannot.

Always report parameter count, actual GPU time, training time, inference latency,
and which cached artifacts or checkpoints were reused. Gate inspection must be
supported by concept removal and prediction-change analysis for the selected model.

### Phase 7 — Reproducible release

**Goal:** Make the final work understandable and repeatable outside the team.

- run the documented smoke-test command;
- complete at least one clean supported hosted-runtime workflow;
- pin the supported Python and dependency versions;
- publish configuration templates and artifact schemas;
- ensure generated notebooks match their generators and contain no saved outputs;
- update status and usage documentation to match the measured implementation;
- report limitations and failed experiments alongside successful results.

## Ownership boundaries

| Area | Ownership scope | Required hand-off |
|---|---|---|
| Data pipeline | Manifest, labels, splits, windowing | Canonical song batches and masks |
| Instrument | Instrument targets, pretraining, 64D branch | Embedding and prediction interface |
| Rhythm | Rhythm schema, extraction, 32D branch | Targets, masks, metrics, embedding |
| Timbre | Timbre schema, extraction, 32D branch | Targets, masks, metrics, embedding |
| Harmony | Temporal schema, teacher evaluation, pseudo-labels, configurable-width branch | Targets, confidence masks, metrics, embedding |
| Shared model | Encoder, pooling, joint losses, fusion | Integrated model and checkpoints |
| Evaluation | Cohorts, metrics, ablations, reports | Comparable predictions and results |

The harmony owner can complete extractor/teacher comparison, pseudo-label creation,
validation, masks, branch code, loss, and standalone evaluation independently. Full
integration requires the canonical waveform manifest and an ordered shared-encoder
interface; one already-pooled song vector is insufficient for chord progressions.

## Definition of project completion

- a clean smoke run passes from manifest creation through evaluation;
- all model variants use the same official split, vocabulary, and comparison cohort;
- best checkpoints are selected using validation data only;
- target schemas, masks, normalization, configurations, seeds, and tag order are
  stored with results;
- the proposed model and required ablations have recorded results;
- every GPU run has a question, cap, lineage, actual usage, and advance/stop decision;
- concept-effect claims are supported by measurements rather than gates alone;
- documentation separates implemented code, verified results, and proposed work.
