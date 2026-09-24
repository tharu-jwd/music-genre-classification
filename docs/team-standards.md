# Team implementation standards

These are the shared contracts every project component must follow. They prevent
independently developed branches from using incompatible data, tensor shapes,
targets, artifacts, or evaluation rules.

Architecture choices are documented in [architecture.md](architecture.md). Work
remaining is documented in [project-plan.md](project-plan.md). This file owns the
cross-team rules and development workflow.

## 1. Canonical data contract

### Song identity and splits

- Normalize every song ID to the same seven-digit string format.
- Use the official MTG-Jamendo `split-0` train, validation, and test partitions.
- Reject duplicate song IDs and conflicting split assignments.
- Join artifacts by `song_id`, never by row position alone.
- Keep the exact song IDs used by every experiment.

### Label vocabularies

- Use the split-specific genre and instrument vocabularies.
- Define one fixed column order for each multi-label target.
- Store that order with target artifacts, predictions, and checkpoints.
- Do not treat a song without instrument annotations as an all-negative example.

### Shared manifest

Concept owners consume one manifest rather than independently reconstructing song
lists. Its minimum logical fields are:

```text
song_id
split
audio_path
logmel_path
waveform_available
genre_available
instrument_available
rhythm_available
timbre_available
harmony_available
```

Paths may be runtime-specific, but song identity, split, and availability must remain
the same.

## 2. Audio and log-Mel contract

The current precomputed-input schema is `mtg_full_audio_logmel_windows_v1`. It
matches the official MTG-Jamendo [Mel extraction
script](https://github.com/MTG/mtg-jamendo-dataset/blob/master/scripts/melspectrograms.py):

| Setting | Frozen value |
|---|---|
| Sample rate | 12,000 Hz mono |
| Analysis | 512-sample Hann frame, 256-sample hop, centered frames |
| Mel representation | 96 Slaney Mel power bands followed by linear-to-dB conversion |
| Source duration | Full-song precomputed Mel, not a single center crop |
| Model window | 1,366 frames (approximately 29.1 seconds) |
| Song cap | 12 ordered windows |
| Long songs | Select evenly spaced non-overlapping chunk indices, including the beginning and end |
| Short/final windows | Right-pad time with zero; mark the window real in the mask |
| Unused slots | All-zero tensor with mask value zero |
| Additional normalization | None in v1; any later change is a separately versioned decision |

Waveform preprocessing for new harmony extraction is a separate contract and must
still freeze channel conversion, amplitude handling, and failure rules before use.
`scripts/export_harmony_regions.py` derives the harmony regions from the same
`logmel_window_plan` used by model segmentation and refuses unfrozen or oversized
development cohorts. Its boundaries are log-Mel frame boundaries at 12,000 Hz and a
256-sample hop; the underlying centered 512-sample analysis context must be retained
in extractor metadata when targets are finally generated.

The proposed model expects the logical input order:

```text
batch × windows × channels × mel_bins × frames
```

A window mask identifies real windows. Padding must never affect pooling, attention,
or loss. Full-song 2D log-Mels must be segmented into actual windows rather than
being treated as one valid window followed by empty padding.

## 3. Concept target contract

Every concept target group must publish a versioned schema containing:

- fixed target names and column order;
- meaning, unit, valid range, and data type for every target;
- source library, algorithm, and extraction settings;
- aggregation method;
- missing-value and extraction-failure rules;
- schema and extractor versions.

Every automatically generated semantic label must additionally record the teacher
name/version, class vocabulary, confidence, and confidence-mask policy. Such labels
are called **pseudo-labels**, not ground truth. Teacher outputs must be produced
without access to genre labels.

Targets must mean what their names claim. An approximation cannot silently replace a
waveform-derived target. If an approximation is retained for comparison, it must use
a separate source label and be described as a baseline.

Extraction must report unreadable inputs, unsupported cases, and missing sources.
It must not silently turn failures into zeros. Each extractor needs deterministic
rerun checks, numerical validation, edge-case fixtures, and representative real-song
sanity checks.

Temporal targets must preserve timestamps and identify the exact waveform region
from which they were computed. Song-level averaging may be retained as a named
baseline, but it must not silently replace a temporal target. Mel-frequency bands
must never be split into pitch-class bins; missing waveform harmony targets remain
missing.

## 4. Normalization and missing data

- Fit continuous-target normalization using training songs only.
- Apply the saved training statistics to validation, test, and inference data.
- Save the mean, scale, target order, and schema version together.
- Define explicit handling for constant or entirely missing columns.
- Use a per-target supervision mask; zero is a valid value, not a missing marker.
- A missing concept target masks only that auxiliary loss. The song may still train
  the genre task and other available concepts.
- Loss code must safely handle a batch with no valid values.

The supervision mask and audio-window mask are different objects and must never be
used interchangeably.

## 5. Model component interface

The shared-model owner must publish the encoder input shape, ordered representation
shape, optional pooled-song width, timestamps, data type, and mask semantics. The
shared interface must be capable of returning:

```text
encoded_sequence
sequence_times
sequence_start_times
sequence_end_times
sequence_mask
sequence_window_index
pooled_song
```

A branch may consume the ordered sequence, the pooled representation, or both. A
branch that claims to model progressions must consume a fine-grained ordered
representation; one token per 1,366-frame/29.1-second input window does not qualify.
Token interval boundaries are required to align temporal pseudo-labels, and window
identity prevents a temporal model from treating gaps between evenly selected song
regions as adjacent audio.
Every concept branch returns the same logical fields:

```text
embedding
predictions
availability
```

The architecture document owns branch dimensions. Code must assert those dimensions
at boundaries and test forward shape, gradients, masking, and checkpoint restore.
Raw branch widths may differ; addition- or attention-based fusion must project them
to a common fusion width.

Concept availability is used when an entire concept is deliberately unavailable to
fusion. Missing supervision alone must not erase a learned branch output.

## 6. Training contract

CUDA execution in hosted training notebooks is locked by default. Before enabling a
GPU, initialize the team ledger once with the actual allocation (do not guess it):

```bash
python scripts/manage_gpu_budget.py init results/gpu_budget.json \
  --total-hours <allocated-hours>
python scripts/manage_gpu_budget.py status results/gpu_budget.json
```

Then create a JSON run record from the checked-in template:

```bash
python scripts/validate_gpu_run_request.py --print-template
python scripts/manage_gpu_budget.py reserve results/gpu_budget.json \
  --run-record path/to/run.json
python scripts/validate_gpu_run_request.py path/to/run.json --expected-job <job>
```

Set `budget_remaining_before_gpu_hours` to the ledger's current
`unreserved_gpu_hours` before reserving. The final validation command must pass
without `--allow-planned`. An approved record requires
one research question, one change from its control, a committed Git hash, a frozen
cohort artifact, evidence that cheaper checks passed, reused-artifact lineage, one
seed, epoch and wall-time caps, a pre-registered validation decision rule, and a
time-zone-qualified approval. Placeholder values are not approval. Paths inside the
record are resolved relative to the record file unless absolute.

Each cheaper check has a human-readable explanation plus a file path and SHA-256.
Reservation validation and the hosted runtime both require those exact files, the
frozen cohort, the pinned control for comparison stages, and every declared reused
artifact to exist. Reused checkpoints/features/targets are recorded as path plus
SHA-256, not a path string alone. Changing a prerequisite after approval invalidates
the run.

For `harmony_branch_screening`, a generic file marked "passed" is not sufficient.
Reservation validation and the hosted runtime parse the evidence and require exactly
one `harmony_branch_screen_decision_v1` with every registered check true and
`decision=advance_to_gpu_registration`. At present it can authorize only
`H1_temporal_chroma`, target `temporal_chroma_v1`, and seed 42. Chord variants and
new harmony configurations stay locked until their own CPU screening path exists.
The unimplemented `joint_training` job is not an accepted GPU job at all. It may be
added only with the joint model, CPU forward/backward smoke evidence, and its own
semantic prerequisite contract. Job/stage mismatches also fail validation.

Approved branch-screening and joint-training comparisons additionally require a
stable comparison ID, a different control run ID, and the exact control prediction
artifact plus its SHA-256. Their practical-effect threshold must be positive. The
validator and hosted runtime both verify that control artifact before CUDA can start;
this prevents spending a candidate run before its reusable control actually exists.

Every record also has a stable `experiment_id` and `attempt` number. Attempt 1 is
the default. At most one repeat (`attempt: 2`) is permitted, and it must name the
original run, state why repeating is justified, and pin the decision/failure
evidence by SHA-256. The budget ledger rejects duplicate experiment-attempt pairs
even when someone changes only the run ID.

A comparison repeat must use an immutable `paired_bootstrap_comparison_v1` report
whose decision is `ambiguous`. The runtime checks that its comparison ID, control
run, repeated candidate run, control prediction hash, metric, effect threshold,
confidence, and bootstrap seed exactly match the repeat request. A generic failure
note or an `advance`/`stop` report cannot authorize a second GPU attempt.

Upload the matching ledger with the record and cohort. CUDA starts only if the run
has a reservation with the same run ID, estimate, and commit, and the budget snapshot
matches. After a run—even a failed one—copy the measured GPU hours from its runtime
JSON into the ledger and release the reservation:

```bash
python scripts/manage_gpu_budget.py complete results/gpu_budget.json \
  --run-id <run-id> --actual-hours <measured-hours>
```

Freeze the exact IDs first; do not hand-edit a cohort JSON:

```bash
python scripts/freeze_experiment_cohort.py dataset/song_manifest.csv \
  --splits train validation \
  --require-available waveform_available \
  --limit train=512 --limit validation=128 \
  --seed 42 \
  --output results/cohorts/harmony_screen_seed42.json
```

Limits are examples, not defaults: choose them once from the available compute
budget and keep the same artifact for every model in that comparison. The tool
selects IDs by a seeded SHA-256 rank without examining model outcomes, records the
source-manifest hash, and refuses to overwrite an existing cohort. Include a test
split only for the final approved evaluation.

Upload that exact record and its frozen cohort artifact to the hosted runtime, then
set `GPU_RUN_RECORD` to the record path before executing the training cell. CPU
preflight remains available without approval. On CUDA, notebooks reject a missing,
planned, wrong-job, malformed, or cohort-less record; they also clamp training to
the approved caps. They verify the current manifest hash and train only on the exact
IDs and splits in the cohort. Checkpoints and runtime ledgers retain the complete record. Do
not edit an approved record in place: a changed question, cohort, code commit, seed,
cap, or test policy requires a new run ID and review.

- Use binary cross-entropy with logits for genre and instrument multi-label outputs.
- Start ordinary continuous concept prediction with masked Smooth L1 loss, but choose
  target-appropriate losses for distributions and sequences. Chord classes use a
  masked classification or soft-target loss rather than continuous regression.
- Make every auxiliary-loss weight configurable and record it with results.
- State how each loss is reduced across targets, samples, and batches.
- Seed Python, NumPy, and PyTorch through one experiment configuration.
- Record device, precision, deterministic settings, and dependency versions.
- Choose checkpoints and tune hyperparameters using validation data only.
- Do not repeatedly inspect test results during development.
- Record estimated and actual GPU time for every run.
- Hosted GPU training notebooks default to a 120-minute total wall-time cap. They
  stop training between batches at 90% of that allowance, reserving the remainder
  for validation, artifact export, and bookkeeping. Override `MAX_GPU_RUN_MINUTES` only when the run record explains
  why the approved question cannot be answered inside that cap.
- Window-based audio notebooks default to `GPU_BATCH_SIZE=2` to avoid losing a run
  to accelerator memory exhaustion. Before the epoch loop, the actual runtime must
  complete one forward/backward preflight without taking an optimizer step. Increase
  batch size only after that preflight succeeds; record the chosen value in the checkpoint and runtime
  ledger.
- Use CPU checks, cached targets, cached frozen-encoder features, a fixed development
  subset, one screening seed, early stopping, and a run cap before full joint training.
- Advance variants through explicit quality gates. Do not run Cartesian sweeps across
  targets, embedding widths, loss weights, fusion methods, and seeds.
- Reuse compatible checkpoints and results when the split, cohort, preprocessing,
  vocabulary, architecture, and evaluation contract match; record the lineage.
- Run repeated seeds and the held-out test only for selected finalists and only when
  the available compute budget permits. Report limited replication honestly.
- Executable notebooks must leave test evaluation off by default. Set
  `EVALUATE_TEST=1` only for the selected final run; ordinary development runs save
  validation predictions instead.
- Before approving another GPU run, compare saved validation predictions with a
  paired song-level bootstrap or another pre-registered paired test. Freeze the
  primary metric, practical-effect threshold, confidence level, and resampling seed
  before viewing the candidate result.
- An ambiguous paired comparison permits at most one targeted repeat of that pair.
  It does not justify opening a new architecture or hyperparameter sweep.

## 7. Evaluation contract

All compared genre models must use the same official split, label order, and
evaluated song cohort. Save predictions and exact evaluated song IDs so this can be
verified.

Validation and test prediction artifacts used for paired comparisons are compressed
NPZ files with these fixed logical fields:

```text
song_ids:   string[songs]
label_names:string[labels]
targets:    binary[songs, labels]
scores:     float[songs, labels]
```

Save raw logits separately if needed; `scores` must have the same ranking meaning in
both compared files. The paired comparison tool aligns by `song_ids` and rejects
cohort, target, or vocabulary mismatches.

- Report the agreed multi-label metrics, including macro ROC-AUC and PR-AUC.
- Exclude undefined per-label metrics from a macro average; do not replace them with
  zero.
- Select decision thresholds using validation data and record the method.
- Report coverage for concept metrics and evaluate only valid targets.
- Compare every continuous branch with a training-set-mean predictor.
- Keep teacher quality, student-to-teacher agreement, and downstream genre utility
  as separate results. Agreement with pseudo-labels is not human-verified accuracy.
- Evaluate a chord teacher on an existing annotated benchmark before accepting its
  pseudo-labels, unless the experiment is explicitly labelled unvalidated.
- Use the same metric implementation for baselines, ablations, and the proposed
  model.

Attention or gate values are inspection signals, not proof of explanation. Stronger
claims require concept accuracy, concept removal, prediction-change analysis, and
appropriate examples.

## 8. Artifact layout and metadata

Large data and experiment outputs remain outside Git. Both hosted workflows use this
logical tree beneath their runtime root:

```text
MTG_Instrument/
├── annotations/
├── dataset/
│   ├── logmel_songs/
│   ├── audio/
│   ├── acousticbrainz/
│   └── song_manifest.csv
├── features/
│   ├── instrument/
│   ├── rhythm/
│   ├── timbre/
│   └── harmony/
├── normalization/
├── checkpoints/
│   ├── baselines/
│   ├── pretraining/
│   └── proposed/
└── results/
    ├── predictions/
    ├── evaluations/
    └── reports/
```

| Runtime | Writable root |
|---|---|
| Colab | `/content/drive/MyDrive/MTG_Instrument` |
| Kaggle | `/kaggle/working/MTG_Instrument` |

Kaggle inputs are read-only and previous notebook outputs are discovered under
`/kaggle/input`. Runtime adapters may resolve different physical paths, but shared
code uses the logical structure above.

The current small development subset uses matching log-Mel and AcousticBrainz shards
`00–02`. Any expansion must change both sources together and record the selected
shards in the experiment configuration.

Every saved artifact must identify or reference its schema version, dataset split,
column order, configuration, input source, creation time, and code commit.

A checkpoint additionally records:

- model and optimizer state;
- epoch and best validation score;
- architecture and training configuration;
- genre and concept target order;
- normalization artifact;
- random seed and dependency environment.

## 9. Code and notebook ownership

- Reusable parsing, preprocessing, extraction, model, loss, and metric logic belongs
  in importable Python modules as it is implemented.
- Colab and Kaggle call the same core functions.
- Runtime-specific notebooks handle environment, storage, and orchestration only.
- The scripts under `scripts/` remain the source of truth for generated notebooks.
- Change a generator first and commit its regenerated notebooks in the same change.
- Do not maintain notebook-only forks.
- Committed notebooks contain no execution outputs.

## 10. Required checks before integration

The project should expose one command that runs these checks:

1. Compile Python modules and notebook generators.
2. Run unit tests for schemas, shapes, masks, losses, and metrics.
3. Run extraction tests and the small shared data/model fixture.
4. Regenerate notebooks twice and confirm deterministic output.
5. Validate every notebook as JSON and compile its Python cells.
6. Confirm notebook outputs are stripped.
7. Check documentation links, paths, names, and status claims.

A component is described as:

- **proposed** before code exists;
- **code present** when code exists but has not passed representative execution;
- **ready** after its agreed tests and smoke run pass;
- **integrated** only after it works in the proposed model.

## 11. Git and research rules

- Use one focused branch per responsibility and update it from `main` before
  integration.
- Keep commits focused and do not commit datasets, targets, checkpoints, results, or
  notebook outputs.
- Update documentation and status in the same change as the implementation.
- Do not claim measured results from unexecuted notebooks.
- Do not present handcrafted descriptors as learned concept embeddings.
- Do not present data preparation, an LLM, or an agent workflow as the deep-learning
  research contribution.
- Report compute, latency, limitations, and failed experiments where relevant.

The research contribution is the shared encoder, supervised learned concept
branches, joint objective, gated fusion, and evidence from controlled evaluation.
