# Harmony evaluation plan

**Status:** Audit and team handoff complete. Read the
`harmony_branch/docs/integration-handoff.md` decision note for the current
contract, evidence, and deferred experiments. No further input is required
from Tharupahan for this handoff.

**Follow-up, 2026-09-30:** Findings about feature redundancy, omitted information,
and potential target sets are documented in the
[feature-selection evaluation](harmony_branch/docs/feature-selection-evaluation.md).
Its candidate derivations and comparison sequence were proposals; the all-45
target wiring is now implemented, but training has not been performed. The completed v3 audit
below remains the record of the implemented baseline.

**All-45 preparation, 2026-09-30:** The trainer now has an `all45` Harmony
configuration and a `selected12` compatibility option. The 45 labels come from
`data/harmony_df.csv`, not a guessed chord teacher. This is code/data preparation
for Dehan, not a training result. Next: run a bounded all-45 smoke test with the
actual log-mel cache, then a matched 12-versus-45 training comparison under the
same split and budget. Compare genre macro AP as the primary outcome and inspect
all per-feature Harmony errors; decide whether to retain, reduce, or replace
targets only after that evidence. The [v4 architecture record](harmony_branch/docs/architecture-versions/v4/README.md)
defines the exact new path.

The next research scope is broader than selecting columns from the old table:
the [52-entry catalog](harmony_branch/docs/harmony-feature-candidate-catalog.md)
records musical hypotheses, and the
[feasibility shortlist](harmony_branch/docs/pseudo-label-feasibility-shortlist.md)
filters them by audio availability, tool capability, compute, and pseudo-label
quality. It defines a proposed ten-value audio pilot and preparation gates for
handoff to Dehan. No new extraction or training has been run.

## Purpose

Evaluate the current harmony targets and model hand-off before proposing changes.
This is an audit of the existing approach, not an instruction to implement every
idea in `harmony_branch/docs/evidence/harmony_branch_evaluation_brief.md`.
Tharupahan owns this evaluation and target preparation; Dehan owns model
training. Any new training comparison is handed to Dehan rather than run as
part of this audit.

## What is currently implemented

- `data/harmony_df.csv` contains 45 automatically extracted, song-level harmony
  measurements for 7,324 tracks. The detailed table is
  `harmony_branch/data/harmony_features_raw.csv`.
- The documented extractor uses full-quality audio, Librosa CQT chroma, and up to
  the first 240 seconds of each track. These measurements are pseudo-labels, not
  human-verified chords or notes.
- The exact extraction notebook was supplied separately and is now at
  `harmony_branch/notebooks/extract_harmony_features_first4min_vastai.ipynb`.
  It was absent from all fetched branches and reachable Git history before this
  addition. Older `06_harmony_features.ipynb` notebooks and
  `scripts/features/extract_harmony.py` exist on historical branches, but use a
  different 18-feature method and cannot reproduce this 45-feature table. The
  current `harmony_branch/src/harmony_branch/features.py` implements a separate
  temporal-chroma candidate, not this song-level CSV extractor.
- The supplied notebook pins Librosa `0.11.0`, decodes mono MP3 at 16 kHz for at
  most 240 s, computes CQT chroma with a 512-sample hop, L1-normalizes valid
  frames, and derives six Tonnetz dimensions plus summary statistics. It does
  **not** perform harmonic/percussive separation; its validity mask is based on
  RMS and finite, nonzero chroma, not a measure of musical tonality. The raw
  table matches its 45-column schema and recorded extractor settings; all 7,324
  rows are `ok`, with 2,583 analyzed to the 240 s cap. The clean CSV is an exact
  column projection of the raw CSV. This consistency does not independently
  prove the notebook was the executed file or validate individual labels against
  source audio.
- `concept_fusion/contract.py` selects 12 of the 45 measurements for current
  joint training. The model predicts those 12 song-level values, standardizes
  targets using training rows, and applies a masked Smooth L1 loss.
- `concept_fusion/harmony_adapter.py` sends the **predicted 12 descriptors** to
  fusion. The temporal 12-pitch-class and optional 25-chord outputs are retained
  in code but are not supervised by the current joint run.
- The supplied `harmony_branch/docs/evidence/harmony_supervised_joint_run_results.json` confirms
  the quoted second-run harmony test scores:
  macro R² `0.4704015` and standardized MAE `0.4791458`. It reports 1,098 test
  tracks with all 12 harmony targets observed, and genre test macro AP `0.7355781`.
  The three selected Tonnetz mean components have R² `0.0718`, `0.0436`, and
  `-0.0042`; Tonnetz movement mean has R² `0.8107`.
- The separately supplied `best.pt` checkpoint matches the report's best epoch
  and validation AP. It confirms the exact 12 target names/order, descriptor
  strategy, harmony supervision, and mel/window settings. Neither artifact
  records a code commit or the exact harmony loss weight used. The current
  code's default is not proof of the historical run setting.
- The team has decided that all audio used for training is under four minutes.
  Treat this as the agreed input policy. The longer durations in the source
  manifest and the 240-second cap in the harmony extractor describe source
  recordings and target extraction, not a request to reopen that decision.

## Evaluation steps

### 1. Verify how the harmony measurements were made

The supplied notebook and committed CSV now establish the written extraction
recipe and internal table consistency. Record the absence of harmonic separation
and that the so-called tonal-validity mask is only an energy/chroma-availability
filter. If independent numerical reproduction is needed, check a few rows
against matching full-quality MP3s. Record any deviation from
`harmony_branch/docs/feature-contract.md`.

**Decision:** keep the existing table if its provenance and calculations are
reproducible. Correct and version only affected features if there is a concrete
error; do not replace the table merely because a different extractor exists.

### 2. Record the agreed audio-span policy

The harmony table summarizes at most the first 240 seconds. The team has
confirmed that all training audio is under four minutes. Use that decision as
the model-input contract; no cache sample is required for this audit. Some
source recordings are longer, and the extractor caps their targets at 240 s,
which is consistent with the agreed training-audio limit.

**Decision:** do not reopen the span question without concrete contradictory
evidence from the actual training input.

### 3. Verify the current model-to-fusion contract

Trace input `(B,T,128)`, valid-token masks, pooled harmony embedding, descriptor
head `(B,12)`, target standardization and mask, loss, and fusion projection.
Record that the current fusion input is 12 **predicted song descriptors**, not 12
pitch-class probabilities. Check the saved run configuration against this code.

**Decision:** treat temporal chroma and chord prediction as a separate proposed
experiment. Do not interchange its 12 probabilities with the current 12
descriptors just because their tensor widths match.

### 4. Check the reported results and weak features

The supplied `harmony_branch/docs/evidence/harmony_supervised_joint_run_results.json`
verifies the reported harmony macro R², standardized MAE, and per-feature
values. The supplied checkpoint confirms the configured target order. All 12
targets are observed on all 1,098 test tracks. The first three selected
Tonnetz means vary in the 5,127-row training split (standard deviations about
0.1324, 0.1256, and 0.0822); low reported R² alone does not establish that a
deeper model is needed.

**Decision:** recommend a target, loss, or model change only when the run record
and feature diagnostics support a specific cause. Preserve the existing baseline
for any later comparison.

### 5. Hand off a concise decision note

Completed in `harmony_branch/docs/integration-handoff.md`: an evidence-backed
note with three labels: **implemented**, **proposed**,
and **rejected/deferred**. Include the exact 12 current target names, extraction
provenance, audio-span finding, verified branch metrics, and what fusion receives.
Give Dehan and Thevindu the current contract. If a temporal-chroma or alternative
descriptor experiment is justified, specify a matched comparison: same tracks,
splits, encoder, budget, genre macro AP, and branch-appropriate diagnostics.
Dehan owns that training experiment. No architecture change follows automatically
from this document.

## Remaining limitations, not user blockers

The checkpoint does not record a code commit or the exact harmony loss weight.
Individual extracted numbers were not independently reproduced from the MP3s;
the notebook/CSV consistency checks are the available evidence. These
limitations are stated in the handoff. No cache files, audio collection, or
other inputs are requested from Tharupahan.

## Completion criteria

- The current 12 targets and fusion input are stated accurately.
- Target provenance is checked against the supplied notebook and CSV. The
  team's under-four-minute training-audio decision is recorded as the input
  policy, without demanding a cache sample.
- Quoted run metrics are checked against the saved report; run provenance is
  confirmed separately or clearly marked unverified.
- The team receives a decision note without silently switching to temporal
  chroma, chords, a different extractor, or a larger model.
