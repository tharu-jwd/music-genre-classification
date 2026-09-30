# Harmony handoff: audited current run

This is the handoff for Dehan (training) and Thevindu (fusion), audited on
2026-09-28. The current joint run uses **12 song-level descriptors**, not 12
pitch-class probabilities and not all 45 columns of the extracted table. Keep
those three contracts separate. The team has decided that training audio is
under four minutes; no further cache-file request is part of this audit.
For complete component diagrams and version boundaries, use the
[Harmony v3 architecture snapshot](architecture-versions/v3/README.md).

## Implemented and checked

The supplied [extraction notebook](../notebooks/extract_harmony_features_first4min_vastai.ipynb)
creates 45 automatic descriptors from the first `min(decoded duration, 240 s)`
of each full-quality mono MP3. It pins Librosa 0.11.0, resamples to 16 kHz,
computes CQT chroma every 512 samples, estimates tuning, normalizes valid
chroma frames to sum to one, and summarizes chroma, Tonnetz, concentration,
entropy, movement, flux, and valid-frame coverage. It does **not** perform
harmonic/percussive separation. Its “valid tonal” filter means sufficient RMS
and finite, nonzero chroma; it is not a calibrated tonality detector.

The [raw table](../data/harmony_features_raw.csv) has 7,324 `ok` rows and the
notebook's 45-column schema, 16 kHz/512-hop/extractor metadata, and 2,583 rows
at the 240-second analysis cap. The [training table](../../data/harmony_df.csv)
is an exact column projection of that raw table. The 12 selected targets, in
checkpoint order, are:

```text
tonal_concentration_mean, tonal_concentration_std,
chroma_entropy_mean, chroma_entropy_std,
chroma_flux_mean, chroma_flux_std,
tonnetz_movement_mean, tonnetz_movement_std,
valid_tonal_ratio, tonnetz_01_mean, tonnetz_02_mean, tonnetz_03_mean
```

In the current [joint trainer](../../scripts/train_joint.py), the shared encoder
provides ordered `(B,T,128)` features and a valid-token mask. The harmony branch
masked-pools them to a song embedding `(B,32)` and predicts standardized
descriptors `(B,12)`. Training targets are standardized using **training rows
only**. The 12 predictions receive masked Smooth L1 supervision, and the
[harmony adapter](../../concept_fusion/harmony_adapter.py) passes those same
predicted values, with availability mask, to the [fusion-owned projection](../../concept_fusion/projections.py)
`Linear(12,64)`. They are *not* softmax probabilities. Temporal chroma logits
also exist `(B,T,12)` but have no chroma target in this run; the chord head is
disabled. The 32D embedding is not the primary fusion input.

The supplied [run report](evidence/harmony_supervised_joint_run_results.json)
records 5,127/1,099/1,098 train/validation/test tracks, best epoch 20,
validation genre macro AP 0.7660, and test genre macro AP 0.7356. Harmony test
macro R² is 0.4704, standardized MAE 0.4791, and standardized RMSE 0.6940,
over all 12 targets on 1,098 test tracks. Selected per-target R² values:

| Descriptor | Test R² |
|---|---:|
| Tonnetz movement mean | 0.811 |
| Chroma flux mean | 0.796 |
| Tonal concentration mean | 0.762 |
| Valid tonal ratio | 0.138 |
| Tonnetz 01 mean | 0.072 |
| Tonnetz 02 mean | 0.044 |
| Tonnetz 03 mean | -0.004 |

The separately supplied `best.pt` checkpoint matches report epoch 20 and
validation AP 0.765960. Its metadata confirms the 12 target names/order above,
`standardized_song_descriptor_regression`, harmony supervision enabled,
16 kHz/512-hop/128-mel/15-second cache metadata, a saved
`window_frames=1366` fallback for the 2D log-Mel loader, and a 12-window
limit. The 3D stacked-cache loader uses the cache's actual frame width rather
than `window_frames`; the checkpoint alone does not establish that width.
SHA-256:
`7916839945e30ca9b4c2d5d6796a9989e1950313b0e5546a4ef8180440e46761`.
The checkpoint remains outside Git, consistent with the repository's checkpoint
ignore policy. The checkpoint and JSON do not record a code commit or the exact
harmony loss weight used; the current code's default of 0.5 must not be
presented as a verified historical run setting.

The three weak Tonnetz means are not constant: their training-set standard
deviations are approximately 0.1324, 0.1256, and 0.0822. Weak R² alone does
not identify a model-depth problem. The reported genre AP is a whole-model
score, **not** evidence that harmony improves genre prediction; a matched
no-harmony comparison is needed for that claim.

## Decision and work boundaries

- **Keep as the implemented baseline:** the 12 standardized song-descriptor
  target and fusion contract above. Do not silently substitute all 45 table
  columns or 12 chroma probabilities.
- **Proposed for Dehan's experiments:** a matched leave-harmony-out run, using
  the same split, encoder, training budget, and genre macro AP evaluation, to
  measure harmony's contribution. If testing a revised target set or loss,
  compare against this baseline and report per-descriptor metrics.
- **Deferred, not implemented in this run:** temporal chroma supervision,
  chord labels/progressions, an Essentia replacement, extra temporal summaries,
  and a deeper harmony model. Each requires its own target-quality check and
  matched experiment; none follows automatically from the weak Tonnetz scores.
- **For Thevindu:** consume the predicted standardized `(B,12)` descriptor
  vector through the current `12→64` projection. The variable name
  `harmony_chroma_projection` is historical; its input in this run is not chroma.

No manual annotation is required for the current baseline. These are
automatically extracted reference measurements, not human-verified chords or
notes. The [review brief](evidence/harmony_branch_evaluation_brief.md) is
evidence to evaluate, not an approved architecture specification. The
[root audit plan](../../plan.md) records the checks and remaining limitations.

This audit used static code/checkpoint inspection, JSON validation, and CSV
consistency checks. It did not retrain the model or recompute features from
source MP3s; PyTorch is not installed in the local audit environment.
