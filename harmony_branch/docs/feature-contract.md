# Harmony feature contract

## Status

This document is the canonical contract for the extracted, track-level harmony
descriptors used by the current baseline. It supersedes older statements in this
directory that describe an unselected 18-value summary or say that full-corpus
harmony extraction has not been completed.

The temporal chroma/chord implementation remains an experimental architecture. It
does not change the schema or provenance of the completed 45-dimensional target
table described here.

## Source audio and extraction policy

| Property | Frozen value |
|---|---:|
| Selected tracks | 7,324 |
| Audio region | First `min(decoded duration, 240 s)` |
| Input | Mono waveform decoded from the full-quality MP3 |
| Sample rate | 16,000 Hz |
| Hop length | 512 samples |
| Chroma representation | `librosa.feature.chroma_cqt` |
| Pitch-class order | C, C#, D, D#, E, F, F#, G, G#, A, A#, B |
| Tonnetz representation | `librosa.feature.tonnetz` from normalized chroma |
| Extractor identifier | `librosa_chroma_cqt_first4min_v1` |

Chroma frames are L1-normalized. Low-energy or unusable tonal frames are excluded
through the extractor's validity mask rather than being treated as observed zero
targets. All statistics are calculated only from valid tonal frames. Flux and
Tonnetz movement use consecutive pairs for which both frames are valid.

These values are automatically extracted reference descriptors, not human harmony
annotations and not causal explanations.

## Feature schema

Every clean row contains `TRACK_ID` followed by exactly 45 finite numeric features.

| Group | Columns | Count | Interpretation |
|---|---|---:|---|
| Chroma means | `chroma_{pitch}_mean` | 12 | Mean relative activity of each pitch class |
| Chroma standard deviations | `chroma_{pitch}_std` | 12 | Track-level variation of each pitch class |
| Tonnetz means | `tonnetz_01_mean` ... `tonnetz_06_mean` | 6 | Mean position in the six-dimensional tonal-centroid space |
| Tonnetz standard deviations | `tonnetz_01_std` ... `tonnetz_06_std` | 6 | Variation in tonal-centroid coordinates |
| Tonal concentration | `tonal_concentration_mean`, `tonal_concentration_std` | 2 | Mean and variation of the strongest normalized chroma bin |
| Chroma entropy | `chroma_entropy_mean`, `chroma_entropy_std` | 2 | Mean and variation of normalized pitch-class uncertainty |
| Chroma flux | `chroma_flux_mean`, `chroma_flux_std` | 2 | Mean and variation of frame-to-frame chroma change |
| Tonnetz movement | `tonnetz_movement_mean`, `tonnetz_movement_std` | 2 | Mean and variation of frame-to-frame tonal-centroid movement |
| Valid tonal coverage | `valid_tonal_ratio` | 1 | Fraction of analyzed frames accepted by the tonal validity mask |
| **Total** |  | **45** |  |

The pitch placeholder is expanded in this fixed order:

```text
c, csharp, d, dsharp, e, f, fsharp, g, gsharp, a, asharp, b
```

The 45 values are an explicit concept bottleneck: each output dimension retains a
defined audio meaning. Downstream normalization must be fitted on the training
split only. The raw values must remain available for interpretation and inverse
transformation even if a later projection is used for fusion.

## Files

| File | Purpose |
|---|---|
| `harmony_branch/data/harmony_features_raw.csv` | Auditable table containing extraction metadata, status, diagnostics, and all 45 descriptors |
| `data/harmony_df.csv` | Training-ready projection containing only `TRACK_ID` and the 45 descriptors |
| `harmony_branch/notebooks/extract_harmony_features_first4min_vastai.ipynb` | Reproducible extraction notebook |

The clean table is an exact feature-value projection of the detailed raw table.
It is not a separately calculated dataset.

## Completed-data audit

The committed extraction was checked against `data/split_csv.csv`:

- 7,324 rows and 7,324 unique track IDs;
- no missing or extra selected tracks;
- all 7,324 extraction statuses are `ok`;
- 45 feature columns and 329,580 finite numeric feature cells;
- no blank, NaN, infinite, or nonnumeric feature values;
- zero feature-value mismatches between the raw and clean tables; and
- one consistent sample rate, hop length, and extractor identifier.

Three short files differ from rounded manifest duration by approximately 0.10 s.
This is consistent with codec/frame boundaries and does not change the first-four-
minute policy.

