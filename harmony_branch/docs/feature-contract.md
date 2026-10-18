# Harmony feature contract

## Status

The current stable harmony branch predicts exactly 12 interpretable,
track-level descriptors. A wider 15-column harmony warehouse is retained for
audit, correlation analysis, and future ablations, but it is not the stable model
input.

The temporal chroma/chord implementation remains an experimental architecture.
It does not change the order or meaning of the stable 12-dimensional descriptor
target.

These values are automatically extracted reference descriptors, not human harmony
annotations and not causal explanations.

## Stable model contract

`harmony_vector` contains exactly the following values in this fixed order:

| Position | Column | Interpretation |
|---:|---|---|
| 1 | `global_key_strength` | Strength of the best global key-profile match |
| 2 | `mode_confidence` | Confidence of the major/minor mode decision |
| 3 | `key_stability` | Consistency of local key evidence through time |
| 4 | `chroma_entropy_mean` | Mean pitch-class uncertainty or tonal complexity |
| 5 | `chroma_entropy_std` | Temporal variability of pitch-class uncertainty |
| 6 | `chroma_flux_mean` | Mean frame-to-frame pitch-class change |
| 7 | `tonnetz_movement_mean` | Mean movement in tonal-centroid space |
| 8 | `tonal_centroid_radius_mean` | Mean tonal concentration in tonal-centroid space |
| 9 | `chroma_recurrence_15s` | Recurrence of chroma structure at a 15-second lag |
| 10 | `chord_strength_mean` | Mean confidence or strength of chord-template matches |
| 11 | `chord_change_rate` | Rate of changes in the estimated chord sequence |
| 12 | `chord_vocabulary_ratio` | Diversity of detected chord classes |

The feature order is part of the serialized data contract. Reordering columns
without rebuilding `harmony_vector` is invalid even when the vector width remains
12.

## Harmony warehouse contract

The warehouse retains 15 harmony descriptors:

| Column | Stable model status |
|---|---|
| `global_key_strength` | Stable 12 |
| `mode_confidence` | Stable 12 |
| `key_stability` | Stable 12 |
| `chroma_entropy_mean` | Stable 12 |
| `chroma_entropy_std` | Stable 12 |
| `chroma_flux_mean` | Stable 12 |
| `chroma_flux_std` | Warehouse only |
| `tonnetz_movement_mean` | Stable 12 |
| `tonnetz_movement_std` | Warehouse only |
| `tonal_centroid_radius_mean` | Stable 12 |
| `tonal_centroid_radius_std` | Warehouse only |
| `chroma_recurrence_15s` | Stable 12 |
| `chord_strength_mean` | Stable 12 |
| `chord_change_rate` | Stable 12 |
| `chord_vocabulary_ratio` | Stable 12 |

The warehouse-only fields are therefore:

```text
chroma_flux_std
tonnetz_movement_std
tonal_centroid_radius_std
```

They must not be silently appended to the stable model vector.

## Stable project dimensions

The current stable dataset and architecture use:

| Branch | Width |
|---|---:|
| Instrument | 41 |
| Rhythm | 10 |
| Timbre | 12 |
| Harmony | 12 |
| Genre targets | 6 |

## Training and normalization

The raw harmony values remain available in the warehouse for interpretation and
inverse transformation. The trainer expands `harmony_vector` using the exact
`HARMONY_FEATURES` order, fits the harmony standardizer using training tracks only,
and applies that fitted transform unchanged to validation and test tracks.

The harmony descriptor head predicts 12 standardized values. Predicted values—not
ground-truth targets—enter fusion through the fusion-owned `Linear(12, 64)`
projection.

## Relevant files

| File | Purpose |
|---|---|
| `concept_fusion/contract.py` | Canonical stable Harmony-12 names and order |
| `data/warehouse_dataset_new.csv` | Auditable 15-column harmony warehouse together with the other branches |
| `data/vector-dataset.csv` | Compact stable vectors used by the Modal paper protocol |
| `data/track_split_assignments.csv` | Fixed train, validation, and test assignments |
