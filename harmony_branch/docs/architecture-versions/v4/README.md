# Harmony v4 — all-45 descriptor training candidate

**State:** Implemented in code on `harmony-feature-evaluation`; no v4 checkpoint,
validation result, or test result has been produced. V3 remains the completed
12-feature reference; see [its detailed diagrams](../v3/README.md). This record
describes the changed contract without reinterpreting that historical run.

## Exact data and tensor path

| Stage | V4 contract |
|---|---|
| Audio/teacher | Same first-`min(duration,240 s)` Librosa CQT-derived table as v3; 7,324 tracks in `data/harmony_df.csv` |
| Label order | 12 chroma means, 12 chroma standard deviations, six Tonnetz means, six Tonnetz standard deviations, eight other tonal summary statistics, one valid-frame ratio; exact order is `HARMONY_FEATURES_45` |
| Shared encoder input | Existing bounded log-mel song windows; no audio preprocessing change |
| Branch input | `(B,T,128)` encoded sequence, valid-token mask, window indices |
| Branch intermediate | Masked song embedding, default `(B,32)` |
| Supervised head | `Linear(32,32)` → GELU → dropout → `Linear(32,45)` descriptor output `(B,45)`; temporal chroma logits `(B,T,12)` remain present but unsupervised; optional chord head remains disabled |
| Targets | `(B,45)` values joined by normalized track ID from authoritative `harmony_df.csv`, standardized per column using **training split only**; observation mask `(B,45)` |
| Loss | Masked Smooth L1 on standardized 45 descriptor values, with configurable `lambda_harmony` (default 0.5); no chord or chroma teacher loss |
| Fusion boundary | **Predicted** descriptor vector `(B,45)` through fusion-owned `Linear(45,64)`, then the existing four-token fusion and genre head |
| Checkpoint | Records `harmony_feature_set=all45`, all 45 target names/order, standardizer statistics, descriptor count, and head/projection weights; incompatible 12-vs-45 layer weights must not be silently loaded |

The concise flow is:

```text
audio -> log-mel windows -> shared encoder (B,T,128)
                                 |
                                 v
                     masked Harmony song embedding (B,32)
                                 |
                    descriptor head (B,45 predictions)
                       /                       \
      masked loss vs standardized labels     fusion Linear(45,64)
      joined from harmony_df.csv                 -> genre prediction
```

No ground-truth descriptor vector enters fusion. Chroma is 12 pitch classes;
the 45 descriptors are song summaries. Chord progression is not represented.
Correlated descriptors are deliberately retained for this candidate, so a wider
target does **not** imply more independent musical information or better genre AP.

## Reproducible comparison contract

The default trainer and Modal entrypoint select `all45`. The historical shape is
still selectable with `--harmony-feature-set selected12`; it continues to use
the 12-column vector/selected columns and `Linear(12,64)`. Keep the same split,
encoder/audio cache, seed, optimizer, epoch budget, concept loss weights, and
genre evaluation protocol when comparing. Record macro AP, Harmony per-feature
R²/MAE, and macro R². The test split remains reserved for the final comparison;
do not claim v4 improvement from this implementation alone.

For compact CSVs, the trainer always reads `data/harmony_df.csv` in all-45 mode
and joins by track ID; Modal requires the same table uploaded under
`dataset/harmony_df.csv`. The compact-dataset builder can also emit 45-element
vectors. It fails on missing or duplicate Harmony IDs instead of filling labels.
Old 12-element compact files remain usable with the all-45 join.

## Delta from v3

Only the target selection, descriptor-head output, Harmony loss width, fusion
projection input, data join, and checkpoint schema change. Encoder, masking,
embedding width, temporal output definitions, other branches, and four-token
fusion width stay the same. V3 evidence cannot be assigned to this version.
