# Harmony v4 — chroma-grounded 45-descriptor branch

Status: **implemented, not yet trained on real audio.** Git branch `harmony-v2`,
based on `main` at `46fecee`. Code: `ChromaGroundedHarmonyBranch` in
[`model.py`](../../../src/harmony_branch/model.py), target contract in
[`constants.py`](../../../src/harmony_branch/constants.py) and
[`descriptors.py`](../../../src/harmony_branch/descriptors.py). No checkpoint or
run report exists for v4 yet; every accuracy statement below about v3 comes from
the v3 evidence report, and v4 must be measured by a Modal run.

## Why v3 was replaced

The v3 run ([evidence](../../evidence/harmony_supervised_joint_run_results.json))
reached test macro R² 0.470 on 12 hand-picked descriptors. It had three problems:

| v3 problem | Evidence | v4 change |
|---|---|---|
| 33 of the 45 extracted descriptors were thrown away, including all 12 chroma means and 12 chroma stds | `HARMONY_FEATURES` had 12 names | All 45 descriptors are targets and fusion inputs |
| Tonnetz means were learned as free regressions and failed | test R² 0.072 / 0.044 / −0.004 | Derived exactly from predicted chroma (see identity below) |
| Mean pooling of a 32-D embedding cannot represent variation; std-type targets were weak | `*_std` R² ≈ 0.43–0.44; `valid_tonal_ratio` R² 0.14 | Attention + mean + std pooling, and token-level chroma statistics |
| Plain z-scores on heavily skewed targets | mean \|skew\| 1.26 over the 45 columns; `valid_tonal_ratio` skew −4.73 | Named log / log1m transforms before the train-only z-score: mean \|skew\| 0.25 |

## Two exact identities used by the model

Both follow from the extractor in
[`extract_harmony_features_first4min_vastai.ipynb`](../../../notebooks/extract_harmony_features_first4min_vastai.ipynb)
and were verified on all 7,324 rows of `data/harmony_df.csv` (max error 0.0):

1. `chroma_<pitch>_mean` is the mean of per-frame L1-normalized chroma, so it is a
   distribution over 12 pitch classes (sums to 1).
2. `tonnetz_<k>_mean = Φ · chroma_mean`, where Φ is librosa's fixed 6×12
   Tonnetz matrix. Tonnetz is linear in L1-normalized chroma, so its mean is Φ
   applied to the chroma mean.

## Architecture

Input: shared-encoder tokens `(B,T,128)`, token mask `(B,T)`, window index
`(B,T)` (unchanged from v3).

1. **Input projection** `Linear(128,96) → LayerNorm → GELU`.
2. **Temporal context:** 4 dilated residual Conv1d blocks (kernel 3, dilations
   1/2/4/8, GELU, dropout 0.1, LayerNorm). Receptive field is 31 tokens, about
   2 s at the 64 ms token stride, which covers typical chord durations. Runs
   inside each sampled window only, so it never crosses a gap between windows.
3. **Token chroma head** `Linear(96,96) → GELU → Linear(96,12)` gives per-token
   pitch-class logits (kept as `chroma_logits`); softmax gives `q_t`.
4. **Exact descriptors (18):** chroma means = masked mean of `q_t`; Tonnetz
   means = Φ · chroma mean. They are transformed and standardized inside the
   model with the fitted target buffers, and have no free output parameters.
5. **Token-chroma statistics (26),** computed with the extractor's own formulas
   on `q_t`: per-class std (12), Tonnetz std (6), max-bin concentration
   mean/std, entropy/log 12 mean/std, L2 chroma-flux mean/std, and L2 Tonnetz
   movement mean/std. Flux and movement use adjacent valid tokens within the
   same window only.
6. **Statistics pooling** of the token context: attention mean, masked mean and
   masked std → `Linear(288,64) → LayerNorm → GELU` gives the 64-D song
   embedding (used by the embedding-fusion ablation).
7. **Learned descriptors (27)** (chroma stds, Tonnetz stds, 9 tonal-dynamics
   values): `LayerNorm → Linear(90,96) → GELU → Dropout → Linear(96,27)` over
   `[embedding, log statistics]`, plus a learned per-descriptor gain on its
   matching statistic (a direct, interpretable path).
8. **Outputs:** `descriptor_values (B,45)` in target space and
   `descriptor_raw (B,45)` in physical units via the inverse transform.
   Unavailable songs are exactly zero.

Parameters: about 202k (v3 was a 64-hidden, 32-embedding reference model).

## Targets, loss, fusion

* Targets: the 45 raw columns of `data/harmony_df.csv`, joined by `TRACK_ID`
  (`--harmony-csv`). The compact vector table's scaled 12-wide `harmony_vector`
  is ignored.
* Transform: `log(x+1e-4)` for positive right-skewed descriptors,
  `−log(1−x+1e-3)` for `valid_tonal_ratio`, identity for the 6 Tonnetz means,
  `chroma_entropy_mean` and `tonal_concentration_std`. Then a z-score fitted on
  **training rows only**. The model stores mean/scale as buffers, so
  checkpoints are self-contained.
* Loss: unchanged masked Smooth L1 over observed values (`--lambda-harmony`,
  default 0.5).
* Fusion: `Linear(45,64)` (`harmony_descriptor_projection`), gated fusion
  unchanged. The 12-bin chroma logits still support the optional aligned
  temporal loss.

## Reported metrics

`results.json` harmony metrics add `group_macro_r2` (chroma_mean, chroma_std,
tonnetz_mean, tonnetz_std, tonal_dynamics) and `v3_subset_macro_r2`. The subset
score uses the same 12 names as v3, but 9 of them are now log-transformed, so it
is indicative rather than an exact comparison with v3's 0.470.

## Compatibility

* v3 checkpoints do not load into v4 (different head and fusion width).
* `TemporalHarmonyBranch` is kept for reference. Its descriptor-less mode can no
  longer feed fusion; the adapter raises a clear error.
* The genre effect of harmony is still unproven until a matched no-harmony
  ablation is run.

## How to evaluate

```bash
modal run modal_app.py --quick --run-name harmony-v4-smoke
modal run --detach modal_app.py --background --run-name harmony-v4-full --epochs 30
```

Compare `test_branch_metrics.harmony` and `test_macro_ap` with the v3 report,
then run the no-harmony ablation before making claims about genre accuracy.
