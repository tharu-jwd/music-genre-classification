# Architecture Decision Record — Concept Fusion, Genre Head, Evaluation, Explainability

**Status:** Harmony integration candidate; team ratification still required

**Owners:** Thevindu (fusion) and harmony branch owner

**Branch:** `harmony`

**Contract:** predicted-concept fusion v0.3 (`predicted_concept_fusion_v1` checkpoint contract)

This record freezes the 11 decisions in the ownership brief §8. Branch owners should object in the meeting, not by silently reshaping tensors later.

---

## Recommendations (ratify these)

| # | Decision | Choice | Why |
|---|---|---|---|
| 1 | Concat baseline | Zero masked tokens, concat 256-D, Linear→128. Uniform gates over enabled concepts. | Required simple control; parameters are *not* matched to gated (report both counts). |
| 2 | Primary fusion | **Masked gated fusion** over LayerNorm tokens. Score is a scalar linear `w⊤ h_k`. Softmax only over enabled branches. Disabled gates are **exactly 0**. | Heterogeneous branches need an explicit availability mask; gating is the proposed contribution. |
| 3 | Self-attention | Keep as named ablation **F-Attn**, not primary. | Attention weights are not explanations ([Jain & Wallace 2019](https://aclanthology.org/N19-1357/)). |
| 4 | Gate type | **Global per-track gates** in the primary model. Genre-conditioned 87×4 gates are a named ablation only. | 87×4 gates on a few thousand tracks will overfit; interpretation gets worse. |
| 5 | Bottleneck units | **Instrument v2:** 40 probabilities + logits; fusion owns `Linear(40,64)`. **Rhythm v1:** mel-derived temporal encoder supplies a learned `(B,64)` token plus ten standardized AcousticBrainz predictions used only for auxiliary loss. **Timbre v2:** 35 standardized descriptors; fusion owns `Linear(35,64)`. **Harmony v1:** temporal chroma `(B,T,12)`, optional chords `(B,T,25)`, and configurable song embedding; fusion owns `Linear(D_harmony,64)`. | Published branch contracts plus the CPU-tested temporal adapters. |
| 6 | Token normalization | **LayerNorm each token before fusion. Non-negotiable.** | Instrument embedding vs rhythm scalars are different geometries. |
| 7 | Concept dropout | **p = 0.15** per branch during **training only**. Never drop all four. All-masked fallback: **learned null token**. | **Hard dependency:** no dropout ⇒ no occlusion faithfulness. Masking at test would be OOD. |
| 8 | Loss weights | Each concept loss is a **mean over observed elements** (unit scale). Start **λ = 1**. Adjust only for a recorded instability or ineffective gradient; Kendall weighting is a named comparison, not a broad sweep. | Stops large target sets from dominating by count while respecting the compute plan. |
| 9 | Thresholds | **Per-tag** threshold maximizing **validation F1**. Tags with support `< 10` use **global 0.5**. Never fit on test. | 87 imbalanced labels; one global threshold is too crude. |
| 10 | Explanations | Report gates **and** occlusion deltas. Rank correlation is required. Attention/gates alone are insufficient. | Same fusion-mask mechanism as dropout. |
| 11 | Runs | JSON `RunRecord` with config hash, seed, git SHA. Tables generated from files. Seeds: **3** for B1 / F-Concat / F-Gated; **1** for everything else. | Colab cannot host 33+ full trainings. |

---

## Mask distinction (do not collapse)

- Fixed branches use `supervision_mask (B, C_k)`. Harmony uses independent temporal chroma/chord masks `(B,T)`. A target enters only its matching auxiliary loss. Missing labels **must not** inner-join the track out of genre training.
- `fusion_mask (B, 1)` per branch, stacked to `(B, 4)`: predicted branch is enabled for fusion (dropout, missing-branch, ablations).

NaN on unobserved targets is **allowed**. NaN on observed targets or on fusion tokens is a **contract error**. Never `nan_to_num(0)`.

## Genre protocol

- Vocabulary: **exactly 87** official split-0 tags, fixed order (Dehan / shared encoder freeze).
- `genre_logits (B, 87)`; `genre_probs = sigmoid(logits)` independently; **never softmax**.
- Primary metric: **macro average precision** = `sklearn.metrics.average_precision_score`, **not** trapezoidal PR-AUC (`auc(recall, precision)`). Both are logged; a unit test proves they differ.
- Always report `n_valid_tags` (exclude all-0 / all-1 tags).

## What is not in the primary model

- No `song_repr → genre` path. That is **F-Shortcut** only (interpretability tax).
- Hidden branch embeddings are **F-Hidden** only, not fusion inputs.
- Do not turn off concept dropout to chase val AP.

## Compute budget

| Experiments | Seeds |
|---|---|
| B1, F-Concat, F-Gated | 0, 1, 2 |
| All others (C-*, F-Attn, F-Hidden, F-Shortcut, leave-one-out) | 0 only |

State this in every results table.

## Literature → one decision each

1. **Koh et al., Concept Bottleneck Models** — predict concepts, then labels; interventions on concept values. → Strict bottleneck; F-Shortcut/F-Hidden measure the tax.
2. **Arevalo et al., GMU** — learned gates on heterogeneous modalities. → Masked gated fusion is primary.
3. **Jain & Wallace, Attention is not explanation** — weights ≠ importance. → Occlusion required.
4. **Wiegreffe & Pinter, Attention is not not explanation** — tests can support a careful claim. → Rank-correlate gates vs occlusion; report counterexamples.
5. **Kendall et al., uncertainty weighting** — balance aux losses. → One comparison run, not the default.
6. **Bogdanov et al., MTG-Jamendo** — official split-0, 87 genre tags, ROC-AUC and PR-AUC. → Frozen cohort + vocab.
7. **Choi et al., Automatic tagging with CNNs** — direct mel-CNN. → B1 protocol baseline.

## Ask each branch owner before real integration

Instrument (published): `concept_values (B,40)`, `logits (B,40)`, masks. **No `fusion_token`.** Vocabulary: `instrument_branch/docs/instrument-vocabulary.json`.

Timbre (published): `z_timbre` / `d_hat_standardized (B,35)` in `FEATURE_COLUMNS` order. **No `fusion_token`. Never fuse `h_audio`.** Feature list: `timbre_branch/src/timbre_branch/constants.py`.

Rhythm v2 consumes only the shared encoder's ordered `(B,T,128)` mel-derived features.
Its ten standardized predictions are `concept_values` and primary fusion owns
`Linear(10,64)`. Its learned 64D embedding is retained for `embedding_fusion` only.
AcousticBrainz values are targets and never branch inputs.

Harmony (integration candidate): `embedding (B,D_harmony)`, temporal chroma logits
`(B,T,12)`, optional chord logits `(B,T,25)`, prediction/target masks, and branch
availability. **No `fusion_token`.** Primary fusion owns `Linear(12,64)` over the
valid-token mean of per-token predicted chroma probabilities. `Linear(D_harmony,64)`
over the embedding remains an ablation. Auxiliary loss uses temporal logits through
`HarmonyTargets`.

---

## Status of this branch (Step 1 + Step 2 on mocks)

Implemented and unit-tested against fixtures, aligned to instrument v2
(`Linear(40,64)`), rhythm v2 (`Linear(10,64)`), timbre v2 (`Linear(35,64)`), and
temporal harmony v2 (`Linear(12,64)`). `python scripts/run_all_fusion.py --quick` trains the full
matrix on fixtures. Harmony's real branch adapter is exercised end to end on
synthetic ordered encoder inputs; official real-audio artifacts are still required.
