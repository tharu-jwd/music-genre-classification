# Architecture Decision Record — Concept Fusion, Genre Head, Evaluation, Explainability

**Status:** Proposed for team ratification (90-minute architecture meeting)  
**Owner:** Thevindu (integration)  
**Branch:** `thevindu-concept-fusion`  
**Contract:** shared architecture v0.1  

This record freezes the 11 decisions in the ownership brief §8. Branch owners should object in the meeting, not by silently reshaping tensors later.

---

## Recommendations (ratify these)

| # | Decision | Choice | Why |
|---|---|---|---|
| 1 | Concat baseline | Zero masked tokens, concat 256-D, Linear→128. Uniform gates over enabled concepts. | Required simple control; parameters are *not* matched to gated (report both counts). |
| 2 | Primary fusion | **Masked gated fusion** over LayerNorm tokens. Score is a scalar linear `w⊤ h_k`. Softmax only over enabled branches. Disabled gates are **exactly 0**. | Heterogeneous branches need an explicit availability mask; gating is the proposed contribution. |
| 3 | Self-attention | Keep as named ablation **F-Attn**, not primary. | Attention weights are not explanations ([Jain & Wallace 2019](https://aclanthology.org/N19-1357/)). |
| 4 | Gate type | **Global per-track gates** in the primary model. Genre-conditioned 87×4 gates are a named ablation only. | 87×4 gates on a few thousand tracks will overfit; interpretation gets worse. |
| 5 | Bottleneck units | **Instrument v2 (published):** branch returns `concept_values (B,40)` probabilities and `logits (B,40)` — **no `fusion_token`**. Fusion owns `Linear(40,64)` and applies `fusion_mask` after that projection. Other branches still supply `(B,64)` tokens until they publish otherwise. Aux instrument loss is **BCE-with-logits**. | Anupama's merged instrument branch; value-level intervention stays in probability space. |
| 6 | Token normalization | **LayerNorm each token before fusion. Non-negotiable.** | Instrument embedding vs rhythm scalars are different geometries. |
| 7 | Concept dropout | **p = 0.15** per branch during **training only**. Never drop all four. All-masked fallback: **learned null token**. | **Hard dependency:** no dropout ⇒ no occlusion faithfulness. Masking at test would be OOD. |
| 8 | Loss weights | Each concept loss is a **mean over observed elements** (unit scale). Start **λ = 1**. Sweep on validation. Compare **Kendall uncertainty weighting** as one run. | Stops 40 instrument tags from dominating 6 timbre tags by count. |
| 9 | Thresholds | **Per-tag** threshold maximizing **validation F1**. Tags with support `< 10` use **global 0.5**. Never fit on test. | 87 imbalanced labels; one global threshold is too crude. |
| 10 | Explanations | Report gates **and** occlusion deltas. Rank correlation is required. Attention/gates alone are insufficient. | Same fusion-mask mechanism as dropout. |
| 11 | Runs | JSON `RunRecord` with config hash, seed, git SHA. Tables generated from files. Seeds: **3** for B1 / F-Concat / F-Gated; **1** for everything else. | Colab cannot host 33+ full trainings. |

---

## Mask distinction (do not collapse)

- `supervision_mask (B, C_k)`: target is observed → may enter aux loss. Missing labels **must not** inner-join the track out of genre training.
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

Instrument (published on `main`): `concept_values (B,40)`, `logits (B,40)`, `supervision_mask (B,40)`, `fusion_mask (B,1)`. **No `fusion_token`.** Hidden diagnostics are detached `(B,128)`. Vocabulary is the official alphabetical 40-tag list in `instrument_branch/docs/instrument-vocabulary.json`.

Rhythm / timbre / harmony still deliver:

- `concept_values (B, C_k)`
- `fusion_token (B, 64)`
- `supervision_mask (B, C_k)`
- `fusion_mask (B, 1)`

Rhythm: C_k = 10. Timbre/harmony: confirm 6 and 18.

---

## Status of this branch (Step 1 + Step 2 on mocks)

Implemented and unit-tested against fixtures, aligned to the published instrument v2 API (fusion-owned `Linear(40,64)`). `python scripts/run_all_fusion.py --quick` trains the full fusion-owned matrix on one frozen fixture cohort. Real encoder + live instrument wiring is the next integration step (C-I). Rhythm/timbre/harmony tokens are still fixtures.
