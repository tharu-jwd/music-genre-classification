# Architecture from first plan to current implementation

This document is the full story of the proposed model: the **original plan**, what **Thevindu implemented** on `thevindu-concept-fusion`, what **Anupama** and **Senindu** published on `main`, and the **current architecture** after those contracts were absorbed.

Fixture scores are not research numbers.

---

## 1. Original plan

Target: explainable multi-label genre classification on MTG-Jamendo official split 0 (87 tags) via a **concept bottleneck**.

Planned flow (`docs/proposed-architecture.md`):

```text
up to 12 log-mel windows → shared CNN → song representation
    → four concept branches (instrument / rhythm / timbre / harmony)
    → gated fusion → 128-D → 87 genre logits
```

Planned branch embeddings were 64 / 32 / 32 / 32. Timbre supervision was six spectral descriptors. Continuous targets standardized on train only. Missing labels masked, not zero-filled.

Baselines stayed as comparisons: direct CNN (B1) and descriptor fusion (learned 64-D instrument embedding + AcousticBrainz / table descriptors). Stage 2 had inner-join, NaN-to-zero, and attention-as-explanation problems.

---

## 2. First fusion implementation

Thevindu froze contract v0.1 and built the stack on **fixtures**:

- Token order: instrument, rhythm, timbre, harmony → `(B, 4, 64)`.
- Every branch, **including instrument and timbre**, was assumed to already emit a `fusion_token (B, 64)`.
- Provisional `C_k`: instrument 40, rhythm 10, timbre **6**, harmony 18.
- Concat / **masked gated** / attention; LayerNorm; global gates; dropout p=0.15; no `song_repr → genre` in the primary model.
- One-command matrix: `python scripts/run_all_fusion.py`.

That first contract is what Anupama and Senindu later revised for their own branches.

---

## 3. Anupama instrument v2 (already on `main`)

`instrument_branch/`: `song_repr (B,128) → Linear(128,128)→ReLU→Dropout→Linear(128,40) → logits / probabilities`.

- **No `fusion_token`.** Fusion owns `Linear(40, 64)`.
- Official **alphabetical** 40-tag vocabulary.
- Weak-closed-world masks; missing annotation rows stay in genre training.
- Hidden `(B,128)` detached diagnostics only.

---

## 4. Senindu timbre v2 (now on `main`)

Merged as PR #4. Package: `timbre_branch/`.

### Architecture

```text
h_audio (B, 128)
  → Linear(128, 128) → LayerNorm → GELU → Dropout(0.20)
  → Linear(128, 64) → GELU
  → Linear(64, 35)
  → z_timbre = d_hat_standardized (B, 35)
```

- Strict bottleneck: fusion must consume **only** the 35 named concepts.
- Concatenating `h_audio` with `z_timbre` is forbidden — that would bypass the bottleneck.
- No sigmoid/softmax: standardized descriptors can be positive or negative.
- Original acoustic units are recovered with a **training-only** standardizer for explanations.
- Loss: masked Smooth L1 over valid descriptor cells.

### Frozen 35-feature order (`FEATURE_COLUMNS`)

Spectral shape (7): centroid/bandwidth mean+std, contrast, flatness, rolloff.  
Harmonic/noise (2): HNR dB, inharmonicity.  
MFCC envelope (26): mfcc_01..13 mean and std.

### What the branch returns

The model returns a tensor `(B, 35)`. Inference helper:

```python
{"z_timbre", "d_hat_standardized", "d_hat_original_units", "feature_names"}
```

There is **no `fusion_token`**.

### Data note

7,324 selected tracks have a complete 35-column target table (gitignored CSV). Real encoder embeddings are still required for paper training.

Main also removed the old hosted Colab/Kaggle notebook set in that merge. Proposed-architecture.md now lists timbre supervision as **35** descriptors, not 6 or 32.

---

## 5. What fusion changed for timbre

| | First fusion contract | After Senindu v2 |
|---|---|---|
| Timbre `C_k` | Provisional **6** | Frozen **35** |
| Timbre fusion input | Branch-owned `fusion_token (B,64)` | `z_timbre (B,35)` standardized |
| Who owns `Linear(35,64)` | Assumed inside the branch | **Fusion** (`TokenAssembler.timbre_projection`) |
| `h_audio` into fusion | Not specified | **Rejected** (bypasses the bottleneck) |
| Feature order | Unspecified | `TIMBRE_FEATURES` loaded from Senindu’s `constants.py` |

Instrument handling is unchanged: still fusion-owned `Linear(40,64)`.

Rhythm and harmony still supply 64-D tokens until they publish.

---

## 6. Current architecture

```text
song_repr (B, 128)                         ← Dehan; not a primary genre input
    ├── Instrument v2 → 40 probs + logits     → Linear(40,64) × fusion_mask
    ├── Rhythm (pending) → token (B,64)
    ├── Timbre v2 → z_timbre (B,35)          → Linear(35,64) × fusion_mask
    └── Harmony (pending) → token (B,64)
                    ↓
            tokens (B, 4, 64)
                    ↓
     dropout p=0.15 → LayerNorm → gated / concat / attention
                    ↓
            fused (B, 128) → 87 genre logits → independent sigmoid
```

Ingest helpers:

- `from_instrument_branch(dict)` — rejects `fusion_token`
- `from_timbre_branch(z_timbre | dict)` — rejects `fusion_token` and `h_audio`

Run:

```powershell
python -m pytest tests -q
python scripts/run_all_fusion.py --quick
```

Next live wiring: **C-I** then **C-T** on official split-0 IDs once Dehan’s `song_repr` exists.

Related: [ADR](adr/0001-concept-fusion-architecture.md), [fusion runbook](concept-fusion-runbook.md), [instrument README](../instrument_branch/README.md), [timbre README](../timbre_branch/README.md).
