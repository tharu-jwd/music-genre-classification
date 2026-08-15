# Project Workflow — Explainable Music Genre Classification Using Instrument Concepts

**Course:** CS3631  
**Target:** Full Team Workflow → Phase 2 (due **23 Aug**)  
**Canonical owners:** Member 5 owns notebook + `.tex`/`.bib` copies (see §0.3)

---

## 0. Before Anything Else — Verify What Actually Exists

Work has been lost before (proposal edits reverted, Stage 1 bug fixes possibly not carried into the current notebook). **Do this first, as a team, before assigning new work.**

### 0.1 Stage 1 notebook — confirm all three fixes

Use the checklist in [`phase2-verification-checklist.md`](phase2-verification-checklist.md):

1. `best_macro_map` is actually **updated** inside the checkpoint-save branch (not compared against a frozen `0.0`).
2. Validation/test split logic does **not** merge test tracks into the validation mask.
3. `resolve_stacked_mel_path` fallback correctly points inside `DRIVE_MEL_ROOT`, not a mismatched folder.

### 0.2 Proposal sources

Confirm the current proposal `.tex`/`.bib` matches the last verified-compiling version:

- 2 pages excluding references
- All 7 citations resolving
- No broken keys

### 0.3 Single owner for canonical copies

| Artifact | Owner | Rule |
|---|---|---|
| Stage 1 / Stage 2 notebooks | **Member 5** | Edits only via this owner or a shared Colab linked from the repo |
| Proposal / Phase 2 paper `.tex`/`.bib` | **Member 5** | Prefer shared Overleaf; no silent local forks |

---

## 1. One-Time Environment Setup

Any one member runs this once; the whole team reuses Drive paths.  
Scripts live under `scripts/colab/`:

| Script | Purpose |
|---|---|
| `01_one_time_drive_setup.py` | Mount Drive, create folders, download mel shards, clone + copy patched baseline |
| `02_session_bootstrap.py` | Every later Colab session: mount, copy mels locally, set paths |

Drive root: `/content/drive/MyDrive/MTG_Instrument`  
Details: [`drive-layout.md`](drive-layout.md)

### 1.1 Folder structure

```
MTG_Instrument/
├── dataset/logmel_songs/
├── checkpoints/          # baseline, stage1, stage2/
├── mlruns/
├── features/             # rhythm, timbre, harmony, instrument embeds
└── patched_baseline_code/
```

### 1.2 Mel shards (one-time)

Download `raw_30s_melspecs-{00,01,02}.tar` into `dataset/logmel_songs`, extract, delete tars. Add more shards later if expanding the subset (§6).

### 1.3 Patched baseline on Drive

Apply the three Stage 1 bug-fix patches, then copy:

`mtg-jamendo-dataset/scripts/baseline` → `MTG_Instrument/patched_baseline_code`

Every session loads code from Drive — never re-clone and re-patch from scratch.

### 1.4 Every session after setup

1. Mount Drive  
2. Copy mels to `/content/local_mels/` (local SSD)  
3. Train against local path  
4. Copy checkpoints back to `MTG_Instrument/checkpoints/`

---

## 2. Architecture Roadmap

```
MP3 audio → 15s windows → log-Mel spectrogram
                │
        Shared CNN Encoder
                │
   ┌────────────┼────────────┬────────────┐
   ▼            ▼            ▼            ▼
Instrument   Rhythm       Timbre       Harmony
(✅ done)    (librosa,    (librosa,    (librosa,
             no training) no training) no training)
   └────────────┴────────────┴────────────┘
                Concatenation / Fusion
                        │            ← Member 1 (NEW)
              Attention-based Pooling
                        │
           Multi-label Genre Classifier  ← Member 1 (NEW)
```

| Component | Status |
|---|---|
| CNN baseline (genre + instrument) | ✅ Done |
| Stage 1 instrument embedding (MIL + attention) | ✅ Done — pending §0 re-verification |
| Rhythm / Timbre / Harmony features | ❌ Not started → scaffolds in `scripts/features/` |
| Fusion layer | ❌ Not started — **top priority** → `scripts/stage2/` |
| Genre classifier on fused representation | ❌ Not started — **top priority** |
| Ablations | ❌ Blocked on fusion + classifier |
| Explainability evaluation | ❌ Blocked on fusion + classifier |

---

## 3. Work Division and Step-by-Step Tasks

### Member 1 — Stage 2: Fusion + Genre Classifier

**Depends on:** nothing to start (placeholders OK); real features from Members 2/3 when ready.

1. Load Stage 1 instrument embedding (64-dim) per song.
2. Fusion A: concatenate concept vectors → single linear layer.
3. Fusion B: concatenate → single-head attention fusion.
4. Genre head: multi-label, 87 tags, sigmoid, BCE.
5. One full train/test run end-to-end (unblocks Members 4 & 5).
6. Save checkpoints to Drive `checkpoints/stage2/`.

Scaffold: `scripts/stage2/fusion_and_genre_classifier.py`

### Member 2 — Rhythm + Timbre Feature Extraction

**Depends on:** nothing — start in parallel.

1. Rhythm (per 15s window): tempo, beat strength, onset density, beat-interval stats.
2. Timbre: spectral centroid, bandwidth, contrast, flatness, RMS, spectral flux.
3. Align track/song IDs to Stage 1 manifest.
4. Document Drive layout (§1 / `drive-layout.md`).

Scaffold: `scripts/features/extract_rhythm_timbre.py`

### Member 3 — Harmony Extraction + Pipeline Integrity

**Depends on:** nothing for extraction; Stage 1 outputs for re-verify.

1. Harmony (per 15s window): chroma, Tonnetz.
2. Re-run §0 checklist; re-train Stage 1 if fixes missing.
3. Confirm no split leakage when Stage 2 joins feature sources.

Scaffold: `scripts/features/extract_harmony.py`

### Member 4 — Ablations, Tuning, Computational Analysis

**Depends on:** Member 1’s first working Stage 2 model.

1. Core ablation: Stage 2 vs CNN baseline (0.7260 ROC-AUC / 0.1592 PR-AUC).
2. Concept-count: instrument-only → +rhythm → full four-concept.
3. Fusion-type: linear vs attention.
4. Stage 2 LR / batch-size sweep (same format as baseline).
5. Params, train time, inference latency vs baseline.

### Member 5 — Explainability Evaluation + Paper Ownership

**Depends on:** Member 1 for explainability; all for paper.

1. Own canonical `.tex`/`.bib` (or shared Overleaf).
2. Qualitative: attention weights / concept % vs what’s audible on held-out tracks.
3. Assemble Phase 2 short paper (4-page limit) as results land.
4. Colour-highlighting + margin comments incrementally.

---

## 4. Suggested Timeline to 23 Aug

| Days | Focus |
|---|---|
| 1–3 | Member 3 re-verifies Stage 1; Members 2 & 3 extract features; Member 1 prototypes fusion with placeholders |
| 4–7 | Member 1 wires real features; Member 4 starts ablations once a checkpoint exists |
| 8–10 | Member 5 assembles paper; full team review before submission |

---

## 5. Definition of Done

- [ ] Metrics on official **`split-0` test** only (never val, never random re-split)
- [ ] No `NaN` into reported macro metrics (exclude undefined per-tag scores; do not zero)
- [ ] Checkpoint = **actual best** validation score, not latest epoch
- [ ] Results + code backed up to Drive (not Colab-only)
- [ ] Every paper citation has a correct `.bib` key

---

## 6. Phase 3 Stretch (not required for 23 Aug)

- More Drive-cached mel shards
- Learned neural embeddings for rhythm/timbre/harmony (vs hand-crafted)
- Full MTG-Jamendo-scale runs + class-wise error analysis
