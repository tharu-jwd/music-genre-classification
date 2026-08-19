# Kaggle Pipeline Plan — MTG-Jamendo Concept-Guided Genre Classification

**Branch:** `thevindu-branch`  
**Runtime:** Kaggle Notebooks (GPU T4/P100 recommended)  
**Deadline focus:** Phase 2 → 23 Aug  

---

## 1. Why Kaggle (vs Colab Drive)

| Concern | Kaggle approach |
|---|---|
| Persistent data | Save outputs as a **Kaggle Dataset** version, or download artifacts |
| Disk | `/kaggle/working` (~20GB writable); attach prior outputs as `/kaggle/input/...` |
| Sessions | Stateless — always re-attach dataset or re-run `00_kaggle_data_download` |
| Splits | Always official MTG-Jamendo **`split-0`** |

Paths used in all notebooks:

```text
ROOT        = /kaggle/working/MTG_Instrument
MEL_DIR     = ROOT/dataset/logmel_songs
ANN_DIR     = ROOT/annotations          # from mtg-jamendo-dataset clone
FEAT_DIR    = ROOT/features
CKPT_DIR    = ROOT/checkpoints
RESULTS_DIR = ROOT/results
```

---

## Before you run — turn Internet ON

In the Kaggle notebook UI: **Settings** (right sidebar) → **Internet** → **On**.

Without Internet you get: `Could not resolve host: github.com`.

### Annotation sources (notebook `00` priority)

1. Attached `/kaggle/input/...` containing MTG `data/` files  
2. `wget` from `raw.githubusercontent.com` (no git clone)  
3. `git clone` last resort  

Also download mel shards from `cdn.freesound.org` (same Internet requirement).

### 2.2 Mel spectrogram shards (large)

CDN (same as official MTG release):

```text
https://cdn.freesound.org/mtg-jamendo/raw_30s/melspecs/raw_30s_melspecs-XX.tar
```

**Phase 2 default:** shards `00`, `01`, `02` (enough for pipeline + paper runs).  
Expand later with `03+` for Phase 3.

### 2.3 Recommended Kaggle workflow

**Kaggle sessions are stateless.** A new notebook does **not** see files from notebook `00`.

1. Run **`00_kaggle_data_download.ipynb`** once (Internet **On**) → `/kaggle/working/MTG_Instrument`.
2. **Save Version** → “Save output” → publish as private dataset `mtg-instrument-cache`.
3. Later notebooks: **Add Data** → attach that dataset. The shared bootstrap copies it into working and **re-downloads split TSVs** if they are still missing.
4. Alternative: paste later stages into the **same** Kaggle session after `00` (same `/kaggle/working`).

---

## 3. Full notebook stage map

| # | Notebook | Stage | Output |
|---|---|---|---|
| 00 | `00_kaggle_data_download.ipynb` | Data | mels + annotations under `ROOT` |
| 01 | `01_preprocessing.ipynb` | Manifest / splits | `song_manifest.csv`, split ID lists |
| 02 | `02_cnn_baseline.ipynb` | Baseline | genre metrics + `checkpoints/baseline/` |
| 03 | `03_instrument_embedding.ipynb` | Stage 1 | 64-d embeds + `checkpoints/stage1/` |
| 04 | `04_rhythm_features.ipynb` | Concepts | `features/rhythm/` |
| 05 | `05_timbre_features.ipynb` | Concepts | `features/timbre/` |
| 06 | `06_harmony_features.ipynb` | Concepts | `features/harmony/` |
| 07 | `07_fusion_genre_classifier.ipynb` | Stage 2 | fused model + test metrics |
| 08 | `08_ablations_and_tuning.ipynb` | Eval | ablation tables vs 0.7260 / 0.1592 |
| 09 | `09_explainability_eval.ipynb` | XAI | attention / concept contribution reports |

Run order is **strict** for 00→01→02→03 and for 04/05/06 (parallel OK) → 07 → 08 → 09.

---

## 4. Definition of done (every training notebook)

- Metrics on **split-0 test only**
- No NaN folded into macro averages (exclude undefined tags)
- Checkpoint = **best validation** score (update `best_*` inside save branch)
- Artifacts written under `ROOT` and optionally versioned as a Kaggle Dataset

---

## 5. Step-by-step execution checklist

1. [ ] Create Kaggle notebook, enable GPU, Internet **on**
2. [ ] Run `00` → verify mel files + TSV splits exist
3. [ ] Save output as dataset `mtg-instrument-cache`
4. [ ] Run `01` → inspect `song_manifest.csv` counts per split
5. [ ] Run `02` → record baseline ROC-AUC / PR-AUC
6. [ ] Run `03` after §0 bug-fix checks → export instrument embeddings
7. [ ] Run `04`/`05`/`06` (CPU OK) → feature CSVs aligned on `song_id`
8. [ ] Run `07` → Stage 2 end-to-end on split-0 test
9. [ ] Run `08` → ablations for paper tables
10. [ ] Run `09` → qualitative figures for explainability section

---

## 6. Repo location

All notebooks: `notebooks/`  
Plan doc: `docs/kaggle-pipeline-plan.md`
