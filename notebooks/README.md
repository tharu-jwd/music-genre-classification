# Kaggle staged notebooks

**Important:** each Kaggle notebook starts with an **empty** `/kaggle/working`. Files from notebook `00` are **not** visible in `01` unless you:

1. Keep running in the **same** session, or  
2. **Save Version → Save output** from `00`, then **Add Data** that dataset in later notebooks.

Every notebook now **auto-finds** files under `/kaggle/working` and `/kaggle/input`, and **re-downloads split TSVs** if they are missing (Internet must be **On**).

| Notebook | Purpose |
|---|---|
| [`00_kaggle_data_download.ipynb`](00_kaggle_data_download.ipynb) | Annotations + mel shards 00–02 |
| [`01_preprocessing.ipynb`](01_preprocessing.ipynb) | `song_manifest.csv` + split-0 join |
| [`02_cnn_baseline.ipynb`](02_cnn_baseline.ipynb) | Multi-label genre CNN |
| [`03_instrument_embedding.ipynb`](03_instrument_embedding.ipynb) | Stage 1 MIL → 64-d embeds |
| [`04_rhythm_features.ipynb`](04_rhythm_features.ipynb) | Rhythm features |
| [`05_timbre_features.ipynb`](05_timbre_features.ipynb) | Timbre features |
| [`06_harmony_features.ipynb`](06_harmony_features.ipynb) | Harmony features |
| [`07_fusion_genre_classifier.ipynb`](07_fusion_genre_classifier.ipynb) | Stage 2 fusion + genre |
| [`08_ablations_and_tuning.ipynb`](08_ablations_and_tuning.ipynb) | Ablations / compute |
| [`09_explainability_eval.ipynb`](09_explainability_eval.ipynb) | XAI templates |

Full plan: [`../docs/kaggle-pipeline-plan.md`](../docs/kaggle-pipeline-plan.md).
