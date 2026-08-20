# Kaggle staged notebooks

Full click-by-click guide: [`../docs/kaggle-how-to.md`](../docs/kaggle-how-to.md)

**Data:** notebook `00` downloads **mel shards 00–09** (not full MP3s).  
**Code:** GitHub `thevindu-branch`. **Do not** commit `.npy` files.

## Pass output to the next notebook

1. Finish the current notebook → **Save Version** with **Always save output**.
2. Open the next notebook → **Add Input** → **Notebook Output**.
3. Attach **`dnn-download-data-1`** (and later notebooks’ outputs as you go).
4. Path: `/kaggle/input/dnn-download-data-1/` (read-only). New files go to `/kaggle/working`.

| Notebook | Purpose | GPU |
|---|---|---|
| [`00_kaggle_data_download.ipynb`](00_kaggle_data_download.ipynb) | Annotations + **10** mel shards | Off |
| [`01_preprocessing.ipynb`](01_preprocessing.ipynb) | Manifest + split-0 | Off |
| [`02_cnn_baseline.ipynb`](02_cnn_baseline.ipynb) | Genre CNN baseline | On |
| [`03_instrument_embedding.ipynb`](03_instrument_embedding.ipynb) | Stage 1 64-d embeds | On |
| [`04_rhythm_features.ipynb`](04_rhythm_features.ipynb) | Rhythm | Off |
| [`05_timbre_features.ipynb`](05_timbre_features.ipynb) | Timbre | Off |
| [`06_harmony_features.ipynb`](06_harmony_features.ipynb) | Harmony | Off |
| [`07_fusion_genre_classifier.ipynb`](07_fusion_genre_classifier.ipynb) | Stage 2 fusion + genre | On |
| [`08_ablations_and_tuning.ipynb`](08_ablations_and_tuning.ipynb) | Ablations | Optional |
| [`09_explainability_eval.ipynb`](09_explainability_eval.ipynb) | XAI templates | Off |
