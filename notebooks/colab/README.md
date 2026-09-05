# Colab + Google Drive notebooks

Shared folder: `/content/drive/MyDrive/MTG_Instrument`

Every notebook: **Mount Drive** first. GPU On only for 02, 03, 07.

| File | What it does | Needs on Drive | Writes |
|---|---|---|---|
| `00_download_to_drive.ipynb` | Labels + mel shards 00–09 | ~25 GB free Drive | `dataset/logmel_songs/`, `annotations/` |
| `01_preprocessing.ipynb` | Split-0 manifest | 00 | `dataset/song_manifest.csv` |
| `02_cnn_baseline.ipynb` | Genre CNN | 00+01 | `checkpoints/baseline/` |
| `03_instrument_embedding.ipynb` | 64-d instrument embeds | 00+01 | `features/instrument/` |
| `04_rhythm_features.ipynb` | Rhythm | 00+01 | `features/rhythm/` |
| `05_timbre_features.ipynb` | Timbre | 00+01 | `features/timbre/` |
| `06_harmony_features.ipynb` | Harmony | 00+01 | `features/harmony/` |
| `07_fusion_genre_classifier.ipynb` | Fusion + genre | 01+03+04+05+06 | `checkpoints/stage2/` |
| `08_ablations_and_tuning.ipynb` | Tables | 02+07 results | `results/08_*.csv` |
| `09_explainability_eval.ipynb` | XAI templates | 01+07 | `results/09_*` |

**Next for you:** finish 00 on Drive (if needed), then run **`01_preprocessing.ipynb`** from this folder.
