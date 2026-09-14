# Colab baseline workflow

Artifacts persist under `/content/drive/MyDrive/MTG_Instrument`.

Run in order:

1. `00_download_to_drive.ipynb`
2. `01_preprocessing.ipynb`
3. `02_direct_cnn_baseline.ipynb`
4. `03_instrument_pretraining.ipynb`
5. `04_rhythm_targets.ipynb`, `05_timbre_targets.ipynb`, `06_harmony_targets.ipynb`
6. `07_descriptor_fusion_baseline.ipynb`
7. `08_baseline_evaluation.ipynb`, `09_baseline_explainability.ipynb`

Use a GPU for `02`, `03`, and `07`. These notebooks establish baselines and prepare supervision targets; they are not the proposed-model implementation.
