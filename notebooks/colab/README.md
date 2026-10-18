# Colab shared-data workflow

Colab persists artifacts under `/content/drive/MyDrive/MTG_Instrument`. The old
numbered baseline notebooks `02`–`03` and `05`–`09` are retired. Training and
evaluation now live in the published packages.

The remaining order is:

1. `00_download_to_drive.ipynb`
2. `01_preprocessing.ipynb`
3. `04_rhythm_targets.ipynb` (AcousticBrainz rhythm schema used by `rhythm_branch`)

Then use the package entry points instead of the old numbered notebooks:

| Retired notebook | Replacement |
|---|---|
| `02` direct CNN | `scripts/train_cnn.py` / `modal_paper.py` (CNN configuration) |
| `03` instrument pretraining | `instrument_branch/notebooks/03_instrument_branch.ipynb` |
| `05` timbre targets | `timbre_branch/notebooks/` |
| `06` harmony targets | `harmony_branch/scripts/` (legacy chroma/chord screening) |
| `07`–`09` descriptor fusion | `scripts/train_joint.py` / `modal_paper.py` |

For the current V2 paper experiments, follow the [paper quickstart](../../docs/paper-quickstart.md).
The AcousticBrainz notebook and harmony CPU ladder describe older target-preparation
routes; they do not define the current 10-rhythm/12-harmony feature schema.
The fusion fixture runner uses synthetic data, not the paper training pipeline.
