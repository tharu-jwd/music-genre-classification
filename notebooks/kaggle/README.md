# Kaggle shared-data workflow

Kaggle does not preserve `/kaggle/working` between notebooks. Save every successful
stage with output enabled and attach it to dependent notebooks using **Add Input →
Notebook Output**.

The old numbered baseline notebooks `02`–`03` and `05`–`09` are retired. Training
and evaluation now live in the published packages.

## Intended dependencies

```text
00 → 01
01 → 04
01 → instrument_branch / timbre_branch / harmony scripts / concept_fusion
```

1. Run `00_kaggle_data_download.ipynb` with Internet enabled and save its output.
2. Attach that output to `01_preprocessing.ipynb` and save the manifest output.
3. Run `04_rhythm_targets.ipynb` for the AcousticBrainz rhythm schema used by
   `rhythm_branch`.
4. Train and evaluate with the published packages, not the retired `02`–`09`
   notebooks.

| Retired notebook | Replacement |
|---|---|
| `02` direct CNN | `concept_fusion` B1 / `scripts/run_all_fusion.py` |
| `03` instrument pretraining | `instrument_branch/notebooks/03_instrument_branch.ipynb` |
| `05` timbre targets | `timbre_branch/notebooks/` |
| `06` harmony targets | `scripts/` harmony CPU ladder |
| `07`–`09` descriptor fusion | `concept_fusion` + `scripts/run_all_fusion.py` |
