# Run the pipeline on Kaggle

## Initial dataset cache

1. Upload `notebooks/kaggle/00_kaggle_data_download.ipynb` to a new Kaggle notebook.
2. Enable Internet and leave the accelerator off.
3. Run all cells. The default downloads shards `00–02`.
4. Save a successful version with output enabled.

## Pass outputs between stages

Kaggle does not preserve `/kaggle/working` across separate notebooks. For every later stage:

1. Create or upload the next numbered notebook.
2. Use **Add Input → Notebook Output** and attach every output it needs.
3. Enable a GPU for notebooks `02`, `03`, and `07`; the other stages can use CPU.
4. Run all cells and save a successful output version.

Minimum dependencies are:

```text
00 → 01
01 → 02 and 03
01 → 04, 05, and 06
01 + 03 + 04 + 05 + 06 → 07
02 + 07 → 08
01 + 03 + 04 + 05 + 06 + 07 → 09
```

The bootstrap searches attached inputs rather than depending on one account-specific path. If needed, set `KAGGLE_KERNEL_SLUG` to the short slug of notebook `00`.

Do not commit downloaded `.npy`, JSON, checkpoint, or result files to Git.
