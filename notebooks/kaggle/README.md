# Kaggle workflow

Use these notebooks when datasets and checkpoints will be passed through saved Kaggle notebook outputs.

Start with `00_kaggle_data_download.ipynb`, which downloads shards `00–02` by default. Save its output, attach it to `01`, and continue through the numbered stages. Enable a GPU for `02`, `03`, and `07`.

See `docs/kaggle-how-to.md` for the dependency graph and exact handoff process. Do not assume files in `/kaggle/working` survive into a new notebook.
