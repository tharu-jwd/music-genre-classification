# Notebook workflows

The repository provides two generated versions of the same staged experiment:

| Directory | Runtime | Persistent storage |
|---|---|---|
| `colab/` | Google Colab | Google Drive |
| `kaggle/` | Kaggle | Saved notebook outputs attached as inputs |

Choose one runtime for a complete run. Intermediate paths are not interchangeable.

Notebook code is generated from `scripts/generate_colab_notebooks.py` and `scripts/generate_kaggle_notebooks.py`. See `docs/development.md` before editing it.
