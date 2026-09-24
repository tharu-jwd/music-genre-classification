# Notebook workflows

The numbered Colab/Kaggle series is now only the shared data path. Concept
training and fusion live in packages, not in notebooks 02–09.

| Directory | Runtime | What remains |
|---|---|---|
| `colab/` | Google Colab | `00` download, `01` preprocessing, `04` rhythm targets |
| `kaggle/` | Kaggle | `00` download, `01` preprocessing, `04` rhythm targets |
| `dataset_split/` | Local | Exploratory split EDA |

Use only one hosted runtime for a complete data pass. Notebook code is generated
by the corresponding script under `scripts/`. Those generators write only the
three essential notebooks.

Branch-owned notebooks stay with their packages:

- `instrument_branch/notebooks/03_instrument_branch.ipynb`
- `timbre_branch/notebooks/`
