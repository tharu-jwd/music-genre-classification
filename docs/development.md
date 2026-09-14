# Development guide

## Notebook source of truth

The files in `scripts/` generate the committed Colab and Kaggle notebooks:

```bash
python3 scripts/generate_colab_notebooks.py
python3 scripts/generate_kaggle_notebooks.py
```

Change generators first and commit regenerated notebooks with them. Do not create notebook-only forks.

## Architecture boundaries

- Existing notebooks are baseline and target-preparation workflows.
- Do not add proposed-model claims until matching executable code exists.
- Shared neural components should move into importable Python modules as the proposed model is implemented.
- Keep runtime-specific storage/bootstrap logic in notebooks, not inside model classes.

## Validation before merging

1. Compile Python modules and generators.
2. Regenerate notebooks twice and confirm deterministic output.
3. Parse every notebook as JSON and Python syntax.
4. Confirm execution outputs are stripped.
5. Run a small data/model smoke test.
6. Check documentation for outdated paths, notebook names, and implementation claims.

`main` is the integration baseline. Data, targets, checkpoints, and results remain outside Git.

The generators use only the Python standard library. Runtime packages used inside notebooks are listed in `requirements.txt`; Colab and Kaggle provide the platform-appropriate PyTorch build.
