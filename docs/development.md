# Development guide

## Source of truth

The Python files in `scripts/` generate the committed notebooks. Pipeline changes must be made there first:

```bash
python3 scripts/generate_colab_notebooks.py
python3 scripts/generate_kaggle_notebooks.py
```

Commit the generator and its regenerated notebooks together. Direct notebook edits are acceptable for exploration, but must be transferred back to the generator before merging.

## Validation before merging

1. Run Python syntax compilation on both generators.
2. Regenerate both notebook sets.
3. Regenerate again and confirm `git diff` is unchanged; generation must be deterministic.
4. Parse every notebook as JSON.
5. Confirm generated code cells have no committed outputs or execution counts.
6. Search documentation and notebook instructions for obsolete branch names, deadlines, user-specific paths, and inconsistent shard counts.
7. For model changes, run the affected notebook on a small subset before a full experiment.

## Branch policy

`main` is the integration baseline. Develop changes on focused branches, keep generated data out of Git, and merge only when documentation reflects the actual implementation.

## Dependencies

The generator scripts require only Python's standard library. Notebook runtime packages are listed in `requirements.txt`; hosted runtimes provide their own appropriate PyTorch build.
