# Timbre branch

This directory contains the complete timbre-branch workstream: audio acquisition, 35-descriptor extraction, the strict 128-to-35 learned concept bottleneck, training utilities, validation tests, and the research proposal.

All mergeable timbre-branch files live here:

```text
timbre_branch/
    README.md
    requirements.txt
    notebooks/
        download_selected_full_quality_audio.ipynb
        extract_timbre_features_from_drive.ipynb
        extract_timbre_features_first_4min.ipynb
    scripts/
        train_timbre_branch.py
    src/timbre_branch/
        constants.py
        data.py
        inference.py
        losses.py
        model.py
        preprocessing.py
        training.py
    tests/
        test_timbre_branch_smoke.py
    docs/
        timbre_branch_proposal.docx
        timbre-branch-implementation.md
    data/
        README.md
```

The two local CSV artifacts are kept under `data/` but intentionally ignored by Git:

- `split_csv.csv` contains the 7,324 selected MTG-Jamendo tracks.
- `timbre_features_raw.csv` contains one verified row and 35 extracted targets per track.

## Architecture

```text
h_audio (B, 128)
  -> Linear(128, 128)
  -> LayerNorm(128)
  -> GELU
  -> Dropout(0.20)
  -> Linear(128, 64)
  -> GELU
  -> Linear(64, 35)
  -> z_timbre = d_hat_standardized (B, 35)
```

The fusion layer receives only the 35 named predicted concepts. There is no direct path from the unrestricted 128-dimensional encoder representation to the timbre fusion input.

## Current data status

The first-four-minutes extractor completed all 7,324 tracks. The resulting target table has 35 finite descriptor columns, no duplicate IDs, no failed rows, and no missing values. Targets remain in original acoustic units until a training-only standardizer is fitted.

## Development smoke test

From this directory:

```powershell
python -m pip install -r requirements.txt
python -m unittest -v tests.test_timbre_branch_smoke
```

The self-contained tests cover 128-to-35 forward shape, masked Smooth L1 loss,
backward optimization, inverse scaling, checkpoint equivalence, invalid-input
rejection, and end-to-end command-line training on disposable synthetic inputs. If
the ignored real target table is present, they also check its 7,324-row contract.

## Real training

The shared-encoder owner must provide a NumPy archive containing:

```text
track_ids   [N]       Unicode strings matching TRACK_ID
embeddings  [N, 128]  float32 shared-encoder representations
```

An official split manifest must contain unique `TRACK_ID` and `split` columns, with split values `train`, `validation`, and `test`.

Run from `timbre_branch/`:

```powershell
python scripts/train_timbre_branch.py `
  --embeddings path/to/shared_encoder_embeddings.npz `
  --splits path/to/official_split_manifest.csv `
  --targets data/timbre_features_raw.csv `
  --output checkpoints/timbre_branch/best.pt
```

Real trained weights and research metrics are not included because the genuine shared-encoder representations are not present. Synthetic smoke-test checkpoints are discarded and must not be reported as experimental results.
