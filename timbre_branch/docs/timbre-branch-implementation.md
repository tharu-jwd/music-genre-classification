# Timbre branch implementation

The implemented branch is a strict learned concept bottleneck. It accepts the shared encoder's one-row-per-track 128-dimensional representation and returns 35 standardized, named timbre predictions. This returned tensor is the timbre input to fusion; the unrestricted 128-dimensional input is never forwarded around it.

## Architecture

```text
h_audio [batch, 128]
  -> Linear(128, 128)
  -> LayerNorm(128)
  -> GELU
  -> Dropout(0.20)
  -> Linear(128, 64)
  -> GELU
  -> Linear(64, 35)
  -> z_timbre = d_hat_standardized [batch, 35]
```

The final layer has no sigmoid or softmax because standardized continuous descriptors can be positive or negative. `d_hat_standardized` is used by fusion and converted to original acoustic units with the saved training-only standardizer for explanations.

## Input artifacts

The training command requires three aligned artifacts:

1. `data/timbre_features_raw.csv`, containing one row and all 35 targets for every `TRACK_ID`.
2. A NumPy `.npz` file containing `track_ids` with shape `[N]` and `embeddings` with shape `[N, 128]`.
3. An official split manifest containing `TRACK_ID` and `split`, where `split` is `train`, `validation`, or `test`.

Rows are aligned by `TRACK_ID`, never by file order. The target standardizer is fitted only on rows assigned to `train` and is saved inside the checkpoint.

## Training

From the repository root:

```bash
python scripts/train_timbre_branch.py \
  --embeddings path/to/shared_encoder_embeddings.npz \
  --splits path/to/official_split_manifest.csv \
  --targets data/timbre_features_raw.csv \
  --output checkpoints/timbre_branch/best.pt
```

Model selection uses validation macro descriptor MAE. The test split is evaluated only after the best validation checkpoint has been restored. The checkpoint stores the architecture configuration, exact feature order, model parameters, training-only scaler, seed, epoch, validation metrics, and optimizer state.

## Fusion contract

```python
d_hat = timbre_branch(h_audio)
z_timbre = d_hat
```

Fusion must consume `z_timbre` only. Concatenating `h_audio` with it would bypass the semantic bottleneck.

For readable predictions:

```python
from timbre_branch import predict_timbre_concepts

result = predict_timbre_concepts(model, h_audio, standardizer)
standardized_for_fusion = result["z_timbre"]
original_units_for_explanation = result["d_hat_original_units"]
```

## Smoke test

```bash
python -m unittest -v tests.test_timbre_branch_smoke
```

The test covers the real 7,324-row descriptor contract, forward and backward passes, masked Smooth L1 loss, one optimization epoch, output shape enforcement, inverse scaling, and checkpoint prediction equivalence.
