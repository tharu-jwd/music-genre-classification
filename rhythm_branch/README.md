# Learned rhythm branch

This branch predicts rhythm from the shared CNN's ordered mel-derived features. The
ten AcousticBrainz descriptors are supervision targets only; they are never passed
to the model as inputs.

## Tensor contract

```text
encoded_sequence       (B,T,128)
sequence_mask          (B,T)
sequence_window_index  (B,T)
        -> gap-aware temporal Conv1d blocks
        -> masked temporal attention pooling
embedding              (B,64)   -> concept fusion
predictions            (B,10)   -> masked Huber auxiliary loss
availability           (B,)
```

The exact target order is frozen in `src/rhythm_branch/constants.py` and matches the
Colab/Kaggle AcousticBrainz extraction notebooks. Target rows join to mel examples
by normalized seven-digit `song_id` and official `split`. Normalization is fitted on
observed training cells only and its statistics, feature order, schema version,
input scope, and explicit exclusions are stored in checkpoints.

The shared encoder selects at most twelve windows over a full song. Temporal
convolutions run independently inside each source window so gaps are not interpreted
as adjacent audio. AcousticBrainz `beats_count` counts beats over the analyzed
recording; `audit_target_coverage(..., input_scope="sampled_windows")` therefore
masks it and records the exclusion. It may be supervised only when the model input
is verified to cover the same full recording.

Run the CPU-only contract smoke test with:

```bash
python scripts/smoke_rhythm_branch.py
pytest -q tests/test_rhythm_branch.py
```

Audit real targets before training with:

```bash
python scripts/audit_rhythm_targets.py \
  --manifest dataset/song_manifest.csv \
  --targets features/rhythm/rhythm_song.csv \
  --input-scope sampled_windows \
  --output results/rhythm_target_audit.json
```

The smoke path uses synthetic tensors solely to verify shapes, masks, gradients,
and integration. It does not produce a research result or trained release checkpoint.
