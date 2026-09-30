# Joint training with CSV concept targets

`scripts/train_joint.py` trains the shared encoder, all four concept branches,
and fusion for the current six genres and 41 instrument labels.

Harmony (v4, `ChromaGroundedHarmonyBranch`) is predicted from the shared audio
sequence and covers all **45** descriptors of `data/harmony_df.csv`, joined by
`TRACK_ID` through `--harmony-csv` (default `DATA_DIR/harmony_df.csv`). The 12
chroma means and 6 Tonnetz means are derived exactly from per-token chroma
predictions; the other 27 come from a statistics-pooled regression head. Targets
get named log/log1m transforms and a training-split-only z-score, and are
optimized with masked Smooth L1 loss. Predicted descriptors, not targets, enter
fusion through `Linear(45,64)`, so validation and test inference stay leakage-free.
See the [v4 snapshot](../harmony_branch/docs/architecture-versions/v4/README.md).

## Local audio setup

`data/logmel_root.txt` contains the actual `logmel_songs` directory resolved from
the Google Drive shortcut. `--logmel-root PATH` overrides it. The original Colab
CSV paths are relocated at load time; the CSV itself is unchanged.

The current cache stores `(windows, 128, 469)` arrays. Stored windows stay separate;
up to `--max-windows` windows are selected evenly across the stored sequence.
`data/logmel_audit.csv` supplies track durations for final-window masking.

Before using this cache for training, create `data/logmel_config.json` with the
confirmed extraction settings: `sample_rate` and `hop_length` as positive integers,
`window_seconds` (the audit indicates 15), `n_mels` (128), and `center` (whether
the STFT was centered). The stored-window loader currently supports centered STFT
only. Do not infer sample rate and hop length from array dimensions alone.

For a 2D full-song cache, the loader retains the existing 96-band, 12 kHz/256-hop
defaults unless overridden by that configuration, and segments into bounded
windows. Input layout and extraction settings must remain consistent across a run.

After confirming the settings, start with a small batch:

```powershell
python scripts/train_joint.py --quick --batch-size 1
```

The quick run uses 32 tracks per split and three epochs. It is a smoke test, not
a final evaluation. A single-window GPU gradient check is not a memory guarantee
for a full batch. Increase batch size only after checking actual peak memory.

## Validation metrics

Validation runs evaluate the fusion output and every supervised branch after each
epoch. The console prints the main branch aggregates, while `results.json` stores
the complete metrics under each epoch's `val_branch_metrics` field:

- genre and instrument: macro/micro average precision, macro/micro F1 at a 0.5
  threshold, binary accuracy, and per-tag AP/F1/support;
- rhythm, timbre, and harmony: masked MAE and RMSE in standardized target units, macro R2,
  and per-feature MAE/RMSE/R2/counts;

Final test metrics use the same schema in `test_branch_metrics`.
