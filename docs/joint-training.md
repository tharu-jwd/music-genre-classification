# Joint training with CSV concept targets

`scripts/train_joint.py` trains the shared encoder, all four concept branches,
and fusion for the current six genres and 41 instrument labels.

Harmony is predicted from the shared audio sequence. The loss compares pooled
predicted chroma against the 12 `chroma_*_mean` columns of `harmony_df.csv`,
normalized to sum to one. These are song-level targets, not frame-level labels.
The other harmony descriptors and chord labels are not supervised by this run.
Missing or invalid chroma rows are masked. Targets never enter fusion, including
at validation and test time. Checkpoints include the learned harmony weights.

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
