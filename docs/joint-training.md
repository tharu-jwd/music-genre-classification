# Joint training with CSV concept targets

`scripts/train_joint.py` trains the shared encoder, all four concept branches,
and fusion. Frozen choices from the published branch contracts:

| Decision | Frozen value | Why |
|---|---|---|
| Genres | 6 scoped tags in `genres_df.csv` | Only labeled table on this cohort. Official split-0 is still 87; do not report scoped scores as official. |
| Instruments | Official **40** split-0 tags | Full TSV has 41 (`ukulele` extra). Official split drops it. Fusion is `Linear(40,64)`. |
| Mel geometry | Encoder v2: 96 mels, 12 kHz, hop 256, 1366 frames | `logmel_config.json` may override; a 128-band cache is a different experiment. |
| Harmony | 12 predicted chroma bins; CSV song-mean aux | Targets never enter fusion. Other 45-D descriptors are not fusion inputs. |
| Rhythm | 10 AcousticBrainz fields; `beats_count` masked | Encoder windows are sampled, not the full AB recording. |

Harmony is predicted from the shared audio sequence. The loss compares pooled
predicted chroma against the 12 `chroma_*_mean` columns of `harmony_df.csv`,
normalized to sum to one. These are song-level targets, not frame-level labels.
Missing or invalid chroma rows are masked. Checkpoints include the learned
harmony weights, validation F1 thresholds, and one-batch gate-vs-occlusion.

The instrument head is `instrument_branch.src.instrument_branch.InstrumentBranch`,
not a trainer-local copy. Fusion remains the only mixer.

## Local audio setup

`data/logmel_root.txt` contains the actual `logmel_songs` directory resolved from
the Google Drive shortcut. `--logmel-root PATH` overrides it. The original Colab
CSV paths are relocated at load time; the CSV itself is unchanged.

The current cache may store `(windows, 128, 469)` arrays. That is **not** the
encoder v2 contract. Stored windows stay separate; up to `--max-windows` windows
are selected evenly. `data/logmel_audit.csv` supplies track durations for
final-window masking.

Before using a stacked cache, create `data/logmel_config.json` with the confirmed
extraction settings: `sample_rate`, `hop_length`, `window_seconds`, `n_mels`, and
`center`. The stored-window loader currently supports centered STFT only.

For a 2D full-song cache, the loader defaults to 96-band, 12 kHz / 256-hop
windows unless overridden.

If `data/full_dataset.csv` is present, the trainer uses that combined table
and official 40 instrument columns. Hosted runs are documented in
[modal-training.md](modal-training.md).

After confirming the settings, start with a small batch:

```powershell
python scripts/train_joint.py --quick --batch-size 1
```

The quick run uses 32 tracks per split and three epochs. It is a smoke test, not
a final evaluation. Metrics are sklearn macro AP plus val-fitted F1 thresholds.
A single-window GPU gradient check is not a memory guarantee for a full batch.
