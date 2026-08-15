# Google Drive Layout — `MTG_Instrument`

Shared root (all members):

```text
/content/drive/MyDrive/MTG_Instrument/
├── dataset/
│   └── logmel_songs/          # extracted raw_30s mel shards (.npy / tar extract)
├── checkpoints/
│   ├── baseline/              # CNN baseline genre / instrument
│   ├── stage1/                # instrument MIL + attention
│   └── stage2/                # fusion + genre classifier
├── mlruns/                    # MLflow (optional)
├── features/
│   ├── instrument/            # Stage 1 song-level embeddings (e.g. 64-d)
│   ├── rhythm/                # Member 2 — aligned to song / window IDs
│   ├── timbre/                # Member 2
│   └── harmony/               # Member 3
└── patched_baseline_code/     # MTG baseline scripts with Stage 1 patches
```

## Conventions

- **Train I/O:** copy mels to `/content/local_mels/` each session; write checkpoints locally then `cp` back to Drive.
- **ID alignment:** every feature file must be joinable on the same track/song IDs as the Stage 1 manifest (document the key column in each feature README or CSV header).
- **Split:** always MTG-Jamendo official **`split-0`**; never invent a random split for reported metrics.
- **Do not** leave the only copy of patched code or best weights in a Colab VM.

## One-time vs every session

| When | Script |
|---|---|
| Once per team | `scripts/colab/01_one_time_drive_setup.py` |
| Every Colab session | `scripts/colab/02_session_bootstrap.py` |
