# Data and artifact layout

Large artifacts are excluded from Git. Each runtime uses the same logical tree under a different root.

```text
MTG_Instrument/
├── annotations/
├── dataset/
│   ├── logmel_songs/
│   ├── acousticbrainz/
│   └── song_manifest.csv
├── features/
│   ├── instrument/
│   ├── rhythm/
│   ├── timbre/
│   └── harmony/
├── checkpoints/
│   ├── baseline/
│   ├── stage1/
│   └── stage2/
└── results/
```

| Runtime | Root |
|---|---|
| Colab | `/content/drive/MyDrive/MTG_Instrument` |
| Kaggle writable output | `/kaggle/working/MTG_Instrument` |
| Kaggle prior-stage input | discovered below `/kaggle/input` |

Kaggle inputs are read-only. Save each stage as a notebook output and attach that output to dependent stages. Colab writes persistent artifacts directly to Drive and may cache individual mel arrays on the VM for faster reads.

The baseline subset uses mel and AcousticBrainz shards `00`, `01`, and `02`. If the subset is expanded, both sources must use matching shard indices.
