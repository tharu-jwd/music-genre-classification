# Data and artifact layout

Large artifacts are excluded from Git. Both hosted runtimes use this logical tree:

```text
MTG_Instrument/
├── annotations/
├── dataset/
│   ├── logmel_songs/
│   ├── acousticbrainz/
│   └── song_manifest.csv
├── features/
│   ├── instrument/                   # pretrained embeddings
│   ├── rhythm/                       # baseline inputs + proposed targets
│   ├── timbre/                       # baseline inputs + proposed targets
│   └── harmony/                      # baseline inputs + proposed targets
├── checkpoints/
│   ├── baselines/
│   │   ├── direct_cnn/
│   │   └── descriptor_fusion/
│   ├── pretraining/
│   │   └── instrument/
│   └── proposed/
└── results/
    ├── baselines/
    └── proposed/
```

| Runtime | Root |
|---|---|
| Colab | `/content/drive/MyDrive/MTG_Instrument` |
| Kaggle writable output | `/kaggle/working/MTG_Instrument` |
| Kaggle previous-stage input | discovered below `/kaggle/input` |

Kaggle inputs are read-only. Colab persists directly to Drive. The default development subset uses matching log-Mel and AcousticBrainz shards `00–02`.
