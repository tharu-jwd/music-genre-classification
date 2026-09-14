# Implemented baseline architectures

The repository retains two baselines. They are comparison systems, not the final proposed contribution.

## Baseline A: direct CNN

Notebook `02` converts each song's stacked log-Mel array into a fixed-size spectrogram, applies a compact CNN, and predicts genre logits directly. It is trained with multi-label binary cross-entropy.

```text
log-Mel song → CNN → genre logits
```

## Instrument pretraining

Notebook `03` encodes each valid song window with a shared CNN, uses masked attention to pool the windows, and learns a 64-dimensional song representation from instrument tags.

This is reusable pretraining for the proposed shared encoder, but the existing checkpoint is not itself the complete proposed model.

## Baseline B: descriptor fusion

Notebook `07` combines:

- the learned 64-dimensional instrument representation;
- AcousticBrainz rhythm descriptors;
- six timbre descriptors;
- twelve chroma and six Tonnetz harmony descriptors.

It supports concatenation with a linear projection or single-head attention over projected concept tokens, followed by a genre head.

```text
instrument embedding + three descriptor groups
                    ↓
          linear/attention fusion
                    ↓
               genre logits
```

This baseline is useful because it tests whether explicit musical information helps genre prediction before the full concept branches are learned end to end.

Rhythm comes from AcousticBrainz/Essentia metadata. Timbre and harmony use waveform-based descriptors when audio is attached; otherwise, the hosted notebooks record that they used log-Mel approximations. Those approximations must not be described as equivalent to waveform-derived features.

Notebook `07` discovers descriptor dimensions from the input tables, intersects songs present in all four sources, selects its checkpoint using validation macro PR-AUC, and evaluates the selected checkpoint on the official test split.

## Baseline data contracts

| Artifact | Contract |
|---|---|
| `song_manifest.csv` | `song_id`, `mel_abs`, `split` |
| stacked log-Mel | 2-D spectrogram or 3-D window stack |
| `instrument_embeddings.npy` | one 64-D row per song |
| `song_ids.json` | IDs in embedding-row order |
| concept CSV | `song_id`, numeric descriptors, optional `source` and `split` |
| checkpoint | model state, validation score, genre tag order |

Every cross-file join uses a normalized seven-digit `song_id`. The instrument embedding width is fixed at 64; other descriptor widths are discovered from their CSV columns.
