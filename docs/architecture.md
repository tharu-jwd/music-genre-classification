# Architecture and data contracts

## Problem

The target is multi-label genre classification: one song can have several genre tags. The repository compares a direct CNN baseline with a concept-guided classifier.

## Direct baseline

Notebook `02` loads each song's stacked log-Mel array, converts it to a fixed-size representation, and applies a convolutional network. The network emits one logit per genre tag and is trained with `BCEWithLogitsLoss`.

## Concept-guided path

### Instrument representation

Notebook `03` treats the windows of a song as a bag:

1. A shared CNN encodes each window.
2. A projection produces a 64-dimensional instrument vector per window.
3. Masked attention pools valid windows into one 64-dimensional song vector.
4. An instrument classifier supervises that vector with multi-label instrument tags.
5. The pooled vectors are exported for every song.

### Other concepts

- Rhythm comes from AcousticBrainz/Essentia JSON and includes BPM, beat count, onset rate, danceability, histogram peaks, and beat-interval statistics when present.
- Timbre includes spectral centroid, bandwidth, contrast, flatness, RMS energy, and spectral flux.
- Harmony includes 12 chroma means and 6 Tonnetz means.

Timbre and harmony use raw audio when it is attached. Otherwise, the hosted notebooks compute explicit log-Mel approximations and record their source in the output CSV. These approximations should not be described as equivalent to waveform-derived librosa features.

### Fusion

Notebook `07` intersects songs available from all four concept sources and supports:

- linear fusion: concatenate every feature vector and project it to 128 dimensions;
- attention fusion: project the four concepts into 64-dimensional tokens, apply single-head self-attention, mean-pool the tokens, and project to 128 dimensions.

A linear head emits genre logits. The model is selected using validation macro PR-AUC, then evaluated once on the official test partition.

## File contracts

| Artifact | Required fields or shape |
|---|---|
| `song_manifest.csv` | `song_id`, `mel_abs`, `split` |
| stacked log-Mel | NumPy array accepted as a 2-D spectrogram or 3-D window stack |
| `instrument_embeddings.npy` | `(number_of_songs, 64)` |
| `song_ids.json` | IDs in the same order as instrument embedding rows |
| concept CSV | `song_id`, numeric feature columns, optional `source` and `split` |
| Stage 2 checkpoint | model state, fusion type, best validation score, genre tag order |

Every join uses a normalized seven-digit `song_id`. Feature dimension discovery is dynamic in Stage 2; the instrument dimension is fixed at 64.

## Evaluation invariants

- Use MTG-Jamendo's official `split-0` train, validation, and test IDs.
- Never combine validation and test IDs.
- Save the checkpoint only when the validation selection metric improves.
- Exclude tags without both positive and negative test examples from macro ROC-AUC/PR-AUC.
- Keep genre tag ordering with the checkpoint.
