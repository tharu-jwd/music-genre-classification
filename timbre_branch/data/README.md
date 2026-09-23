# Local timbre data

Place the following local artifacts in this directory:

- `split_csv.csv`
- `timbre_features_raw.csv`
- the official `TRACK_ID,split` manifest when available
- shared-encoder `.npz` exports when they are not stored elsewhere

The two first files currently exist in this working copy and are ignored by Git. The target CSV contains 7,324 unique tracks and 35 finite raw timbre descriptors. Large audio, embeddings, checkpoints, and other generated artifacts must not be committed.
