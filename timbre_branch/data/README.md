# Local timbre data

Place the following local artifacts in this directory:

- `split_csv.csv`
- `timbre_features_raw.csv`
- the official `TRACK_ID,split` manifest when available
- shared-encoder `.npz` exports when they are not stored elsewhere

The first two files are legacy local artifacts and are ignored by Git. Current
training uses the 12 timbre columns in `../../data/warehouse_dataset_new.csv`,
whose order matches both joint vector CSVs. The old target CSV lacks the current
skewness/flux fields. Large audio, embeddings, checkpoints, and other generated
artifacts must not be committed.
