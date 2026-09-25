"""Generate the final deterministic multi-label track-splitting notebook."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "dataset_split" / "02_create_final_track_splits.ipynb"


def src(text):
    return [line + "\n" for line in text.strip("\n").splitlines()]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": src(text)}


def code(text):
    return {
        "cell_type": "code", "execution_count": None, "metadata": {},
        "outputs": [], "source": src(text),
    }


cells = [
    md("""
# Create final track-level train/validation/test assignments

This notebook creates the final `track_id,split` CSV for the 7,324-track experiment.

- Exact sizes: 5,127 train, 1,099 validation, 1,098 test
- Multi-label stratification over six target genres
- Seed: 42
- No artist or album grouping for the current experiments
- Every complete track, including all its log-Mel windows, belongs to one split
"""),
    code("""
%pip install -q iterative-stratification
"""),
    code("""
from pathlib import Path
import ast
import json
import re

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

SEED = 42
EXPECTED_TRACKS = 7_324
TARGET_GENRES = ["classical", "electronic", "folk", "hiphop", "jazz", "rock"]
EXPECTED_SIZES = {"train": 5_127, "validation": 1_099, "test": 1_098}
"""),
    code("""
try:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
except ImportError:
    print("Not running in Colab; using a local data directory if available.")

DRIVE_DATA_ROOT = Path("/content/drive/MyDrive/music-genre-classification/data")
if DRIVE_DATA_ROOT.is_dir():
    DATA_ROOT = DRIVE_DATA_ROOT
else:
    DATA_ROOT = next(
        (p for p in (Path.cwd() / "data", Path.cwd().parent / "data")
         if (p / "split_csv.csv").is_file()),
        None,
    )

assert DATA_ROOT is not None, "Could not locate the project data directory."
METADATA_CSV = DATA_ROOT / "split_csv.csv"
GENRES_CSV = DATA_ROOT / "genres_df.csv"
OUTPUT_CSV = DATA_ROOT / "track_split_assignments.csv"

assert METADATA_CSV.is_file()
assert GENRES_CSV.is_file()
print("Data root:", DATA_ROOT)
print("Output:", OUTPUT_CSV)
"""),
    code("""
def canonical_track_id(value):
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if re.fullmatch(r"\\d+\\.0", text):
        text = text[:-2]
    text = re.sub(r"^track[_\\-\\s]*", "", text, flags=re.IGNORECASE)
    return str(int(text)) if re.fullmatch(r"\\d+", text) else None


metadata = pd.read_csv(METADATA_CSV, dtype=str, keep_default_na=False)
genres = pd.read_csv(GENRES_CSV, dtype=str, keep_default_na=False)

assert "TRACK_ID" in metadata and "TRACK_ID" in genres
metadata["_key"] = metadata["TRACK_ID"].map(canonical_track_id)
genres["_key"] = genres["TRACK_ID"].map(canonical_track_id)

assert len(metadata) == EXPECTED_TRACKS
assert metadata["_key"].notna().all() and metadata["_key"].is_unique
assert genres["_key"].notna().all() and genres["_key"].is_unique
assert set(metadata["_key"]) == set(genres["_key"])
"""),
    code("""
# Preserve metadata order and exact original TRACK_ID formatting.
working = metadata[["TRACK_ID", "_key"]].merge(
    genres[["_key"] + TARGET_GENRES], on="_key", how="left", validate="one_to_one"
)

Y = working[TARGET_GENRES].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.int8)
assert np.isin(Y, [0, 1]).all()
assert (Y.sum(axis=1) >= 1).all()

print("Tracks:", len(working))
display(pd.Series(Y.sum(axis=0), index=TARGET_GENRES, name="positives").to_frame())
"""),
    md("## Stage 1: 5,127 training tracks and a 2,197-track holdout"),
    code("""
stage1 = MultilabelStratifiedShuffleSplit(
    n_splits=1,
    test_size=EXPECTED_SIZES["validation"] + EXPECTED_SIZES["test"],
    random_state=SEED,
)
train_idx, holdout_idx = next(stage1.split(np.zeros(len(working)), Y))

assert len(train_idx) == EXPECTED_SIZES["train"]
assert len(holdout_idx) == EXPECTED_SIZES["validation"] + EXPECTED_SIZES["test"]
"""),
    md("## Stage 2: divide the holdout into validation and test"),
    code("""
stage2 = MultilabelStratifiedShuffleSplit(
    n_splits=1,
    test_size=EXPECTED_SIZES["test"],
    random_state=SEED + 1,
)
validation_local, test_local = next(
    stage2.split(np.zeros(len(holdout_idx)), Y[holdout_idx])
)
validation_idx = holdout_idx[validation_local]
test_idx = holdout_idx[test_local]

assert len(validation_idx) == EXPECTED_SIZES["validation"]
assert len(test_idx) == EXPECTED_SIZES["test"]
"""),
    code("""
assignment = np.empty(len(working), dtype=object)
assignment[train_idx] = "train"
assignment[validation_idx] = "validation"
assignment[test_idx] = "test"

result = pd.DataFrame({
    "track_id": working["TRACK_ID"],
    "split": assignment,
})

display(result.head())
display(result["split"].value_counts().reindex(["train", "validation", "test"]).to_frame("tracks"))
"""),
    md("## Validate separation and genre balance"),
    code("""
assert len(result) == EXPECTED_TRACKS
assert result["track_id"].is_unique
assert result["split"].notna().all()
assert result["split"].value_counts().to_dict() == EXPECTED_SIZES

split_sets = {
    name: set(result.loc[result["split"] == name, "track_id"])
    for name in EXPECTED_SIZES
}
assert split_sets["train"].isdisjoint(split_sets["validation"])
assert split_sets["train"].isdisjoint(split_sets["test"])
assert split_sets["validation"].isdisjoint(split_sets["test"])
assert set().union(*split_sets.values()) == set(metadata["TRACK_ID"])

audit = working[["TRACK_ID"] + TARGET_GENRES].copy()
audit["split"] = assignment
counts = audit.groupby("split")[TARGET_GENRES].sum().reindex(["train", "validation", "test"])
prevalence = counts.div(audit["split"].value_counts(), axis=0)
overall = audit[TARGET_GENRES].mean()
deviation = prevalence.subtract(overall, axis=1)

print("Positive-label counts:")
display(counts.astype(int))
print("Genre prevalence:")
display(prevalence.round(5))
print("Absolute prevalence deviation from the full dataset:")
display(deviation.abs().round(5))
print("Maximum absolute deviation:", float(deviation.abs().to_numpy().max()))
"""),
    md("## Determinism check"),
    code("""
def recreate_assignment():
    s1 = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=2_197, random_state=SEED
    )
    tr, ho = next(s1.split(np.zeros(len(working)), Y))
    s2 = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=1_098, random_state=SEED + 1
    )
    va_local, te_local = next(s2.split(np.zeros(len(ho)), Y[ho]))
    recreated = np.empty(len(working), dtype=object)
    recreated[tr] = "train"
    recreated[ho[va_local]] = "validation"
    recreated[ho[te_local]] = "test"
    return recreated

assert np.array_equal(assignment, recreate_assignment())
print("PASS: seed and algorithm reproduce the same assignments.")
"""),
    md("## Save the final two-column CSV"),
    code("""
result.to_csv(OUTPUT_CSV, index=False)
saved = pd.read_csv(OUTPUT_CSV, dtype=str, keep_default_na=False)

assert list(saved.columns) == ["track_id", "split"]
assert len(saved) == EXPECTED_TRACKS
assert saved["track_id"].is_unique
assert saved["split"].value_counts().to_dict() == EXPECTED_SIZES

print("Saved:", OUTPUT_CSV)
print("Rows:", len(saved))
print(saved["split"].value_counts())
"""),
    code("""
# Optional Colab download.
try:
    from google.colab import files
    files.download(str(OUTPUT_CSV))
except ImportError:
    print("Output is available at:", OUTPUT_CSV)
"""),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print("Wrote", OUTPUT)
