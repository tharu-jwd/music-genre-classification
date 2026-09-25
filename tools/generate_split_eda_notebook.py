"""Generate the EDA-only notebook used before assigning dataset splits."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "dataset_split" / "01_split_readiness_eda.ipynb"


def lines(text: str) -> list[str]:
    text = text.strip("\n")
    return [line + "\n" for line in text.splitlines()]


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": lines(text)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines(text),
    }


cells = [
    markdown(
        """
# Train/validation/test split-readiness EDA

This notebook audits the 7,324 selected MTG-Jamendo records before producing the final two-column `track_id,split` file.

Current experimental policy:

- split unit: complete track;
- target ratio: 70% train, 15% validation, 15% test;
- preserve the six-genre multi-label distribution as closely as possible;
- do **not** constrain artists or albums for these initial experiments;
- never split individual log-Mel windows independently;
- do not write a final split CSV in this notebook.

Run every cell and share the final summary, genre tables, co-occurrence table, and readiness verdict before generating the split.
"""
    ),
    code(
        """
from pathlib import Path
import ast
import json
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.max_columns", 100)
pd.set_option("display.max_rows", 100)
sns.set_theme(style="whitegrid")

SEED = 42
EXPECTED_TRACKS = 7_324
SPLIT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}
TARGET_GENRES = ["classical", "electronic", "folk", "hiphop", "jazz", "rock"]
"""
    ),
    markdown("## 1. Mount Google Drive"),
    code(
        """
try:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    print("Google Drive is mounted.")
except ImportError:
    print(
        "google.colab is unavailable. This is expected outside Colab; "
        "set DATA_ROOT to a local repository data folder instead."
    )
"""
    ),
    markdown(
        """
## 2. Locate and validate the project data

The primary location is the project data folder created in Google Drive:

`/content/drive/MyDrive/music-genre-classification/data`

All six CSV files are validated before the EDA continues.
"""
    ),
    code(
        """
DRIVE_DATA_ROOT = Path(
    "/content/drive/MyDrive/music-genre-classification/data"
)

if DRIVE_DATA_ROOT.is_dir():
    DATA_ROOT = DRIVE_DATA_ROOT
else:
    local_candidates = [
        Path.cwd() / "data",
        Path.cwd().parent / "data",
        Path("/content/music-genre-classification/data"),
    ]
    DATA_ROOT = next(
        (path for path in local_candidates if (path / "split_csv.csv").is_file()),
        None,
    )

assert DATA_ROOT is not None, (
    "Data folder not found. In Colab, confirm Drive is mounted and the folder is "
    "/content/drive/MyDrive/music-genre-classification/data"
)

FILES = {
    "metadata": DATA_ROOT / "split_csv.csv",
    "genres": DATA_ROOT / "genres_df.csv",
    "logmels": DATA_ROOT / "logmel_metadata.csv",
    "rhythm": DATA_ROOT / "rhythm_df.csv",
    "timbre": DATA_ROOT / "timbre_df.csv",
    "instrument": DATA_ROOT / "instrument_df.csv",
}

print("Data root:", DATA_ROOT.resolve())
for name, path in FILES.items():
    print(f"{name:10s} | exists={path.is_file()} | {path}")

missing_files = [path.name for path in FILES.values() if not path.is_file()]
assert not missing_files, f"Required data files are missing: {missing_files}"
print("All six required CSV files are available.")
"""
    ),
    markdown("## 3. Helpers and table loading"),
    code(
        """
def normalize_track_id(value):
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if re.fullmatch(r"\\d+\\.0", text):
        text = text[:-2]
    text = re.sub(r"^track[_\\-\\s]*", "", text, flags=re.IGNORECASE)
    # Canonical numeric representation removes harmless leading-zero
    # differences such as track_0000382 versus 382.
    return str(int(text)) if re.fullmatch(r"\\d+", text) else None


def parse_label_list(value):
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None or pd.isna(value) or not str(value).strip():
        return []
    text = str(value).strip()
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
            if isinstance(parsed, (list, tuple, set)):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
            pass
    return [item.strip() for item in text.split("|") if item.strip()]


def load_table(path):
    frame = pd.read_csv(path, dtype={"TRACK_ID": str, "track_id": str})
    id_column = next((c for c in ("TRACK_ID", "track_id", "song_id") if c in frame), None)
    assert id_column is not None, f"No track ID column in {path.name}"
    frame["track_id"] = frame[id_column].map(normalize_track_id)
    return frame


tables = {
    name: load_table(path)
    for name, path in FILES.items()
    if path.is_file()
}

for name, frame in tables.items():
    print(f"{name:10s}: {frame.shape[0]:,} rows x {frame.shape[1]:,} columns")
"""
    ),
    markdown("## 3. Track-ID integrity and cross-table coverage"),
    code(
        """
integrity_rows = []
metadata_ids = set(tables["metadata"]["track_id"].dropna())

for name, frame in tables.items():
    ids = frame["track_id"]
    current_ids = set(ids.dropna())
    integrity_rows.append({
        "table": name,
        "rows": len(frame),
        "unique_track_ids": ids.nunique(dropna=True),
        "missing_track_ids": int(ids.isna().sum()),
        "duplicate_track_ids": int(ids.duplicated(keep=False).sum()),
        "missing_vs_metadata": len(metadata_ids - current_ids),
        "extra_vs_metadata": len(current_ids - metadata_ids),
    })

integrity = pd.DataFrame(integrity_rows).set_index("table")
display(integrity)

assert len(tables["metadata"]) == EXPECTED_TRACKS
assert tables["metadata"]["track_id"].nunique() == EXPECTED_TRACKS
"""
    ),
    code(
        """
coverage = pd.DataFrame({"track_id": sorted(metadata_ids)})
for name, frame in tables.items():
    if name == "metadata":
        continue
    available = set(frame["track_id"].dropna())
    coverage[f"has_{name}"] = coverage["track_id"].isin(available)

coverage_columns = [column for column in coverage if column.startswith("has_")]
display(coverage[coverage_columns].sum().to_frame("covered_tracks"))

if coverage_columns:
    incomplete = coverage.loc[~coverage[coverage_columns].all(axis=1)]
    print("Tracks missing at least one required modality:", len(incomplete))
    display(incomplete.head(20))
"""
    ),
    markdown(
        """
## 4. Genre-label integrity

`target_genres` in the source metadata and the six binary columns in `genres_df.csv` are checked independently and compared.
"""
    ),
    code(
        """
metadata = tables["metadata"].copy()
assert "target_genres" in metadata, "metadata must contain target_genres"
metadata["target_genres_parsed"] = metadata["target_genres"].map(parse_label_list)

unknown_labels = sorted({
    label
    for labels in metadata["target_genres_parsed"]
    for label in labels
    if label not in TARGET_GENRES
})

label_cardinality = metadata["target_genres_parsed"].map(len)
print("Tracks with no target genre:", int((label_cardinality == 0).sum()))
print("Tracks with multiple target genres:", int((label_cardinality > 1).sum()))
print("Maximum labels on one track:", int(label_cardinality.max()))
print("Unknown target labels:", unknown_labels)
display(label_cardinality.value_counts().sort_index().rename("tracks").to_frame())
"""
    ),
    code(
        """
if "genres" in tables:
    genres = tables["genres"].copy()
    independent_genre_table = True
else:
    print("genres_df.csv not found; reconstructing labels from target_genres.")
    genres = metadata[["track_id", "target_genres_parsed"]].copy()
    for genre in TARGET_GENRES:
        genres[genre] = genres["target_genres_parsed"].map(
            lambda labels, selected=genre: int(selected in labels)
        )
    genres = genres.drop(columns="target_genres_parsed")
    tables["genres"] = genres.copy()
    independent_genre_table = False
missing_genre_columns = sorted(set(TARGET_GENRES) - set(genres.columns))
assert not missing_genre_columns, f"Missing genre columns: {missing_genre_columns}"

genre_matrix = genres.set_index("track_id")[TARGET_GENRES].apply(
    pd.to_numeric, errors="coerce"
)

print("NaN genre cells:", int(genre_matrix.isna().sum().sum()))
print("Non-binary genre cells:", int((~genre_matrix.isin([0, 1])).sum().sum()))
print("All-zero genre rows:", int((genre_matrix.sum(axis=1) == 0).sum()))
print("Multi-label genre rows:", int((genre_matrix.sum(axis=1) > 1).sum()))
display(genre_matrix.sum().sort_values(ascending=False).rename("track_count").to_frame())
"""
    ),
    code(
        """
if independent_genre_table:
    metadata_matrix = (
        metadata.set_index("track_id")["target_genres_parsed"]
        .map(lambda labels: {label: int(label in labels) for label in TARGET_GENRES})
        .apply(pd.Series)
        .loc[:, TARGET_GENRES]
    )
    common_ids = genre_matrix.index.intersection(metadata_matrix.index)
    mismatch_cells = (
        genre_matrix.loc[common_ids].astype(int)
        != metadata_matrix.loc[common_ids].astype(int)
    )
    print("Tracks compared:", len(common_ids))
    print("Mismatching genre cells:", int(mismatch_cells.sum().sum()))
    print("Tracks with any genre mismatch:", int(mismatch_cells.any(axis=1).sum()))
else:
    mismatch_cells = pd.DataFrame(
        False, index=genre_matrix.index, columns=TARGET_GENRES
    )
    print("Independent comparison skipped: genres were reconstructed from metadata.")
"""
    ),
    markdown("## 5. Genre imbalance and co-occurrence"),
    code(
        """
genre_counts = genre_matrix.sum().sort_values(ascending=False)
genre_prevalence = (genre_counts / len(genre_matrix)).rename("prevalence")
genre_summary = pd.concat(
    [genre_counts.rename("tracks"), genre_prevalence], axis=1
)
display(genre_summary)

ax = genre_counts.sort_values().plot.barh(figsize=(8, 4), color="#4267B2")
ax.set(title="Target-genre support", xlabel="Tracks", ylabel="Genre")
plt.tight_layout()
plt.show()
"""
    ),
    code(
        """
cooccurrence = genre_matrix.T.dot(genre_matrix).astype(int)
display(cooccurrence)

plt.figure(figsize=(7, 6))
sns.heatmap(cooccurrence, annot=True, fmt="d", cmap="Blues")
plt.title("Genre-label co-occurrence")
plt.tight_layout()
plt.show()
"""
    ),
    markdown(
        """
## 6. Proposed 70/15/15 sizes and per-genre feasibility

These are expected counts, not final assignments. The final notebook should use multi-label stratification and seed 42.
"""
    ),
    code(
        """
train_size = round(EXPECTED_TRACKS * SPLIT_RATIOS["train"])
validation_size = round(EXPECTED_TRACKS * SPLIT_RATIOS["validation"])
test_size = EXPECTED_TRACKS - train_size - validation_size

planned_sizes = pd.Series({
    "train": train_size,
    "validation": validation_size,
    "test": test_size,
}, name="tracks")
display(planned_sizes.to_frame())
print("Total:", int(planned_sizes.sum()))

expected_genre_counts = pd.DataFrame({
    split: genre_counts * ratio
    for split, ratio in SPLIT_RATIOS.items()
}).round(1)
display(expected_genre_counts)

minimum_expected = expected_genre_counts.min(axis=1)
print("Genres expected to have fewer than 20 positives in a split:")
display(minimum_expected[minimum_expected < 20].rename("smallest_expected_count").to_frame())
"""
    ),
    markdown("## 7. Feature-table numerical quality"),
    code(
        """
feature_quality = []
for name in ("rhythm", "timbre", "instrument"):
    if name not in tables:
        continue
    frame = tables[name]
    numeric = frame.drop(columns=[c for c in ("TRACK_ID", "track_id", "song_id") if c in frame], errors="ignore")
    numeric = numeric.apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=float)
    feature_quality.append({
        "table": name,
        "rows": len(frame),
        "feature_columns": numeric.shape[1],
        "missing_values": int(np.isnan(values).sum()),
        "infinite_values": int(np.isinf(values).sum()),
        "constant_columns": int((numeric.nunique(dropna=False) <= 1).sum()),
    })

if feature_quality:
    display(pd.DataFrame(feature_quality).set_index("table"))
else:
    print(
        "No optional rhythm, timbre, or instrument tables were found. "
        "Skipping numerical feature-quality checks. This does not block "
        "creation of a track_id,split file from split_csv.csv."
    )
"""
    ),
    markdown("## 8. Log-Mel manifest checks"),
    code(
        """
logmels = tables.get("logmels")
if logmels is None:
    print("No logmel_metadata.csv was found.")
else:
    path_column = next(
        (column for column in ("logmel_path", "mel_path", "path") if column in logmels),
        None,
    )
    assert path_column is not None, "Cannot identify the log-Mel path column"
    paths = logmels[path_column].astype(str).str.strip()
    print("Rows:", len(logmels))
    print("Unique track IDs:", logmels["track_id"].nunique())
    print("Blank paths:", int(paths.eq("").sum()))
    print("Duplicate paths:", int(paths.duplicated().sum()))
    print("Non-.npy paths:", int((~paths.str.lower().str.endswith(".npy")).sum()))

    # Existence is meaningful only when Drive is mounted at the recorded location.
    if len(paths) and Path(paths.iloc[0]).anchor:
        existence_sample = paths.sample(min(100, len(paths)), random_state=SEED)
        exists = existence_sample.map(lambda value: Path(value).is_file())
        print("Existing paths in a 100-file local sample:", int(exists.sum()), "/", len(exists))
        if not exists.all():
            print("A failed local existence check can simply mean this notebook is not running in Colab with Drive mounted.")
"""
    ),
    markdown("## 9. Final readiness report"),
    code(
        """
optional_tables = ["logmels", "rhythm", "timbre", "instrument"]
checks = {
    "metadata_has_7324_rows": len(tables["metadata"]) == EXPECTED_TRACKS,
    "metadata_ids_are_unique": tables["metadata"]["track_id"].nunique() == EXPECTED_TRACKS,
    "genre_labels_available": "genres" in tables,
    "available_optional_tables_cover_selected_tracks": all(
        set(tables[name]["track_id"].dropna()) == metadata_ids
        for name in optional_tables
        if name in tables
    ),
    "genre_matrix_is_complete": genre_matrix.notna().all().all(),
    "genre_matrix_is_binary": genre_matrix.isin([0, 1]).all().all(),
    "every_track_has_a_target_genre": (genre_matrix.sum(axis=1) >= 1).all(),
    "metadata_and_genre_table_agree": not mismatch_cells.any().any(),
    "split_sizes_sum_to_7324": int(planned_sizes.sum()) == EXPECTED_TRACKS,
}

report = pd.Series(checks, name="passed").to_frame()
display(report)

print("FINAL VERDICT:", "READY" if report["passed"].all() else "REVIEW FAILED CHECKS")
missing_optional = [name for name in optional_tables if name not in tables]
print("Optional tables not audited:", missing_optional or "none")
print()
print("Share these outputs before creating the final split CSV:")
print("1. integrity table")
print("2. coverage totals")
print("3. genre summary and label-cardinality table")
print("4. co-occurrence matrix")
print("5. expected per-split genre counts")
print("6. feature-quality table")
print("7. final readiness report")
"""
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(f"Wrote {OUTPUT}")
