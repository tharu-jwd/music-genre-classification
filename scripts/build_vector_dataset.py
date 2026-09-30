"""Group full_dataset.csv feature columns into JSON vectors in dataset.csv."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "rhythm_branch" / "src", ROOT / "timbre_branch" / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from concept_fusion.contract import (
    GENRE_TAGS,
    HARMONY_FEATURES,
    HARMONY_FEATURE_SETS,
    INSTRUMENT_TAGS,
    RHYTHM_FEATURES,
    TIMBRE_FEATURES,
)
VECTOR_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("instrument_vector", tuple(INSTRUMENT_TAGS)),
    ("rhythm_vector", tuple(RHYTHM_FEATURES)),
    ("timbre_vector", tuple(TIMBRE_FEATURES)),
    ("harmony_vector", HARMONY_FEATURES),
    ("genre", tuple(GENRE_TAGS)),
)


def _json_vector(values: np.ndarray) -> str:
    return json.dumps(values.tolist(), separators=(",", ":"), allow_nan=False)


def build_vector_dataset(
    source: Path, output: Path, *, overwrite: bool = False,
    harmony_feature_set: str = "all45", harmony_csv: Path | None = None,
) -> dict[str, object]:
    source, output = Path(source), Path(output)
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} already exists; pass --overwrite to replace it")

    frame = pd.read_csv(source, dtype={"TRACK_ID": str})
    if harmony_feature_set not in HARMONY_FEATURE_SETS:
        raise ValueError(f"Unknown harmony feature set: {harmony_feature_set}")
    harmony_features = HARMONY_FEATURE_SETS[harmony_feature_set]
    groups = tuple((name, harmony_features if name == "harmony_vector" else columns)
                   for name, columns in VECTOR_GROUPS)
    if harmony_feature_set == "all45":
        harmony_path = Path(harmony_csv) if harmony_csv is not None else source.parent / "harmony_df.csv"
        if not harmony_path.is_file():
            raise FileNotFoundError(f"45-feature Harmony table missing: {harmony_path}")
        harmony = pd.read_csv(harmony_path, dtype={"TRACK_ID": str})
        if harmony["TRACK_ID"].isna().any() or harmony["TRACK_ID"].duplicated().any():
            raise ValueError("Harmony TRACK_ID must be non-null and unique")
        absent = sorted(set(harmony_features) - set(harmony.columns))
        if absent:
            raise ValueError(f"Harmony table is missing columns: {absent}")
        frame = frame.drop(columns=[name for name in harmony_features if name in frame.columns])
        frame = frame.merge(harmony[["TRACK_ID", *harmony_features]], on="TRACK_ID", how="left", validate="one_to_one")
        missing_ids = frame.loc[frame[list(harmony_features)].isna().all(axis=1), "TRACK_ID"]
        if not missing_ids.empty:
            raise ValueError(f"Harmony table has no targets for {len(missing_ids)} tracks; first: {missing_ids.iloc[0]}")
    required = {"TRACK_ID", "logmel_path"}
    for _, columns in groups:
        required.update(columns)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Source dataset is missing columns: {missing}")
    if frame["TRACK_ID"].isna().any() or frame["TRACK_ID"].duplicated().any():
        raise ValueError("TRACK_ID must be non-null and unique")

    result = frame[["TRACK_ID", "logmel_path"]].rename(
        columns={"TRACK_ID": "track_id", "logmel_path": "path"}
    )
    for vector_name, columns in groups:
        values = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            bad = np.argwhere(~np.isfinite(values))[0]
            raise ValueError(
                f"{vector_name} has a missing/non-numeric value at row {bad[0] + 2}, "
                f"feature {columns[int(bad[1])]!r}"
            )
        result[vector_name] = [_json_vector(row) for row in values]

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    result.to_csv(temporary, index=False)
    temporary.replace(output)
    return {
        "source": str(source),
        "output": str(output),
        "rows": len(result),
        "columns": list(result.columns),
        "vector_lengths": {name: len(columns) for name, columns in groups},
        "harmony_feature_set": harmony_feature_set,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/full_dataset.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "data/dataset.csv")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--harmony-feature-set", choices=tuple(HARMONY_FEATURE_SETS), default="all45")
    parser.add_argument("--harmony-csv", type=Path, help="Canonical 45-column table; defaults to source directory/harmony_df.csv")
    args = parser.parse_args()
    print(json.dumps(build_vector_dataset(
        args.source, args.output, overwrite=args.overwrite,
        harmony_feature_set=args.harmony_feature_set, harmony_csv=args.harmony_csv,
    ), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
