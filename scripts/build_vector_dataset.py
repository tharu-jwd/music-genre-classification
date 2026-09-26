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


def build_vector_dataset(source: Path, output: Path, *, overwrite: bool = False) -> dict[str, object]:
    source, output = Path(source), Path(output)
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} already exists; pass --overwrite to replace it")

    frame = pd.read_csv(source, dtype={"TRACK_ID": str})
    required = {"TRACK_ID", "logmel_path"}
    for _, columns in VECTOR_GROUPS:
        required.update(columns)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Source dataset is missing columns: {missing}")
    if frame["TRACK_ID"].isna().any() or frame["TRACK_ID"].duplicated().any():
        raise ValueError("TRACK_ID must be non-null and unique")

    result = frame[["TRACK_ID", "logmel_path"]].rename(
        columns={"TRACK_ID": "track_id", "logmel_path": "path"}
    )
    for vector_name, columns in VECTOR_GROUPS:
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
        "vector_lengths": {name: len(columns) for name, columns in VECTOR_GROUPS},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/full_dataset.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "data/dataset.csv")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build_vector_dataset(args.source, args.output, overwrite=args.overwrite), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
