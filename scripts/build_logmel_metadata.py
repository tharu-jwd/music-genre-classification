"""Build trainer metadata for the repository's stacked log-mel cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LOGMEL_CONFIG = {
    "sample_rate": 16000,
    "hop_length": 512,
    "window_seconds": 15.0,
    "n_mels": 128,
    "center": True,
}


def build(source: Path, output_dir: Path) -> dict[str, object]:
    source, output_dir = Path(source), Path(output_dir)
    frame = pd.read_csv(source, dtype={"TRACK_ID": str})
    required = {"TRACK_ID", "DURATION"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Duration source is missing columns: {missing}")
    if frame["TRACK_ID"].isna().any() or frame["TRACK_ID"].duplicated().any():
        raise ValueError("TRACK_ID must be non-null and unique")
    durations = pd.to_numeric(frame["DURATION"], errors="coerce")
    if durations.isna().any() or (durations <= 0).any():
        raise ValueError("Every track must have a positive finite duration")

    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "logmel_config.json"
    audit_path = output_dir / "logmel_audit.csv"
    config_path.write_text(json.dumps(LOGMEL_CONFIG, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame({
        "TRACK_ID": frame["TRACK_ID"],
        "track_duration_sec": durations,
    }).to_csv(audit_path, index=False)
    return {
        "config": str(config_path),
        "audit": str(audit_path),
        "tracks": len(frame),
        "logmel_config": LOGMEL_CONFIG,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path,
        default=ROOT / "notebooks/dataset_split/split_csv.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.output_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
