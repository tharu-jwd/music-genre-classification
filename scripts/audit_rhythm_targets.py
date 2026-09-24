"""Audit real AcousticBrainz rhythm coverage and interval compatibility."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rhythm_branch" / "src"))

from rhythm_branch.preprocessing import (
    align_manifest_and_targets,
    audit_target_coverage,
    load_rhythm_targets,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument(
        "--input-scope",
        choices=("full_recording", "centered_excerpt", "sampled_windows"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest, dtype={"song_id": str, "split": str})
    targets = load_rhythm_targets(args.targets)
    joined, values, mask = align_manifest_and_targets(manifest, targets)
    audit = audit_target_coverage(joined, values, mask, input_scope=args.input_scope)
    audit.pop("validity_mask")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
