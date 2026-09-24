"""Freeze an exact, deterministic song cohort before training or comparison."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path


SCHEMA_VERSION = "experiment_cohort_v1"
VALID_SPLITS = ("train", "validation", "test")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_bool(value: str, *, field: str, song_id: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"{field} for song {song_id} must be true/false or 1/0, got {value!r}")


def _selection_key(song_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{song_id}".encode()).hexdigest()


def parse_limits(values: list[str]) -> dict[str, int]:
    limits: dict[str, int] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"limit must use split=count syntax, got {value!r}")
        split, raw_count = value.split("=", 1)
        if split not in VALID_SPLITS or split in limits:
            raise ValueError(f"invalid or duplicate split limit: {split!r}")
        try:
            count = int(raw_count)
        except ValueError as error:
            raise ValueError(f"limit for {split} must be an integer") from error
        if count < 1:
            raise ValueError(f"limit for {split} must be positive")
        limits[split] = count
    return limits


def freeze_cohort(
    manifest_path: Path,
    *,
    splits: tuple[str, ...] = VALID_SPLITS,
    required_available: tuple[str, ...] = (),
    limits: dict[str, int] | None = None,
    seed: int = 42,
) -> dict:
    """Return a versioned cohort selected without inspecting model outcomes."""
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if not splits or any(split not in VALID_SPLITS for split in splits) or len(set(splits)) != len(splits):
        raise ValueError("splits must be unique members of train, validation, test")
    limits = dict(limits or {})
    if set(limits) - set(splits):
        raise ValueError("limits may only name selected splits")

    with manifest_path.open(newline="", encoding="utf-8", errors="strict") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"song_id", "split", *required_available}
        missing_columns = required_columns - set(reader.fieldnames or ())
        if missing_columns:
            raise ValueError(f"manifest is missing columns: {sorted(missing_columns)}")
        rows = list(reader)
    if not rows:
        raise ValueError("manifest is empty")

    seen: set[str] = set()
    eligible = {split: [] for split in splits}
    excluded_for_availability = {split: 0 for split in splits}
    for row_number, row in enumerate(rows, start=2):
        song_id = str(row.get("song_id", "")).strip()
        if not re.fullmatch(r"\d{7}", song_id):
            raise ValueError(f"manifest row {row_number} has invalid song_id {song_id!r}")
        if song_id in seen:
            raise ValueError(f"manifest contains duplicate song_id {song_id}")
        seen.add(song_id)
        split = str(row.get("split", "")).strip()
        if split not in VALID_SPLITS:
            raise ValueError(f"manifest row {row_number} has unsupported split {split!r}")
        if split not in eligible:
            continue
        if not all(_strict_bool(row[field], field=field, song_id=song_id) for field in required_available):
            excluded_for_availability[split] += 1
            continue
        eligible[split].append(song_id)

    selected: dict[str, list[str]] = {}
    for split in splits:
        ranked = sorted(eligible[split], key=lambda song_id: (_selection_key(song_id, seed), song_id))
        limit = limits.get(split)
        if limit is not None and len(ranked) < limit:
            raise ValueError(f"{split} has only {len(ranked)} eligible songs; requested {limit}")
        chosen = ranked if limit is None else ranked[:limit]
        if not chosen:
            raise ValueError(f"no eligible songs remain in {split}")
        selected[split] = chosen

    canonical = json.dumps(selected, sort_keys=True, separators=(",", ":")).encode()
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": _sha256_file(manifest_path),
        },
        "selection": {
            "method": "sha256_rank_v1",
            "seed": seed,
            "required_available": list(required_available),
            "limits": {split: limits.get(split) for split in splits},
            "eligible_counts": {split: len(eligible[split]) for split in splits},
            "excluded_for_availability": excluded_for_availability,
        },
        "splits": selected,
        "counts": {split: len(song_ids) for split, song_ids in selected.items()},
        "cohort_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--splits", nargs="+", choices=VALID_SPLITS, default=list(VALID_SPLITS))
    parser.add_argument("--require-available", action="append", default=[])
    parser.add_argument("--limit", action="append", default=[], metavar="SPLIT=COUNT")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; frozen cohorts are immutable")
    artifact = freeze_cohort(
        args.manifest,
        splits=tuple(args.splits),
        required_available=tuple(args.require_available),
        limits=parse_limits(args.limit),
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "counts": artifact["counts"],
        "cohort_sha256": artifact["cohort_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
