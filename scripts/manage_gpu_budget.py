"""Create and update the shared GPU-hour budget ledger."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

try:
    from scripts.gpu_run_contract import (
        load_gpu_budget,
        validate_gpu_run_record,
        verify_gpu_run_artifacts,
    )
except ModuleNotFoundError:
    from gpu_run_contract import load_gpu_budget, validate_gpu_run_record, verify_gpu_run_artifacts


def new_ledger(total_gpu_hours: float) -> dict:
    if total_gpu_hours <= 0:
        raise ValueError("total GPU hours must be positive")
    return {
        "schema_version": "gpu_budget_v1",
        "total_gpu_hours": float(total_gpu_hours),
        "completed_runs": [],
        "reservations": [],
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def reserve_run(ledger: dict, record: dict) -> dict:
    validated = validate_gpu_run_record(record)
    _, summary = load_gpu_budget_from_object(ledger)
    if summary["overcommitted"]:
        raise ValueError("GPU budget is already overcommitted")
    existing_ids = {
        item["run_id"]
        for field in ("completed_runs", "reservations")
        for item in ledger.get(field, [])
    }
    if validated["run_id"] in existing_ids:
        raise ValueError(f"run ID already exists in budget ledger: {validated['run_id']}")
    available = summary["unreserved_gpu_hours"]
    if abs(float(validated["budget_remaining_before_gpu_hours"]) - available) > 1e-9:
        raise ValueError("run budget snapshot does not equal current unreserved GPU hours")
    if float(validated["estimated_gpu_hours"]) > available:
        raise ValueError("run estimate exceeds unreserved GPU hours")
    updated = json.loads(json.dumps(ledger))
    updated["reservations"].append({
        "run_id": validated["run_id"],
        "experiment_id": validated["experiment_id"],
        "attempt": validated["attempt"],
        "estimated_gpu_hours": float(validated["estimated_gpu_hours"]),
        "code_commit": validated["code_commit"],
        "reserved_at": datetime.now(timezone.utc).isoformat(),
    })
    updated["updated_at"] = datetime.now(timezone.utc).isoformat()
    load_gpu_budget_from_object(updated)
    return updated


def complete_run(ledger: dict, run_id: str, actual_gpu_hours: float) -> dict:
    if actual_gpu_hours < 0:
        raise ValueError("actual GPU hours must be non-negative")
    reservations = ledger.get("reservations", [])
    matches = [item for item in reservations if item.get("run_id") == run_id]
    if len(matches) != 1:
        raise ValueError(f"expected one reservation for {run_id!r}")
    updated = json.loads(json.dumps(ledger))
    updated["reservations"] = [item for item in updated["reservations"] if item["run_id"] != run_id]
    reservation = matches[0]
    updated["completed_runs"].append({
        "run_id": run_id,
        "experiment_id": reservation["experiment_id"],
        "attempt": reservation["attempt"],
        "actual_gpu_hours": float(actual_gpu_hours),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    })
    updated["updated_at"] = datetime.now(timezone.utc).isoformat()
    load_gpu_budget_from_object(updated)
    return updated


def load_gpu_budget_from_object(ledger: dict):
    with tempfile.TemporaryDirectory() as temporary_directory:
        path = Path(temporary_directory) / "ledger.json"
        path.write_text(json.dumps(ledger))
        return load_gpu_budget(path)


def _atomic_write(path: Path, value: dict, *, create: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if create and path.exists():
        raise FileExistsError(f"budget ledger already exists: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, indent=2)
            handle.write("\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    initialize = subparsers.add_parser("init")
    initialize.add_argument("ledger", type=Path)
    initialize.add_argument("--total-hours", required=True, type=float)
    reserve = subparsers.add_parser("reserve")
    reserve.add_argument("ledger", type=Path)
    reserve.add_argument("--run-record", required=True, type=Path)
    complete = subparsers.add_parser("complete")
    complete.add_argument("ledger", type=Path)
    complete.add_argument("--run-id", required=True)
    complete.add_argument("--actual-hours", required=True, type=float)
    status = subparsers.add_parser("status")
    status.add_argument("ledger", type=Path)
    args = parser.parse_args()

    if args.command == "init":
        ledger = new_ledger(args.total_hours)
        _atomic_write(args.ledger, ledger, create=True)
        _, summary = load_gpu_budget(args.ledger)
    else:
        ledger, summary = load_gpu_budget(args.ledger)
        if args.command == "reserve":
            record = json.loads(args.run_record.read_text())
            verify_gpu_run_artifacts(record, args.run_record)
            ledger = reserve_run(ledger, record)
            _atomic_write(args.ledger, ledger)
            _, summary = load_gpu_budget(args.ledger)
        elif args.command == "complete":
            ledger = complete_run(ledger, args.run_id, args.actual_hours)
            _atomic_write(args.ledger, ledger)
            _, summary = load_gpu_budget(args.ledger)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
