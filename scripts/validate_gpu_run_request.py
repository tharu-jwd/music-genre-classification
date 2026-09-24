"""Validate a GPU run request before consuming hosted accelerator time."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scripts.gpu_run_contract import (
        load_cohort_artifact,
        load_gpu_budget,
        template_record,
        validate_gpu_run_record,
        verify_gpu_run_artifacts,
    )
except ModuleNotFoundError:
    from gpu_run_contract import (
        load_cohort_artifact,
        load_gpu_budget,
        template_record,
        validate_gpu_run_record,
        verify_gpu_run_artifacts,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", nargs="?", type=Path)
    parser.add_argument("--expected-job")
    parser.add_argument("--allow-planned", action="store_true")
    parser.add_argument("--print-template", action="store_true")
    args = parser.parse_args()
    if args.print_template:
        print(json.dumps(template_record(), indent=2))
        return 0
    if args.record is None:
        parser.error("record is required unless --print-template is used")
    record = json.loads(args.record.read_text())
    validated = validate_gpu_run_record(
        record,
        expected_job=args.expected_job,
        require_approved=not args.allow_planned,
    )
    validated, cohort, prerequisite_summary = verify_gpu_run_artifacts(record, args.record)
    budget_summary = None
    if validated["status"] == "approved":
        budget_path = Path(validated["budget_ledger"])
        if not budget_path.is_absolute():
            budget_path = args.record.parent / budget_path
        if not budget_path.is_file():
            raise FileNotFoundError(f"GPU budget ledger does not exist: {budget_path}")
        _, budget_summary = load_gpu_budget(budget_path, validated)
    print(json.dumps({
        "valid": True,
        "run_id": validated["run_id"],
        "experiment_id": validated["experiment_id"],
        "attempt": validated["attempt"],
        "status": validated["status"],
        "job": validated["job"],
        "max_epochs": validated["max_epochs"],
        "max_wall_minutes": validated["max_wall_minutes"],
        "evaluate_test": validated["evaluate_test"],
        "estimated_gpu_hours": validated["estimated_gpu_hours"],
        "budget_remaining_before_gpu_hours": validated.get("budget_remaining_before_gpu_hours"),
        "cohort_counts": cohort["counts"],
        "cohort_sha256": cohort["cohort_sha256"],
        "comparison_id": validated.get("comparison_id"),
        "control": prerequisite_summary["control"],
        "cheaper_check_artifacts": prerequisite_summary["cheaper_checks"],
        "reused_artifacts": prerequisite_summary["reused_artifacts"],
        "budget_summary": budget_summary,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
