"""Run every fusion-owned experiment on one frozen fixture cohort.

Usage (repo root):

    python scripts/run_all_fusion.py --quick
    python scripts/run_all_fusion.py
    python scripts/run_all_fusion.py --only F-Gated --seed 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from concept_fusion.pipeline import run_all, run_single


def main() -> None:
    p = argparse.ArgumentParser(description="Run concept-fusion experiments (fixtures until branches land)")
    p.add_argument("--quick", action="store_true", help="One seed each, tiny cohort, few steps")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--out", default="results/proposed/mock")
    p.add_argument("--only", default=None, help="Single experiment ID, e.g. F-Gated")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lambda-rhythm", type=float, default=1.0)
    args = p.parse_args()

    if args.only:
        rec = run_single(
            args.only,
            seed=args.seed,
            out_dir=args.out,
            steps=args.steps or (5 if args.quick else 30),
            lambda_rhythm=args.lambda_rhythm,
        )
        print("wrote", rec.checkpoint_path)
        print("val_macro_ap", rec.metrics.get("val_macro_ap"), "test_macro_ap", rec.metrics.get("test_macro_ap"))
        return

    run_all(
        quick=args.quick,
        out_dir=args.out,
        steps=args.steps,
        lambda_rhythm=args.lambda_rhythm,
    )


if __name__ == "__main__":
    main()
