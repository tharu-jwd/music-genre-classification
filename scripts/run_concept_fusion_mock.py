"""CLI: train one fusion experiment on fixtures (thin wrapper over the full pipeline)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from concept_fusion.experiments import experiment_specs
from concept_fusion.pipeline import run_single


def main() -> None:
    ids = [s.experiment_id for s in experiment_specs()]
    p = argparse.ArgumentParser(description="Mock concept-fusion stack (no branch owner required)")
    p.add_argument("--experiment-id", default="F-Gated", choices=ids)
    p.add_argument("--fusion", default=None, help="Ignored; fusion is taken from the experiment spec")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--out", default="results/proposed/mock")
    args = p.parse_args()
    rec = run_single(args.experiment_id, seed=args.seed, out_dir=args.out, steps=args.steps)
    print("wrote", rec.checkpoint_path)
    print("macro_ap", rec.metrics.get("macro_ap"), "n_valid_tags", rec.n_valid_tags)
    print("loss", rec.metrics.get("loss_terms"))


if __name__ == "__main__":
    main()
