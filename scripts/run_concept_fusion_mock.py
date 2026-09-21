"""CLI: train the fusion stack on fixtures and write a RunRecord."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from concept_fusion.compute import parameter_count, synced_infer_ms
from concept_fusion.contract import CONCEPT_DROPOUT_P, N_GENRE_TAGS
from concept_fusion.fixtures import make_bundle, make_concept_targets, make_genre_batch
from concept_fusion.joint_loss import JointLossOrchestrator
from concept_fusion.metrics import ranking_metrics, sigmoid_np
from concept_fusion.model import ConceptBottleneckModel
from concept_fusion.run_schema import RunConfig, RunRecord, seeds_for
from concept_fusion.thresholds import f1_precision_recall, fit_thresholds, apply_thresholds


def main() -> None:
    p = argparse.ArgumentParser(description="Mock concept-fusion stack (no branch owner required)")
    p.add_argument("--experiment-id", default="F-Gated")
    p.add_argument("--fusion", default="gated", choices=("concat", "gated", "attention"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--out", default="results/proposed/mock")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    bundle = make_bundle(args.batch, seed=args.seed)
    y = make_genre_batch(args.batch, seed=args.seed + 1)
    targets = make_concept_targets(bundle, seed=args.seed + 2)

    model = ConceptBottleneckModel(fusion=args.fusion, dropout_p=CONCEPT_DROPOUT_P)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = JointLossOrchestrator()

    model.train()
    last = None
    for _ in range(args.steps):
        opt.zero_grad()
        logits, _ = model.from_bundle(bundle)
        br = loss_fn(logits, y, bundle, targets)
        br.total.backward()
        opt.step()
        last = br

    model.eval()
    with torch.no_grad():
        logits, fout = model.from_bundle(bundle, apply_dropout=False)
    probs = torch.sigmoid(logits)
    rank = ranking_metrics(y, probs)
    thr = fit_thresholds(y, probs)  # mock only — real runs must use validation split
    pred = apply_thresholds(probs, thr)
    clf = f1_precision_recall(y, pred)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / f"{args.experiment_id}_seed{args.seed}.pt"
    torch.save({"model": model.state_dict(), "fusion": args.fusion, "n_tags": N_GENRE_TAGS}, ckpt)

    cfg = RunConfig(experiment_id=args.experiment_id, fusion=args.fusion, seed=args.seed)
    rec = RunRecord.create(
        cfg,
        metrics={
            **rank.__dict__,
            **clf,
            "loss_terms": last.terms if last else {},
            "n_observed": last.n_observed if last else {},
            "params": parameter_count(model),
            "infer_ms": synced_infer_ms(model, bundle.tokens(), bundle.fusion_mask(), warmup=2, steps=10),
            "mean_gates": fout.gates.mean(0).tolist(),
        },
        n_valid_tags=rank.n_valid_tags,
        checkpoint_path=str(ckpt),
        extra={"note": "FIXTURE RUN — not a paper result", "seeds_policy": list(seeds_for(args.experiment_id))},
    )
    rec.test_evaluated = False
    rec.write(out_dir / f"{args.experiment_id}_seed{args.seed}.json")
    print("wrote", ckpt)
    print("macro_ap", rank.macro_ap, "n_valid_tags", rank.n_valid_tags)
    print("loss", last.terms if last else None)


if __name__ == "__main__":
    main()
