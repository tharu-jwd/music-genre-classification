"""Train, select on validation, evaluate test, write RunRecords — one API for all experiments."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from concept_fusion.compute import checkpoint_size_bytes, parameter_count
from concept_fusion.contract import FUSION_CONTRACT_VERSION, N_GENRE_TAGS
from concept_fusion.direct import DirectAudioBaseline
from concept_fusion.experiments import ExperimentSpec, experiment_specs, jobs_for
from concept_fusion.fixtures import FixtureCohort, FixtureSplit, make_cohort
from concept_fusion.interventions import gate_vs_occlusion_correlation, occlude_each_concept
from concept_fusion.joint_loss import JointLossOrchestrator, LossWeights
from concept_fusion.metrics import ranking_metrics
from concept_fusion.model import ConceptBottleneckModel
from concept_fusion.run_schema import RunConfig, RunRecord, validate_fusion_checkpoint_metadata
from concept_fusion.tables import comparison_table, dataframe_markdown, mean_std_table
from concept_fusion.thresholds import apply_thresholds, f1_precision_recall, fit_thresholds


@dataclass
class PipelineConfig:
    out_dir: Path = Path("results/proposed/mock")
    steps: int = 30
    batch_size: int = 16
    lr: float = 1e-3
    n_train: int = 48
    n_val: int = 24
    n_test: int = 24
    cohort_seed: int = 0
    warmup: int = 2
    infer_steps: int = 10
    fixture: bool = True
    lambda_rhythm: float = 1.0

    def __post_init__(self) -> None:
        if self.lambda_rhythm < 0:
            raise ValueError("lambda_rhythm must be non-negative")


def _loss_weights(spec: ExperimentSpec, *, lambda_rhythm: float = 1.0) -> LossWeights:
    if not spec.aux:
        return LossWeights(
            instrument=0.0,
            rhythm=0.0,
            timbre=0.0,
            harmony=0.0,
            use_kendall=spec.use_kendall,
        )
    return LossWeights(rhythm=lambda_rhythm, use_kendall=spec.use_kendall)


def _build_model(spec: ExperimentSpec) -> nn.Module:
    if spec.model_kind == "direct":
        return DirectAudioBaseline()
    return ConceptBottleneckModel(
        fusion=spec.fusion,  # type: ignore[arg-type]
        dropout_p=spec.dropout_p,
        allow_shortcut=spec.allow_shortcut,
        use_hidden=spec.use_hidden,
        allow_no_dropout=spec.allow_no_dropout,
        fusion_input_mode=spec.fusion_input_mode,
    )


def _prepare_split(split: FixtureSplit, spec: ExperimentSpec) -> FixtureSplit:
    if spec.model_kind == "direct":
        return split
    return FixtureSplit(
        song_ids=split.song_ids,
        bundle=split.bundle.with_enabled_concepts(spec.enabled_concepts),
        genre=split.genre,
        concept_targets=split.concept_targets,
        song_repr=split.song_repr,
    )


def _minibatches(n: int, batch_size: int, generator: torch.Generator) -> list[torch.Tensor]:
    perm = torch.randperm(n, generator=generator)
    return [perm[i : i + batch_size] for i in range(0, n, batch_size)]


def _slice_split(split: FixtureSplit, idx: torch.Tensor) -> FixtureSplit:
    from concept_fusion.fixtures import _index_bundle, _index_targets

    return FixtureSplit(
        song_ids=[split.song_ids[int(i)] for i in idx.tolist()],
        bundle=_index_bundle(split.bundle, idx),
        genre=split.genre[idx],
        concept_targets=_index_targets(split.concept_targets, idx),
        song_repr=split.song_repr[idx],
    )


def _forward(
    model: nn.Module,
    split: FixtureSplit,
    spec: ExperimentSpec,
    *,
    train: bool,
) -> tuple[torch.Tensor, Any]:
    if spec.model_kind == "direct":
        logits = model(split.song_repr)
        return logits, None
    kwargs: dict[str, Any] = {"apply_dropout": train and spec.dropout_p > 0}
    if spec.allow_shortcut:
        kwargs["song_repr"] = split.song_repr
    return model.from_bundle(split.bundle, **kwargs)


def _eval_split(
    model: nn.Module, split: FixtureSplit, spec: ExperimentSpec
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, Any]:
    model.eval()
    with torch.no_grad():
        logits, fout = _forward(model, split, spec, train=False)
    probs = torch.sigmoid(logits)
    rank = ranking_metrics(split.genre, probs)
    out: dict[str, Any] = {
        **rank.__dict__,
        "n_tracks": len(split.song_ids),
    }
    if fout is not None:
        out["mean_gates"] = fout.gates.mean(0).tolist()
    return out, logits, probs, fout


def train_one(
    spec: ExperimentSpec,
    seed: int,
    cohort: FixtureCohort,
    cfg: PipelineConfig,
) -> RunRecord:
    torch.manual_seed(seed)
    train = _prepare_split(cohort.train, spec)
    val = _prepare_split(cohort.val, spec)
    test = _prepare_split(cohort.test, spec)

    model = _build_model(spec)
    loss_fn = JointLossOrchestrator(weights=_loss_weights(spec, lambda_rhythm=cfg.lambda_rhythm))
    params = list(model.parameters()) + list(loss_fn.parameters())
    opt = torch.optim.Adam(params, lr=cfg.lr)
    g = torch.Generator().manual_seed(seed + 99)

    model.train()
    last_terms: dict[str, float] = {}
    last_obs: dict[str, int] = {}
    t0 = time.perf_counter()
    step = 0
    while step < cfg.steps:
        for idx in _minibatches(train.bundle.batch if spec.model_kind != "direct" else train.genre.shape[0], cfg.batch_size, g):
            if step >= cfg.steps:
                break
            batch = _slice_split(train, idx)
            opt.zero_grad()
            if spec.model_kind == "direct":
                logits = model(batch.song_repr)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, batch.genre)
                last_terms = {"genre": float(loss.detach().item())}
                last_obs = {"genre": int(batch.genre.numel())}
                loss.backward()
            else:
                logits, _ = _forward(model, batch, spec, train=True)
                br = loss_fn(logits, batch.genre, batch.bundle, batch.concept_targets)
                last_terms = br.terms
                last_obs = br.n_observed
                br.total.backward()
            opt.step()
            step += 1
    train_s = time.perf_counter() - t0

    val_metrics, val_logits, val_probs, _ = _eval_split(model, val, spec)
    # Thresholds from validation only. Test labels are not used here.
    thr = fit_thresholds(val.genre, val_probs)
    val_clf = f1_precision_recall(val.genre, apply_thresholds(val_probs, thr))

    test_metrics, test_logits, test_probs, fout = _eval_split(model, test, spec)
    test_clf = f1_precision_recall(test.genre, apply_thresholds(test_probs, thr))

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = cfg.out_dir / f"{spec.experiment_id}_seed{seed}.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "spec": asdict(spec),
            "seed": seed,
            "n_tags": N_GENRE_TAGS,
            "thresholds": thr,
            "test_song_ids": test.song_ids,
            "fixture": cfg.fixture,
            "lambda_rhythm": cfg.lambda_rhythm,
            "fusion_contract_version": FUSION_CONTRACT_VERSION,
            "fusion_input_mode": spec.fusion_input_mode,
        },
        ckpt,
    )

    infer_ms = _infer_ms(model, test, spec, warmup=cfg.warmup, steps=cfg.infer_steps)
    run_cfg = RunConfig(
        experiment_id=spec.experiment_id,
        fusion=spec.fusion,
        seed=seed,
        dropout_p=spec.dropout_p,
        allow_shortcut=spec.allow_shortcut,
        lambda_rhythm=cfg.lambda_rhythm,
        fusion_contract_version=FUSION_CONTRACT_VERSION,
        fusion_input_mode=spec.fusion_input_mode,
        notes=spec.purpose,
    )
    rec = RunRecord.create(
        run_cfg,
        metrics={
            "macro_ap": test_metrics["macro_ap"],
            "micro_ap": test_metrics["micro_ap"],
            "macro_roc_auc": test_metrics["macro_roc_auc"],
            "trapezoidal_macro_pr_auc": test_metrics["trapezoidal_macro_pr_auc"],
            "val_macro_ap": val_metrics["macro_ap"],
            "val_micro_ap": val_metrics["micro_ap"],
            "test_macro_ap": test_metrics["macro_ap"],
            "val_macro_f1": val_clf["macro_f1"],
            "test_macro_f1": test_clf["macro_f1"],
            "test_micro_f1": test_clf["micro_f1"],
            "n_valid_tags_val": val_metrics["n_valid_tags"],
            "n_valid_tags_test": test_metrics["n_valid_tags"],
            "loss_terms": last_terms,
            "n_observed": last_obs,
            "params": parameter_count(model),
            "checkpoint_bytes": checkpoint_size_bytes(ckpt),
            "train_seconds": train_s,
            "infer_ms": infer_ms,
            "mean_gates": test_metrics.get("mean_gates"),
            "enabled_concepts": list(spec.enabled_concepts),
            "use_hidden": spec.use_hidden,
            "aux": spec.aux,
            "use_kendall": spec.use_kendall,
            "dropout_was_trained": spec.dropout_was_trained,
            "lambda_rhythm": cfg.lambda_rhythm,
            "fusion_contract_version": FUSION_CONTRACT_VERSION,
            "fusion_input_mode": spec.fusion_input_mode,
        },
        n_valid_tags=test_metrics["n_valid_tags"],
        checkpoint_path=str(ckpt),
        extra={
            "note": "FIXTURE RUN — not a paper result" if cfg.fixture else "",
            "purpose": spec.purpose,
            "test_song_ids": test.song_ids,
            "n_train": len(train.song_ids),
            "n_val": len(val.song_ids),
            "n_test": len(test.song_ids),
            "threshold_split": "validation",
            "model_kind": spec.model_kind,
        },
    )
    rec.test_evaluated = True
    rec.split_used_for_selection = "validation"
    rec.write(cfg.out_dir / f"{spec.experiment_id}_seed{seed}.json")
    _ = val_logits, test_logits, fout
    return rec


def _infer_ms(model: nn.Module, split: FixtureSplit, spec: ExperimentSpec, *, warmup: int, steps: int) -> float:
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _forward(model, split, spec, train=False)
        t0 = time.perf_counter()
        for _ in range(max(1, steps)):
            _forward(model, split, spec, train=False)
    return (time.perf_counter() - t0) / max(1, steps) * 1000.0


def write_reports(out_dir: Path, *, faithfulness: dict[str, Any] | None = None) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmp = comparison_table(out_dir)
    summary = mean_std_table(out_dir)
    paths = {
        "comparison_csv": out_dir / "comparison.csv",
        "summary_csv": out_dir / "summary.csv",
        "comparison_md": out_dir / "comparison.md",
        "summary_md": out_dir / "summary.md",
        "report": out_dir / "REPORT.md",
    }
    cmp.to_csv(paths["comparison_csv"], index=False)
    summary.to_csv(paths["summary_csv"], index=False)
    paths["comparison_md"].write_text(dataframe_markdown(cmp), encoding="utf-8")
    paths["summary_md"].write_text(dataframe_markdown(summary), encoding="utf-8")
    if faithfulness is not None:
        (out_dir / "faithfulness.json").write_text(json.dumps(faithfulness, indent=2), encoding="utf-8")
    lines = [
        "# Concept fusion run report",
        "",
        "**These numbers are fixture runs unless `fixture` is false. Do not put them in the paper.**",
        "",
        "## Mean ± std (from stored JSON)",
        "",
        dataframe_markdown(summary),
        "",
        "## Every seed",
        "",
        dataframe_markdown(cmp),
        "",
    ]
    if faithfulness is not None:
        lines += ["## Faithfulness (F-Gated, dropout required)", "", "```json", json.dumps(faithfulness, indent=2), "```", ""]
    paths["report"].write_text("\n".join(lines), encoding="utf-8")
    return paths


def run_faithfulness(spec: ExperimentSpec, seed: int, cohort: FixtureCohort, cfg: PipelineConfig) -> dict[str, Any]:
    if not spec.dropout_was_trained:
        raise RuntimeError("refusing faithfulness without concept dropout")
    ckpt_path = cfg.out_dir / f"{spec.experiment_id}_seed{seed}.pt"
    model = _build_model(spec)
    try:
        blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        blob = torch.load(ckpt_path, map_location="cpu")
    validate_fusion_checkpoint_metadata(blob, expected_input_mode=spec.fusion_input_mode)
    model.load_state_dict(blob["model"])
    test = _prepare_split(cohort.test, spec)
    tokens = model.assemble_tokens(test.bundle)
    mask = test.bundle.fusion_mask()
    occ = occlude_each_concept(model, tokens, mask, dropout_was_trained=True)
    model.eval()
    with torch.no_grad():
        _, fout = model(tokens, mask, apply_dropout=False)
    corr = gate_vs_occlusion_correlation(fout.gates, occ)
    return {
        "experiment_id": spec.experiment_id,
        "seed": seed,
        "dropout_was_trained": True,
        "n_test": len(test.song_ids),
        "per_concept": {c: {"abs_mean_delta": occ[c].abs_mean_delta} for c in occ},
        **corr,
    }


def run_all(
    *,
    quick: bool = False,
    out_dir: Path | str = "results/proposed/mock",
    steps: int | None = None,
    lambda_rhythm: float = 1.0,
) -> list[RunRecord]:
    cfg = PipelineConfig(out_dir=Path(out_dir), lambda_rhythm=lambda_rhythm)
    if quick:
        cfg.steps = 5 if steps is None else steps
        cfg.batch_size = 8
        cfg.n_train, cfg.n_val, cfg.n_test = 16, 8, 8
        cfg.warmup, cfg.infer_steps = 1, 3
    elif steps is not None:
        cfg.steps = steps

    cohort = make_cohort(n_train=cfg.n_train, n_val=cfg.n_val, n_test=cfg.n_test, seed=cfg.cohort_seed)
    records: list[RunRecord] = []
    jobs = jobs_for(quick=quick)
    for spec, seed in jobs:
        rec = train_one(spec, seed, cohort, cfg)
        records.append(rec)
        print(
            f"{spec.experiment_id} seed={seed} val_ap={rec.metrics.get('val_macro_ap'):.4f} "
            f"test_ap={rec.metrics.get('test_macro_ap'):.4f} params={rec.metrics.get('params')}"
        )

    faith = None
    gated = next(s for s in experiment_specs() if s.experiment_id == "F-Gated")
    try:
        faith = run_faithfulness(gated, 0, cohort, cfg)
    except Exception as exc:  # noqa: BLE001 — report, do not hide a failed matrix
        faith = {"error": str(exc)}

    write_reports(cfg.out_dir, faithfulness=faith)
    (cfg.out_dir / "cohort_ids.json").write_text(
        json.dumps(
            {
                "train": cohort.train.song_ids,
                "val": cohort.val.song_ids,
                "test": cohort.test.song_ids,
                "note": "same test IDs for every experiment in this run",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("wrote reports under", cfg.out_dir)
    return records


def run_single(experiment_id: str, seed: int = 0, **kwargs: Any) -> RunRecord:
    spec = next(s for s in experiment_specs() if s.experiment_id == experiment_id)
    cfg = PipelineConfig(**{k: v for k, v in kwargs.items() if k in PipelineConfig.__dataclass_fields__})
    if "out_dir" in kwargs:
        cfg.out_dir = Path(kwargs["out_dir"])
    if "steps" in kwargs:
        cfg.steps = int(kwargs["steps"])
    cohort = make_cohort(n_train=cfg.n_train, n_val=cfg.n_val, n_test=cfg.n_test, seed=cfg.cohort_seed)
    rec = train_one(spec, seed, cohort, cfg)
    write_reports(cfg.out_dir)
    return rec
