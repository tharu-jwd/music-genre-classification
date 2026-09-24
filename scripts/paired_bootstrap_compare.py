"""Compare two models from saved validation predictions without using a GPU.

Both NPZ files must contain:

- ``song_ids``: one unique string per row;
- ``label_names``: the fixed multi-label vocabulary;
- ``targets``: binary array shaped ``songs × labels``;
- ``scores``: probability or ranking-score array with the same shape.

The script aligns rows by song ID, verifies identical targets and label order, then
bootstraps the paired difference in macro PR-AUC or ROC-AUC.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


REQUIRED_KEYS = {"song_ids", "label_names", "targets", "scores"}
SCHEMA_VERSION = "paired_bootstrap_comparison_v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_predictions(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        missing = REQUIRED_KEYS - set(data.files)
        if missing:
            raise ValueError(f"{path} is missing NPZ keys: {sorted(missing)}")
        result = {key: np.asarray(data[key]) for key in REQUIRED_KEYS}

    song_ids = result["song_ids"].astype(str)
    label_names = result["label_names"].astype(str)
    targets = result["targets"]
    scores = result["scores"]
    if targets.ndim != 2 or scores.shape != targets.shape:
        raise ValueError(f"{path} targets/scores must share shape songs × labels")
    if len(song_ids) != targets.shape[0] or len(label_names) != targets.shape[1]:
        raise ValueError(f"{path} IDs or labels do not match prediction dimensions")
    if len(set(song_ids.tolist())) != len(song_ids):
        raise ValueError(f"{path} contains duplicate song IDs")
    if not np.isin(targets, [0, 1]).all():
        raise ValueError(f"{path} targets must be binary")
    if not np.isfinite(scores).all():
        raise ValueError(f"{path} scores contain non-finite values")
    return {
        "song_ids": song_ids,
        "label_names": label_names,
        "targets": targets.astype(np.int8, copy=False),
        "scores": scores.astype(np.float64, copy=False),
    }


def align_pair(
    control: dict[str, np.ndarray], candidate: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not np.array_equal(control["label_names"], candidate["label_names"]):
        raise ValueError("Control and candidate label_names differ")

    candidate_rows = {sid: i for i, sid in enumerate(candidate["song_ids"].tolist())}
    if set(control["song_ids"].tolist()) != set(candidate_rows):
        raise ValueError("Control and candidate song-ID cohorts differ")
    order = np.asarray([candidate_rows[sid] for sid in control["song_ids"]], dtype=np.int64)
    candidate_targets = candidate["targets"][order]
    if not np.array_equal(control["targets"], candidate_targets):
        raise ValueError("Control and candidate targets differ after song-ID alignment")
    return (
        control["song_ids"],
        control["targets"],
        control["scores"],
        candidate["scores"][order],
    )


def macro_metric(targets: np.ndarray, scores: np.ndarray, metric: str) -> float:
    values: list[float] = []
    for column in range(targets.shape[1]):
        y_true = targets[:, column]
        if np.unique(y_true).size < 2:
            continue
        if metric == "macro_pr_auc":
            value = average_precision_score(y_true, scores[:, column])
        elif metric == "macro_roc_auc":
            value = roc_auc_score(y_true, scores[:, column])
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        if np.isfinite(value):
            values.append(float(value))
    if not values:
        raise ValueError("No labels have both positive and negative validation examples")
    return float(np.mean(values))


def compare(
    targets: np.ndarray,
    control_scores: np.ndarray,
    candidate_scores: np.ndarray,
    *,
    metric: str,
    confidence: float,
    n_resamples: int,
    seed: int,
    min_effect: float,
) -> dict[str, Any]:
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if n_resamples < 100:
        raise ValueError("n_resamples must be at least 100")
    if min_effect < 0:
        raise ValueError("min_effect must be non-negative")

    control_metric = macro_metric(targets, control_scores, metric)
    candidate_metric = macro_metric(targets, candidate_scores, metric)
    point_difference = candidate_metric - control_metric

    rng = np.random.default_rng(seed)
    differences = np.empty(n_resamples, dtype=np.float64)
    completed = 0
    attempts = 0
    max_attempts = n_resamples * 10
    while completed < n_resamples and attempts < max_attempts:
        attempts += 1
        rows = rng.integers(0, len(targets), size=len(targets))
        try:
            control_value = macro_metric(targets[rows], control_scores[rows], metric)
            candidate_value = macro_metric(targets[rows], candidate_scores[rows], metric)
        except ValueError:
            continue
        differences[completed] = candidate_value - control_value
        completed += 1
    if completed < n_resamples:
        raise ValueError("Too few valid bootstrap resamples; validation cohort is too sparse")

    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(differences, [alpha, 1.0 - alpha]).tolist()
    if point_difference >= min_effect and lower > 0:
        decision = "advance"
    elif upper < min_effect:
        decision = "stop"
    else:
        decision = "ambiguous"

    return {
        "schema_version": SCHEMA_VERSION,
        "metric": metric,
        "control_metric": control_metric,
        "candidate_metric": candidate_metric,
        "point_difference": point_difference,
        "confidence": confidence,
        "interval": [float(lower), float(upper)],
        "min_effect": min_effect,
        "n_resamples": n_resamples,
        "seed": seed,
        "decision": decision,
        "decision_rule": {
            "advance": "point_difference >= min_effect and interval lower bound > 0",
            "stop": "interval upper bound < min_effect",
            "ambiguous": "otherwise; permits at most one targeted repeat",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--comparison-id", required=True)
    parser.add_argument("--control-run-id", required=True)
    parser.add_argument("--candidate-run-id", required=True)
    parser.add_argument("--metric", required=True, choices=["macro_pr_auc", "macro_roc_auc"])
    parser.add_argument("--min-effect", required=True, type=float)
    parser.add_argument("--confidence", required=True, type=float)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--n-resamples", type=int, default=2000)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output and args.output.exists():
        raise FileExistsError("output already exists; paired decisions are immutable")
    for field in ("comparison_id", "control_run_id", "candidate_run_id"):
        if not str(getattr(args, field)).strip():
            raise ValueError(f"{field} must be non-empty")
    if args.control_run_id == args.candidate_run_id:
        raise ValueError("control-run-id and candidate-run-id must differ")
    control = load_predictions(args.control)
    candidate = load_predictions(args.candidate)
    song_ids, targets, control_scores, candidate_scores = align_pair(control, candidate)
    result = compare(
        targets,
        control_scores,
        candidate_scores,
        metric=args.metric,
        confidence=args.confidence,
        n_resamples=args.n_resamples,
        seed=args.seed,
        min_effect=args.min_effect,
    )
    result.update(
        {
            "comparison_id": args.comparison_id,
            "control_run_id": args.control_run_id,
            "candidate_run_id": args.candidate_run_id,
            "control": str(args.control.resolve()),
            "control_sha256": _sha256_file(args.control),
            "candidate": str(args.candidate.resolve()),
            "candidate_sha256": _sha256_file(args.candidate),
            "n_songs": int(len(song_ids)),
            "n_labels": int(targets.shape[1]),
        }
    )
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
