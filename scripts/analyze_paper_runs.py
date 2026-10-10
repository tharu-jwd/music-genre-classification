"""Produce tables, paired differences, bootstrap intervals and figures from saved predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from scripts import train_joint as j
from scripts.paper_artifacts import METRIC_VERSION, write_json
from scripts.paper_experiments import load_manifest, CONFIGURATIONS, SEEDS
from scripts.paired_bootstrap_compare import load_predictions, align_pair, macro_metric


def group_summary(frame, groups, metrics):
    rows = []
    for keys, group in frame.groupby(groups, sort=False, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(groups, keys))
        row["n_seeds"] = len(group)
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[metric + "_n_valid"] = int(values.notna().sum())
            row[metric + "_mean"] = float(values.mean()) if values.notna().any() else None
            row[metric + "_sd"] = float(values.std(ddof=1)) if values.notna().sum() > 1 else None
        rows.append(row)
    return pd.DataFrame(rows)


def paired_seed_average_bootstrap(pairs, *, n_resamples=2000, confidence=0.95, seed=2026):
    """Resample the same tracks across every seed/model; do not ensemble predictions."""
    if n_resamples < 100 or not 0 < confidence < 1:
        raise ValueError("Use at least 100 resamples and confidence in (0,1)")
    targets = pairs[0][0]
    if any(not np.array_equal(targets, pair[0]) for pair in pairs):
        raise ValueError("Paired seeds have different target rows")
    def difference(rows):
        return float(np.mean([macro_metric(y[rows], candidate[rows], "macro_pr_auc") -
                              macro_metric(y[rows], control[rows], "macro_pr_auc")
                              for y, control, candidate in pairs]))
    point = difference(np.arange(len(targets)))
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_resamples * 10):
        try:
            samples.append(difference(rng.integers(0, len(targets), len(targets))))
        except ValueError:
            continue
        if len(samples) == n_resamples:
            break
    if len(samples) != n_resamples:
        raise ValueError("Too few valid bootstrap resamples")
    alpha = (1 - confidence) / 2
    lower, upper = np.quantile(samples, [alpha, 1 - alpha])
    return dict(mean_seed_difference=point, lower=float(lower), upper=float(upper),
                confidence=confidence, n_resamples=n_resamples, n_seeds=len(pairs),
                interpretation="Track-sampling uncertainty conditional on fitted models; not training-seed uncertainty.",
                sampling_unit="track; assumes track independence, not artist-cluster resampling")


def analyze(manifest_path, *, split="test", out_dir="results/paper-analysis", allow_incomplete=False,
            n_resamples=0, plots=True, runs_root=None):
    manifest = load_manifest(manifest_path)
    destination = Path(out_dir)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("Analysis directory is not empty; choose a fresh --out-dir")
    rows, genres, concepts, features, missing = [], [], [], [], []
    predictions, reference_ids, reference_targets, reference_names = {}, None, None, None
    dataset_hashes = {}
    source_hashes = None
    for run in manifest["runs"]:
        directory = Path(runs_root) / run["run_id"] if runs_root else Path(run["config"]["out_dir"])
        path = directory / f"{split}_predictions.npz"
        result_path = directory / "results.json"
        if not path.exists() or not result_path.exists():
            missing.append(run["run_id"])
            continue
        metadata = json.loads(result_path.read_text(encoding="utf-8"))
        if source_hashes is None:
            source_hashes = metadata.get("code_fingerprints")
        elif metadata.get("code_fingerprints") != source_hashes:
            raise ValueError("Runs used different model/evaluation source versions")
        for name, fingerprint in metadata.get("dataset_fingerprints", {}).items():
            if name in dataset_hashes and dataset_hashes[name] != fingerprint["sha256"]:
                raise ValueError(f"Run datasets differ for {name}")
            dataset_hashes[name] = fingerprint["sha256"]
        if metadata.get("metric_version") != METRIC_VERSION or metadata["seed"] != run["seed"] \
                or tuple(metadata["branches"]) != tuple(run["config"]["branches"]):
            raise ValueError(f"Run metadata differs from manifest: {run['run_id']}")
        for key in ("epochs", "batch_size", "lr", "weight_decay", "max_windows", "window_frames",
                    "fusion", "genre_predictor", "genre_hidden_dim", "genre_dropout", "concept_dropout", "lambda_aux"):
            if metadata["config"].get(key) != run["config"][key]:
                raise ValueError(f"Run {run['run_id']} has a different {key}")
        evaluation_path = directory / f"evaluation_{split}.json"
        if evaluation_path.exists():
            evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
            from scripts.paper_artifacts import sha256
            if evaluation.get("prediction_sha256") is not None:
                if evaluation["prediction_sha256"] != sha256(path):
                    raise ValueError("Prediction file differs from evaluation report")
            elif not (directory / "best.pt").exists():
                raise ValueError("Older evaluation export needs best.pt; re-export evaluation with prediction hashes")
            if (directory / "best.pt").exists() and evaluation["checkpoint_sha256"] != sha256(directory / "best.pt"):
                raise ValueError("Evaluation belongs to a different checkpoint")
        else:
            evaluation = {}
        with np.load(path, allow_pickle=False) as artifact:
            if str(artifact["metric_version"]) != METRIC_VERSION:
                raise ValueError("Predictions use an older metric definition")
            data = {key: artifact[key] for key in artifact.files}
        loaded = load_predictions(path)
        if reference_ids is None:
            reference_ids = loaded["song_ids"]
            reference_targets, reference_names = loaded["targets"], loaded["label_names"]
        reference = dict(song_ids=reference_ids, targets=reference_targets, label_names=reference_names,
                         scores=np.zeros_like(reference_targets, dtype=float))
        _, truth, _, scores = align_pair(reference, loaded)
        metrics = j._multilabel_metrics(torch.from_numpy(scores.copy()), torch.from_numpy(truth.copy()),
                                       reference_names.tolist())
        identity = dict(run_id=run["run_id"], configuration=run["configuration"], seed=run["seed"])
        rows.append({**identity, **{key: metrics[key] for key in
                    ("macro_average_precision", "micro_average_precision", "macro_f1", "micro_f1")},
                     "parameter_count": metadata["parameter_count"], "n_tracks": len(truth),
                     "best_epoch": metadata["best_epoch"],
                     "gpu_name": metadata.get("environment", {}).get("gpu_name"),
                     "train_seconds": sum(epoch.get("train_seconds", 0) for epoch in metadata["log"]),
                     "validation_seconds": sum(epoch.get("validation_seconds", 0) for epoch in metadata["log"]),
                     "seconds_per_track": evaluation.get("seconds_per_track"),
                     "peak_gpu_bytes": evaluation.get("peak_gpu_bytes")})
        predictions[run["configuration"], run["seed"]] = (truth, scores)
        for name, values in metrics["per_tag"].items():
            genres.append({**identity, "genre": name, **values})
        for branch in run["config"]["branches"]:
            required = [f"{branch}_{key}" for key in ("predictions", "targets", "mask", "names", "baseline_predictions")]
            if any(key not in data for key in required):
                raise ValueError(f"Missing concept exports for {run['run_id']}/{branch}")
            for source, prediction_key in (("model", "predictions"), ("training-reference", "baseline_predictions")):
                predicted = torch.from_numpy(data[f"{branch}_{prediction_key}"].copy())
                target = torch.from_numpy(data[f"{branch}_targets"].copy())
                mask = torch.from_numpy(data[f"{branch}_mask"].copy()).bool() & torch.isfinite(predicted)
                names = data[f"{branch}_names"].tolist()
                if branch == "instrument":
                    result = j._multilabel_metrics(predicted, target, names, mask=mask)
                    aggregate = {key: result[key] for key in ("macro_average_precision", "macro_f1", "n_observed")}
                    per_feature = result["per_tag"]
                else:
                    result = j._masked_regression_metrics(predicted, target, mask, names)
                    aggregate = {key: result[key] for key in ("mae_standardized", "rmse_standardized", "macro_r2", "n_observed")}
                    per_feature = result["per_feature"]
                concepts.append({**identity, "branch": branch, "source": source, **aggregate})
                features.extend({**identity, "branch": branch, "source": source, "feature": name, **values}
                                for name, values in per_feature.items())
    if missing and not allow_incomplete:
        raise FileNotFoundError(f"Missing {len(missing)} runs: {missing}; use --allow-incomplete for exploratory output")
    if not rows:
        raise ValueError("No completed prediction exports found")
    runs = pd.DataFrame(rows)
    if (runs.groupby("configuration")["parameter_count"].nunique() > 1).any():
        raise ValueError("Parameter counts differ across seeds of the same configuration")
    capacity = runs.groupby("configuration")["parameter_count"].first()
    if "full" in capacity and "no-supervision" in capacity and capacity["full"] != capacity["no-supervision"]:
        raise ValueError("Full and no-supervision controls do not have matched capacity")
    summary = group_summary(runs, ["configuration"], ["macro_average_precision", "micro_average_precision",
                            "macro_f1", "micro_f1", "parameter_count", "train_seconds", "seconds_per_track"])
    genre_frame = pd.DataFrame(genres)
    genre_summary = group_summary(genre_frame, ["configuration", "genre"], ["average_precision", "f1"])
    differences, genre_differences, intervals = [], [], {}
    comparisons = [("concept-supervision", "no-supervision", "full"), ("full-vs-cnn", "cnn", "full"),
                   *((f"removal-{name}", f"without-{name}", "full") for name in j.CONCEPT_ORDER)]
    for comparison, control, candidate in comparisons:
        pairs = []
        for seed in SEEDS:
            if (control, seed) not in predictions or (candidate, seed) not in predictions:
                continue
            truth, control_scores = predictions[control, seed]
            _, candidate_scores = predictions[candidate, seed]
            pairs.append((truth, control_scores, candidate_scores))
            delta = macro_metric(truth, candidate_scores, "macro_pr_auc") - macro_metric(truth, control_scores, "macro_pr_auc")
            differences.append(dict(comparison=comparison, seed=seed, delta_macro_ap=delta))
            for index, name in enumerate(reference_names):
                candidate_ap = j._average_precision(torch.from_numpy(candidate_scores[:, index].copy()), torch.from_numpy(truth[:, index].copy()))
                control_ap = j._average_precision(torch.from_numpy(control_scores[:, index].copy()), torch.from_numpy(truth[:, index].copy()))
                genre_differences.append(dict(comparison=comparison, seed=seed, genre=str(name),
                    delta_ap=None if candidate_ap is None else candidate_ap - control_ap))
        if pairs and n_resamples:
            intervals[comparison] = paired_seed_average_bootstrap(pairs, n_resamples=n_resamples)
    destination.mkdir(parents=True, exist_ok=True)
    outputs = {"runs": runs, "summary": summary, "per_genre": genre_frame,
               "per_genre_summary": genre_summary, "concepts": pd.DataFrame(concepts),
               "concept_features": pd.DataFrame(features), "paired_differences": pd.DataFrame(differences),
               "per_genre_differences": pd.DataFrame(genre_differences)}
    if differences:
        outputs["paired_summary"] = group_summary(pd.DataFrame(differences), ["comparison"], ["delta_macro_ap"])
    if concepts:
        frame = pd.DataFrame(concepts)
        for key in ("macro_average_precision", "macro_f1", "mae_standardized", "rmse_standardized", "macro_r2"):
            if key not in frame:
                frame[key] = np.nan
        outputs["concept_summary"] = group_summary(frame, ["configuration", "branch", "source"],
                           ["macro_average_precision", "macro_f1", "mae_standardized", "rmse_standardized", "macro_r2"])
    for name, frame in outputs.items():
        frame.to_csv(destination / f"{name}.csv", index=False)
    lines = ["| Configuration | Seeds | Macro AP | Micro AP | Macro F1 | Parameters |",
             "|---|---:|---:|---:|---:|---:|"]
    def format_metric(row, key):
        mean, sd = row[key + "_mean"], row[key + "_sd"]
        if pd.isna(mean):
            return "—"
        return f"{mean:.4f} ± {sd:.4f}" if pd.notna(sd) else f"{mean:.4f} (SD unavailable)"
    for row in summary.to_dict("records"):
        lines.append(f"| {row['configuration']} | {row['n_seeds']} | {format_metric(row, 'macro_average_precision')} | "
                     f"{format_metric(row, 'micro_average_precision')} | {format_metric(row, 'macro_f1')} | {int(row['parameter_count_mean']):,} |")
    (destination / "main_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(destination / "analysis.json", dict(split=split, missing_runs=missing, exploratory=bool(missing),
        metric_version=METRIC_VERSION, bootstrap=intervals,
        notes=["Standard deviations describe training-seed variation; bootstrap intervals describe track-sampling variation.",
               "Branch-removal differences include capacity changes.", "F1 uses threshold 0.5.",
               "No classifier-weight causal interpretation or multiple-comparison significance claims."]))
    if plots:
        make_plots(destination, summary, genre_summary, pd.DataFrame(genre_differences))
    return outputs


def make_plots(destination, summary, genres, differences):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10})
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(summary))
    ax.bar(x, summary["macro_average_precision_mean"], yerr=summary["macro_average_precision_sd"].fillna(0), capsize=4)
    ax.set_xticks(x, summary["configuration"], rotation=25, ha="right")
    ax.set_ylabel("Macro average precision (mean ± seed SD)")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(destination / "genre_performance.pdf")
    fig.savefig(destination / "genre_performance.png", dpi=200)
    plt.close(fig)
    if not differences.empty:
        removals = differences[differences["comparison"].str.startswith("removal-")]
        if not removals.empty:
            matrix = removals.pivot_table(index="comparison", columns="genre", values="delta_ap")
            if matrix.empty:
                return
            fig, ax = plt.subplots(figsize=(9, 4))
            limit = max(float(np.nanmax(np.abs(matrix.to_numpy()))), 0.001)
            im = ax.imshow(matrix, cmap="RdBu", vmin=-limit, vmax=limit)
            ax.set_xticks(np.arange(len(matrix.columns)), matrix.columns, rotation=25, ha="right")
            ax.set_yticks(np.arange(len(matrix.index)), matrix.index)
            fig.colorbar(im, ax=ax, label="Full model AP − branch-removal AP")
            fig.tight_layout()
            fig.savefig(destination / "branch_removal_heatmap.pdf")
            fig.savefig(destination / "branch_removal_heatmap.png", dpi=200)
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("experiments/paper21.json"))
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, help="Downloaded run folders, overriding remote /runs paths")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--n-resamples", type=int, default=0)
    parser.add_argument("--no-plots", action="store_true")
    args = vars(parser.parse_args())
    args["plots"] = not args.pop("no_plots")
    manifest = args.pop("manifest")
    analyze(manifest, **args)


if __name__ == "__main__":
    main()
