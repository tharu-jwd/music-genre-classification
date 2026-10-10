"""Plan the seven configurations and three seeds; execution is opt-in."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import train_joint as j
from scripts.paper_artifacts import METRIC_VERSION, write_json

SEEDS = (42, 123, 456)
CONFIGURATIONS = (
    ("cnn", (), 0.0, "core"),
    ("full", j.CONCEPT_ORDER, 1.0, "core"),
    ("no-supervision", j.CONCEPT_ORDER, 0.0, "core"),
    *((f"without-{name}", tuple(n for n in j.CONCEPT_ORDER if n != name), 1.0, "ablation")
      for name in ("instrument", "timbre", "rhythm", "harmony")),
)


def make_manifest(*, out_root="results/paper21", data_dir="data", dataset_csv=None,
                  split_csv=None, logmel_root=None, epochs=30, batch_size=1,
                  max_windows=16, window_frames=1366, device="cuda", num_workers=2):
    if epochs < 1 or batch_size < 1 or max_windows < 1 or window_frames < 1 or num_workers < 0:
        raise ValueError("Invalid training budget or geometry")
    runs = []
    for phase in ("core", "ablation"):
        for seed in SEEDS:
            for name, branches, weight, run_phase in CONFIGURATIONS:
                if phase != run_phase:
                    continue
                run_id = f"paper-v2-{name}-s{seed}"
                cfg = j.TrainConfig(
                    branches=branches, seed=seed, lambda_aux=weight, evaluate_test=False,
                    out_dir=(Path(out_root) / run_id).as_posix(), data_dir=Path(data_dir),
                    dataset_csv=Path(dataset_csv) if dataset_csv else None,
                    split_csv=Path(split_csv) if split_csv else None,
                    logmel_root=Path(logmel_root) if logmel_root else None,
                    epochs=epochs, batch_size=batch_size, max_windows=max_windows,
                    window_frames=window_frames, device=device, num_workers=num_workers)
                config = {key: value.as_posix() if isinstance(value, Path) else value for key, value in asdict(cfg).items()}
                runs.append(dict(run_id=run_id, configuration=name, phase=phase, seed=seed, config=config))
    return dict(schema_version="paper_experiments_v2", metric_version=METRIC_VERSION,
                seeds=list(SEEDS), primary_metric="macro_average_precision",
                threshold=0.5, runs=runs,
                analysis_notes=["Branch removals change capacity; they are not pure concept-importance estimates.",
                                "All test evaluations use validation-selected checkpoints."])


def load_manifest(path):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "paper_experiments_v2" or manifest.get("metric_version") != METRIC_VERSION:
        raise ValueError("Unsupported manifest or metric definition")
    if manifest.get("seeds") != list(SEEDS):
        raise ValueError("Manifest seed list differs from the study protocol")
    if len(manifest["runs"]) != 21 or len({run["run_id"] for run in manifest["runs"]}) != 21:
        raise ValueError("Manifest must contain 21 unique runs")
    expected = {(name, seed): (branches, weight, phase)
                for name, branches, weight, phase in CONFIGURATIONS for seed in SEEDS}
    actual = {(run["configuration"], run["seed"]) for run in manifest["runs"]}
    if actual != set(expected):
        raise ValueError("Manifest does not contain the declared configurations and seeds")
    budgets = set()
    destinations = set()
    for run in manifest["runs"]:
        cfg = run["config"]
        branches, weight, phase = expected[run["configuration"], run["seed"]]
        if tuple(cfg["branches"]) != branches or cfg["lambda_aux"] != weight or cfg["seed"] != run["seed"] \
                or run["phase"] != phase or cfg["evaluate_test"] or cfg["quick"]:
            raise ValueError("Manifest run differs from fixed experiment definition")
        if any(cfg[f"lambda_{name}"] is not None for name in j.CONCEPT_ORDER):
            raise ValueError("Paper protocol does not use branch-specific loss weights")
        budgets.add(tuple(str(cfg[key]) for key in
                          ("epochs", "batch_size", "lr", "weight_decay", "max_windows", "window_frames",
                           "data_dir", "dataset_csv", "split_csv", "logmel_root", "fusion", "genre_predictor",
                           "genre_hidden_dim", "genre_dropout", "concept_dropout")))
        destinations.add(cfg["out_dir"])
    if len(budgets) != 1 or len(destinations) != 21:
        raise ValueError("Training protocol must be shared and destinations unique")
    return manifest


def selected_runs(manifest, *, phase="all", seed=None, run_id=None):
    runs = [run for run in manifest["runs"] if (phase == "all" or run["phase"] == phase)
            and (seed is None or run["seed"] == seed) and (run_id is None or run["run_id"] == run_id)]
    if not runs:
        raise ValueError("Selection contains no runs")
    return runs


def train_command(run):
    cfg = run["config"]
    command = [sys.executable, str(ROOT / "scripts/train_joint.py"), "--branches",
               *(cfg["branches"] or ["none"]), "--skip-test"]
    keys = ("seed", "epochs", "batch_size", "lr", "weight_decay", "max_windows", "window_frames",
            "num_workers", "device", "out_dir", "data_dir", "dataset_csv", "split_csv", "logmel_root",
            "fusion", "genre_predictor", "genre_hidden_dim", "genre_dropout", "concept_dropout", "lambda_aux")
    for key in keys:
        if cfg[key] is not None:
            command.extend(["--" + key.replace("_", "-"), str(cfg[key])])
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--manifest", type=Path, default=Path("experiments/paper21.json"))
    for key, default in (("out-root", "results/paper21"), ("data-dir", "data"),
                         ("dataset-csv", "data/full_dataset.csv"), ("split-csv", None), ("logmel-root", None)):
        plan.add_argument("--" + key, default=default)
    for key, default in (("epochs", 30), ("batch-size", 1), ("max-windows", 16),
                         ("window-frames", 1366), ("num-workers", 2)):
        plan.add_argument("--" + key, type=int, default=default)
    plan.add_argument("--device", default="cuda")
    run = sub.add_parser("run")
    run.add_argument("--manifest", type=Path, default=Path("experiments/paper21.json"))
    run.add_argument("--phase", choices=("core", "ablation", "all"), default="core")
    run.add_argument("--seed", type=int, choices=SEEDS)
    run.add_argument("--run-id")
    run.add_argument("--execute", action="store_true")
    args = vars(parser.parse_args())
    action, path = args.pop("action"), args.pop("manifest")
    if action == "plan":
        if path.exists():
            raise FileExistsError(f"Manifest exists: {path}")
        write_json(path, make_manifest(**args))
        print(f"Prepared 21 runs: {path}")
    else:
        execute = args.pop("execute")
        for item in selected_runs(load_manifest(path), **args):
            command = train_command(item)
            print(json.dumps(dict(run_id=item["run_id"], command=command)))
            if execute:
                subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
