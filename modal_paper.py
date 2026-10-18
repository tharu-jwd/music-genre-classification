# Modal entry point for the fixed 21-run paper protocol.
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modal_app import (app, train_remote, train_and_evaluate_remote, evaluate_remote, pilot_remote, paper_status_remote,
                       DATA_MOUNT, AUDIO_MOUNT, RUNS_MOUNT, RUNS_VOLUME_NAME, DATASET_FILENAME)


@app.local_entrypoint()
def paper(action: str = "plan", phase: str = "core", seed: int = 0,
          run_name: str = "", manifest_path: str = "experiments/paper21-modal.json",
          gpu: str = "A10", execute: bool = False, split: str = "test"):
    """Preview or explicitly execute the fixed paper protocol on Modal."""
    from scripts.paper_experiments import SEEDS, load_manifest, selected_runs
    if action not in ("plan", "train", "evaluate", "pilot", "run"):
        raise ValueError("action must be plan, train, evaluate, pilot or run")
    if action == "run" and (not run_name or split != "test"):
        raise ValueError("run requires --run-name and evaluates the test split")
    if phase not in ("core", "ablation", "all") or seed not in (0, *SEEDS):
        raise ValueError("Invalid phase or seed")
    if split not in ("validation", "test"):
        raise ValueError("Invalid evaluation split")
    if action == "pilot":
        if execute:
            print(json.dumps(pilot_remote.with_options(gpu=gpu).remote(run_name or "paper-pilot-v2"), indent=2))
        else:
            print("Preview: bounded Modal pilot; add --execute to allocate a GPU")
        return
    manifest = load_manifest(manifest_path)
    runs = selected_runs(manifest, phase=phase, seed=seed or None, run_id=run_name or None)
    statuses = paper_status_remote.remote([run["run_id"] for run in runs]) if execute and action != "plan" else {}
    for run in runs:
        print(json.dumps(dict(action=action, run_id=run["run_id"], configuration=run["configuration"],
                              seed=run["seed"], config=run["config"])), flush=True)
        if not execute or action == "plan":
            continue
        status = statuses[run["run_id"]]
        if status["state"] == "partial":
            raise RuntimeError(f"Partial run exists: {run['run_id']}. Inspect its artifacts; automatic overwriting/resume is disabled.")
        if status["state"] == "complete":
            from scripts.paper_artifacts import METRIC_VERSION
            if status["metric_version"] != METRIC_VERSION:
                raise ValueError("Completed run used an older metric definition")
            for key, expected_value in run["config"].items():
                if key == "device":
                    continue
                if status["config"].get(key) != expected_value:
                    raise ValueError(f"Completed run has different {key}: {run['run_id']}")
            if action == "train" or status["evaluations"].get(split):
                print(f"Skipping completed {action}: {run['run_id']}", flush=True)
                continue
        elif action == "evaluate":
            raise FileNotFoundError(f"Run has not completed training: {run['run_id']}")
        if action == "evaluate":
            result = evaluate_remote.with_options(gpu=gpu).remote(run["run_id"], split)
        else:
            cfg = run["config"]
            expected = dict(data_dir=str(DATA_MOUNT / "dataset"),
                            dataset_csv=str(DATA_MOUNT / "dataset" / "vector-dataset.csv"),
                            split_csv=str(DATA_MOUNT / "dataset" / "track_split_assignments.csv"),
                            logmel_root=str(AUDIO_MOUNT / "logmel_songs"),
                            out_dir=str(RUNS_MOUNT / run["run_id"]))
            if any(cfg[key] != value for key, value in expected.items()):
                raise ValueError("Manifest paths differ from this Modal runner's Volume layout")
            arguments = dict(
                run_name=run["run_id"], epochs=cfg["epochs"], batch_size=cfg["batch_size"],
                learning_rate=cfg["lr"], num_workers=cfg["num_workers"], max_windows=cfg["max_windows"],
                quick=False, skip_test=True, model="joint" if cfg["branches"] else "cnn",
                branches=",".join(cfg["branches"]), seed=run["seed"], fusion=cfg["fusion"],
                genre_predictor=cfg["genre_predictor"], genre_hidden_dim=cfg["genre_hidden_dim"],
                genre_dropout=cfg["genre_dropout"], concept_dropout=cfg["concept_dropout"],
                lambda_aux=cfg["lambda_aux"], window_frames=cfg["window_frames"], weight_decay=cfg["weight_decay"],
                architecture=cfg["architecture"],
                dataset_filename="vector-dataset.csv")
            if action == "run":
                call = train_and_evaluate_remote.with_options(gpu=gpu).spawn(
                    arguments, train_completed=status["state"] == "complete")
                result = dict(status="submitted", run_name=run["run_id"],
                              function_call_id=call.object_id,
                              results_volume=RUNS_VOLUME_NAME)
            else:
                result = train_remote.with_options(gpu=gpu).remote(**arguments)
        print(json.dumps(result, indent=2), flush=True)
