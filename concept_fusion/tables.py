"""Tables from stored run JSON — never hand-typed notebook numbers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from concept_fusion.run_schema import load_records


def comparison_table(results_dir: Path) -> pd.DataFrame:
    recs = load_records(results_dir)
    rows = []
    for r in recs:
        m = r.get("metrics") or {}
        rows.append(
            {
                "experiment_id": r.get("experiment_id"),
                "seed": r.get("seed"),
                "fusion": (r.get("config") or {}).get("fusion"),
                "macro_ap": m.get("macro_ap"),
                "macro_roc_auc": m.get("macro_roc_auc"),
                "micro_ap": m.get("micro_ap"),
                "n_valid_tags": r.get("n_valid_tags"),
                "config_hash": r.get("config_hash"),
                "git_sha": r.get("git_sha"),
                "test_evaluated": r.get("test_evaluated"),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["experiment_id", "seed"]).reset_index(drop=True)


def mean_std_table(results_dir: Path) -> pd.DataFrame:
    df = comparison_table(results_dir)
    if df.empty:
        return df
    g = df.groupby("experiment_id", as_index=False).agg(
        n_seeds=("seed", "nunique"),
        macro_ap_mean=("macro_ap", "mean"),
        macro_ap_std=("macro_ap", "std"),
        macro_roc_mean=("macro_roc_auc", "mean"),
        n_valid_tags=("n_valid_tags", "max"),
    )
    return g
