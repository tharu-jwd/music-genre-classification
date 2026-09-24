"""Ranking metrics. Macro AP is sklearn average_precision_score, not trapezoidal PR-AUC."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    auc as sk_auc,
)


@dataclass
class RankingMetrics:
    macro_ap: float
    micro_ap: float
    macro_roc_auc: float
    trapezoidal_macro_pr_auc: float
    n_valid_tags: int
    n_tags_total: int
    per_tag_ap: list[float | None]


def _as_numpy(y):
    if hasattr(y, "detach"):
        y = y.detach().cpu().numpy()
    return np.asarray(y)


def valid_tag_indices(y_true: np.ndarray) -> list[int]:
    """Tags that are not all-0 or all-1 (undefined ranking metrics)."""
    ok = []
    for k in range(y_true.shape[1]):
        s = float(y_true[:, k].sum())
        if s in (0.0, float(len(y_true))):
            continue
        ok.append(k)
    return ok


def trapezoidal_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    p, r, _ = precision_recall_curve(y_true, y_score)
    return float(sk_auc(r, p))


def ranking_metrics(y_true, y_prob) -> RankingMetrics:
    yt = _as_numpy(y_true).astype(np.float64)
    yp = _as_numpy(y_prob).astype(np.float64)
    if yt.shape != yp.shape:
        raise ValueError(f"shape mismatch {yt.shape} vs {yp.shape}")
    valid = valid_tag_indices(yt)
    per_tag: list[float | None] = [None] * yt.shape[1]
    aps, rocs, traps = [], [], []
    for k in valid:
        ap = float(average_precision_score(yt[:, k], yp[:, k]))
        per_tag[k] = ap
        aps.append(ap)
        try:
            rocs.append(float(roc_auc_score(yt[:, k], yp[:, k])))
        except ValueError:
            pass
        traps.append(trapezoidal_pr_auc(yt[:, k], yp[:, k]))
    micro = float("nan")
    if valid:
        cols = np.array(valid)
        micro = float(average_precision_score(yt[:, cols].ravel(), yp[:, cols].ravel()))
    return RankingMetrics(
        macro_ap=float(np.mean(aps)) if aps else float("nan"),
        micro_ap=micro,
        macro_roc_auc=float(np.mean(rocs)) if rocs else float("nan"),
        trapezoidal_macro_pr_auc=float(np.mean(traps)) if traps else float("nan"),
        n_valid_tags=len(valid),
        n_tags_total=int(yt.shape[1]),
        per_tag_ap=per_tag,
    )


def sigmoid_np(logits) -> np.ndarray:
    z = _as_numpy(logits)
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))
