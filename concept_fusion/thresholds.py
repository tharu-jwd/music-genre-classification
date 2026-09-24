"""Validation-only per-tag F1 thresholds. Test labels must never enter this module's callers."""

from __future__ import annotations

import numpy as np

from concept_fusion.contract import GLOBAL_THRESHOLD_FALLBACK, SUPPORT_FLOOR
from concept_fusion.metrics import _as_numpy, valid_tag_indices


def _best_f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    # Unique scores as candidate thresholds.
    cand = np.unique(np.concatenate([p, np.array([0.0, 1.0, 0.5])]))
    best_t, best_f1 = GLOBAL_THRESHOLD_FALLBACK, -1.0
    for t in cand:
        pred = p >= t
        tp = float((pred & (y == 1)).sum())
        fp = float((pred & (y == 0)).sum())
        fn = float((~pred & (y == 1)).sum())
        prec = tp / (tp + fp + 1e-12)
        rec = tp / (tp + fn + 1e-12)
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def fit_thresholds(
    y_true_val,
    y_prob_val,
    *,
    support_floor: int = SUPPORT_FLOOR,
    global_fallback: float = GLOBAL_THRESHOLD_FALLBACK,
) -> dict:
    """Fit on validation only. Returns per-tag thresholds and a global fallback."""
    y = _as_numpy(y_true_val).astype(np.int32)
    p = _as_numpy(y_prob_val).astype(np.float64)
    valid = valid_tag_indices(y)
    thr = np.full(y.shape[1], global_fallback, dtype=np.float64)
    used_global = []
    for k in range(y.shape[1]):
        support = int(y[:, k].sum())
        if k not in valid or support < support_floor:
            used_global.append(k)
            continue
        thr[k] = _best_f1_threshold(y[:, k], p[:, k])
    return {
        "per_tag": thr.tolist(),
        "global_fallback": global_fallback,
        "support_floor": support_floor,
        "n_global_fallback_tags": len(used_global),
        "fallback_tag_indices": used_global,
        "n_valid_tags": len(valid),
    }


def apply_thresholds(y_prob, spec: dict) -> np.ndarray:
    p = _as_numpy(y_prob)
    thr = np.asarray(spec["per_tag"], dtype=np.float64)
    return (p >= thr.reshape(1, -1)).astype(np.int32)


def f1_precision_recall(y_true, y_pred) -> dict:
    y = _as_numpy(y_true).astype(np.int32)
    pred = _as_numpy(y_pred).astype(np.int32)
    valid = valid_tag_indices(y)
    f1s, ps, rs = [], [], []
    for k in valid:
        tp = float(((pred[:, k] == 1) & (y[:, k] == 1)).sum())
        fp = float(((pred[:, k] == 1) & (y[:, k] == 0)).sum())
        fn = float(((pred[:, k] == 0) & (y[:, k] == 1)).sum())
        prec = tp / (tp + fp + 1e-12)
        rec = tp / (tp + fn + 1e-12)
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
        f1s.append(f1)
        ps.append(prec)
        rs.append(rec)
    tp = float(((pred == 1) & (y == 1)).sum())
    fp = float(((pred == 1) & (y == 0)).sum())
    fn = float(((pred == 0) & (y == 1)).sum())
    micro_p = tp / (tp + fp + 1e-12)
    micro_r = tp / (tp + fn + 1e-12)
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r + 1e-12)
    return {
        "macro_f1": float(np.mean(f1s)) if f1s else float("nan"),
        "macro_precision": float(np.mean(ps)) if ps else float("nan"),
        "macro_recall": float(np.mean(rs)) if rs else float("nan"),
        "micro_f1": float(micro_f1),
        "micro_precision": float(micro_p),
        "micro_recall": float(micro_r),
        "n_valid_tags": len(valid),
    }
