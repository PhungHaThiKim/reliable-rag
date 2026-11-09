"""Classification metrics with an explicit policy for `uncertain` (plan 12.1).

  strict        uncertain counts as WRONG (malicious -> FN, benign -> FP): full-coverage metric
  conservative  uncertain is flagged for review, i.e. treated as malicious (security triage view)
  answered      only answered samples (report together with `coverage`; never alone)
Positive class = malicious. Unknown gold labels are excluded before calling.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def _prf(tp, fp, fn, tn) -> Dict[str, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    fnr = fn / (fn + tp) if fn + tp else 0.0
    tnr = 1 - fpr if fp + tn else 0.0
    pb = tn / (tn + fn) if tn + fn else 0.0
    rb = tnr
    f1b = 2 * pb * rb / (pb + rb) if pb + rb else 0.0
    n = tp + fp + fn + tn
    return {"n": int(n), "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "precision_malicious": p, "recall_malicious": r, "f1_malicious": f1,
            "precision_benign": pb, "recall_benign": rb, "f1_benign": f1b, "macro_f1": (f1 + f1b) / 2,
            "fpr": fpr, "fnr": fnr, "accuracy": (tp + tn) / n if n else 0.0, "balanced_accuracy": (r + tnr) / 2}


def confusion(y: Sequence[str], p: Sequence[str], policy: str) -> Dict[str, float]:
    tp = fp = fn = tn = 0
    for t, q in zip(y, p):
        if q == "uncertain":
            if policy == "answered":
                continue
            q = ("benign" if t == "malicious" else "malicious") if policy == "strict" else "malicious"
        if t == "malicious":
            tp += q == "malicious"; fn += q != "malicious"
        else:
            fp += q == "malicious"; tn += q != "malicious"
    return _prf(tp, fp, fn, tn)


def classification_metrics(y_true: Sequence[str], y_pred: Sequence[str]) -> Dict:
    n = len(y_true)
    unc = sum(1 for p in y_pred if p == "uncertain")
    return {"n": n, "coverage": 1 - unc / n if n else 0.0, "abstention_rate": unc / n if n else 0.0,
            "strict": confusion(y_true, y_pred, "strict"), "conservative": confusion(y_true, y_pred, "conservative"),
            "answered": confusion(y_true, y_pred, "answered")}


def selective_curve(conf: Sequence[float], correct: Sequence[int], n_total: int, n_points: int = 20,
                    points=(0.2, 0.4, 0.6, 0.8, 1.0)) -> Dict:
    """Risk-coverage over ALL n_total items: answered items (conf, correct) are ranked by confidence; coverage
    = answered so far / n_total (so a method that abstains ends its curve early), risk = error rate among them.
    aurc = mean risk along the answered ranking; risk_at[c] only for coverages the method reaches."""
    rows = sorted(((c if c is not None else 0.5), int(k)) for c, k in zip(conf, correct))[::-1]
    if not rows or not n_total:
        return {"aurc": float("nan"), "max_coverage": 0.0, "curve": [], "risk_at": {}}
    ok = np.array([r[1] for r in rows], dtype=float)
    risk = 1 - np.cumsum(ok) / np.arange(1, len(ok) + 1)
    cov = np.arange(1, len(ok) + 1) / n_total
    idx = sorted({int(round(x)) for x in np.linspace(0, len(ok) - 1, min(n_points, len(ok)))})
    return {"aurc": float(risk.mean()), "max_coverage": float(cov[-1]),
            "curve": [[float(cov[i]), float(risk[i])] for i in idx],
            "risk_at": {str(c): float(risk[int(np.searchsorted(cov, c - 1e-9))]) for c in points if c <= cov[-1] + 1e-9}}


def risk_coverage(y_true: Sequence[str], y_pred: Sequence[str], conf: Sequence[float], points=(0.2, 0.4, 0.6, 0.8, 1.0)) -> Dict:
    """Error rate among the most-confident ANSWERED predictions, coverage measured over all samples; AURC."""
    rows = [(c, int(p == t)) for t, p, c in zip(y_true, y_pred, conf) if p != "uncertain"]
    return selective_curve([r[0] for r in rows], [r[1] for r in rows], len(y_true), points=points)


def bootstrap_metric_ci(y_true: List[str], y_pred: List[str], key=("strict", "f1_malicious"), n_boot: int = 2000,
                        seed: int = 0) -> Dict:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    if n == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        m = classification_metrics([y_true[i] for i in idx], [y_pred[i] for i in idx])
        vals.append(m[key[0]][key[1]])
    point = classification_metrics(y_true, y_pred)[key[0]][key[1]]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"mean": float(point), "ci_low": float(lo), "ci_high": float(hi)}
