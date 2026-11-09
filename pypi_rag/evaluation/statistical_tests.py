"""Paired comparisons on IDENTICAL sample ids (plan 6.2 / 12.5): paired bootstrap of metric differences
(including non-additive F1) and McNemar's exact test on per-sample correctness.
`paired_bootstrap` and `mcnemar` follow graph_rag/significance.py.

With `groups` (sample -> group id, e.g. meta.group_id of the grouped split, or the org entity of a QA
question) the bootstrap resamples whole GROUPS (cluster bootstrap): near-duplicate packages or questions
about the same entity are not independent, and resampling them one by one would make the CI too narrow."""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np

from .classification import classification_metrics


def _members(groups: Optional[Sequence], n: int) -> Optional[List[np.ndarray]]:
    """Index arrays per group, or None when every sample is its own group."""
    if groups is None:
        return None
    by: Dict = {}
    for i, g in enumerate(groups):
        by.setdefault(g, []).append(i)
    if len(by) >= n:
        return None
    return [np.asarray(v) for v in by.values()]


def _draw(rng, n: int, members: Optional[List[np.ndarray]]) -> np.ndarray:
    if members is None:
        return rng.integers(0, n, n)
    pick = rng.integers(0, len(members), len(members))
    return np.concatenate([members[j] for j in pick])


def paired_bootstrap(scores_a, scores_b, n_bootstrap: int = 10000, seed: int = 42, groups: Optional[Sequence] = None
                     ) -> Dict:
    a = np.asarray(scores_a, dtype=float); b = np.asarray(scores_b, dtype=float)
    assert a.shape == b.shape, "paired arrays must align"
    n = len(a)
    if n == 0:
        return {"delta": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "p_value": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    members = _members(groups, n)
    if members is None:
        idx = rng.integers(0, n, size=(n_bootstrap, n))
        deltas = a[idx].mean(1) - b[idx].mean(1)
    else:
        deltas = np.empty(n_bootstrap)
        for i in range(n_bootstrap):
            ix = _draw(rng, n, members)
            deltas[i] = a[ix].mean() - b[ix].mean()
    d = float(a.mean() - b.mean())
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p = (deltas <= 0).mean() if d >= 0 else (deltas >= 0).mean()
    return {"mean_a": float(a.mean()), "mean_b": float(b.mean()), "delta": d, "ci_low": float(lo),
            "ci_high": float(hi), "p_value": float(min(2 * p, 1.0)), "n": int(n),
            "resample_unit": "group" if members is not None else "sample",
            "n_groups": len(members) if members is not None else int(n)}


def mcnemar(hits_a, hits_b) -> Dict:
    a = np.asarray(hits_a, dtype=int); b = np.asarray(hits_b, dtype=int)
    n10 = int(((a == 1) & (b == 0)).sum()); n01 = int(((a == 0) & (b == 1)).sum())
    n = n10 + n01
    if n == 0:
        p = 1.0
    else:
        k = min(n10, n01)
        p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"n10_a_only": n10, "n01_b_only": n01, "p_value": float(p)}


def paired_f1_bootstrap(y: List[str], pa: List[str], pb: List[str], key=("strict", "f1_malicious"),
                        n_bootstrap: int = 2000, seed: int = 42, groups: Optional[Sequence] = None) -> Dict:
    n = len(y)
    if n == 0:
        return {"delta": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "p_value": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    members = _members(groups, n)
    f = lambda yy, pp: classification_metrics(yy, pp)[key[0]][key[1]]  # noqa: E731
    d_obs = f(y, pa) - f(y, pb)
    ds = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = _draw(rng, n, members)
        yy = [y[j] for j in idx]
        ds[i] = f(yy, [pa[j] for j in idx]) - f(yy, [pb[j] for j in idx])
    lo, hi = np.percentile(ds, [2.5, 97.5])
    p = (ds <= 0).mean() if d_obs >= 0 else (ds >= 0).mean()
    return {"metric": "/".join(key), "delta": float(d_obs), "ci_low": float(lo), "ci_high": float(hi),
            "p_value": float(min(2 * p, 1.0)), "n": n, "resample_unit": "group" if members is not None else "sample"}


def compare_methods(y: Dict[str, str], pred_a: Dict[str, str], pred_b: Dict[str, str], n_bootstrap: int = 2000,
                    groups: Optional[Dict[str, object]] = None) -> Dict:
    ids = sorted(set(y) & set(pred_a) & set(pred_b))
    yy = [y[i] for i in ids]; pa = [pred_a[i] for i in ids]; pb = [pred_b[i] for i in ids]
    ca = [int(p == t) for p, t in zip(pa, yy)]; cb = [int(p == t) for p, t in zip(pb, yy)]
    g = [groups.get(i, i) for i in ids] if groups else None
    out = {"n": len(ids),
           "f1_malicious_strict": paired_f1_bootstrap(yy, pa, pb, ("strict", "f1_malicious"), n_bootstrap, groups=g),
           "macro_f1_strict": paired_f1_bootstrap(yy, pa, pb, ("strict", "macro_f1"), n_bootstrap, groups=g),
           "fpr_strict": paired_f1_bootstrap(yy, pa, pb, ("strict", "fpr"), n_bootstrap, groups=g),
           "accuracy": paired_bootstrap(ca, cb, n_bootstrap, groups=g), "mcnemar": mcnemar(ca, cb)}
    b = out["f1_malicious_strict"]
    out["significant_0.05"] = bool(b["ci_low"] > 0 or b["ci_high"] < 0)
    return out
