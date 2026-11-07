"""One metric suite for every method.

Input: a list of per-question `records` (dicts). Required keys:

    qid, coverage, precision, f1, hit_ctx, cost
Optional keys (used when present):
    n_triples, n_tokens, confidence, latency_ms, chosen_k, las_hit1,
    llm_hit, llm_f1, llm_hit1, hops, n_gold, n_candidates, answer_in_subgraph

Metric families (names used in tables)
--------------------------------------
Retrieval quality  : cov (answer recall), prec, f1 (entity-level, GNN-RAG style),
                     hit_ctx (any gold entity inside selected context)
Cost               : cost (#paths), n_triples, n_tokens, latency_ms (mean/p50/p95)
Calibration        : ece, ece_adaptive, mce, brier (vs hit_ctx), brier_recall (vs coverage),
                     nll, aurc, sel_acc@80, sel_acc@50, overconf_rate
Answer-level       : las_hit1 (PRISM student top-1), llm_hit / llm_f1 / llm_hit1 (reader)
Uncertainty        : bootstrap 95% CI for f1, hit_ctx, llm_hit
Oracle ceilings    : answer_in_subgraph rate (upper bound on cov for ANY selector)
Stratified         : by hops / n_gold / n_candidates
Paired comparison  : paired bootstrap + McNemar (significance.py)
Multi-seed         : mean ± std across seeds
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np

from prism.metrics import expected_calibration_error, brier_score
from prism.metrics_full import (adaptive_ece, maximum_calibration_error,
                                negative_log_likelihood, area_under_risk_coverage,
                                selective_accuracy_at_coverage, bootstrap_ci)
from significance import paired_bootstrap, mcnemar


def _mean(xs: Sequence[float]) -> float:
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else float("nan")


def _pct(xs: Sequence[float], p: float) -> float:
    xs = [x for x in xs if x is not None]
    return float(np.percentile(xs, p)) if xs else float("nan")


def calibration_block(confidences: Sequence[float], correct: Sequence[int],
                      recall: Optional[Sequence[float]] = None) -> Dict[str, float]:
    confs = [float(c) for c in confidences]
    corr = [int(c) for c in correct]
    if not confs:
        return {}
    out = {
        "ece": expected_calibration_error(confs, corr),
        "ece_adaptive": adaptive_ece(confs, corr),
        "mce": maximum_calibration_error(confs, corr),
        "brier": brier_score(confs, corr),
        "nll": negative_log_likelihood(confs, corr),
        "aurc": area_under_risk_coverage(confs, corr),
        "sel_acc@80": selective_accuracy_at_coverage(confs, corr, 0.8),
        "sel_acc@50": selective_accuracy_at_coverage(confs, corr, 0.5),
        "mean_conf": float(np.mean(confs)),
        "overconf_rate": float(np.mean([(c > 0.7 and y == 0) for c, y in zip(confs, corr)])),
    }
    if recall is not None:
        out["brier_recall"] = float(np.mean((np.array(confs) - np.array(recall, dtype=float)) ** 2))
    return out


def summarize_records(records: List[Dict], n_bootstrap: int = 1000, seed: int = 0) -> Dict:
    """Aggregate per-question records into one metric dict."""
    if not records:
        return {"n": 0}
    g = lambda k: [r.get(k) for r in records]
    out: Dict[str, float] = {
        "n": len(records),
        "cov": _mean(g("coverage")),
        "prec": _mean(g("precision")),
        "f1": _mean(g("f1")),
        "hit_ctx": _mean(g("hit_ctx")),
        "cost": _mean(g("cost")),
        "cost_p50": _pct(g("cost"), 50),
        "cost_p95": _pct(g("cost"), 95),
        "n_triples": _mean(g("n_triples")),
        "n_tokens": _mean(g("n_tokens")),
        "latency_ms": _mean(g("latency_ms")),
        "latency_p95_ms": _pct(g("latency_ms"), 95),
        "chosen_k": _mean(g("chosen_k")),
        "empty_rate": float(np.mean([1 if (r.get("cost") or 0) == 0 else 0 for r in records])),
    }
    # oracle ceiling
    ais = [r.get("answer_in_subgraph") for r in records if r.get("answer_in_subgraph") is not None]
    if ais:
        out["answer_in_subgraph"] = float(np.mean(ais))
        out["cov_given_reachable"] = _mean([r["coverage"] for r in records if r.get("answer_in_subgraph")])
    # answer-level
    for k in ("las_hit1", "llm_hit", "llm_f1", "llm_hit1", "llm_acc", "llm_unsupported_rate",
              "n_tokens_prompt", "mean_step_kl", "n_steps"):
        vals = [r[k] for r in records if r.get(k) is not None]
        if vals:
            out[k] = float(np.mean(vals))
    # calibration (vs hit_ctx; PRISM/CGAR confidence heads predict answer recall)
    confs = [r["confidence"] for r in records if r.get("confidence") is not None]
    if len(confs) == len(records):
        out.update(calibration_block(confs, g("hit_ctx"), g("coverage")))
        llm = [r.get("llm_hit") for r in records]
        if all(v is not None for v in llm):
            cb = calibration_block(confs, llm)
            out.update({f"{k}_llm": v for k, v in cb.items() if k in ("ece", "brier", "aurc", "sel_acc@80")})
    # bootstrap CIs
    for k, src in (("f1", "f1"), ("hit_ctx", "hit_ctx"), ("llm_hit", "llm_hit")):
        vals = [r[src] for r in records if r.get(src) is not None]
        if len(vals) == len(records) and vals:
            _, lo, hi = bootstrap_ci(vals, np.mean, n_bootstrap=n_bootstrap, seed=seed)
            out[f"{k}_ci_low"], out[f"{k}_ci_high"] = lo, hi
    return out


def stratify_records(records: List[Dict], key: str = "hops",
                     bins: Optional[Dict] = None) -> Dict[str, Dict]:
    """Group by a record field (hops, n_gold, n_candidates, answer_in_subgraph)
    and summarise each group. `bins` maps raw value → label (e.g. {3: '3+'})."""
    groups = defaultdict(list)
    for r in records:
        v = r.get(key)
        if bins and v in bins:
            v = bins[v]
        groups[str(v)].append(r)
    return {k: summarize_records(v, n_bootstrap=200) for k, v in sorted(groups.items())}


def compare_records(records_a: List[Dict], records_b: List[Dict], metric: str = "f1",
                    n_bootstrap: int = 10000, seed: int = 42) -> Dict:
    """Paired comparison of two methods on common qids (A − B)."""
    by_a = {r["qid"]: r for r in records_a}
    by_b = {r["qid"]: r for r in records_b}
    qids = sorted(set(by_a) & set(by_b))
    a = [by_a[q][metric] for q in qids if by_a[q].get(metric) is not None and by_b[q].get(metric) is not None]
    b = [by_b[q][metric] for q in qids if by_a[q].get(metric) is not None and by_b[q].get(metric) is not None]
    if not a:
        return {"n": 0}
    res = {"metric": metric, "n": len(a),
           "bootstrap": paired_bootstrap(a, b, n_bootstrap=n_bootstrap, seed=seed)}
    if set(a) | set(b) <= {0, 1, 0.0, 1.0}:
        res["mcnemar"] = mcnemar([int(x) for x in a], [int(x) for x in b])
    res["significant_0.05"] = res["bootstrap"]["p_value"] < 0.05
    return res


def aggregate_seeds(per_seed: Dict[int, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """{seed: summary} → {metric: {mean, std, n}}."""
    if not per_seed:
        return {}
    keys = set().union(*[set(v) for v in per_seed.values()])
    out = {}
    for k in sorted(keys):
        vals = [v[k] for v in per_seed.values() if isinstance(v.get(k), (int, float)) and not np.isnan(v[k])]
        if not vals:
            continue
        out[k] = {"mean": float(np.mean(vals)),
                  "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                  "n": len(vals)}
    return out


def pareto_frontier(points: List[Dict], cost_key: str = "cost", quality_key: str = "f1") -> List[Dict]:
    pts = sorted(points, key=lambda p: (p[cost_key], -p[quality_key]))
    frontier, best = [], -float("inf")
    for p in pts:
        if p[quality_key] > best:
            frontier.append(p); best = p[quality_key]
    return frontier


def matched_cost_table(summaries: Dict[str, Dict], target_method: str,
                       tol: float = 0.5) -> List[Dict]:
    """Rows of baselines whose mean cost is within `tol` paths of the target
    method's cost — the fair 'same budget' comparison reviewers ask for."""
    if target_method not in summaries:
        return []
    c = summaries[target_method]["cost"]
    rows = []
    for m, s in summaries.items():
        if abs(s.get("cost", float("nan")) - c) <= tol:
            rows.append({"method": m, **{k: s.get(k) for k in ("cost", "f1", "cov", "prec", "hit_ctx")}})
    return rows
