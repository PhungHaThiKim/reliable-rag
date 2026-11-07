"""Extended metric suite — every measurement PRISM paper reports.

Augments [metrics.py](metrics.py) with:
  - MCE (Maximum Calibration Error)
  - NLL (Negative Log-Likelihood) under predicted confidence
  - AURC (Area Under Risk-Coverage curve)
  - Selective accuracy at threshold
  - Cross-LLM agreement metrics
  - Bootstrap confidence intervals
  - Statistical significance tests (paired bootstrap, McNemar)

All functions accept numpy arrays or lists; return float / dict / pd.DataFrame.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .metrics import (brier_score, expected_calibration_error, f1_score,
                       hit_at_k, reliability_diagram)


# ─────────────────────────────────────────────────────────────────────
# Calibration metrics
# ─────────────────────────────────────────────────────────────────────

def maximum_calibration_error(confidences: Sequence[float],
                                correctness: Sequence[int],
                                n_bins: int = 10) -> float:
    """MCE: worst per-bin calibration gap. Stricter than ECE."""
    confs = np.asarray(confidences); corrs = np.asarray(correctness)
    bins = np.linspace(0, 1, n_bins + 1)
    max_gap = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confs >= lo) & (confs < hi if i < n_bins - 1 else confs <= hi)
        if mask.sum() == 0:
            continue
        gap = abs(corrs[mask].mean() - confs[mask].mean())
        max_gap = max(max_gap, gap)
    return float(max_gap)


def negative_log_likelihood(confidences: Sequence[float],
                              correctness: Sequence[int],
                              eps: float = 1e-9) -> float:
    """NLL of binary correctness under predicted confidence."""
    c = np.clip(np.asarray(confidences), eps, 1 - eps)
    y = np.asarray(correctness)
    return float(-(y * np.log(c) + (1 - y) * np.log(1 - c)).mean())


def adaptive_ece(confidences: Sequence[float],
                  correctness: Sequence[int],
                  n_bins: int = 10) -> float:
    """ECE with EQUAL-MASS bins (each bin contains same # samples).
    More robust to non-uniform confidence distributions than fixed-width ECE."""
    confs = np.asarray(confidences); corrs = np.asarray(correctness)
    n = len(confs)
    if n < n_bins:
        return expected_calibration_error(confidences, correctness, n_bins)
    sorted_idx = np.argsort(confs)
    edges = np.array_split(sorted_idx, n_bins)
    ece = 0.0
    for bin_idx in edges:
        if len(bin_idx) == 0:
            continue
        bin_acc = corrs[bin_idx].mean()
        bin_conf = confs[bin_idx].mean()
        ece += (len(bin_idx) / n) * abs(bin_acc - bin_conf)
    return float(ece)


def calibration_report(confidences: Sequence[float],
                         correctness: Sequence[int]) -> Dict[str, float]:
    """One-stop calibration metrics dict — paper Section 5.3 table."""
    return {
        "ece":          expected_calibration_error(confidences, correctness),
        "ece_adaptive": adaptive_ece(confidences, correctness),
        "mce":          maximum_calibration_error(confidences, correctness),
        "brier":        brier_score(confidences, correctness),
        "nll":          negative_log_likelihood(confidences, correctness),
        "n":            len(confidences),
        "mean_conf":    float(np.mean(confidences)),
        "mean_acc":     float(np.mean(correctness)),
        "conf_acc_gap": float(np.mean(confidences) - np.mean(correctness)),
    }


# ─────────────────────────────────────────────────────────────────────
# Selective prediction — risk/coverage
# ─────────────────────────────────────────────────────────────────────

def area_under_risk_coverage(confidences: Sequence[float],
                               correctness: Sequence[int]) -> float:
    """AURC — lower is better. Computes integral over coverage levels."""
    pairs = sorted(zip(confidences, correctness), key=lambda x: -x[0])
    n = len(pairs)
    if n == 0:
        return 0.0
    cum_wrong = 0
    auc = 0.0
    for i, (_, ok) in enumerate(pairs, 1):
        cum_wrong += (1 - int(ok))
        risk_at_i = cum_wrong / i
        auc += risk_at_i / n
    return float(auc)


def selective_accuracy_at_coverage(confidences: Sequence[float],
                                     correctness: Sequence[int],
                                     coverage: float = 0.8) -> float:
    """Accuracy on the top-`coverage` fraction by confidence."""
    pairs = sorted(zip(confidences, correctness), key=lambda x: -x[0])
    n = len(pairs)
    if n == 0:
        return 0.0
    keep = max(1, int(n * coverage))
    kept = [int(ok) for _, ok in pairs[:keep]]
    return float(np.mean(kept))


# ─────────────────────────────────────────────────────────────────────
# Bootstrap confidence intervals
# ─────────────────────────────────────────────────────────────────────

def bootstrap_ci(values: Sequence[float],
                  stat_fn=np.mean,
                  n_bootstrap: int = 1000,
                  ci: float = 0.95,
                  seed: int = 0) -> Tuple[float, float, float]:
    """Returns (point_estimate, lower, upper)."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n == 0:
        return 0.0, 0.0, 0.0
    samples = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        samples[b] = stat_fn(arr[idx])
    alpha = (1 - ci) / 2
    lo = float(np.quantile(samples, alpha))
    hi = float(np.quantile(samples, 1 - alpha))
    return float(stat_fn(arr)), lo, hi


def paired_bootstrap_diff(a: Sequence[float], b: Sequence[float],
                            n_bootstrap: int = 10000,
                            seed: int = 0) -> Dict[str, float]:
    """Paired bootstrap for mean(a) - mean(b). Returns CI + p-value."""
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    assert len(a_arr) == len(b_arr), "paired"
    n = len(a_arr)
    rng = np.random.default_rng(seed)
    obs = a_arr.mean() - b_arr.mean()
    deltas = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        deltas[i] = a_arr[idx].mean() - b_arr[idx].mean()
    ci_low, ci_high = np.quantile(deltas, [0.025, 0.975])
    p = (deltas >= 0).mean() if obs < 0 else (deltas <= 0).mean()
    return {
        "mean_a": float(a_arr.mean()),
        "mean_b": float(b_arr.mean()),
        "delta": float(obs),
        "ci_low": float(ci_low), "ci_high": float(ci_high),
        "p_value": float(min(2 * p, 1.0)),
        "n": int(n),
    }


def mcnemar_test(hits_a: Sequence[int], hits_b: Sequence[int]) -> Dict[str, float]:
    """Exact binomial McNemar's test on paired binary outcomes."""
    a = np.asarray(hits_a, dtype=int); b = np.asarray(hits_b, dtype=int)
    n10 = int(((a == 1) & (b == 0)).sum())
    n01 = int(((a == 0) & (b == 1)).sum())
    n11 = int(((a == 1) & (b == 1)).sum())
    n00 = int(((a == 0) & (b == 0)).sum())
    disc = n10 + n01
    if disc == 0:
        return {"n10": n10, "n01": n01, "p_value": 1.0,
                "method": "no_discordant"}
    from math import comb
    k = max(n10, n01)
    p_one = sum(comb(disc, i) * (0.5 ** disc) for i in range(k, disc + 1))
    return {
        "n10_a_wins": n10, "n01_b_wins": n01,
        "n11_both": n11, "n00_neither": n00,
        "p_value": float(min(2 * p_one, 1.0)),
        "method": "exact_binomial",
    }


# ─────────────────────────────────────────────────────────────────────
# Cross-method agreement
# ─────────────────────────────────────────────────────────────────────

def top_k_agreement(probs_a: torch.Tensor, probs_b: torch.Tensor,
                      k: int = 5) -> float:
    """|top_k(A) ∩ top_k(B)| / k. Both inputs [C]."""
    if probs_a.numel() != probs_b.numel():
        return 0.0
    k_eff = min(k, probs_a.numel())
    set_a = set(probs_a.topk(k_eff).indices.tolist())
    set_b = set(probs_b.topk(k_eff).indices.tolist())
    return len(set_a & set_b) / k_eff


def kl_divergence_distribution(probs_a: torch.Tensor, probs_b: torch.Tensor,
                                  eps: float = 1e-9) -> float:
    """KL(A || B) for two probability distributions."""
    a = probs_a.clamp(min=eps); b = probs_b.clamp(min=eps)
    return float((a * (a.log() - b.log())).sum().item())


def js_divergence(probs_a: torch.Tensor, probs_b: torch.Tensor,
                   eps: float = 1e-9) -> float:
    """Symmetric Jensen-Shannon divergence."""
    m = 0.5 * (probs_a + probs_b)
    return 0.5 * (kl_divergence_distribution(probs_a, m, eps) +
                  kl_divergence_distribution(probs_b, m, eps))


def cross_method_agreement(method_outputs: Dict[str, List[torch.Tensor]]
                              ) -> Dict[str, float]:
    """Pairwise top-1, top-5, KL, JS between any number of methods."""
    methods = list(method_outputs.keys())
    out = {}
    for i, m1 in enumerate(methods):
        for m2 in methods[i + 1:]:
            tops_1, tops_5, kls, jss = [], [], [], []
            for p1, p2 in zip(method_outputs[m1], method_outputs[m2]):
                tops_1.append(top_k_agreement(p1, p2, k=1))
                tops_5.append(top_k_agreement(p1, p2, k=5))
                kls.append(kl_divergence_distribution(p1, p2))
                jss.append(js_divergence(p1, p2))
            out[f"{m1}_vs_{m2}_top1"] = float(np.mean(tops_1))
            out[f"{m1}_vs_{m2}_top5"] = float(np.mean(tops_5))
            out[f"{m1}_vs_{m2}_kl"] = float(np.mean(kls))
            out[f"{m1}_vs_{m2}_js"] = float(np.mean(jss))
    return out


# ─────────────────────────────────────────────────────────────────────
# Multi-seed aggregation
# ─────────────────────────────────────────────────────────────────────

def aggregate_seeds(seed_results: Dict[int, Dict[str, float]]
                      ) -> Dict[str, Dict[str, float]]:
    """Across multiple seeds, compute mean ± std for each metric."""
    if not seed_results:
        return {}
    metric_keys = next(iter(seed_results.values())).keys()
    agg = {}
    for k in metric_keys:
        values = [seed_results[s].get(k) for s in seed_results]
        values = [v for v in values if isinstance(v, (int, float))]
        if not values:
            agg[k] = {"mean": None, "std": None, "n": 0}
            continue
        agg[k] = {
            "mean": float(np.mean(values)),
            "std":  float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "n":    len(values),
        }
    return agg


# ─────────────────────────────────────────────────────────────────────
# Per-difficulty stratification
# ─────────────────────────────────────────────────────────────────────

def stratify_metrics(metrics_per_sample: List[Dict],
                       strat_key: str = "hop_count") -> Dict:
    """Group per-sample metrics by a stratification key (e.g. hop_count)."""
    by_stratum = defaultdict(list)
    for m in metrics_per_sample:
        key = m.get(strat_key, "unknown")
        by_stratum[key].append(m)
    out = {}
    for stratum, items in by_stratum.items():
        out[str(stratum)] = {
            "n": len(items),
            "hit": float(np.mean([m["hit"] for m in items])),
            "f1":  float(np.mean([m["f1"] for m in items])),
            "ece": expected_calibration_error(
                [m["confidence"] for m in items],
                [m["hit"] for m in items],
            ),
            "mean_paths": float(np.mean([m["n_paths"] for m in items])),
            "mean_K": float(np.mean([m["chosen_K"] for m in items])),
        }
    return out


def estimate_hop_count(sample) -> int:
    """Cheap proxy: longest non-padded path length."""
    paths = sample.paths  # [N, L]
    non_pad = (paths != 0).sum(dim=-1)  # [N]
    return int(non_pad.max().item())


# ─────────────────────────────────────────────────────────────────────
# Comprehensive report — produces JSON-ready dict for paper tables
# ─────────────────────────────────────────────────────────────────────

def full_report(confidences: Sequence[float],
                 correctness: Sequence[int],
                 n_paths_list: Sequence[int],
                 f1s: Sequence[float],
                 chosen_K_list: Sequence[int],
                 latency_ms_per_q: float = 0.0,
                 ) -> Dict[str, float]:
    """One-shot full metric dump for paper Section 5 main table."""
    rep = calibration_report(confidences, correctness)
    rep.update({
        "hit_at_1":    float(np.mean(correctness) * 100),
        "f1":          float(np.mean(f1s) * 100),
        "mean_paths":  float(np.mean(n_paths_list)),
        "mean_K":      float(np.mean(chosen_K_list)),
        "p50_paths":   float(np.percentile(n_paths_list, 50)),
        "p95_paths":   float(np.percentile(n_paths_list, 95)),
        "latency_ms":  float(latency_ms_per_q),
        "aurc":        area_under_risk_coverage(confidences, correctness),
        "sel_acc_80":  selective_accuracy_at_coverage(confidences, correctness, 0.8),
        "sel_acc_50":  selective_accuracy_at_coverage(confidences, correctness, 0.5),
    })
    # Bootstrap CI for headline numbers
    hit_pe, hit_lo, hit_hi = bootstrap_ci(correctness, np.mean,
                                            n_bootstrap=1000, seed=0)
    # BUGFIX: previously `hit_pe*100 if hit_at_1 < 1 else hit_lo*100` — the
    # condition compared a percentage with 1, so the lower bound was wrong.
    rep["hit_ci_low"] = hit_lo * 100
    rep["hit_ci_high"] = hit_hi * 100
    return rep
