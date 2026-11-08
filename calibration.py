"""Calibration and selective-prediction metrics used by kgqa_bench.metrics and kgqa_bench.report.

Copied verbatim from the research repo (prism/metrics.py and prism/metrics_full.py) so this
folder has no dependency on the PRISM package.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def expected_calibration_error(confidences: List[float], correctness: List[float],
                                n_bins: int = 10) -> float:
    """ECE with equal-width bins. Lower is better."""
    confs = np.array(confidences); corrs = np.array(correctness)
    bins = np.linspace(0, 1, n_bins + 1)
    n = len(confs)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confs >= lo) & (confs < hi if i < n_bins - 1 else confs <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = corrs[mask].mean()
        bin_conf = confs[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def brier_score(confidences: List[float], correctness: List[float]) -> float:
    return float(np.mean((np.array(confidences) - np.array(correctness)) ** 2))


def reliability_diagram(confidences: List[float], correctness: List[float],
                         n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    confs = np.array(confidences); corrs = np.array(correctness)
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    accs = np.zeros(n_bins); counts = np.zeros(n_bins)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confs >= lo) & (confs < hi if i < n_bins - 1 else confs <= hi)
        if mask.sum() > 0:
            accs[i] = corrs[mask].mean()
            counts[i] = mask.sum()
    return bin_centers, accs, counts


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
