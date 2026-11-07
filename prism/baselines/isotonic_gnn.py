"""Isotonic regression calibration baseline (Zadrozny-Elkan 2002).

Stronger than temperature scaling because non-parametric — fits arbitrary
monotone mapping from raw_score → calibrated_prob. Used as a second
calibration baseline in PRISM Experiment 4.

If PRISM still wins against isotonic, calibration claim is strong.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def fit_isotonic(raw_scores: Sequence[float],
                  correctness: Sequence[int]):
    """Returns sklearn IsotonicRegression fitted on (raw_scores, correctness)."""
    try:
        from sklearn.isotonic import IsotonicRegression
    except ImportError as e:
        raise RuntimeError("install scikit-learn: pip install scikit-learn") from e
    iso = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1)
    iso.fit(list(raw_scores), list(correctness))
    return iso


class IsotonicGNN:
    """Isotonic regression on GNN-RAG raw scores. Drop-in calibrator."""

    def __init__(self):
        self.iso = None

    def fit(self, raw_scores: Sequence[float],
            correctness: Sequence[int]) -> "IsotonicGNN":
        self.iso = fit_isotonic(raw_scores, correctness)
        return self

    def calibrate(self, raw_score: float) -> float:
        if self.iso is None:
            raise RuntimeError("call .fit() first")
        return float(self.iso.transform([raw_score])[0])

    def calibrate_batch(self, raw_scores: Sequence[float]) -> list:
        if self.iso is None:
            raise RuntimeError("call .fit() first")
        return self.iso.transform(list(raw_scores)).tolist()
