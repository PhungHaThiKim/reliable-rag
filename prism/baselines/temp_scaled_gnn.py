"""Temperature-scaled GNN-RAG calibration baseline (Guo et al. ICML 2017).

Concern C response: comparing PRISM's calibrated confidence against the RAW
GNN-RAG score is cherry-picking. The fair baseline is GNN-RAG + post-hoc
temperature scaling — a single scalar T fit on dev set so σ(z/T) matches
empirical accuracy.

If PRISM's ECE is still substantially lower than temp-scaled GNN-RAG, the
calibration claim is solid. If not, the claim collapses (or needs reframe).
"""
from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn


def fit_temp_scaling(confidences: Sequence[float],
                       correctness: Sequence[int],
                       n_iter: int = 200, lr: float = 0.05) -> float:
    """Fit a single scalar T so σ(logit(conf)/T) is calibrated against
    correctness. Returns T."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_T = nn.Parameter(torch.zeros(1, device=device))
    c = torch.tensor(confidences, dtype=torch.float32, device=device)
    y = torch.tensor(correctness, dtype=torch.float32, device=device)
    eps = 1e-6
    logit = torch.log(c.clamp(eps, 1 - eps) / (1 - c).clamp(eps, 1 - eps))
    opt = torch.optim.LBFGS([log_T], lr=lr, max_iter=n_iter)

    def closure():
        opt.zero_grad()
        T = log_T.exp()
        p = torch.sigmoid(logit / T)
        loss = -(y * p.clamp(eps).log() +
                 (1 - y) * (1 - p).clamp(eps).log()).mean()
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_T.exp().item())


class TempScaledGNN:
    """Wraps any GNN-RAG-style retriever to produce calibrated confidences.

    Workflow:
      1. Run retriever on dev set → collect (raw_score, correctness) pairs
      2. Call .fit(...) on dev set to learn T
      3. .calibrate(raw_score) → calibrated probability

    `raw_score` should be the max GNN score among selected paths, normalised
    to [0,1] (we add sigmoid internally if needed).
    """

    def __init__(self):
        self.T: float = 1.0
        self.fitted: bool = False

    def fit(self, raw_scores: Sequence[float], correctness: Sequence[int],
            n_iter: int = 200, lr: float = 0.05) -> "TempScaledGNN":
        # Project raw_scores into (0,1) if not already
        s = torch.tensor(raw_scores, dtype=torch.float32)
        if s.min() < 0 or s.max() > 1:
            s = torch.sigmoid(s)
            raw_scores = s.tolist()
        self.T = fit_temp_scaling(raw_scores, correctness, n_iter=n_iter, lr=lr)
        self.fitted = True
        return self

    def calibrate(self, raw_score: float) -> float:
        if not self.fitted:
            raise RuntimeError("call .fit() first")
        eps = 1e-6
        s = max(eps, min(1 - eps, raw_score))
        logit = torch.log(torch.tensor(s / (1 - s)))
        return float(torch.sigmoid(logit / self.T).item())

    def calibrate_batch(self, raw_scores: Sequence[float]) -> list:
        return [self.calibrate(s) for s in raw_scores]
