"""Path-count heuristic confidence (Concern E baseline).

Simple confidence proxy: questions with MORE retrieved paths supporting a
single answer entity get higher confidence. Naive but a fair baseline for
the risk-coverage curve comparison.
"""
from __future__ import annotations

from typing import List

import torch

from ..data import Sample


class PathCountConfidence:
    """Confidence = max candidate's path-frequency / total paths.

    Intuition: if many paths converge on the same candidate, we trust it more.
    """

    @staticmethod
    def score(sample: Sample, selected_path_idx: List[int]) -> float:
        if not selected_path_idx:
            return 0.0
        sel_cands = sample.path_to_candidate[
            torch.tensor(selected_path_idx, dtype=torch.long)
        ]
        # Count frequency per candidate
        unique, counts = torch.unique(sel_cands, return_counts=True)
        max_count = int(counts.max().item())
        total = len(selected_path_idx)
        return float(max_count / max(total, 1))
