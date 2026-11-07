"""GNN-RAG top-K baseline retriever — pure GNN scoring, no learned filter.

Replicates the original GNN-RAG (Mavromatis ACL 2025) by:
  1. Taking all paths from path_to_candidate
  2. Sorting by max GNN score of the candidate
  3. Returning top-K paths (K configurable for Pareto sweep)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch

from ..config import CFG
from ..data import Sample


@dataclass
class GNNRAGTopKTrace:
    """Matches ControllerTrace API for compatibility."""
    selected_path_idx: List[int]
    chosen_K: int
    final_probs: torch.Tensor       # placeholder — GNN-RAG uses LLM downstream
    confidence: torch.Tensor
    stopped_reason: str = "fixed_topk"
    step_kl: list = None
    log_probs: list = None

    def __post_init__(self):
        if self.step_kl is None: self.step_kl = []
        if self.log_probs is None: self.log_probs = []


@torch.no_grad()
def gnnrag_topk_select(sample: Sample, K: int = 10,
                         use_path_score: bool = True) -> GNNRAGTopKTrace:
    """Selects top-K paths by their corresponding candidate's GNN score.

    Args:
        sample: PRISM Sample object
        K: number of paths to keep
        use_path_score: if True, sort by candidate_gnn_scores[p_to_c]; else
            sort by path-frequency per candidate (S-Path-RAG style).
    """
    n_paths = sample.paths.size(0)
    if n_paths == 0:
        empty = sample.candidate_ids.new_zeros(sample.candidate_ids.size(0),
                                                dtype=torch.float32)
        return GNNRAGTopKTrace(
            selected_path_idx=[], chosen_K=0,
            final_probs=empty / max(empty.sum(), 1),
            confidence=torch.tensor(0.0),
        )

    if use_path_score:
        # Score each path by its candidate's GNN score
        scores = sample.candidate_gnn_scores[sample.path_to_candidate]
    else:
        # Path-frequency: more paths to candidate → higher score
        unique, counts = torch.unique(sample.path_to_candidate, return_counts=True)
        freq_lookup = torch.zeros(sample.candidate_ids.size(0))
        for u, c in zip(unique.tolist(), counts.tolist()):
            freq_lookup[u] = c
        scores = freq_lookup[sample.path_to_candidate]

    # Top-K path indices
    k_eff = min(K, n_paths)
    top_idx = scores.topk(k_eff).indices.tolist()

    # "Final probs" = normalised candidate_gnn_scores (proxy for downstream)
    final_probs = sample.candidate_gnn_scores.float()
    final_probs = final_probs / max(final_probs.sum().item(), 1e-9)

    # Confidence = max GNN score among selected
    sel_scores = sample.candidate_gnn_scores[
        sample.path_to_candidate[torch.tensor(top_idx, dtype=torch.long)]
    ] if top_idx else torch.tensor([0.0])
    confidence = sel_scores.max() if top_idx else torch.tensor(0.0)

    return GNNRAGTopKTrace(
        selected_path_idx=sorted(top_idx),
        chosen_K=k_eff,
        final_probs=final_probs,
        confidence=confidence,
    )


class GNNRAGTopK:
    """Wrapper exposing same API as CGAR/PRISM controllers."""

    def __init__(self, K: int = 10, use_path_score: bool = True):
        self.K = K
        self.use_path_score = use_path_score

    def select(self, sample: Sample) -> GNNRAGTopKTrace:
        return gnnrag_topk_select(sample, K=self.K,
                                    use_path_score=self.use_path_score)
