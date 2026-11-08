"""Adapter: BenchDataset -> GNN-RAG-style feature store used by every retriever
(GNN-RAG all / top-K, SBERT top-K, CGAR, CGAR-G), so all methods see the SAME
candidate ranking and the SAME shortest relation paths.
"""
from __future__ import annotations

from .data import BenchDataset


def to_gnn_features(dataset: BenchDataset):
    """Return a `grl_rag.GNNHeuristic` whose `.features` is built from the
    BenchDataset (no pickle round-trip). Works with extract_candidate_paths
    and CGARRetriever unchanged."""
    from grl_rag import GNNHeuristic
    gnn = GNNHeuristic.__new__(GNNHeuristic)   # skip the (noisy) file-loading __init__
    gnn.features = {q.qid: q.to_feature_dict() for q in dataset}
    return gnn
