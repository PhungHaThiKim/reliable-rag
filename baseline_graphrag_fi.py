"""GraphRAG-FI baseline reimplementation (Yang et al., EMNLP 2025).

Two-stage filtering applied to GNN-RAG candidate paths:
  Stage 1 (Filter): drop paths whose mean GNN score is below the τ_drop quantile.
  Stage 2 (Keep):   keep top-K_keep highest-scoring remaining paths.

This is an APPROXIMATION of the full GraphRAG-FI paper, which also has an
LLM-side "Integration" stage (logits-based knowledge fusion). We compare
retrieval quality only — Integration is orthogonal and could be applied to
any retriever including CGAR.

Default hyperparameters are taken from §3.2 of the paper:
    τ_drop = 0.3   (drop bottom 30% of paths)
    K_keep = 8     (keep top 8 after filtering)

Usage matches the GreedyGNN / CGAR retrievers — drop into P6 Cell 6 loop.
"""
from grl_rag import extract_candidate_paths, path_to_string


class GraphRAGFIRetriever:
    """Reimpl of EMNLP 2025 GraphRAG-FI filtering stage."""

    def __init__(self, gnn, tau_drop=0.3, k_keep=8, max_candidates=20,
                 max_paths_per_pair=3):
        self.gnn = gnn
        self.tau_drop = tau_drop
        self.k_keep = k_keep
        self.max_candidates = max_candidates
        self.max_paths_per_pair = max_paths_per_pair

    def retrieve(self, qid):
        paths = extract_candidate_paths(
            qid, self.gnn,
            max_candidates=self.max_candidates,
            max_paths_per_pair=self.max_paths_per_pair,
        )
        if not paths:
            return {'triples': [], 'cost': 0, 'context': '',
                    'selected_paths': []}

        # Stage 1: drop paths below τ_drop quantile of GNN score
        scores = sorted([p['gnn_score'] for p in paths])
        cutoff_idx = int(len(scores) * self.tau_drop)
        cutoff = scores[cutoff_idx] if scores else 0.0
        kept = [p for p in paths if p['gnn_score'] >= cutoff]

        # Stage 2: keep top-K_keep by score
        kept.sort(key=lambda p: -p['gnn_score'])
        kept = kept[:self.k_keep]

        triples = [t for p in kept for t in p['path_triples']]
        return {
            'triples': triples,
            'cost': len(kept),
            'context': '\n'.join(p['path_text'] for p in kept),
            'selected_paths': kept,
            'n_total_paths': len(paths),
            'n_selected': len(kept),
        }
