"""
Shared GraphRAG utilities used by CGAR and the baselines:
GNN-RAG feature store (GNNHeuristic), SBERT / hash embedders, candidate-path
extraction from the GNN candidate ranking, and the reward used by CGAR Stage 2.

Trimmed copy of the research repo's grl_rag.py: the RL traversal / path-selection
policies (GreedyGNNRetriever, BudgetKGEnvironment, Traversal*, PathSelection*) are
not used by any Seminar 2 result and were removed.
"""

import os
import pickle
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # optional: only needed for the SBERT embedder
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - exercised when SBERT is not installed
    SentenceTransformer = None


# ==============================================================
# 1. GNN Heuristic (loads P2 output)
# ==============================================================

class GNNHeuristic:
    """Wrapper for GNN features from Phase 2."""

    def __init__(self, features_path=None):
        self.features = {}
        if features_path and os.path.exists(features_path):
            with open(features_path, 'rb') as f:
                self.features = pickle.load(f)
            print(f'Loaded GNN features for {len(self.features)} questions')
        else:
            print('GNNHeuristic: No features loaded')

    def get(self, qid):
        return self.features.get(qid)

    def get_node_score(self, qid, entity):
        qdata = self.features.get(qid, {})
        return qdata.get('node_scores', {}).get(entity, 0.0)

    def get_local_graph(self, qid):
        qdata = self.features.get(qid)
        if qdata is None:
            return nx.Graph()
        G = nx.Graph()
        for triplet in qdata['subgraph_tuples']:
            h, r, t = triplet
            if isinstance(h, dict): h = h['text']
            if isinstance(r, dict): r = r['text']
            if isinstance(t, dict): t = t['text']
            G.add_edge(str(h), str(t), relation=str(r).strip())
        return G

    def get_seeds(self, qid):
        qdata = self.features.get(qid, {})
        result = []
        for e in qdata.get('q_entities', []):
            result.append(e.get('text', str(e)) if isinstance(e, dict) else str(e))
        return result

    def get_answers(self, qid):
        qdata = self.features.get(qid, {})
        result = []
        for a in qdata.get('answers', []):
            result.append(a.get('text', a.get('kb_id', str(a))) if isinstance(a, dict) else str(a))
        return result


# ==============================================================
# 2. Embedder (SBERT with cache)
# ==============================================================

class HashEmbedder:
    """Deterministic, dependency-free stand-in for SBERT.

    Maps a string to a fixed random unit vector seeded by its hash. Used for
    unit tests / CPU smoke runs; NOT meaningful for paper numbers.
    """

    def __init__(self, emb_dim=384, seed=0):
        self.emb_dim = emb_dim
        self.seed = seed
        self.cache = {}

    def _vec(self, text):
        import hashlib
        h = int(hashlib.md5(f"{self.seed}|{text}".encode()).hexdigest(), 16) % (2 ** 31)
        g = torch.Generator().manual_seed(h)
        v = torch.randn(self.emb_dim, generator=g)
        return v / (v.norm() + 1e-8)

    def encode(self, text):
        if text is None:
            return torch.zeros(self.emb_dim)
        text = str(text)
        if text not in self.cache:
            self.cache[text] = self._vec(text)
        return self.cache[text]

    def encode_batch(self, texts):
        return torch.stack([self.encode(t) for t in texts]) if texts else torch.zeros(0, self.emb_dim)

    def save_cache(self, path):
        pass


class Embedder:
    """SBERT encoder for entities AND relations. Cached.

    If `sentence_transformers` is unavailable and `fallback='hash'`, a
    HashEmbedder is used instead (tests / smoke runs only).
    """

    def __init__(self, model_name='all-MiniLM-L6-v2', cache_path=None, fallback=None):
        if SentenceTransformer is None:
            if fallback == 'hash':
                self._fallback = HashEmbedder()
                self.model = None
                self.emb_dim = self._fallback.emb_dim
                self.cache = self._fallback.cache
                return
            raise ImportError("sentence_transformers not installed; pass fallback='hash' for tests")
        self._fallback = None
        self.model = SentenceTransformer(model_name)
        self.emb_dim = self.model.get_sentence_embedding_dimension()
        self.cache = {}
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                self.cache = pickle.load(f)
            print(f'Loaded {len(self.cache)} cached embeddings')

    def encode(self, text):
        if self._fallback is not None:
            return self._fallback.encode(text)
        if text is None:
            return torch.zeros(self.emb_dim)
        text = str(text)
        if text not in self.cache:
            emb = self.model.encode(text, convert_to_tensor=True)
            self.cache[text] = emb.cpu()
        return self.cache[text]

    def encode_batch(self, texts):
        if self._fallback is not None:
            return self._fallback.encode_batch(list(texts))
        texts = [str(t) if t is not None else None for t in texts]
        uncached = [t for t in texts if t not in self.cache and t is not None]
        if uncached:
            embs = self.model.encode(uncached, convert_to_tensor=True, show_progress_bar=False)
            for name, emb in zip(uncached, embs):
                self.cache[name] = emb.cpu()
        return torch.stack([
            self.cache.get(t, torch.zeros(self.emb_dim)) if t else torch.zeros(self.emb_dim)
            for t in texts
        ])

    def save_cache(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.cache, f)
        print(f'Saved {len(self.cache)} embeddings to {path}')


# ==============================================================
# 3. Utility functions
# ==============================================================

def path_to_string(triples):
    if not triples:
        return ''
    parts = []
    for h, r, t in triples:
        if not parts:
            parts.append(str(h))
        parts.append(str(r))
        parts.append(str(t))
    return ' -> '.join(parts)


def extract_candidate_paths(qid, gnn, max_candidates=10, max_paths_per_pair=3):
    """Extract all candidate paths for a question (like GNN-RAG)."""
    qdata = gnn.get(qid)
    if qdata is None:
        return []

    G = gnn.get_local_graph(qid)
    seeds = gnn.get_seeds(qid)
    candidates = qdata.get('top_candidates', [])[:max_candidates]

    all_paths = []
    seen = set()

    for seed in seeds:
        if seed not in G:
            continue
        for cand in candidates:
            cand_str = str(cand)
            if cand_str not in G or cand_str == seed:
                continue
            try:
                paths_gen = nx.all_shortest_paths(G, seed, cand_str)
                count = 0
                for p in paths_gen:
                    if count >= max_paths_per_pair:
                        break
                    triples = []
                    for i in range(len(p) - 1):
                        u, v = p[i], p[i + 1]
                        rel = G[u][v].get('relation', 'related_to')
                        triples.append((u, rel, v))
                    path_key = tuple((h, r, t) for h, r, t in triples)
                    if path_key not in seen:
                        seen.add(path_key)
                        all_paths.append({
                            'path_triples': triples,
                            'path_text': path_to_string(triples),
                            'candidate': cand_str,
                            'gnn_score': gnn.get_node_score(qid, cand),
                            'length': len(triples),
                        })
                    count += 1
            except nx.NetworkXNoPath:
                continue

    return all_paths


def extract_candidate_paths_by_k(qid, gnn, k_options=(5, 10, 20, 50), max_paths_per_pair=3):
    """Paths for every K in `k_options` from ONE extraction.

    `extract_candidate_paths(qid, gnn, max_candidates=K)` for increasing K
    yields nested supersets (candidates are a ranked prefix), so computing
    the K=max set once and slicing by candidate rank is equivalent to the
    old 4x extraction but ~len(k_options)x cheaper.

    Returns {K: [path_dict, ...]} with the same path dicts as
    extract_candidate_paths, plus a 'cand_rank' key.
    """
    k_options = sorted(k_options)
    qdata = gnn.get(qid)
    if qdata is None:
        return {K: [] for K in k_options}
    candidates = [str(c) for c in qdata.get('top_candidates', [])[:k_options[-1]]]
    rank = {c: i for i, c in enumerate(candidates)}
    all_paths = extract_candidate_paths(qid, gnn, max_candidates=k_options[-1],
                                        max_paths_per_pair=max_paths_per_pair)
    for p in all_paths:
        p['cand_rank'] = rank.get(str(p.get('candidate')), k_options[-1])
    return {K: [p for p in all_paths if p['cand_rank'] < K] for K in k_options}


# ==============================================================
# 4. Reward Function (used by CGAR Stage 2 / SCST)
# ==============================================================

def compute_reward(collected_triples, gold_answers,
                   budget_used, total_budget,
                   alpha=0.6, beta=0.2, gamma=0.2,
                   question_emb=None, embedder=None):
    """Compute reward for RL episode."""
    if not collected_triples:
        return -0.1

    collected_entities = set()
    for h, r, t in collected_triples:
        collected_entities.add(str(h))
        collected_entities.add(str(t))

    gold_set = set()
    for a in gold_answers:
        gold_set.add(a.get('text', a.get('kb_id', str(a))) if isinstance(a, dict) else str(a))

    overlap = len(collected_entities & gold_set)
    answer_quality = min(overlap / max(len(gold_set), 1), 1.0)

    efficiency = 1.0 - (budget_used / total_budget)

    relevance = 0.0
    if question_emb is not None and embedder is not None:
        ctx = path_to_string(collected_triples)
        ctx_emb = embedder.model.encode(ctx, convert_to_tensor=True)
        q_emb = question_emb if isinstance(question_emb, torch.Tensor) else torch.tensor(question_emb)
        relevance = max(0, F.cosine_similarity(ctx_emb.unsqueeze(0), q_emb.unsqueeze(0)).item())

    return alpha * answer_quality + beta * efficiency + gamma * relevance
