"""
GRL-RAG: GNN-Guided RL Retriever for GraphRAG
All shared classes for P4a (architecture) and P4b (training).
"""

import os
import pickle
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict, deque

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
# 4. GreedyGNN Baseline
# ==============================================================

class GreedyGNNRetriever:
    """Baseline: greedily follow highest GNN-scored neighbors."""

    def __init__(self, gnn_heuristic, max_steps=10, budget=50):
        self.gnn = gnn_heuristic
        self.max_steps = max_steps
        self.budget = budget

    def retrieve(self, qid):
        qdata = self.gnn.get(qid)
        if qdata is None:
            return {'triples': [], 'cost': 0, 'context': ''}

        G = self.gnn.get_local_graph(qid)
        seeds = self.gnn.get_seeds(qid)

        visited = set(seeds)
        all_triples = []
        frontier = list(seeds)
        steps = 0

        while frontier and steps < self.max_steps and len(visited) < self.budget:
            candidates = []
            for node in frontier:
                if node not in G:
                    continue
                for neighbor in G.neighbors(node):
                    if neighbor in visited:
                        continue
                    score = self.gnn.get_node_score(qid, neighbor)
                    rel = G[node][neighbor].get('relation', 'related_to')
                    candidates.append((node, rel, neighbor, score))

            if not candidates:
                break

            candidates.sort(key=lambda x: -x[3])
            next_frontier = []
            for src, rel, dst, score in candidates:
                if len(visited) >= self.budget:
                    break
                if dst not in visited:
                    visited.add(dst)
                    all_triples.append((src, rel, dst))
                    next_frontier.append(dst)

            frontier = next_frontier
            steps += 1

        context = path_to_string(all_triples)
        return {'triples': all_triples, 'cost': len(visited), 'context': context}


# ==============================================================
# 5. Budget KG Environment (for RL traversal)
# ==============================================================

class BudgetKGEnvironment:
    """MDP environment on local subgraph with shared budget across seeds."""

    STOP = '__STOP__'

    def __init__(self, gnn, max_steps=10, budget=50, top_k=20):
        self.gnn = gnn
        self.max_steps = max_steps
        self.budget = budget
        self.top_k = top_k
        self.qid = None
        self.graph = None
        self.current_node = None
        self.visited = set()
        self.path = []
        self.triples = []
        self.step_count = 0

    def reset(self, qid):
        self.qid = qid
        self.graph = self.gnn.get_local_graph(qid)
        self.visited = set()
        self.path = []
        self.triples = []
        self.step_count = 0
        self.current_node = None

    def set_start(self, entity):
        self.current_node = entity
        self.visited.add(entity)
        self.path.append(entity)

    def get_actions(self):
        if self.current_node is None or self.current_node not in self.graph:
            return [(self.STOP, None, None, 0.0)]

        neighbors = []
        for nb in self.graph.neighbors(self.current_node):
            if nb in self.visited:
                continue
            rel = self.graph[self.current_node][nb].get('relation', 'related_to')
            score = self.gnn.get_node_score(self.qid, nb)
            neighbors.append((rel, nb, rel, score))

        neighbors.sort(key=lambda x: -x[3])
        neighbors = neighbors[:self.top_k]
        actions = [(rel, nb, rel, score) for rel, nb, rel, score in neighbors]
        actions.append((self.STOP, None, None, 0.0))
        return actions

    def step(self, action_idx):
        actions = self.get_actions()
        action_idx = min(action_idx, len(actions) - 1)
        rel, next_node, _, gnn_score = actions[action_idx]

        if (rel == self.STOP or self.step_count >= self.max_steps
                or len(self.visited) >= self.budget):
            return True

        self.triples.append((self.current_node, rel, next_node))
        self.current_node = next_node
        self.visited.add(next_node)
        self.path.append(next_node)
        self.step_count += 1

        done = (self.step_count >= self.max_steps
                or len(self.visited) >= self.budget
                or len(self.get_actions()) <= 1)
        return done

    @property
    def budget_ratio(self):
        return max(0, (self.budget - len(self.visited)) / self.budget)

    @property
    def step_ratio(self):
        return self.step_count / self.max_steps


# ==============================================================
# 6. Traversal Policy
# ==============================================================

class TraversalPolicy(nn.Module):
    """Policy for path traversal with relation encoding."""

    def __init__(self, emb_dim=384, hidden_dim=256, num_heads=4):
        super().__init__()
        self.emb_dim = emb_dim

        state_dim = emb_dim * 3 + 2
        self.state_enc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        action_dim = hidden_dim + emb_dim * 2 + 1
        self.action_scorer = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.history_attn = nn.MultiheadAttention(
            embed_dim=emb_dim, num_heads=num_heads, batch_first=True
        )

        self.stop_entity_emb = nn.Parameter(torch.randn(emb_dim))
        self.stop_relation_emb = nn.Parameter(torch.randn(emb_dim))

    def encode_history(self, query_emb, history_embs):
        if history_embs.shape[1] == 0:
            return torch.zeros(query_emb.shape[0], self.emb_dim, device=query_emb.device)
        q = query_emb.unsqueeze(1)
        out, _ = self.history_attn(q, history_embs, history_embs)
        return out.squeeze(1)

    def forward(self, query_emb, node_emb, history_embs,
                budget_ratio, step_ratio,
                action_entity_embs, action_relation_embs, gnn_scores):
        n_actions = action_entity_embs.shape[1]

        history_emb = self.encode_history(query_emb, history_embs)
        state = torch.cat([query_emb, node_emb, history_emb,
                           budget_ratio, step_ratio], dim=-1)
        state_repr = self.state_enc(state)

        state_exp = state_repr.unsqueeze(1).expand(-1, n_actions, -1)
        action_input = torch.cat([
            state_exp,
            action_entity_embs,
            action_relation_embs,
            gnn_scores,
        ], dim=-1)

        scores = self.action_scorer(action_input).squeeze(-1)
        return F.softmax(scores, dim=-1)


# ==============================================================
# 7. Traversal Retriever
# ==============================================================

class TraversalRetriever:
    """RL traversal retriever with shared budget across seeds."""

    def __init__(self, gnn, policy, embedder, max_steps=10, budget=50,
                 top_k=20, device='cpu'):
        self.gnn = gnn
        self.policy = policy.to(device)
        self.embedder = embedder
        self.max_steps = max_steps
        self.budget = budget
        self.top_k = top_k
        self.device = device
        self.env = BudgetKGEnvironment(gnn, max_steps, budget, top_k)

    def _encode_actions(self, actions):
        entity_embs = []
        relation_embs = []
        scores = []
        for rel, entity, rel_text, gnn_score in actions:
            if entity is None:
                entity_embs.append(self.policy.stop_entity_emb.detach().cpu())
                relation_embs.append(self.policy.stop_relation_emb.detach().cpu())
            else:
                entity_embs.append(self.embedder.encode(entity))
                relation_embs.append(self.embedder.encode(rel_text))
            scores.append(gnn_score)

        return (
            torch.stack([e.cpu() for e in entity_embs]).unsqueeze(0).to(self.device),
            torch.stack([e.cpu() for e in relation_embs]).unsqueeze(0).to(self.device),
            torch.tensor(scores).unsqueeze(0).unsqueeze(-1).float().to(self.device),
        )

    def retrieve(self, qid, greedy=True):
        qdata = self.gnn.get(qid)
        if qdata is None:
            return {'triples': [], 'cost': 0, 'log_probs': [], 'context': ''}

        query_emb = self.embedder.encode(qdata['question']).to(self.device)
        seeds = self.gnn.get_seeds(qid)

        self.env.reset(qid)
        all_log_probs = []

        for seed in seeds:
            if seed in self.env.visited and seed != seeds[0]:
                continue
            if len(self.env.visited) >= self.budget:
                break

            self.env.set_start(seed)
            done = False

            while not done:
                actions = self.env.get_actions()
                if len(actions) <= 1:
                    break

                node_emb = self.embedder.encode(self.env.current_node).unsqueeze(0).to(self.device)
                history = self.embedder.encode_batch(self.env.path).unsqueeze(0).to(self.device)
                budget_r = torch.tensor([[self.env.budget_ratio]]).to(self.device)
                step_r = torch.tensor([[self.env.step_ratio]]).to(self.device)
                act_ent, act_rel, gnn_sc = self._encode_actions(actions)

                if greedy:
                    with torch.no_grad():
                        probs = self.policy(query_emb.unsqueeze(0), node_emb,
                                            history, budget_r, step_r,
                                            act_ent, act_rel, gnn_sc)
                    action_idx = probs.argmax(dim=-1).item()
                else:
                    probs = self.policy(query_emb.unsqueeze(0), node_emb,
                                        history, budget_r, step_r,
                                        act_ent, act_rel, gnn_sc)
                    dist = torch.distributions.Categorical(probs)
                    action_idx = dist.sample().item()
                    all_log_probs.append(dist.log_prob(
                        torch.tensor(action_idx, device=self.device)))

                done = self.env.step(action_idx)

        context = path_to_string(self.env.triples)
        return {
            'triples': self.env.triples,
            'cost': len(self.env.visited),
            'log_probs': all_log_probs,
            'context': context,
        }


# ==============================================================
# 8. Path Selection Policy
# ==============================================================

class PathSelectionPolicy(nn.Module):
    """Score each candidate path for selection."""

    def __init__(self, emb_dim=384, hidden_dim=256):
        super().__init__()
        self.emb_dim = emb_dim
        self.scorer = nn.Sequential(
            nn.Linear(emb_dim * 2 + 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, query_emb, path_embs, gnn_scores, path_lengths):
        n_paths = path_embs.shape[1]
        query_exp = query_emb.unsqueeze(1).expand(-1, n_paths, -1)
        x = torch.cat([query_exp, path_embs, gnn_scores, path_lengths], dim=-1)
        scores = self.scorer(x).squeeze(-1)
        return torch.sigmoid(scores)


# ==============================================================
# 9. Path Selection Retriever
# ==============================================================

class PathSelectionRetriever:
    """RL-based path selection on top of GNN-RAG candidate paths."""

    def __init__(self, gnn, policy, embedder, max_paths=10, device='cpu'):
        self.gnn = gnn
        self.policy = policy.to(device)
        self.embedder = embedder
        self.max_paths = max_paths
        self.device = device

    def retrieve(self, qid, greedy=True):
        qdata = self.gnn.get(qid)
        if qdata is None:
            return {'triples': [], 'cost': 0, 'log_probs': [], 'context': '',
                    'selected_paths': [], 'all_paths': []}

        all_paths = extract_candidate_paths(qid, self.gnn)
        if not all_paths:
            return {'triples': [], 'cost': 0, 'log_probs': [], 'context': '',
                    'selected_paths': [], 'all_paths': all_paths}

        query_emb = self.embedder.encode(qdata['question']).unsqueeze(0).to(self.device)

        path_embs = []
        gnn_scores = []
        path_lengths = []
        for p in all_paths:
            entities_in_path = set()
            for h, r, t in p['path_triples']:
                entities_in_path.add(h)
                entities_in_path.add(t)
            embs = self.embedder.encode_batch(list(entities_in_path))
            path_embs.append(embs.mean(dim=0))
            gnn_scores.append(p['gnn_score'])
            path_lengths.append(p['length'] / 5.0)

        path_embs_t = torch.stack(path_embs).unsqueeze(0).to(self.device)
        gnn_scores_t = torch.tensor(gnn_scores).unsqueeze(0).unsqueeze(-1).float().to(self.device)
        path_lengths_t = torch.tensor(path_lengths).unsqueeze(0).unsqueeze(-1).float().to(self.device)

        if greedy:
            with torch.no_grad():
                selection_probs = self.policy(query_emb, path_embs_t,
                                              gnn_scores_t, path_lengths_t)
        else:
            selection_probs = self.policy(query_emb, path_embs_t,
                                          gnn_scores_t, path_lengths_t)

        probs = selection_probs.squeeze(0)

        log_probs_list = []
        if greedy:
            _, top_indices = probs.topk(min(self.max_paths, len(all_paths)))
            selected_indices = top_indices.tolist()
        else:
            selections = torch.bernoulli(probs)
            selected_indices = selections.nonzero(as_tuple=True)[0].tolist()
            if len(selected_indices) > self.max_paths:
                selected_indices = selected_indices[:self.max_paths]
            if not selected_indices:
                selected_indices = [probs.argmax().item()]

            for i in range(len(all_paths)):
                if i in selected_indices:
                    log_probs_list.append(torch.log(probs[i] + 1e-8))
                else:
                    log_probs_list.append(torch.log(1 - probs[i] + 1e-8))

        selected_paths = [all_paths[i] for i in selected_indices]
        all_triples = []
        for p in selected_paths:
            all_triples.extend(p['path_triples'])

        context_lines = [p['path_text'] for p in selected_paths]
        context = '\n'.join(context_lines)

        return {
            'triples': all_triples,
            'cost': len(selected_paths),
            'log_probs': log_probs_list,
            'context': context,
            'selected_paths': selected_paths,
            'all_paths': all_paths,
        }


# ==============================================================
# 10. Reward Function
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
