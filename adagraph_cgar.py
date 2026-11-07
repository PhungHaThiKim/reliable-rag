"""
CGAR: Confidence-Guided Adaptive Retrieval
=============================================
Fuses OptStop (confidence-based stopping on candidate expansion)
with AdaGraph (relevance-based path ranking) in a two-level controller
on top of GNN-RAG.

Level 1 — Candidate Budget (OptStop):
  Iteratively expand K (5→10→20→50 candidates).
  At each K, estimate confidence that answer is reachable.
  SPRT: stop expanding when confidence > β.

Level 2 — Path Relevance Ranking (AdaGraph):
  Score each path by query-relevance (not marginal utility).
  Penalize redundancy via pairwise cosine similarity.
  Select top-M paths under budget.

Why this works when P7 didn't:
  P7 selected 5 from 6 paths → no room for improvement.
  CGAR varies K (5→50) → decision space 10× larger.
  CGAR ranks by relevance → LLM-aligned (not diversity-first).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import defaultdict

from grl_rag import (
    GNNHeuristic, Embedder, extract_candidate_paths,
    path_to_string, compute_reward
)


# ══════════════════════════════════════════════════════════════
# 1. Confidence Head — OptStop component
# ══════════════════════════════════════════════════════════════

class CandidateConfidenceHead(nn.Module):
    """Estimates P(answer reachable | top-K candidates, query).

    Input: [evidence_state, query_emb, K_features]
    Output: confidence ∈ (0, 1)

    When confidence > β → stop expanding candidates.
    """
    def __init__(self, emb_dim=384, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim * 2 + 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, evidence, query_emb, k_features):
        """
        Args:
            evidence: (B, emb_dim) — accumulated evidence
            query_emb: (B, emb_dim)
            k_features: (B, 4) — [K/K_max, n_paths/max_paths,
                                   mean_gnn_score, coverage_estimate]
        """
        x = torch.cat([evidence, query_emb, k_features], dim=-1)
        return self.net(x).squeeze(-1)


# ══════════════════════════════════════════════════════════════
# 2. Path Relevance Head — AdaGraph component
# ══════════════════════════════════════════════════════════════

class PathRelevanceHead(nn.Module):
    """Scores each path by query-relevance (LLM-aligned).

    Unlike MarginalUtilityHead which optimizes diversity,
    this optimizes RELEVANCE to question → better LLM answers.
    """
    def __init__(self, emb_dim=384, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim * 2 + 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, path_embs, query_emb, features):
        """
        Args:
            path_embs: (B, N, emb_dim) — path embeddings
            query_emb: (B, emb_dim)
            features: (B, N, 2) — [gnn_score, path_length_norm]
        Returns:
            scores: (B, N) — relevance scores (raw logits)
        """
        n = path_embs.size(1)
        q_exp = query_emb.unsqueeze(1).expand(-1, n, -1)
        x = torch.cat([path_embs, q_exp, features], dim=-1)
        return self.net(x).squeeze(-1)


# ══════════════════════════════════════════════════════════════
# 3. Evidence Accumulator (GRU)
# ══════════════════════════════════════════════════════════════

class EvidenceGRU(nn.Module):
    """GRU accumulates evidence across candidate expansion rounds."""
    def __init__(self, emb_dim=384):
        super().__init__()
        self.gru = nn.GRUCell(emb_dim, emb_dim)
        self.init_proj = nn.Linear(emb_dim, emb_dim)

    def initialize(self, query_emb):
        return self.init_proj(query_emb)

    def update(self, state, new_evidence):
        return self.gru(new_evidence, state)


# ══════════════════════════════════════════════════════════════
# 4. CGAR Controller — Two-Level Unified
# ══════════════════════════════════════════════════════════════

class CGARController(nn.Module):
    """Confidence-Guided Adaptive Retrieval Controller.

    Two levels:
      Level 1 (OptStop): Decide how many candidates K to explore.
      Level 2 (AdaGraph): Rank and select paths by relevance.

    Training:
      Stage 1: Supervised (confidence calibration + relevance ranking)
      Stage 2: SCST (vary K stochastically → real advantage signal)
    """

    # Candidate budget options — provides real variance for SCST
    K_OPTIONS = [5, 10, 20, 50]

    def __init__(self, emb_dim=384, hidden_dim=256,
                 max_paths=10, sprt_beta=0.85,
                 redundancy_lambda=0.3):
        super().__init__()
        self.emb_dim = emb_dim
        self.max_paths = max_paths
        self.sprt_beta = sprt_beta
        self.redundancy_lambda = redundancy_lambda

        # Level 1: OptStop
        self.confidence_head = CandidateConfidenceHead(emb_dim, hidden_dim // 2)
        self.evidence_gru = EvidenceGRU(emb_dim)

        # Level 2: AdaGraph
        self.relevance_head = PathRelevanceHead(emb_dim, hidden_dim)

        # K-selection policy (for SCST stochastic sampling)
        self.k_policy = nn.Sequential(
            nn.Linear(emb_dim * 2 + 4, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, len(self.K_OPTIONS)),
        )

    def forward(self, query_emb, all_paths_by_k, embedder,
                deterministic=True, return_details=False):
        """
        Args:
            query_emb: (1, emb_dim) — query embedding
            all_paths_by_k: dict {K: [path_dicts]} — paths for each K option
            embedder: Embedder instance for encoding paths
            deterministic: greedy (True) or stochastic (False)

        Returns:
            selected_paths: list of selected path dicts
            details: dict with per-level info
        """
        dev = query_emb.device
        evidence = self.evidence_gru.initialize(query_emb)

        # ═══════════════════════════════════
        # LEVEL 1: Candidate expansion (OptStop)
        # ═══════════════════════════════════
        level1_details = []
        chosen_k = self.K_OPTIONS[-1]  # default: max K
        chosen_paths = []
        k_confidences = []

        for ki, K in enumerate(self.K_OPTIONS):
            paths_k = all_paths_by_k.get(K, [])
            if not paths_k:
                k_confidences.append(torch.tensor([0.0], device=dev))
                continue

            # Encode paths at this K level
            path_embs_list = []
            gnn_scores_list = []
            for p in paths_k:
                texts = []
                for h, r, t in p['path_triples']:
                    texts.extend([str(h), str(r), str(t)])
                texts = list(set(texts))
                embs = embedder.encode_batch(texts)
                path_embs_list.append(embs.mean(dim=0).clone())
                gnn_scores_list.append(p['gnn_score'])

            if not path_embs_list:
                k_confidences.append(torch.tensor([0.0], device=dev))
                continue

            path_embs_k = torch.stack(path_embs_list).to(dev)  # (N_k, D)
            mean_path_emb = path_embs_k.mean(0, keepdim=True)  # (1, D)

            # Update evidence with paths at this K
            evidence = self.evidence_gru.update(evidence, mean_path_emb)

            # K-level features
            mean_gnn = np.mean(gnn_scores_list) if gnn_scores_list else 0.0
            k_features = torch.tensor([[
                K / max(self.K_OPTIONS),
                len(paths_k) / self.max_paths,
                mean_gnn,
                ki / len(self.K_OPTIONS),
            ]], device=dev, dtype=torch.float32)

            # Confidence: is answer reachable with top-K candidates?
            conf = self.confidence_head(evidence, query_emb, k_features)
            k_confidences.append(conf)

            level1_details.append({
                'K': K, 'n_paths': len(paths_k),
                'confidence': conf, 'mean_gnn': mean_gnn,
            })

            # SPRT stopping check
            if deterministic and conf.item() > self.sprt_beta:
                chosen_k = K
                chosen_paths = paths_k
                break

            chosen_k = K
            chosen_paths = paths_k

        # Stochastic K selection (for SCST)
        if not deterministic and k_confidences:
            conf_stack = torch.stack([c.squeeze() for c in k_confidences
                                      if c.numel() > 0])
            k_logits = self.k_policy(torch.cat([
                evidence, query_emb,
                torch.tensor([[
                    len(chosen_paths) / self.max_paths,
                    conf_stack.max().item(),
                    conf_stack.mean().item(),
                    len(conf_stack) / len(self.K_OPTIONS),
                ]], device=dev, dtype=torch.float32)
            ], dim=-1))

            valid_k_count = len([c for c in k_confidences if c.numel() > 0 and c.item() > 0])
            if valid_k_count > 0:
                k_dist = torch.distributions.Categorical(
                    logits=k_logits[:, :len(self.K_OPTIONS)])
                k_idx = k_dist.sample().item()
                k_idx = min(k_idx, len(self.K_OPTIONS) - 1)
                chosen_k = self.K_OPTIONS[k_idx]
                chosen_paths = all_paths_by_k.get(chosen_k, chosen_paths)

        if not chosen_paths:
            if return_details:
                return [], {'level1': level1_details, 'level2': [],
                            'chosen_k': 0, 'n_selected': 0,
                            'final_confidence': torch.tensor([0.0], device=dev)}
            return []

        # ═══════════════════════════════════
        # LEVEL 2: Path relevance ranking (AdaGraph)
        # ═══════════════════════════════════

        # Encode chosen paths
        path_embs_list = []
        gnn_scores_list = []
        lengths_list = []
        for p in chosen_paths:
            texts = []
            for h, r, t in p['path_triples']:
                texts.extend([str(h), str(r), str(t)])
            texts = list(set(texts))
            embs = embedder.encode_batch(texts)
            path_embs_list.append(embs.mean(dim=0).clone())
            gnn_scores_list.append(p['gnn_score'])
            lengths_list.append(float(p['length']))

        path_embs = torch.stack(path_embs_list).unsqueeze(0).to(dev)  # (1, N, D)
        features = torch.stack([
            torch.tensor(gnn_scores_list, device=dev),
            torch.tensor(lengths_list, device=dev) / 5.0,
        ], dim=-1).unsqueeze(0)  # (1, N, 2)

        # Relevance scores
        rel_scores = self.relevance_head(path_embs, query_emb, features)  # (1, N)

        # Pairwise redundancy penalty
        path_norm = F.normalize(path_embs.squeeze(0), dim=-1)  # (N, D)
        sim_matrix = torch.mm(path_norm, path_norm.t())  # (N, N)

        # Greedy selection with redundancy penalty (MMR-style)
        n_paths = len(chosen_paths)
        n_select = min(self.max_paths, n_paths)

        selected_indices = []
        remaining = set(range(n_paths))
        adjusted_scores = rel_scores.squeeze(0).clone()

        for _ in range(n_select):
            if not remaining:
                break

            # Mask already selected
            mask = torch.full((n_paths,), float('-inf'), device=dev)
            for idx in remaining:
                mask[idx] = adjusted_scores[idx]

            best_idx = mask.argmax().item()
            if mask[best_idx] == float('-inf'):
                break

            selected_indices.append(best_idx)
            remaining.discard(best_idx)

            # Penalize similar paths (MMR)
            for idx in list(remaining):
                penalty = self.redundancy_lambda * sim_matrix[best_idx, idx]
                adjusted_scores[idx] = adjusted_scores[idx] - penalty

        selected_paths = [chosen_paths[i] for i in selected_indices]

        # Final confidence
        final_conf_features = torch.tensor([[
            chosen_k / max(self.K_OPTIONS),
            len(selected_paths) / self.max_paths,
            np.mean([p['gnn_score'] for p in selected_paths]) if selected_paths else 0,
            len(selected_paths) / max(len(chosen_paths), 1),
        ]], device=dev, dtype=torch.float32)
        final_conf = self.confidence_head(evidence, query_emb, final_conf_features)

        if return_details:
            return selected_paths, {
                'level1': level1_details,
                'chosen_k': chosen_k,
                'n_candidates': len(chosen_paths),
                'n_selected': len(selected_paths),
                'relevance_scores': rel_scores.detach(),
                'final_confidence': final_conf,
                'k_confidences': k_confidences,
            }
        return selected_paths


# ══════════════════════════════════════════════════════════════
# 5. CGAR Retriever
# ══════════════════════════════════════════════════════════════

class CGARRetriever:
    """End-to-end retriever: GNN-RAG heuristic → CGAR Controller → paths."""

    def __init__(self, gnn, controller, embedder,
                 max_paths_per_pair=3, device='cpu'):
        self.gnn = gnn
        self.controller = controller.to(device)
        self.embedder = embedder
        self.max_paths_per_pair = max_paths_per_pair
        self.device = device

    def retrieve(self, qid, deterministic=True, return_details=False):
        qdata = self.gnn.get(qid)
        if qdata is None:
            return {'triples': [], 'cost': 0, 'context': '',
                    'selected_paths': [], 'all_paths': {}}

        # Pre-compute paths for ALL K options
        all_paths_by_k = {}
        for K in self.controller.K_OPTIONS:
            paths = extract_candidate_paths(
                qid, self.gnn,
                max_candidates=K,
                max_paths_per_pair=self.max_paths_per_pair)
            all_paths_by_k[K] = paths

        query_emb = self.embedder.encode(
            qdata['question']).clone().unsqueeze(0).to(self.device)

        if return_details:
            selected, details = self.controller(
                query_emb, all_paths_by_k, self.embedder,
                deterministic=deterministic, return_details=True)
        else:
            selected = self.controller(
                query_emb, all_paths_by_k, self.embedder,
                deterministic=deterministic)
            details = None

        all_triples = []
        for p in selected:
            all_triples.extend(p['path_triples'])

        result = {
            'triples': all_triples,
            'cost': len(selected),
            'context': '\n'.join(p['path_text'] for p in selected),
            'selected_paths': selected,
            'all_paths': all_paths_by_k,
            'n_total_paths': sum(len(v) for v in all_paths_by_k.values()),
            'n_selected': len(selected),
        }
        if details:
            result['details'] = details
        return result


# ══════════════════════════════════════════════════════════════
# 6. Loss Function
# ══════════════════════════════════════════════════════════════

class CGARLoss(nn.Module):
    """Loss for CGAR Controller.

    L = w_conf * L_confidence + w_rel * L_relevance

    L_confidence: Calibrate candidate confidence to actual recall at each K
    L_relevance:  BCE — gold-entity paths should score higher
    """
    def __init__(self, w_conf=0.5, w_rel=0.5):
        super().__init__()
        self.w_conf = w_conf
        self.w_rel = w_rel

    def forward(self, details, gold_entities, all_paths_by_k):
        device = details['final_confidence'].device

        # ── L_confidence: calibrate at each K ──
        conf_loss = torch.tensor(0.0, device=device)
        n_conf = 0
        for ld in details['level1']:
            K = ld['K']
            paths_k = all_paths_by_k.get(K, [])
            # Actual: what fraction of gold entities reachable at this K?
            reachable = set()
            for p in paths_k:
                for h, r, t in p['path_triples']:
                    reachable.add(str(h)); reachable.add(str(t))
            actual = len(reachable & gold_entities) / max(len(gold_entities), 1)
            target = torch.tensor([actual], device=device)
            conf_loss = conf_loss + F.mse_loss(ld['confidence'], target)
            n_conf += 1
        conf_loss = conf_loss / max(n_conf, 1)

        # ── L_confidence final ──
        # What fraction of gold in selected paths?
        # (passed via details from training loop)
        final_conf = details['final_confidence']
        # Placeholder — actual recall computed externally
        conf_loss = conf_loss + F.mse_loss(final_conf,
                                            torch.tensor([0.5], device=device)) * 0.1

        # ── L_relevance: paths with gold entities should score high ──
        rel_loss = torch.tensor(0.0, device=device)
        if 'relevance_scores' in details and details['relevance_scores'] is not None:
            chosen_k = details['chosen_k']
            paths = all_paths_by_k.get(chosen_k, [])
            if paths:
                targets = torch.zeros(1, len(paths), device=device)
                for i, p in enumerate(paths):
                    ents = set()
                    for h, r, t in p['path_triples']:
                        ents.add(str(h)); ents.add(str(t))
                    if ents & gold_entities:
                        targets[0, i] = 1.0
                # BCE on relevance scores
                rel_scores = details['relevance_scores']
                if rel_scores.size(-1) == targets.size(-1):
                    rel_loss = F.binary_cross_entropy_with_logits(
                        rel_scores, targets, reduction='mean')

        total = self.w_conf * conf_loss + self.w_rel * rel_loss
        return {
            'total': total,
            'confidence': conf_loss.item(),
            'relevance': rel_loss.item() if isinstance(rel_loss, torch.Tensor) else rel_loss,
        }


# ══════════════════════════════════════════════════════════════
# 7. SCST Trainer
# ══════════════════════════════════════════════════════════════

class CGARSCSTTrainer:
    """SCST for CGAR.

    Key advantage over P7: K varies stochastically among {5,10,20,50}
    → greedy and sampled produce DIFFERENT paths → real advantage signal.
    """
    def __init__(self, controller, gnn, embedder,
                 alpha=0.6, beta=0.2, gamma=0.2,
                 n_samples=8, lr=1e-5, device='cpu'):
        self.controller = controller
        self.gnn = gnn
        self.embedder = embedder
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.n_samples = n_samples
        self.device = device
        self.optimizer = torch.optim.AdamW(
            controller.parameters(), lr=lr, weight_decay=0.01)

    def compute_reward(self, selected_paths, gold_entities, max_paths):
        if not selected_paths:
            return -0.1
        collected = set()
        for p in selected_paths:
            for h, r, t in p['path_triples']:
                collected.add(str(h)); collected.add(str(t))
        overlap = len(collected & gold_entities)
        coverage = min(overlap / max(len(gold_entities), 1), 1.0)
        efficiency = 1.0 - len(selected_paths) / max_paths
        return self.alpha * coverage + self.beta * efficiency + self.gamma * 0.5

    def train_step(self, qid):
        qdata = self.gnn.get(qid)
        if qdata is None:
            return None

        gold_entities = set(str(a) for a in self.gnn.get_answers(qid))
        if not gold_entities:
            return None

        # Pre-compute all K paths
        all_paths_by_k = {}
        for K in self.controller.K_OPTIONS:
            all_paths_by_k[K] = extract_candidate_paths(
                qid, self.gnn, max_candidates=K, max_paths_per_pair=3)

        if not any(all_paths_by_k.values()):
            return None

        query_emb = self.embedder.encode(
            qdata['question']).clone().unsqueeze(0).to(self.device)

        # Greedy baseline
        self.controller.eval()
        with torch.no_grad():
            greedy_paths = self.controller(
                query_emb, all_paths_by_k, self.embedder, deterministic=True)
        baseline_r = self.compute_reward(
            greedy_paths, gold_entities, self.controller.max_paths)

        # Stochastic samples — K varies → different paths → real variance
        self.controller.train()
        loss_terms = []
        sample_rewards = []

        for _ in range(self.n_samples):
            sel_paths, details = self.controller(
                query_emb, all_paths_by_k, self.embedder,
                deterministic=False, return_details=True)
            reward = self.compute_reward(
                sel_paths, gold_entities, self.controller.max_paths)
            sample_rewards.append(reward)

            # Actual recall for confidence target
            if sel_paths:
                collected = set()
                for p in sel_paths:
                    for h, r, t in p['path_triples']:
                        collected.add(str(h)); collected.add(str(t))
                actual_recall = len(collected & gold_entities) / max(len(gold_entities), 1)
            else:
                actual_recall = 0.0

            # Loss: advantage-weighted confidence calibration
            target = torch.tensor([actual_recall], device=self.device)
            conf_loss = F.mse_loss(details['final_confidence'], target)
            advantage = reward - baseline_r
            loss_terms.append(advantage * conf_loss + 0.01 * conf_loss)

        scst_loss = sum(loss_terms) / len(loss_terms)
        avg_reward = sum(sample_rewards) / len(sample_rewards)

        self.optimizer.zero_grad()
        scst_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.controller.parameters(), 0.5)
        self.optimizer.step()

        return {
            'avg_reward': avg_reward,
            'baseline': baseline_r,
            'loss': scst_loss.item(),
        }


# ══════════════════════════════════════════════════════════════
# 8. Evaluation
# ══════════════════════════════════════════════════════════════

def evaluate_cgar(retriever, gnn, qids, verbose=False):
    """Evaluate CGAR retriever."""
    results = defaultdict(list)
    for qid in qids:
        gold = set(str(a) for a in gnn.get_answers(qid))
        if not gold:
            continue
        ret = retriever.retrieve(qid, deterministic=True)
        collected = set()
        for h, r, t in ret['triples']:
            collected.add(str(h)); collected.add(str(t))
        overlap = len(collected & gold)
        cov = overlap / max(len(gold), 1)
        prec = overlap / max(len(collected), 1)
        f1 = 2*prec*cov / max(prec+cov, 1e-8)
        results['coverage'].append(cov)
        results['precision'].append(prec)
        results['recall'].append(cov)
        results['f1'].append(f1)
        results['cost'].append(ret['cost'])
        results['chosen_k'].append(ret.get('details', {}).get('chosen_k', 0)
                                    if ret.get('details') else 0)
        if verbose and len(results['f1']) <= 5:
            det = ret.get('details', {})
            print(f"  Q={qid}: K={det.get('chosen_k','?')} "
                  f"sel={ret['n_selected']}/{ret.get('n_total_paths','?')} "
                  f"cov={cov:.3f} prec={prec:.3f} f1={f1:.3f}")
    return {k: np.mean(v) if v else 0 for k, v in results.items()}
