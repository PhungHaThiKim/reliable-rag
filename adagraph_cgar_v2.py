"""
CGAR v2.1: Confidence-Guided Adaptive Retrieval
=================================================
Fixes from v1:
  1. relevance_scores NOT detached → gradient flows to relevance_head
  2. SCST uses epsilon-greedy (ε=0.3) instead of pure softmax over K
  3. conf_loss uses actual_recall (not placeholder 0.5)
  4. adjusted_scores in-place fix (clone before MMR penalty)
  5. HotpotQA support added
  6. CWQ loading enforces separate pkl (no glob fallback to WebQSP)

Fixes from v2 → v2.1 (this file):
  A. SCST is now PROPER REINFORCE: -advantage * log_prob(action) + calibration
     (v2 used `advantage * conf_loss` which anti-learns when advantage<0)
  B. K-policy sampling stores log_prob (no more .item() detach)
     stochastic path selection via Categorical-on-MMR-adjusted scores
  C. compute_reward.gamma term is now a real diversity score
     (Jaccard-based path overlap), no more `* 0.5` placeholder

Root cause of v1 failure:
  relevance_scores.detach() on line 357 → BCE loss=log(2)=0.693 constant
  → relevance_head NEVER received gradient → path ranking = random
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random as _random
from collections import defaultdict

from grl_rag import (
    GNNHeuristic, Embedder, extract_candidate_paths,
    extract_candidate_paths_by_k, path_to_string, compute_reward
)


# ══════════════════════════════════════════════════════════════
# 1. Heads
# ══════════════════════════════════════════════════════════════

class CandidateConfidenceHead(nn.Module):
    """P(answer reachable | top-K candidates, query)."""
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
        return self.net(torch.cat([evidence, query_emb, k_features], -1)).squeeze(-1)


class PathRelevanceHead(nn.Module):
    """Score each path by query-relevance."""
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
        n = path_embs.size(1)
        q_exp = query_emb.unsqueeze(1).expand(-1, n, -1)
        return self.net(torch.cat([path_embs, q_exp, features], -1)).squeeze(-1)


class EvidenceGRU(nn.Module):
    """GRU evidence accumulator."""
    def __init__(self, emb_dim=384):
        super().__init__()
        self.gru = nn.GRUCell(emb_dim, emb_dim)
        self.init_proj = nn.Linear(emb_dim, emb_dim)
    def initialize(self, query_emb):
        return self.init_proj(query_emb)
    def update(self, state, new_evidence):
        return self.gru(new_evidence, state)


# ══════════════════════════════════════════════════════════════
# 2. CGAR Controller v2
# ══════════════════════════════════════════════════════════════

class CGARController(nn.Module):
    """Confidence-Guided Adaptive Retrieval Controller v2.

    Key fix: relevance_scores kept LIVE (not detached) in return dict.
    """
    K_OPTIONS = [5, 10, 20, 50]

    def __init__(self, emb_dim=384, hidden_dim=256,
                 max_paths=10, sprt_beta=0.85,
                 redundancy_lambda=0.3, epsilon=0.3,
                 rel_threshold=None, min_select=1):
        """
        rel_threshold: optional logit threshold for ADAPTIVE PATH COUNT.
            v2.1 always selected min(max_paths, n_paths) paths, so the only
            adaptive quantity was K (the candidate pool). With a threshold,
            the MMR loop stops once the best remaining adjusted relevance
            logit drops below it (deterministic mode only; keeps >= min_select).
            0.0 == "stop when P(path relevant) < 0.5". None == legacy behaviour.
        """
        super().__init__()
        self.emb_dim = emb_dim
        self.max_paths = max_paths
        self.sprt_beta = sprt_beta
        self.redundancy_lambda = redundancy_lambda
        self.epsilon = epsilon  # for epsilon-greedy SCST
        self.rel_threshold = rel_threshold
        self.min_select = min_select

        self.confidence_head = CandidateConfidenceHead(emb_dim, hidden_dim // 2)
        self.evidence_gru = EvidenceGRU(emb_dim)
        self.relevance_head = PathRelevanceHead(emb_dim, hidden_dim)
        self.k_policy = nn.Sequential(
            nn.Linear(emb_dim * 2 + 4, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, len(self.K_OPTIONS)),
        )

    @staticmethod
    def _path_embs(paths, embedder, cache):
        """Mean entity/relation embedding per path, memoised per forward call
        (the same path dict appears at every K level >= its cand_rank)."""
        out = []
        for p in paths:
            key = p.get('path_text') or id(p)
            if key not in cache:
                texts = list(set(str(x) for triple in p['path_triples'] for x in triple))
                cache[key] = embedder.encode_batch(texts).mean(dim=0).clone()
            out.append(cache[key])
        return out

    def forward(self, query_emb, all_paths_by_k, embedder,
                deterministic=True, return_details=False):
        dev = query_emb.device
        evidence = self.evidence_gru.initialize(query_emb)
        emb_cache = {}

        # ═══ LEVEL 1: Candidate expansion (OptStop) ═══
        level1_details = []
        chosen_k = self.K_OPTIONS[-1]
        chosen_paths = []
        k_confidences = []
        # *** REINFORCE: track log_prob of K choice (None if no policy used)
        log_prob_k = None

        for ki, K in enumerate(self.K_OPTIONS):
            paths_k = all_paths_by_k.get(K, [])
            if not paths_k:
                k_confidences.append(torch.tensor([0.0], device=dev))
                continue

            path_embs_list = self._path_embs(paths_k, embedder, emb_cache)
            gnn_scores_list = [p['gnn_score'] for p in paths_k]

            if not path_embs_list:
                k_confidences.append(torch.tensor([0.0], device=dev))
                continue

            path_embs_k = torch.stack(path_embs_list).to(dev)
            mean_path_emb = path_embs_k.mean(0, keepdim=True)
            evidence = self.evidence_gru.update(evidence, mean_path_emb)

            mean_gnn = np.mean(gnn_scores_list) if gnn_scores_list else 0.0
            k_features = torch.tensor([[
                K / max(self.K_OPTIONS), len(paths_k) / self.max_paths,
                mean_gnn, ki / len(self.K_OPTIONS),
            ]], device=dev, dtype=torch.float32)

            conf = self.confidence_head(evidence, query_emb, k_features)
            k_confidences.append(conf)
            level1_details.append({
                'K': K, 'n_paths': len(paths_k),
                'confidence': conf, 'mean_gnn': mean_gnn,
            })

            if deterministic and conf.item() > self.sprt_beta:
                chosen_k = K
                chosen_paths = paths_k
                break
            chosen_k = K
            chosen_paths = paths_k

        # Stochastic K: EPSILON-GREEDY with proper log_prob bookkeeping
        if not deterministic and k_confidences:
            valid_confs = [c for c in k_confidences if c.numel() > 0 and c.item() > 0]
            # Always run k_policy so we have access to the distribution for log_prob
            conf_feats = torch.tensor([[
                len(chosen_paths) / self.max_paths,
                max(c.item() for c in valid_confs) if valid_confs else 0,
                np.mean([c.item() for c in valid_confs]) if valid_confs else 0,
                len(valid_confs) / max(len(self.K_OPTIONS), 1),
            ]], device=dev, dtype=torch.float32)
            k_logits = self.k_policy(torch.cat([evidence, query_emb, conf_feats], -1))
            k_dist = torch.distributions.Categorical(logits=k_logits.squeeze(0))

            if valid_confs and _random.random() < self.epsilon:
                # Random K (uniform exploration)
                k_idx = _random.randint(0, len(self.K_OPTIONS) - 1)
            else:
                # Greedy K (sample from policy)
                k_idx_t = k_dist.sample()
                k_idx = k_idx_t.item()
            # *** Bug B fix: keep gradient via log_prob of action taken ***
            k_idx_tensor = torch.tensor(min(k_idx, len(self.K_OPTIONS) - 1),
                                        device=dev, dtype=torch.long)
            log_prob_k = k_dist.log_prob(k_idx_tensor)

            k_idx = min(k_idx, len(self.K_OPTIONS) - 1)
            chosen_k = self.K_OPTIONS[k_idx]
            chosen_paths = all_paths_by_k.get(chosen_k, chosen_paths)

        if not chosen_paths:
            empty_details = {'level1': level1_details, 'chosen_k': 0,
                             'n_candidates': 0, 'n_selected': 0,
                             'relevance_scores': None,
                             'final_confidence': torch.tensor([0.0], device=dev),
                             'k_confidences': k_confidences,
                             'log_prob_k': log_prob_k,
                             'log_prob_paths': [],
                             'selected_path_embs': None}
            return ([], empty_details) if return_details else []

        # ═══ LEVEL 2: Path relevance ranking ═══
        path_embs_list = self._path_embs(chosen_paths, embedder, emb_cache)
        gnn_scores_list = [p['gnn_score'] for p in chosen_paths]
        lengths_list = [float(p['length']) for p in chosen_paths]

        path_embs = torch.stack(path_embs_list).unsqueeze(0).to(dev)
        features = torch.stack([
            torch.tensor(gnn_scores_list, device=dev),
            torch.tensor(lengths_list, device=dev) / 5.0,
        ], dim=-1).unsqueeze(0)

        # Relevance scores — LIVE tensor (gradient flows!)
        rel_scores = self.relevance_head(path_embs, query_emb, features)

        # MMR selection — deterministic: argmax; stochastic: Categorical sample
        n_paths = len(chosen_paths)
        n_select = min(self.max_paths, n_paths)
        path_norm = F.normalize(path_embs.squeeze(0), dim=-1)
        sim_matrix = torch.mm(path_norm, path_norm.t())

        selected_indices = []
        log_prob_paths = []  # *** REINFORCE: log_prob of each selection step
        rel_live = rel_scores.squeeze(0)  # (n_paths,) — gradient flows

        # Cumulative redundancy tensor (live, shape n_paths) — penalises already-selected
        cum_redundancy = torch.zeros(n_paths, device=dev)

        for step in range(n_select):
            # Live, gradient-carrying adjusted scores
            adjusted = rel_live - self.redundancy_lambda * cum_redundancy
            # Mask out already-selected
            if selected_indices:
                mask_idx = torch.tensor(selected_indices, device=dev, dtype=torch.long)
                adjusted = adjusted.index_fill(0, mask_idx, float('-1e9'))

            if deterministic:
                best_idx = adjusted.argmax().item()
                if adjusted[best_idx].item() <= -1e8:
                    break
                # Adaptive path count: stop when the best remaining path is
                # no longer judged relevant (logit < threshold).
                if (self.rel_threshold is not None
                        and len(selected_indices) >= self.min_select
                        and adjusted[best_idx].item() < self.rel_threshold):
                    break
            else:
                # Categorical sampling for REINFORCE
                step_dist = torch.distributions.Categorical(logits=adjusted)
                idx_t = step_dist.sample()
                log_prob_paths.append(step_dist.log_prob(idx_t))
                best_idx = idx_t.item()
                if adjusted[best_idx].item() <= -1e8:
                    break

            selected_indices.append(best_idx)
            # Add similarity row of newly-selected to cum_redundancy (gradient through sim_matrix)
            cum_redundancy = cum_redundancy + sim_matrix[best_idx]

        selected_paths = [chosen_paths[i] for i in selected_indices]
        # Embeddings of selected paths (for diversity reward / external use)
        if selected_indices:
            sel_emb_idx = torch.tensor(selected_indices, device=dev, dtype=torch.long)
            selected_path_embs = path_embs.squeeze(0).index_select(0, sel_emb_idx)
        else:
            selected_path_embs = None

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
                'relevance_scores': rel_scores,    # *** LIVE — NOT detached ***
                'final_confidence': final_conf,    # *** LIVE ***
                'k_confidences': k_confidences,
                'log_prob_k': log_prob_k,          # *** REINFORCE: K-policy log_prob (or None)
                'log_prob_paths': log_prob_paths,  # *** REINFORCE: per-step path log_prob
                'selected_path_embs': selected_path_embs,  # for diversity reward
                'selected_indices': selected_indices,
            }
        return selected_paths


# ══════════════════════════════════════════════════════════════
# 3. Retriever
# ══════════════════════════════════════════════════════════════

class CGARRetriever:
    def __init__(self, gnn, controller, embedder, max_paths_per_pair=3, device='cpu'):
        self.gnn = gnn
        self.controller = controller.to(device)
        self.embedder = embedder
        self.max_paths_per_pair = max_paths_per_pair
        self.device = device

    def retrieve(self, qid, deterministic=True, return_details=False):
        qdata = self.gnn.get(qid)
        if qdata is None:
            return {'triples': [], 'cost': 0, 'context': '',
                    'selected_paths': [], 'all_paths': {}, 'n_total_paths': 0, 'n_selected': 0}

        all_paths_by_k = extract_candidate_paths_by_k(
            qid, self.gnn, k_options=self.controller.K_OPTIONS,
            max_paths_per_pair=self.max_paths_per_pair)

        query_emb = self.embedder.encode(qdata['question']).clone().unsqueeze(0).to(self.device)

        if return_details:
            selected, details = self.controller(
                query_emb, all_paths_by_k, self.embedder,
                deterministic=deterministic, return_details=True)
        else:
            selected = self.controller(
                query_emb, all_paths_by_k, self.embedder,
                deterministic=deterministic)
            details = None

        all_triples = [t for p in selected for t in p['path_triples']]
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
# 4. Loss (FIXED — no more placeholder targets)
# ══════════════════════════════════════════════════════════════

class CGARLoss(nn.Module):
    """v2: actual_recall passed in, relevance_scores are LIVE."""
    def __init__(self, w_conf=0.4, w_rel=0.6):
        super().__init__()
        self.w_conf = w_conf
        self.w_rel = w_rel

    def forward(self, details, gold_entities, all_paths_by_k, actual_recall=None):
        device = details['final_confidence'].device

        # L_confidence: calibrate at each K level
        conf_loss = torch.tensor(0.0, device=device)
        n_conf = 0
        for ld in details['level1']:
            paths_k = all_paths_by_k.get(ld['K'], [])
            reachable = set()
            for p in paths_k:
                for h, r, t in p['path_triples']:
                    reachable.add(str(h)); reachable.add(str(t))
            actual = len(reachable & gold_entities) / max(len(gold_entities), 1)
            target = torch.tensor([actual], device=device)
            conf_loss = conf_loss + F.mse_loss(ld['confidence'], target)
            n_conf += 1
        conf_loss = conf_loss / max(n_conf, 1)

        # Final confidence calibration with actual recall
        if actual_recall is not None:
            conf_loss = conf_loss + F.mse_loss(
                details['final_confidence'],
                torch.tensor([actual_recall], device=device))

        # L_relevance: BCE on LIVE relevance scores
        rel_loss = torch.tensor(0.0, device=device)
        rel_scores = details.get('relevance_scores')
        if rel_scores is not None:
            chosen_k = details['chosen_k']
            paths = all_paths_by_k.get(chosen_k, [])
            if paths and rel_scores.size(-1) == len(paths):
                targets = torch.zeros(1, len(paths), device=device)
                for i, p in enumerate(paths):
                    ents = set()
                    for h, r, t in p['path_triples']:
                        ents.add(str(h)); ents.add(str(t))
                    if ents & gold_entities:
                        targets[0, i] = 1.0
                # Weighted BCE: gold paths are rare, upweight them
                n_pos = targets.sum().item()
                n_neg = targets.numel() - n_pos
                if n_pos > 0 and n_neg > 0:
                    pos_weight = torch.tensor([n_neg / n_pos], device=device)
                    rel_loss = F.binary_cross_entropy_with_logits(
                        rel_scores, targets, pos_weight=pos_weight, reduction='mean')
                else:
                    rel_loss = F.binary_cross_entropy_with_logits(
                        rel_scores, targets, reduction='mean')

        total = self.w_conf * conf_loss + self.w_rel * rel_loss
        return {
            'total': total,
            'confidence': conf_loss.item() if isinstance(conf_loss, torch.Tensor) else conf_loss,
            'relevance': rel_loss.item() if isinstance(rel_loss, torch.Tensor) else rel_loss,
        }


# ══════════════════════════════════════════════════════════════
# 5. SCST Trainer (FIXED — epsilon-greedy)
# ══════════════════════════════════════════════════════════════

class CGARSCSTTrainer:
    """v2.1: PROPER REINFORCE with log_prob from policy + diversity reward.

    Loss = -mean(advantage * log_prob_total) + w_calib * conf_calibration
    where log_prob_total = log_prob(K) + sum_step log_prob(path_t)
    Optional entropy bonus encourages exploration.
    """
    def __init__(self, controller, gnn, embedder,
                 alpha=0.6, beta=0.2, gamma=0.2,
                 n_samples=8, lr=5e-5, device='cpu',
                 w_calib=0.1, entropy_coef=0.01, max_paths_per_pair=3):
        self.controller = controller
        self.gnn = gnn
        self.embedder = embedder
        self.max_paths_per_pair = max_paths_per_pair
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.n_samples = n_samples
        self.device = device
        self.w_calib = w_calib
        self.entropy_coef = entropy_coef
        self.optimizer = torch.optim.AdamW(controller.parameters(), lr=lr, weight_decay=0.01)

    @staticmethod
    def _diversity_jaccard(selected_paths):
        """Bug C fix: real diversity = 1 - mean Jaccard overlap of path entity sets."""
        if not selected_paths or len(selected_paths) <= 1:
            return 1.0
        ent_sets = []
        for p in selected_paths:
            ents = set()
            for h, _r, t in p['path_triples']:
                ents.add(str(h)); ents.add(str(t))
            ent_sets.append(ents)
        n = len(ent_sets)
        sims = []
        for i in range(n):
            for j in range(i + 1, n):
                inter = len(ent_sets[i] & ent_sets[j])
                union = len(ent_sets[i] | ent_sets[j])
                sims.append(inter / union if union > 0 else 0.0)
        avg_sim = sum(sims) / len(sims) if sims else 0.0
        return 1.0 - avg_sim

    def compute_reward(self, selected_paths, gold_entities, max_paths):
        if not selected_paths:
            return -0.1
        collected = set()
        for p in selected_paths:
            for h, _r, t in p['path_triples']:
                collected.add(str(h)); collected.add(str(t))
        overlap = len(collected & gold_entities)
        coverage = min(overlap / max(len(gold_entities), 1), 1.0)
        efficiency = 1.0 - len(selected_paths) / max_paths
        diversity = self._diversity_jaccard(selected_paths)  # Bug C fix
        return self.alpha * coverage + self.beta * efficiency + self.gamma * diversity

    def train_step(self, qid):
        qdata = self.gnn.get(qid)
        if qdata is None:
            return None
        gold = set(str(a) for a in self.gnn.get_answers(qid))
        if not gold:
            return None

        all_paths_by_k = extract_candidate_paths_by_k(
            qid, self.gnn, k_options=self.controller.K_OPTIONS,
            max_paths_per_pair=self.max_paths_per_pair)
        if not any(all_paths_by_k.values()):
            return None

        query_emb = self.embedder.encode(qdata['question']).clone().unsqueeze(0).to(self.device)

        # Baseline (greedy)
        self.controller.eval()
        with torch.no_grad():
            greedy_paths = self.controller(query_emb, all_paths_by_k, self.embedder, deterministic=True)
        baseline_r = self.compute_reward(greedy_paths, gold, self.controller.max_paths)

        # Samples (stochastic — k_policy + Categorical path sampling)
        self.controller.train()
        pg_terms = []         # policy-gradient terms (-advantage * log_prob)
        calib_terms = []      # confidence calibration MSE (always pulls toward target)
        sample_rewards = []

        for _ in range(self.n_samples):
            sel, details = self.controller(
                query_emb, all_paths_by_k, self.embedder,
                deterministic=False, return_details=True)
            reward = self.compute_reward(sel, gold, self.controller.max_paths)
            sample_rewards.append(reward)

            # Actual recall (for confidence calibration target)
            if sel:
                collected = set()
                for p in sel:
                    for h, _r, t in p['path_triples']:
                        collected.add(str(h)); collected.add(str(t))
                actual_recall = len(collected & gold) / max(len(gold), 1)
            else:
                actual_recall = 0.0

            # *** Bug A fix: PROPER REINFORCE ***
            # Total log_prob over the trajectory (K choice + per-step path choices)
            log_prob_total = torch.tensor(0.0, device=self.device)
            lp_k = details.get('log_prob_k')
            if lp_k is not None:
                log_prob_total = log_prob_total + lp_k.sum()
            for lp in details.get('log_prob_paths', []):
                log_prob_total = log_prob_total + lp
            advantage = reward - baseline_r  # scalar
            # Negative because we minimise loss but want to maximise expected reward
            pg_terms.append(-advantage * log_prob_total)

            # Calibration term — always positive, pulls confidence toward actual recall
            target = torch.tensor([actual_recall], device=self.device)
            calib_terms.append(F.mse_loss(details['final_confidence'], target))

        pg_loss = torch.stack(pg_terms).mean() if pg_terms else torch.tensor(0.0, device=self.device)
        calib_loss = torch.stack(calib_terms).mean() if calib_terms else torch.tensor(0.0, device=self.device)
        scst_loss = pg_loss + self.w_calib * calib_loss
        avg_reward = sum(sample_rewards) / len(sample_rewards)

        self.optimizer.zero_grad()
        scst_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.controller.parameters(), 0.5)
        self.optimizer.step()

        return {
            'avg_reward': avg_reward,
            'baseline': baseline_r,
            'loss': scst_loss.item(),
            'pg_loss': pg_loss.item(),
            'calib_loss': calib_loss.item(),
            'advantage_nonzero': sum(1 for r in sample_rewards if abs(r - baseline_r) > 0.01),
        }


# ══════════════════════════════════════════════════════════════
# 6. Evaluation
# ══════════════════════════════════════════════════════════════

def evaluate_cgar(retriever, gnn, qids, verbose=False):
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
        f1 = 2 * prec * cov / max(prec + cov, 1e-8)
        results['coverage'].append(cov)
        results['precision'].append(prec)
        results['recall'].append(cov)
        results['f1'].append(f1)
        results['cost'].append(ret['cost'])
        det = ret.get('details', {})
        results['chosen_k'].append(det.get('chosen_k', 0) if det else 0)
        if verbose and len(results['f1']) <= 5:
            print(f"  Q={qid}: K={det.get('chosen_k','?') if det else '?'} "
                  f"sel={ret['n_selected']}/{ret.get('n_total_paths','?')} "
                  f"cov={cov:.3f} f1={f1:.3f}")
    return {k: np.mean(v) if v else 0 for k, v in results.items()}


def path_entities(paths):
    ents = set()
    for p in paths:
        for h, _r, t in p['path_triples']:
            ents.add(str(h)); ents.add(str(t))
    return ents


def evaluate_cgar_records(retriever, gnn, qids, verbose=False):
    """Per-question records (for calibration / significance / stratification).

    Each record: qid, coverage (= answer recall), precision, f1, hit_ctx
    (any gold entity inside the selected context), cost (#paths), n_triples,
    chosen_k, confidence (final head output), latency_ms, selected_paths
    (texts). Aggregate with kgqa_bench.metrics.summarize_records.
    """
    import time as _time
    records = []
    for qid in qids:
        gold = set(str(a) for a in gnn.get_answers(qid))
        if not gold:
            continue
        t0 = _time.perf_counter()
        with torch.no_grad():
            ret = retriever.retrieve(qid, deterministic=True, return_details=True)
        dt = (_time.perf_counter() - t0) * 1000
        collected = path_entities(ret['selected_paths'])
        overlap = len(collected & gold)
        cov = overlap / max(len(gold), 1)
        prec = overlap / max(len(collected), 1)
        f1 = 2 * prec * cov / max(prec + cov, 1e-8)
        det = ret.get('details') or {}
        conf = det.get('final_confidence')
        records.append({
            'qid': qid,
            'coverage': cov, 'precision': prec, 'f1': f1,
            'hit_ctx': int(overlap > 0),
            'cost': ret['cost'], 'n_triples': len(ret['triples']),
            'n_total_paths': ret.get('n_total_paths', 0),
            'chosen_k': int(det.get('chosen_k', 0) or 0),
            'confidence': float(conf.item()) if conf is not None else None,
            'latency_ms': dt,
            'selected_paths': [p['path_text'] for p in ret['selected_paths']],
        })
        if verbose and len(records) <= 5:
            r = records[-1]
            print(f"  Q={qid}: K={r['chosen_k']} sel={r['cost']}/{r['n_total_paths']} "
                  f"cov={cov:.3f} f1={f1:.3f} conf={r['confidence']}")
    return records


def evaluate_baseline(retriever, gnn, qids, verbose=False):
    """Evaluate non-CGAR retrievers (GreedyGNN, GNN-RAG baselines)."""
    results = defaultdict(list)
    for qid in qids:
        gold = set(str(a) for a in gnn.get_answers(qid))
        if not gold:
            continue
        ret = retriever.retrieve(qid)
        collected = set()
        for h, r, t in ret['triples']:
            collected.add(str(h)); collected.add(str(t))
        overlap = len(collected & gold)
        cov = overlap / max(len(gold), 1)
        prec = overlap / max(len(collected), 1)
        f1 = 2 * prec * cov / max(prec + cov, 1e-8)
        results['coverage'].append(cov)
        results['precision'].append(prec)
        results['recall'].append(cov)
        results['f1'].append(f1)
        results['cost'].append(ret['cost'])
        if verbose and len(results['f1']) <= 5:
            print(f"  Q={qid}: cov={cov:.3f} f1={f1:.3f} cost={ret['cost']}")
    return {k: np.mean(v) if v else 0 for k, v in results.items()}
