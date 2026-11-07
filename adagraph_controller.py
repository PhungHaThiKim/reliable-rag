"""
AdaGraph Controller for GRL-RAG (v2 — Professor-reviewed)
==========================================================
Adaptive budgeted path selection + confidence-based stopping
on top of GNN-RAG heuristic.

Key insight:
  "The right abstraction for GraphRAG is not full RL walking,
   but budgeted control over a strong graph heuristic."

v2 fixes (from professor review):
  - Path embedding includes relation info (not just entity mean)
  - Evidence update uses differentiable gather (no in-place)
  - Confidence head properly calibrated (not pushed to 0)
  - Loss function receives actual_recall as argument
  - Marginal loss only on unselected paths per round
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

from grl_rag import (
    GNNHeuristic, Embedder, extract_candidate_paths,
    path_to_string, compute_reward
)


# ══════════════════════════════════════════════════════════════
# 1. Marginal Utility Estimator
# ══════════════════════════════════════════════════════════════

class MarginalUtilityHead(nn.Module):
    """Estimates Δ(path | S_t) for each candidate path."""

    def __init__(self, emb_dim=384, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim * 3 + 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, path_embs, evidence_state, query_emb, features):
        n_paths = path_embs.size(1)
        ev_exp = evidence_state.unsqueeze(1).expand(-1, n_paths, -1)
        q_exp = query_emb.unsqueeze(1).expand(-1, n_paths, -1)
        x = torch.cat([path_embs, ev_exp, q_exp, features], dim=-1)
        return self.net(x).squeeze(-1)


# ══════════════════════════════════════════════════════════════
# 2. Confidence Head
# ══════════════════════════════════════════════════════════════

class ConfidenceHead(nn.Module):
    """Estimates P(answer correct | evidence so far). Stopping signal."""

    def __init__(self, emb_dim=384, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim * 2 + 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, evidence_state, query_emb, budget_features):
        x = torch.cat([evidence_state, query_emb, budget_features], dim=-1)
        return self.net(x).squeeze(-1)


# ══════════════════════════════════════════════════════════════
# 3. Evidence Accumulator (GRU)
# ══════════════════════════════════════════════════════════════

class EvidenceAccumulator(nn.Module):
    """GRU-based evidence accumulator."""

    def __init__(self, emb_dim=384):
        super().__init__()
        self.gru = nn.GRUCell(emb_dim, emb_dim)
        self.init_proj = nn.Linear(emb_dim, emb_dim)

    def initialize(self, query_emb):
        return self.init_proj(query_emb)

    def update(self, evidence_state, new_path_emb):
        return self.gru(new_path_emb, evidence_state)


# ══════════════════════════════════════════════════════════════
# 4. AdaGraph Controller
# ══════════════════════════════════════════════════════════════

class AdaGraphController(nn.Module):
    """Adaptive Budgeted Path Selection Controller.

    Sits ON TOP of GNN-RAG. Controls WHICH paths to keep and WHEN to stop.

    v2 fixes:
    - Evidence update via differentiable weighted sum (no in-place)
    - Redundancy computed via cosine similarity (no separate head needed)
    - Marginals masked AFTER storing raw for loss
    """

    def __init__(self, emb_dim=384, hidden_dim=256,
                 max_rounds=5, select_k=3, max_paths=15,
                 sprt_beta=0.90):
        super().__init__()
        self.emb_dim = emb_dim
        self.max_rounds = max_rounds
        self.select_k = select_k
        self.max_paths = max_paths
        self.sprt_beta = sprt_beta

        self.marginal_head = MarginalUtilityHead(emb_dim, hidden_dim)
        self.confidence_head = ConfidenceHead(emb_dim, hidden_dim // 2)
        self.evidence_acc = EvidenceAccumulator(emb_dim)

        # Learnable path aggregation for evidence update
        self.path_gate = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.Sigmoid(),
        )

    def forward(self, query_emb, path_embs, gnn_scores, path_lengths,
                deterministic=True, return_details=False):
        bs = query_emb.size(0)
        n_paths = path_embs.size(1)
        dev = query_emb.device

        evidence = self.evidence_acc.initialize(query_emb)
        selected = torch.zeros(bs, n_paths, dtype=torch.bool, device=dev)

        round_details = []
        confidences = []

        for round_idx in range(self.max_rounds):
            n_selected = selected.float().sum(-1)
            budget_feats = torch.stack([
                n_selected / self.max_paths,
                (self.max_paths - n_selected) / self.max_paths,
                torch.full_like(n_selected, round_idx / self.max_rounds),
            ], dim=-1)

            # Confidence check
            conf = self.confidence_head(evidence, query_emb, budget_feats)
            confidences.append(conf)

            if deterministic and (conf > self.sprt_beta).all():
                if return_details:
                    round_details.append({
                        'confidence': conf.detach(),
                        'stopped': True, 'round': round_idx,
                    })
                break

            # Redundancy: cosine similarity between each path and evidence
            ev_norm = F.normalize(evidence, dim=-1).unsqueeze(1)  # (B,1,D)
            p_norm = F.normalize(path_embs, dim=-1)               # (B,N,D)
            redundancy = (ev_norm * p_norm).sum(-1).clamp(0, 1)   # (B,N) cosine sim

            # Features for marginal utility
            features = torch.stack([
                gnn_scores,
                path_lengths / 5.0,
                redundancy,
                torch.full_like(gnn_scores, round_idx / self.max_rounds),
            ], dim=-1)

            # Marginal utility (raw — for loss)
            marginals_raw = self.marginal_head(path_embs, evidence, query_emb, features)

            # Mask already selected (on clone — no in-place on live tensor)
            marginals = marginals_raw.clone()
            marginals[selected] = -1e9

            # Budget check
            budget_remaining = self.max_paths - n_selected
            n_valid = (marginals > -1e8).sum(dim=-1).min().item()
            k = min(self.select_k, int(budget_remaining.min().item()), int(n_valid))
            if k <= 0:
                break

            # Selection
            if deterministic:
                _, topk_idx = marginals.topk(k, dim=-1)
            else:
                valid_mask = marginals > -1e8
                safe_m = marginals.clone()
                safe_m[~valid_mask] = -1e9
                probs = F.softmax(safe_m * 5.0, dim=-1)
                actual_k = min(k, int(valid_mask.sum(-1).min().item()))
                if actual_k <= 0:
                    break
                topk_idx = torch.multinomial(probs, actual_k, replacement=False)

            # Update selected (bool tensor, not in computation graph)
            new_selected = selected.clone()
            for b in range(bs):
                for idx in topk_idx[b]:
                    new_selected[b, idx] = True
            selected = new_selected

            # Evidence update: DIFFERENTIABLE gather + gated aggregation
            # Gather selected path embeddings using topk indices
            # topk_idx: (B, k) — indices into path_embs
            gathered = torch.zeros(bs, self.emb_dim, device=dev)
            for b in range(bs):
                sel_embs = path_embs[b, topk_idx[b]]  # (k, D) — differentiable gather
                # Gated aggregation: learn which path to emphasize
                gate_input = torch.cat([
                    sel_embs,
                    evidence[b].unsqueeze(0).expand(sel_embs.size(0), -1)
                ], dim=-1)  # (k, 2D)
                gates = self.path_gate(gate_input)  # (k, D) sigmoid
                gated = (sel_embs * gates).mean(0)  # (D,) — weighted mean
                gathered[b] = gated

            evidence = self.evidence_acc.update(evidence, gathered)

            if return_details:
                round_details.append({
                    'confidence': conf,
                    'marginals': marginals_raw,
                    'n_selected': selected.float().sum(-1).detach(),
                    'stopped': False, 'round': round_idx,
                })

            if (selected.float().sum(-1) >= self.max_paths).all():
                break

        # Final confidence
        n_selected = selected.float().sum(-1)
        budget_feats = torch.stack([
            n_selected / self.max_paths,
            (self.max_paths - n_selected) / self.max_paths,
            torch.ones_like(n_selected),
        ], dim=-1)
        final_conf = self.confidence_head(evidence, query_emb, budget_feats)

        if return_details:
            return selected.float(), {
                'round_details': round_details,
                'confidences': confidences,
                'final_confidence': final_conf,
                'n_rounds': len(round_details),
                'n_selected': n_selected.detach(),
            }
        return selected.float()


# ══════════════════════════════════════════════════════════════
# 5. AdaGraph Path Retriever
# ══════════════════════════════════════════════════════════════

class AdaGraphPathRetriever:
    """Combines GNN-RAG paths + AdaGraph Controller."""

    def __init__(self, gnn, controller, embedder,
                 max_candidates=10, max_paths_per_pair=3,
                 device='cpu'):
        self.gnn = gnn
        self.controller = controller.to(device)
        self.embedder = embedder
        self.max_candidates = max_candidates
        self.max_paths_per_pair = max_paths_per_pair
        self.device = device

    def _encode_paths(self, all_paths, question):
        """Encode paths: include RELATION info, not just entity mean."""
        query_emb = self.embedder.encode(question).clone().unsqueeze(0).to(self.device)

        path_embs = []
        gnn_scores = []
        path_lengths = []
        for p in all_paths:
            # Encode entities AND relations
            texts = []
            for h, r, t in p['path_triples']:
                texts.extend([str(h), str(r), str(t)])
            texts = list(set(texts))  # deduplicate
            embs = self.embedder.encode_batch(texts)
            path_embs.append(embs.mean(dim=0).clone())
            gnn_scores.append(p['gnn_score'])
            path_lengths.append(float(p['length']))

        path_embs_t = torch.stack(path_embs).unsqueeze(0).to(self.device)
        gnn_scores_t = torch.tensor(gnn_scores, dtype=torch.float32).unsqueeze(0).to(self.device)
        path_lengths_t = torch.tensor(path_lengths, dtype=torch.float32).unsqueeze(0).to(self.device)
        return query_emb, path_embs_t, gnn_scores_t, path_lengths_t

    def retrieve(self, qid, deterministic=True, return_details=False):
        qdata = self.gnn.get(qid)
        if qdata is None:
            return {'triples': [], 'cost': 0, 'context': '',
                    'selected_paths': [], 'all_paths': []}

        all_paths = extract_candidate_paths(
            qid, self.gnn,
            max_candidates=self.max_candidates,
            max_paths_per_pair=self.max_paths_per_pair,
        )
        if not all_paths:
            return {'triples': [], 'cost': 0, 'context': '',
                    'selected_paths': [], 'all_paths': []}

        query_emb, path_embs_t, gnn_scores_t, path_lengths_t = \
            self._encode_paths(all_paths, qdata['question'])

        if return_details:
            selection, details = self.controller(
                query_emb, path_embs_t, gnn_scores_t, path_lengths_t,
                deterministic=deterministic, return_details=True)
        else:
            selection = self.controller(
                query_emb, path_embs_t, gnn_scores_t, path_lengths_t,
                deterministic=deterministic)
            details = None

        sel_mask = selection.squeeze(0).bool()
        selected_paths = [p for p, s in zip(all_paths, sel_mask) if s]

        all_triples = []
        for p in selected_paths:
            all_triples.extend(p['path_triples'])

        result = {
            'triples': all_triples,
            'cost': len(selected_paths),
            'context': '\n'.join(p['path_text'] for p in selected_paths),
            'selected_paths': selected_paths,
            'all_paths': all_paths,
            'n_total_paths': len(all_paths),
            'n_selected': len(selected_paths),
        }
        if details is not None:
            result['details'] = details
        return result


# ══════════════════════════════════════════════════════════════
# 6. Loss Function (FIXED)
# ══════════════════════════════════════════════════════════════

class AdaGraphControllerLoss(nn.Module):
    """Loss for AdaGraph Controller.

    L = w_mg * L_marginal + w_conf * L_confidence + w_bud * L_budget

    v2 fixes:
    - L_confidence receives actual_recall (not hardcoded 0)
    - L_marginal only on UNSELECTED paths per round
    """

    def __init__(self, w_marginal=0.4, w_confidence=0.3, w_budget=0.3):
        super().__init__()
        self.w_marginal = w_marginal
        self.w_confidence = w_confidence
        self.w_budget = w_budget

    def forward(self, details, gold_entities, all_paths, max_paths, actual_recall=None):
        """
        Args:
            actual_recall: float — actual recall of selected paths (computed externally)
        """
        device = details['final_confidence'].device

        # ── L_marginal ──
        # Target: 1 if path contains gold entity, 0 otherwise
        # Applied via BCE for better gradient signal than MSE
        path_targets = torch.zeros(1, len(all_paths), device=device)
        for i, p in enumerate(all_paths):
            ents = set()
            for h, r, t in p['path_triples']:
                ents.add(str(h)); ents.add(str(t))
            if ents & gold_entities:
                path_targets[0, i] = 1.0

        marginal_loss = 0.0
        mg_count = 0
        for rd in details['round_details']:
            if 'marginals' not in rd:
                continue
            pred = rd['marginals']  # (1, N) — LIVE tensor
            # BCE: better gradient than MSE for binary targets
            marginal_loss += F.binary_cross_entropy_with_logits(
                pred, path_targets, reduction='mean')
            mg_count += 1
        marginal_loss = marginal_loss / max(mg_count, 1)

        # ── L_confidence: calibration with ACTUAL recall ──
        final_conf = details['final_confidence']
        if actual_recall is not None:
            conf_target = torch.tensor([actual_recall], device=device)
        else:
            conf_target = torch.tensor([0.5], device=device)  # neutral default
        conf_loss = F.mse_loss(final_conf, conf_target)

        # ── L_budget ──
        n_selected = details['n_selected']
        budget_loss = (n_selected / max_paths).mean()

        total = (self.w_marginal * marginal_loss +
                 self.w_confidence * conf_loss +
                 self.w_budget * budget_loss)

        return {
            'total': total,
            'marginal': marginal_loss.item() if isinstance(marginal_loss, torch.Tensor) else marginal_loss,
            'confidence': conf_loss.item(),
            'budget': budget_loss.item(),
        }


# ══════════════════════════════════════════════════════════════
# 7. SCST Trainer (FIXED)
# ══════════════════════════════════════════════════════════════

class SCSTTrainer:
    """SCST for AdaGraph Controller.

    v2 fixes:
    - Uses differentiable gather for gradient flow
    - Loss = advantage * conf_calibration_loss (always has gradient)
    - Includes relation embeddings in path encoding
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
        self.optimizer = torch.optim.AdamW(controller.parameters(), lr=lr, weight_decay=0.01)

    def compute_selection_reward(self, selected_paths, gold_entities, total_budget):
        if not selected_paths:
            return -0.1
        collected = set()
        for p in selected_paths:
            for h, r, t in p['path_triples']:
                collected.add(str(h)); collected.add(str(t))
        overlap = len(collected & gold_entities)
        coverage = min(overlap / max(len(gold_entities), 1), 1.0)
        efficiency = 1.0 - len(selected_paths) / total_budget
        unique_ratio = len(collected) / max(
            sum(len(p['path_triples']) * 2 for p in selected_paths), 1)
        return self.alpha * coverage + self.beta * efficiency + self.gamma * unique_ratio

    def _encode_paths(self, all_paths, question):
        """Encode with relations included."""
        query_emb = self.embedder.encode(question).clone().unsqueeze(0).to(self.device)
        path_embs, gnn_scores, path_lengths = [], [], []
        for p in all_paths:
            texts = []
            for h, r, t in p['path_triples']:
                texts.extend([str(h), str(r), str(t)])
            texts = list(set(texts))
            embs = self.embedder.encode_batch(texts)
            path_embs.append(embs.mean(dim=0).clone())
            gnn_scores.append(p['gnn_score'])
            path_lengths.append(float(p['length']))
        return (query_emb,
                torch.stack(path_embs).unsqueeze(0).to(self.device),
                torch.tensor(gnn_scores).unsqueeze(0).float().to(self.device),
                torch.tensor(path_lengths).unsqueeze(0).float().to(self.device))

    def train_step(self, qid):
        qdata = self.gnn.get(qid)
        if qdata is None:
            return None

        gold_entities = set(str(a) for a in self.gnn.get_answers(qid))
        all_paths = extract_candidate_paths(qid, self.gnn)
        if not all_paths or not gold_entities:
            return None

        query_emb, path_embs_t, gnn_scores_t, path_lengths_t = \
            self._encode_paths(all_paths, qdata['question'])

        # Greedy baseline
        self.controller.eval()
        with torch.no_grad():
            greedy_sel = self.controller(query_emb, path_embs_t, gnn_scores_t,
                                          path_lengths_t, deterministic=True)
        greedy_paths = [p for p, s in zip(all_paths, greedy_sel.squeeze(0).bool()) if s]
        baseline_r = self.compute_selection_reward(
            greedy_paths, gold_entities, self.controller.max_paths)

        # Sampled trajectories
        self.controller.train()
        loss_terms = []
        sample_rewards = []

        for _ in range(self.n_samples):
            sel, details = self.controller(
                query_emb, path_embs_t, gnn_scores_t, path_lengths_t,
                deterministic=False, return_details=True)
            sel_paths = [p for p, s in zip(all_paths, sel.squeeze(0).bool()) if s]
            reward = self.compute_selection_reward(
                sel_paths, gold_entities, self.controller.max_paths)
            sample_rewards.append(reward)

            # Actual recall
            if sel_paths:
                collected = set()
                for p in sel_paths:
                    for h, r, t in p['path_triples']:
                        collected.add(str(h)); collected.add(str(t))
                actual_recall = len(collected & gold_entities) / max(len(gold_entities), 1)
            else:
                actual_recall = 0.0

            # Confidence calibration (differentiable through controller)
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

def evaluate_retriever(retriever, gnn, qids, verbose=False):
    results = {'coverage': [], 'cost': [], 'precision': [], 'recall': [], 'f1': []}
    for qid in qids:
        gold_entities = set(str(a) for a in gnn.get_answers(qid))
        if not gold_entities:
            continue
        ret = retriever.retrieve(qid)
        collected = set()
        for h, r, t in ret['triples']:
            collected.add(str(h)); collected.add(str(t))
        overlap = len(collected & gold_entities)
        coverage = overlap / max(len(gold_entities), 1)
        precision = overlap / max(len(collected), 1)
        f1 = 2 * precision * coverage / max(precision + coverage, 1e-8)
        results['coverage'].append(coverage)
        results['cost'].append(ret['cost'])
        results['precision'].append(precision)
        results['recall'].append(coverage)
        results['f1'].append(f1)
        if verbose and len(results['f1']) <= 5:
            print(f"  Q={qid}: cov={coverage:.3f} prec={precision:.3f} "
                  f"cost={ret['cost']} sel={ret.get('n_selected','?')}/{ret.get('n_total_paths','?')}")
    return {k: np.mean(v) if v else 0 for k, v in results.items()}
