"""Information-theoretic controller derived from LAS.

No new parameters: every decision (when to stop, which path to add) is computed
from the LAS answer distribution. This is the core claim of PRISM — one
mechanism, four abilities.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple

import torch

from .config import CFG
from .data import Sample
from .las import LAS


@dataclass
class ControllerTrace:
    """Records what the controller did per query — used for analysis & RL."""
    selected_path_idx: List[int]
    chosen_K: int
    final_probs: torch.Tensor
    confidence: torch.Tensor
    stopped_reason: str
    step_kl: List[float]
    log_probs: List[torch.Tensor]
    entropies: List[torch.Tensor] = field(default_factory=list)  # policy entropy per step (differentiable)


def _kl(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    return (p * (p.clamp(min=eps).log() - q.clamp(min=eps).log())).sum()


@torch.no_grad()
def deterministic_select(las: LAS, sample: Sample, cfg=CFG) -> ControllerTrace:
    """Greedy controller using LAS predictions only.

    Loop:
      1. Compute current P_LAS over candidates.
      2. For each unselected path p, compute counterfactual KL impact of adding p.
      3. Pick top path by (impact - λ · redundancy).
      4. Stop if KL(P_t || P_{t-1}) < ε  OR |S| = max_select_paths.
    """
    device = sample.query_emb.device
    n_paths = sample.paths.size(0)
    selected: List[int] = []
    step_kl: List[float] = []

    probs_prev, conf, _ = las(sample.query_emb, sample.paths,
                              sample.path_to_candidate, sample.candidate_gnn_scores,
                              selected_path_idx=[])
    final_probs, final_conf = probs_prev, conf

    available = list(range(n_paths))
    chosen_K = cfg.k_options[0]
    reason = "max_paths"

    for step in range(cfg.max_select_paths):
        impacts = torch.zeros(len(available), device=device)
        new_probs_per_path = []
        for j, p_idx in enumerate(available):
            trial_sel = selected + [p_idx]
            p_trial, _, _ = las(sample.query_emb, sample.paths,
                                sample.path_to_candidate, sample.candidate_gnn_scores,
                                selected_path_idx=trial_sel)
            new_probs_per_path.append(p_trial)
            impacts[j] = _kl(p_trial, probs_prev)

        if len(selected) > 0:
            sel_idx_tensor = torch.tensor(selected, device=device)
            sel_cand = sample.path_to_candidate[sel_idx_tensor]
            avail_cand = sample.path_to_candidate[torch.tensor(available, device=device)]
            redundancy = (avail_cand.unsqueeze(1) == sel_cand.unsqueeze(0)).float().mean(1)
            scores = impacts - cfg.diversity_lambda * redundancy
        else:
            scores = impacts

        best = int(scores.argmax().item())
        best_path = available[best]
        new_probs = new_probs_per_path[best]

        kl_step = float(_kl(new_probs, probs_prev).item())
        step_kl.append(kl_step)

        selected.append(best_path)
        available.pop(best)
        probs_prev = new_probs
        final_probs = new_probs

        n_cand_expanded = len({sample.path_to_candidate[i].item() for i in selected})
        for K in cfg.k_options:
            if n_cand_expanded <= K:
                chosen_K = K
                break

        if kl_step < cfg.stopping_eps and step >= 2:
            reason = "saturated"
            break

        if len(available) == 0:
            reason = "exhausted"
            break

    _, final_conf, _ = las(sample.query_emb, sample.paths,
                           sample.path_to_candidate, sample.candidate_gnn_scores,
                           selected_path_idx=selected)

    return ControllerTrace(
        selected_path_idx=selected, chosen_K=chosen_K,
        final_probs=final_probs, confidence=final_conf,
        stopped_reason=reason, step_kl=step_kl, log_probs=[],
    )


def stochastic_select(las: LAS, sample: Sample, cfg=CFG, temp: float = 1.0
                       ) -> ControllerTrace:
    """Stochastic version used during RL training. Same impact-based scoring,
    but Categorical sampling so we can collect log-probs for REINFORCE."""
    device = sample.query_emb.device
    n_paths = sample.paths.size(0)
    selected: List[int] = []
    log_probs: List[torch.Tensor] = []
    step_kl: List[float] = []

    probs_prev, _, _ = las(sample.query_emb, sample.paths,
                           sample.path_to_candidate, sample.candidate_gnn_scores,
                           selected_path_idx=[])
    final_probs = probs_prev

    available = list(range(n_paths))
    chosen_K = cfg.k_options[0]
    reason = "max_paths"

    for step in range(cfg.max_select_paths):
        impacts_list = []
        new_probs_per_path = []
        for p_idx in available:
            trial_sel = selected + [p_idx]
            p_trial, _, _ = las(sample.query_emb, sample.paths,
                                sample.path_to_candidate, sample.candidate_gnn_scores,
                                selected_path_idx=trial_sel)
            new_probs_per_path.append(p_trial)
            impacts_list.append(_kl(p_trial, probs_prev))
        impacts = torch.stack(impacts_list)

        if len(selected) > 0:
            sel_idx_tensor = torch.tensor(selected, device=device)
            sel_cand = sample.path_to_candidate[sel_idx_tensor]
            avail_cand = sample.path_to_candidate[torch.tensor(available, device=device)]
            redundancy = (avail_cand.unsqueeze(1) == sel_cand.unsqueeze(0)).float().mean(1)
            scores = impacts - cfg.diversity_lambda * redundancy
        else:
            scores = impacts

        logits = scores / temp
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_probs.append(dist.log_prob(action))

        best = int(action.item())
        best_path = available[best]
        new_probs = new_probs_per_path[best]

        step_kl.append(float(_kl(new_probs, probs_prev).item()))
        selected.append(best_path)
        available.pop(best)
        probs_prev = new_probs
        final_probs = new_probs

        n_cand_expanded = len({sample.path_to_candidate[i].item() for i in selected})
        for K in cfg.k_options:
            if n_cand_expanded <= K:
                chosen_K = K
                break

        if step_kl[-1] < cfg.stopping_eps and step >= 2:
            reason = "saturated"
            break
        if len(available) == 0:
            reason = "exhausted"
            break

    _, final_conf, _ = las(sample.query_emb, sample.paths,
                           sample.path_to_candidate, sample.candidate_gnn_scores,
                           selected_path_idx=selected)

    return ControllerTrace(
        selected_path_idx=selected, chosen_K=chosen_K,
        final_probs=final_probs, confidence=final_conf,
        stopped_reason=reason, step_kl=step_kl, log_probs=log_probs,
    )
