"""Evaluation metrics: Hit, F1, ECE, Brier, Pareto cost-quality."""
from typing import List, Tuple

import numpy as np
import torch


def hit_at_k(probs: torch.Tensor, candidate_ids: torch.Tensor,
             gold_entities: torch.Tensor, k: int = 1) -> float:
    topk = probs.topk(min(k, probs.numel())).indices
    pred_ents = candidate_ids[topk]
    gold_set = set(gold_entities.tolist())
    return float(any(int(e) in gold_set for e in pred_ents))


def f1_score(probs: torch.Tensor, candidate_ids: torch.Tensor,
             gold_entities: torch.Tensor, threshold: float = 0.1) -> float:
    pred_mask = probs >= threshold
    pred_ents = set(candidate_ids[pred_mask].tolist())
    gold_set = set(gold_entities.tolist())
    if not pred_ents and not gold_set:
        return 1.0
    if not pred_ents or not gold_set:
        return 0.0
    tp = len(pred_ents & gold_set)
    if tp == 0:
        return 0.0
    p = tp / len(pred_ents)
    r = tp / len(gold_set)
    return 2 * p * r / (p + r)


def expected_calibration_error(confidences: List[float], correctness: List[float],
                                n_bins: int = 10) -> float:
    """ECE with equal-width bins. Lower is better."""
    confs = np.array(confidences); corrs = np.array(correctness)
    bins = np.linspace(0, 1, n_bins + 1)
    n = len(confs)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confs >= lo) & (confs < hi if i < n_bins - 1 else confs <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = corrs[mask].mean()
        bin_conf = confs[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def brier_score(confidences: List[float], correctness: List[float]) -> float:
    return float(np.mean((np.array(confidences) - np.array(correctness)) ** 2))


def coverage(selected_path_idx: List[int], path_to_cand: torch.Tensor,
             cand_ids: torch.Tensor, gold_entities: torch.Tensor) -> float:
    if not selected_path_idx:
        return 0.0
    sel_cands = path_to_cand[torch.tensor(selected_path_idx)]
    sel_ents = set(cand_ids[sel_cands].tolist())
    gold_set = set(gold_entities.tolist())
    if not gold_set:
        return 1.0
    return len(sel_ents & gold_set) / len(gold_set)


def diversity(selected_path_idx: List[int], path_to_cand: torch.Tensor) -> float:
    if len(selected_path_idx) < 2:
        return 1.0
    cands = path_to_cand[torch.tensor(selected_path_idx)].tolist()
    return len(set(cands)) / len(cands)


def reliability_diagram(confidences: List[float], correctness: List[float],
                         n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    confs = np.array(confidences); corrs = np.array(correctness)
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    accs = np.zeros(n_bins); counts = np.zeros(n_bins)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confs >= lo) & (confs < hi if i < n_bins - 1 else confs <= hi)
        if mask.sum() > 0:
            accs[i] = corrs[mask].mean()
            counts[i] = mask.sum()
    return bin_centers, accs, counts


def pareto_frontier(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Each point = (cost, quality). Return sorted Pareto-optimal points
    (lower cost, higher quality)."""
    pts = sorted(points, key=lambda x: (x[0], -x[1]))
    frontier, best_q = [], -float("inf")
    for c, q in pts:
        if q > best_q:
            frontier.append((c, q))
            best_q = q
    return frontier
