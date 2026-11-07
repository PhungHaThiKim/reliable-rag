"""Experiment 5 — Cost-Quality Pareto Frontier (Paper §5.5).

Sweep stopping ε ∈ {0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5} for PRISM.
Compare against:
  - GNN-RAG fixed K ∈ {5, 10, 20, 50}
  - GraphRAG-FI threshold sweep (if available)

Headline claim: PRISM operating points strictly dominate GNN-RAG top-K at
all matched cost levels.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from ..baselines.gnn_rag_topk import gnnrag_topk_select
from ..config import CFG
from ..controller_batched import deterministic_select as prism_select
from ..las import LAS
from ..metrics import hit_at_k, f1_score


@torch.no_grad()
def _eval_at_cost(select_fn, dataset, label, verbose=True):
    hits, costs, f1s = [], [], []
    iterator = tqdm(range(len(dataset)), desc=label) if verbose else range(len(dataset))
    for i in iterator:
        sample = dataset[i]
        trace = select_fn(sample)
        hit = hit_at_k(trace.final_probs, sample.candidate_ids,
                        sample.gold_entities, k=1)
        f1 = f1_score(trace.final_probs, sample.candidate_ids,
                        sample.gold_entities, 0.05)
        hits.append(int(hit))
        f1s.append(float(f1))
        costs.append(len(trace.selected_path_idx))
    return {
        "hit_at_1": float(np.mean(hits)),
        "f1": float(np.mean(f1s)),
        "mean_cost": float(np.mean(costs)),
        "p50_cost": float(np.percentile(costs, 50)),
        "p95_cost": float(np.percentile(costs, 95)),
        "n": len(hits),
        "label": label,
    }


def run_exp5(las: LAS, eval_set, cfg=CFG, out_dir: str = "./results",
              eps_grid: Optional[List[float]] = None,
              K_grid: Optional[List[int]] = None) -> Dict:
    """Sweep ε for PRISM, K for GNN-RAG, return Pareto data."""
    os.makedirs(out_dir, exist_ok=True)
    eps_grid = eps_grid or [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    K_grid = K_grid or [3, 5, 7, 10, 15, 20, 30, 50]

    # Move data to device
    device = next(las.parameters()).device

    def _prepare(sample):
        for f in ["query_emb", "paths", "path_to_candidate",
                  "candidate_gnn_scores", "candidate_ids", "gold_entities"]:
            setattr(sample, f, getattr(sample, f).to(device))
        return sample

    # ─── PRISM sweep ───
    prism_points = []
    orig_eps = cfg.stopping_eps
    for eps in eps_grid:
        cfg.stopping_eps = eps
        def sel_fn(sample, _eps=eps):
            sample = _prepare(sample)
            return prism_select(las, sample, cfg)
        pt = _eval_at_cost(sel_fn, eval_set, f"PRISM ε={eps}")
        pt["epsilon"] = eps
        pt["method"] = "PRISM"
        prism_points.append(pt)
    cfg.stopping_eps = orig_eps

    # ─── GNN-RAG fixed-K sweep ───
    gnn_points = []
    for K in K_grid:
        def sel_fn(sample, _K=K):
            sample = _prepare(sample)
            return gnnrag_topk_select(sample, K=_K)
        pt = _eval_at_cost(sel_fn, eval_set, f"GNN-RAG K={K}")
        pt["K"] = K
        pt["method"] = "GNN-RAG"
        gnn_points.append(pt)

    # Pareto frontier — points NOT dominated
    all_points = prism_points + gnn_points
    pareto = _pareto_frontier(all_points)

    out = {
        "prism_sweep": prism_points,
        "gnn_rag_sweep": gnn_points,
        "pareto_frontier": pareto,
        "n_samples": len(eval_set),
    }
    with open(os.path.join(out_dir, "exp5_pareto.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n✓ Saved → {os.path.join(out_dir, 'exp5_pareto.json')}")

    _plot_pareto(prism_points, gnn_points, pareto,
                  out_path=os.path.join(out_dir, "exp5_pareto.pdf"))
    return out


def _pareto_frontier(points: List[Dict]) -> List[Dict]:
    """Lower cost = better, higher quality = better. Returns non-dominated points."""
    sorted_pts = sorted(points, key=lambda p: (p["mean_cost"], -p["hit_at_1"]))
    frontier = []
    best_q = -float("inf")
    for p in sorted_pts:
        if p["hit_at_1"] > best_q:
            frontier.append(p)
            best_q = p["hit_at_1"]
    return frontier


def _plot_pareto(prism_pts, gnn_pts, pareto, out_path):
    import matplotlib
    matplotlib.rcParams.update({'font.size': 12, 'figure.dpi': 150,
                                 'font.family': 'serif'})
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5.5))

    px = [p["mean_cost"] for p in prism_pts]
    py = [p["hit_at_1"] * 100 for p in prism_pts]
    ax.plot(px, py, 'D-', color='#d62728', linewidth=2.5, markersize=9,
             label='PRISM (ε sweep)', zorder=4)
    for p in prism_pts:
        if p["epsilon"] in (0.05, 0.5):
            ax.annotate(f"ε={p['epsilon']:.2f}",
                        (p['mean_cost'], p['hit_at_1'] * 100),
                        xytext=(5, 8), textcoords='offset points', fontsize=9,
                        color='#d62728', fontweight='bold')

    gx = [p["mean_cost"] for p in gnn_pts]
    gy = [p["hit_at_1"] * 100 for p in gnn_pts]
    ax.plot(gx, gy, 'o-', color='#1f77b4', linewidth=1.8, markersize=8,
             label='GNN-RAG (K sweep)', alpha=0.85)
    for p in gnn_pts:
        if p["K"] in (5, 50):
            ax.annotate(f"K={p['K']}",
                        (p['mean_cost'], p['hit_at_1'] * 100),
                        xytext=(5, -12), textcoords='offset points',
                        fontsize=9, color='#1f77b4')

    # Pareto dashed
    par_x = [p["mean_cost"] for p in pareto]
    par_y = [p["hit_at_1"] * 100 for p in pareto]
    ax.plot(par_x, par_y, '--', color='#555', linewidth=1.5, alpha=0.5,
             label='Pareto frontier')

    ax.set_xlabel('Mean #paths (cost)')
    ax.set_ylabel('Hit@1 (%)')
    ax.set_title('Cost-Quality Pareto — PRISM vs GNN-RAG', fontsize=13)
    ax.grid(alpha=0.3)
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.savefig(out_path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✓ Pareto figure → {out_path}")
