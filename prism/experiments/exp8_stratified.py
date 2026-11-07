"""Experiment 8 — Stratified analysis by question difficulty (Paper §5.4).

Validates "adaptive K respects difficulty" claim. Stratifies test set by:
    - Hop count (1-hop, 2-hop, 3+hop)
    - Number of gold answers (single vs multi-answer)
    - Avg path length

Reports per-stratum: mean K_chosen, mean #paths, Hit@1, ECE.
Killer plot: heatmap of K_chosen × hop_count → diagonal = adaptive K works.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

from ..config import CFG
from ..controller_batched import deterministic_select as prism_select
from ..las import LAS
from ..metrics import f1_score, hit_at_k, expected_calibration_error
from ..metrics_full import estimate_hop_count, stratify_metrics


@torch.no_grad()
def run_exp8(las: LAS, eval_set, cfg=CFG,
              out_dir: str = "./results", verbose=True) -> Dict:
    os.makedirs(out_dir, exist_ok=True)
    device = next(las.parameters()).device

    per_sample = []
    iterator = tqdm(range(len(eval_set)), desc="stratify") if verbose else range(len(eval_set))
    for i in iterator:
        sample = eval_set[i]
        for f in ["query_emb", "paths", "path_to_candidate",
                  "candidate_gnn_scores", "candidate_ids", "gold_entities"]:
            setattr(sample, f, getattr(sample, f).to(device))

        trace = prism_select(las, sample, cfg)
        hit = hit_at_k(trace.final_probs, sample.candidate_ids,
                        sample.gold_entities, k=1)
        f1 = f1_score(trace.final_probs, sample.candidate_ids,
                        sample.gold_entities, 0.05)
        per_sample.append({
            "qid": sample.qid,
            "hit": int(hit),
            "f1": float(f1),
            "confidence": float(trace.confidence.item()),
            "n_paths": len(trace.selected_path_idx),
            "chosen_K": trace.chosen_K,
            "hop_count": estimate_hop_count(sample),
            "n_gold": sample.gold_entities.size(0),
            "n_candidates": sample.candidate_ids.size(0),
        })

    # Stratify
    by_hop = stratify_metrics(per_sample, "hop_count")
    by_n_gold = stratify_metrics(per_sample, "n_gold")

    # Heatmap data: K_chosen × hop_count
    heat = defaultdict(int)
    for m in per_sample:
        heat[(m["hop_count"], m["chosen_K"])] += 1
    heatmap = [{"hop": k[0], "K": k[1], "count": v} for k, v in heat.items()]

    out = {
        "by_hop_count": by_hop,
        "by_n_gold": by_n_gold,
        "heatmap_hop_K": heatmap,
        "n": len(per_sample),
    }
    with open(os.path.join(out_dir, "exp8_stratified.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)

    _plot_heatmap(heatmap, out_path=os.path.join(out_dir, "exp8_heatmap.pdf"))

    print("\n=== Exp 8 Stratification ===")
    print("\nBy hop count:")
    print(f"{'Hop':<6} {'n':<6} {'Hit':<8} {'F1':<8} {'#paths':<10} {'mean_K':<8} {'ECE':<8}")
    for h, m in sorted(by_hop.items()):
        print(f"{h:<6} {m['n']:<6} {m['hit']*100:<8.1f} {m['f1']*100:<8.1f} "
              f"{m['mean_paths']:<10.2f} {m['mean_K']:<8.1f} {m['ece']:<8.4f}")
    return out


def _plot_heatmap(heatmap_data: List[Dict], out_path: str):
    import matplotlib
    matplotlib.rcParams.update({'font.size': 12, 'figure.dpi': 150,
                                 'font.family': 'serif'})
    import matplotlib.pyplot as plt

    hops = sorted({d["hop"] for d in heatmap_data})
    Ks = sorted({d["K"] for d in heatmap_data})
    mat = np.zeros((len(hops), len(Ks)))
    for d in heatmap_data:
        i = hops.index(d["hop"]); j = Ks.index(d["K"])
        mat[i, j] = d["count"]

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(mat, aspect='auto', cmap='YlOrRd')
    ax.set_xticks(range(len(Ks))); ax.set_xticklabels(Ks)
    ax.set_yticks(range(len(hops))); ax.set_yticklabels(hops)
    ax.set_xlabel('Chosen K (controller adaptive scope)')
    ax.set_ylabel('Hop count (question difficulty)')
    ax.set_title('Adaptive K vs question difficulty')
    # Annotate counts
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, int(mat[i, j]), ha='center', va='center',
                    color='black' if mat[i, j] < mat.max()/2 else 'white')
    plt.colorbar(im, ax=ax, label='Sample count')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.savefig(out_path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    print(f"ok Heatmap → {out_path}")
