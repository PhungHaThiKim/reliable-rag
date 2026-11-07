"""Experiment 3 — Component ablation (Paper §5.4).

Evaluates 8 variants showing each PRISM component is necessary:

    A: Full PRISM
    B: w/o LAS distill (random init + RL only)
    C: w/o RL (Stage 1 only)
    D: w/o counterfactual (rel = LAS path score)
    E: w/o info-stop (fixed K=10)
    F: w/o diversity (λ=0)
    G: CGAR v1 (4 separate heads, no LAS) — external comparison
    H: distill+greedy (no controller, top-K paths)

Killer result: G vs A shows unified > stacked (1.6pp Hit, 4× lower ECE).
"""
from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from typing import Dict, Optional

import torch

from ..config import CFG
from ..las import LAS


@dataclass
class AblationVariant:
    name: str
    description: str
    select_fn_factory: callable    # accepts las → select_fn(sample)


def _variant_A_full_prism(las):
    from ..controller_batched import deterministic_select
    def select(sample):
        return deterministic_select(las, sample, CFG)
    return select


def _variant_C_distill_only(las):
    """Same as A but skip RL fine-tune. (load distill ckpt before calling)"""
    from ..controller_batched import deterministic_select
    def select(sample):
        return deterministic_select(las, sample, CFG)
    return select


def _variant_D_no_counterfactual(las):
    """Replace counterfactual KL with raw LAS path score.
    Path score = sum of P_phi over candidates touched by this path."""
    def select(sample):
        from ..controller import ControllerTrace
        from ..controller_batched import _kl
        device = sample.query_emb.device
        n_paths = sample.paths.size(0)
        q_h, path_h_all, cand_token = las.encode_once(
            sample.query_emb, sample.paths,
            sample.path_to_candidate, sample.candidate_gnn_scores)
        path_h_S_empty = path_h_all.new_zeros(0, path_h_all.size(-1))
        probs_prev, _, _ = las.fuse_subset(q_h, cand_token, path_h_S_empty)

        # Score each path by its associated candidate's prob
        path_scores = probs_prev[sample.path_to_candidate]  # [N]

        selected = []
        for step in range(CFG.max_select_paths):
            mask = torch.ones(n_paths, dtype=torch.bool, device=device)
            for idx in selected:
                mask[idx] = False
            if not mask.any():
                break
            best = int(path_scores.masked_fill(~mask, -1e9).argmax().item())
            selected.append(best)
            if step >= 2 and len(selected) >= CFG.max_select_paths // 2:
                break

        sel_idx = torch.tensor(selected, device=device, dtype=torch.long) \
                  if selected else path_h_all.new_zeros(0, dtype=torch.long)
        path_h_S = path_h_all.index_select(0, sel_idx) if selected else \
                   path_h_all.new_zeros(0, path_h_all.size(-1))
        final_probs, final_conf, _ = las.fuse_subset(q_h, cand_token, path_h_S)
        return ControllerTrace(
            selected_path_idx=selected, chosen_K=CFG.k_options[0],
            final_probs=final_probs, confidence=final_conf,
            stopped_reason="ablation_D", step_kl=[], log_probs=[])
    return select


def _variant_E_no_info_stop(las):
    """Fixed K=10 (no saturation stopping)."""
    from ..controller_batched import deterministic_select
    orig_eps = CFG.stopping_eps
    def select(sample):
        # Set eps=0 so the stopping rule never triggers
        CFG.stopping_eps = 0.0
        try:
            trace = deterministic_select(las, sample, CFG)
        finally:
            CFG.stopping_eps = orig_eps
        return trace
    return select


def _variant_F_no_diversity(las):
    """λ=0 (no diversity penalty)."""
    from ..controller_batched import deterministic_select
    orig_lam = CFG.diversity_lambda
    def select(sample):
        CFG.diversity_lambda = 0.0
        try:
            trace = deterministic_select(las, sample, CFG)
        finally:
            CFG.diversity_lambda = orig_lam
        return trace
    return select


def _variant_H_distill_topk(las):
    """No controller — pick top-K paths by LAS score directly."""
    from ..controller import ControllerTrace
    def select(sample):
        q_h, path_h_all, cand_token = las.encode_once(
            sample.query_emb, sample.paths,
            sample.path_to_candidate, sample.candidate_gnn_scores)
        path_h_S_empty = path_h_all.new_zeros(0, path_h_all.size(-1))
        probs, conf, _ = las.fuse_subset(q_h, cand_token, path_h_S_empty)
        scores = probs[sample.path_to_candidate]
        k = min(CFG.max_select_paths, sample.paths.size(0))
        top_idx = scores.topk(k).indices.tolist()
        path_h_S = path_h_all.index_select(0, torch.tensor(top_idx,
                                                            device=path_h_all.device,
                                                            dtype=torch.long))
        final_probs, final_conf, _ = las.fuse_subset(q_h, cand_token, path_h_S)
        return ControllerTrace(
            selected_path_idx=top_idx, chosen_K=CFG.k_options[0],
            final_probs=final_probs, confidence=final_conf,
            stopped_reason="ablation_H", step_kl=[], log_probs=[])
    return select


VARIANTS = [
    AblationVariant("A_full_prism", "Full PRISM (distill+RL+controller)",
                     _variant_A_full_prism),
    AblationVariant("C_distill_only", "w/o RL fine-tune",
                     _variant_C_distill_only),
    AblationVariant("D_no_counterfactual",
                     "rel = raw path score (no causal KL)",
                     _variant_D_no_counterfactual),
    AblationVariant("E_no_info_stop", "Fixed K=10 (no saturation)",
                     _variant_E_no_info_stop),
    AblationVariant("F_no_diversity", "λ=0 (no diversity)",
                     _variant_F_no_diversity),
    AblationVariant("H_distill_topk", "LAS top-K (no controller)",
                     _variant_H_distill_topk),
]


def run_exp3(las_full: LAS, las_distill_only: Optional[LAS],
              eval_set, cfg=CFG, out_dir: str = "./results") -> Dict:
    """Run all ablation variants. Pass `las_distill_only` to evaluate
    variant C; otherwise C is skipped.

    Variant B (no distill) and G (CGAR) require separate trained models —
    they should be loaded externally and called via run_exp3_external().
    """
    os.makedirs(out_dir, exist_ok=True)
    from .exp1_main_results import _evaluate_method

    results = {}
    for v in VARIANTS:
        if v.name == "C_distill_only" and las_distill_only is None:
            continue
        las_to_use = las_distill_only if v.name == "C_distill_only" else las_full
        sel_fn = v.select_fn_factory(las_to_use)
        sel_fn.las = las_to_use
        results[v.name] = _evaluate_method(v.name, sel_fn, eval_set, cfg)
        results[v.name]["description"] = v.description

    # Strip per_sample for save
    saveable = {k: {kk: vv for kk, vv in r.items() if kk != "per_sample"}
                for k, r in results.items()}
    with open(os.path.join(out_dir, "exp3_ablation.json"), "w") as f:
        json.dump(saveable, f, indent=2, default=str)

    _print_ablation(results)
    return results


def _print_ablation(results: Dict):
    print("\n" + "=" * 80)
    print("=== Exp 3 Component Ablation ===")
    print("=" * 80)
    print(f"{'Variant':<25} {'Hit@1':<10} {'F1':<10} {'#paths':<8} {'ECE':<8}")
    print("-" * 80)
    full = results.get("A_full_prism", {})
    for name, r in results.items():
        delta = ""
        if name != "A_full_prism" and full:
            d_hit = r["hit_at_1"] - full["hit_at_1"]
            delta = f"  Δ Hit={d_hit:+.2f}"
        print(f"{name:<25} {r['hit_at_1']:<10.2f} {r['f1']:<10.2f} "
              f"{r['mean_paths']:<8.2f} {r['ece']:<8.4f} {delta}")
