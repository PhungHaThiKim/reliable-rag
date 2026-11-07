"""Extra Experiment — Multi-seed (×3) for error bars.

Re-runs Stage 2 RL with 3 seeds, holding Stage 1 distill checkpoint frozen.
Reports mean ± std for all key metrics.

Required by EMNLP reviewers for any RL paper.
"""
from __future__ import annotations

import json
import os
import random
from copy import deepcopy
from typing import Dict, List, Optional

import numpy as np
import torch

from ..config import CFG
from ..improvements import train_rl_improved
from ..las import LAS
from ..metrics_full import aggregate_seeds


def run_multi_seed(las_init_state_dict: Dict,
                     train_set, eval_set, cfg=CFG,
                     seeds: Optional[List[int]] = None,
                     out_dir: str = "./results",
                     eval_fn=None) -> Dict:
    """Re-run Stage 2 RL with each seed; aggregate metrics.

    Args:
        las_init_state_dict: Stage-1-distilled LAS state (anchored start)
        eval_fn: callable(las) -> {"hit_at_1": ..., "ece": ..., ...}
                   default = uses exp1's _evaluate_method
    """
    seeds = seeds or [42, 1337, 2024]
    os.makedirs(out_dir, exist_ok=True)

    per_seed = {}
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    for seed in seeds:
        print(f"\n=== Seed {seed} ===")
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

        # Fresh LAS from Stage 1 ckpt
        las = LAS(cfg).to(device)
        las.load_state_dict(las_init_state_dict)

        train_rl_improved(
            las, train_set, eval_set, cfg=cfg,
            reward_mode="calib_shaped", entropy_coef=0.02,
            use_batched=True, eval_every=100, log_every=50,
        )

        # Save per-seed ckpt
        torch.save(las.state_dict(),
                   os.path.join(out_dir, f"rl_seed{seed}.pt"))

        # Evaluate
        if eval_fn is None:
            from .exp1_main_results import _evaluate_method
            from ..controller_batched import deterministic_select
            def fn(sample):
                return deterministic_select(las, sample, cfg)
            fn.las = las
            metrics = _evaluate_method(f"seed{seed}", fn, eval_set, cfg,
                                          verbose=False)
        else:
            metrics = eval_fn(las)
        # Strip non-numeric
        clean = {k: v for k, v in metrics.items()
                 if isinstance(v, (int, float))}
        per_seed[seed] = clean

    aggregated = aggregate_seeds(per_seed)

    out = {"per_seed": per_seed, "aggregated": aggregated, "seeds": seeds}
    with open(os.path.join(out_dir, "exp_extra_seed.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)

    print("\n=== Multi-seed Aggregate (n=" + str(len(seeds)) + ") ===")
    print(f"{'Metric':<25} {'Mean':<12} {'Std':<12}")
    for k, v in aggregated.items():
        if v["mean"] is None: continue
        print(f"{k:<25} {v['mean']:<12.4f} {v['std']:<12.4f}")
    return out
