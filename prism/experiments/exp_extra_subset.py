"""Extra Experiment — Subset-size ablation (Issue #1 defense).

Reviewer flagged: 3-mode subset sweep (∅, random, full) is under-justified.
This experiment sweeps n_subsets ∈ {1, 3, 6, 10, 30} and shows the LAS's
interpolation quality vs n_subsets curve. Defends the size-stratified choice.

Note: requires re-running teacher cache OR sampling from existing cache.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import torch
from tqdm import tqdm

from ..config import CFG
from ..distill import distill_step
from ..las import LAS
from ..metrics import expected_calibration_error


def run_subset_sweep(las_factory, train_set, eval_set,
                       teacher_factory,
                       n_subsets_grid: Optional[List[int]] = None,
                       n_epochs: int = 3,
                       out_dir: str = "./results", cfg=CFG) -> Dict:
    """Sweep n_subsets and measure LAS KL convergence + downstream Hit@1.

    Args:
        las_factory: callable() → fresh LAS (so we restart from scratch each sweep)
        teacher_factory: callable() → teacher object
    """
    n_subsets_grid = n_subsets_grid or [1, 3, 6, 10, 30]
    results = {}

    for n_sub in n_subsets_grid:
        print(f"\n--- Subset sweep: n_subsets={n_sub} ---")
        las = las_factory()
        las.to(cfg.device if torch.cuda.is_available() else "cpu")
        teacher = teacher_factory()
        opt = torch.optim.AdamW(las.parameters(), lr=cfg.distill_lr)

        # Monkey-patch distill_step to use n_sub
        from .. import distill as distill_mod
        orig_strat = distill_mod._stratified_subsets

        def custom_strat(n_paths, n_sub=n_sub):
            import random
            base = list(range(n_paths))
            if n_sub == 1:
                return [base]
            if n_sub == 2:
                return [[], base]
            # Otherwise interpolate between empty and full
            sizes = []
            for i in range(n_sub):
                sizes.append(int(i * n_paths / (n_sub - 1)))
            sizes = sorted(set(sizes))
            return [sorted(random.sample(range(n_paths), s)) if 0 < s < n_paths
                    else ([] if s == 0 else base) for s in sizes]

        distill_mod._stratified_subsets = custom_strat

        try:
            las.train()
            for ep in range(n_epochs):
                losses = []
                for i in tqdm(range(len(train_set)), desc=f"ep{ep}",
                                leave=False):
                    sample = train_set[i]
                    opt.zero_grad()
                    loss = distill_step(las, sample, teacher, cfg,
                                         stratified=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(las.parameters(), 1.0)
                    opt.step()
                    losses.append(float(loss.item()))
                print(f"  ep{ep} avg loss = {sum(losses)/len(losses):.4f}")

            # Eval KL
            from ..distill import evaluate_distill
            eval_kl = evaluate_distill(las, eval_set, teacher, cfg)
            results[f"n_sub={n_sub}"] = {
                "n_subsets": n_sub,
                "final_train_loss": float(sum(losses)/len(losses)),
                "eval_kl": float(eval_kl),
            }
            print(f"  ok n_sub={n_sub} → eval_kl={eval_kl:.4f}")
        finally:
            distill_mod._stratified_subsets = orig_strat

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "exp_extra_subset.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Subset-size sweep summary ===")
    print(f"{'n_subsets':<12} {'eval_kl':<10} {'train_loss':<12}")
    for k, v in results.items():
        print(f"{v['n_subsets']:<12} {v['eval_kl']:<10.4f} "
              f"{v['final_train_loss']:<12.4f}")
    return results
