"""Extra Experiment — Scoring-mode agreement (Issue #2 defense).

Reviewer flagged: first-token scoring breaks on Freebase MIDs. We replaced
with letter-scoring. This experiment validates letter ≈ sum_logprob on 200
samples, so the cheap mode is also accurate.

Output: 3-way agreement matrix (top-1, top-5, KL, JS) across modes.
"""
from __future__ import annotations

import json
import os
import random
from typing import Dict, List

import torch
from tqdm import tqdm

from ..config import CFG
from ..metrics_full import (cross_method_agreement, top_k_agreement,
                              kl_divergence_distribution, js_divergence)
from ..teacher_llm import LiveTeacherLLM, TeacherConfig


def run_scoring_agreement(eval_set, n_samples: int = 200,
                             modes: List[str] = None,
                             model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
                             dtype: str = "bf16",
                             out_dir: str = "./results",
                             rel_lookup=None, ent_lookup=None,
                             question_lookup=None) -> Dict:
    """For each scoring_mode in {letter, first_token, sum_logprob}, score the
    same 200 samples and measure pairwise agreement.

    Returns dict of agreement metrics + per-method probability dumps.
    """
    modes = modes or ["letter", "first_token", "sum_logprob"]
    os.makedirs(out_dir, exist_ok=True)

    # Randomly sample N samples
    rng = random.Random(0)
    indices = rng.sample(range(len(eval_set)), min(n_samples, len(eval_set)))
    samples = [eval_set[i] for i in indices]

    method_outputs: Dict[str, List[torch.Tensor]] = {m: [] for m in modes}

    for mode in modes:
        print(f"\n--- Scoring with mode={mode} ---")
        tcfg = TeacherConfig(model_name=model_name, dtype=dtype,
                              scoring_mode=mode)
        teacher = LiveTeacherLLM(tcfg)

        for s in tqdm(samples, desc=mode):
            q_text = None
            if question_lookup:
                q_text = question_lookup.get(s.qid)
            elif hasattr(s, "question_text"):
                q_text = s.question_text
            # Use FULL subset for the comparison (P_M(. | q, P))
            probs = teacher.score_subgraph(
                s, list(range(s.paths.size(0))),
                question_text=q_text,
                rel_lookup=rel_lookup, ent_lookup=ent_lookup,
            )
            method_outputs[mode].append(probs.cpu())

        # Free GPU
        del teacher
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Pairwise agreement
    agreement = cross_method_agreement(method_outputs)

    # Detailed: per-pair top-1, top-5, KL, JS
    summary = {"n_samples": len(samples), "modes": modes, **agreement}

    with open(os.path.join(out_dir, "exp_extra_scoring.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n=== Scoring-mode agreement ===")
    for k, v in agreement.items():
        print(f"  {k:<40} {v:.4f}")
    return summary
