"""Stage 2: REINFORCE fine-tune of LAS via stochastic controller.

Reward = w_cov * coverage + w_eff * efficiency + w_div * diversity
Baseline = greedy controller's reward (SCST-style).
Online calibration loss keeps the confidence head matched to actual recall
under the policy's induced distribution.
"""
import os
from typing import List

import torch
from torch.optim import AdamW
from tqdm import tqdm

from .config import CFG
from .controller import deterministic_select, stochastic_select
from .data import Sample, SyntheticKGQA
from .las import LAS
from .metrics import coverage as cov_metric, diversity as div_metric


def _reward(sample: Sample, selected: List[int], cfg=CFG) -> float:
    cov = cov_metric(selected, sample.path_to_candidate, sample.candidate_ids,
                     sample.gold_entities)
    eff = 1.0 - len(selected) / max(1, cfg.max_select_paths)
    div = div_metric(selected, sample.path_to_candidate)
    return cfg.rl_w_coverage * cov + cfg.rl_w_efficiency * eff + cfg.rl_w_diversity * div


def rl_step(las: LAS, sample: Sample, cfg=CFG):
    """Returns (loss, mean_reward, baseline_reward) for one sample."""
    with torch.no_grad():
        baseline_trace = deterministic_select(las, sample, cfg)
        baseline_reward = _reward(sample, baseline_trace.selected_path_idx, cfg)

    pg_loss = 0.0
    calib_loss = 0.0
    rewards = []
    for _ in range(cfg.rl_n_samples):
        trace = stochastic_select(las, sample, cfg)
        r = _reward(sample, trace.selected_path_idx, cfg)
        rewards.append(r)
        advantage = r - baseline_reward
        if trace.log_probs:
            log_p = torch.stack(trace.log_probs).sum()
            pg_loss = pg_loss + (-advantage * log_p)

        gold_set = set(sample.gold_entities.tolist())
        gold_mask = torch.tensor(
            [int(e.item()) in gold_set for e in sample.candidate_ids],
            dtype=torch.float32, device=trace.final_probs.device,
        )
        recall = float((trace.final_probs * gold_mask).sum().item())
        calib_loss = calib_loss + (trace.confidence - recall) ** 2

    n = max(1, cfg.rl_n_samples)
    loss = pg_loss / n + cfg.rl_calib_weight * calib_loss / n
    return loss, sum(rewards) / n, baseline_reward


def train_rl(las: LAS, train_set: SyntheticKGQA, eval_set: SyntheticKGQA,
              cfg=CFG, log_every: int = 25):
    opt = AdamW(las.parameters(), lr=cfg.rl_lr, weight_decay=1e-4)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    las.to(device)

    os.makedirs(cfg.out_dir, exist_ok=True)
    ckpt_path = os.path.join(cfg.out_dir, cfg.ckpt_rl)
    best_reward = -float("inf")
    evals_since_improvement = 0

    eval_every = getattr(cfg, "rl_eval_every", 100)
    patience = getattr(cfg, "rl_patience", 3)
    min_steps = getattr(cfg, "rl_min_steps", 200)
    min_delta = getattr(cfg, "rl_min_delta", 1e-3)
    grad_accum = max(1, getattr(cfg, "rl_grad_accum", 1))

    pbar = tqdm(range(cfg.rl_steps), desc="rl")
    running_r, running_b = 0.0, 0.0
    opt.zero_grad()
    for step in pbar:
        sample = train_set[step % len(train_set)]
        sample.query_emb = sample.query_emb.to(device)
        sample.paths = sample.paths.to(device)
        sample.path_to_candidate = sample.path_to_candidate.to(device)
        sample.candidate_gnn_scores = sample.candidate_gnn_scores.to(device)
        sample.candidate_ids = sample.candidate_ids.to(device)
        sample.gold_entities = sample.gold_entities.to(device)

        loss, mean_r, base_r = rl_step(las, sample, cfg)
        # Scale loss by grad_accum so total gradient norm matches single-sample case
        (loss / grad_accum).backward()

        running_r = 0.95 * running_r + 0.05 * mean_r
        running_b = 0.95 * running_b + 0.05 * base_r

        # Optimizer step every grad_accum queries
        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(las.parameters(), cfg.rl_clip_norm)
            opt.step()
            opt.zero_grad()

        if step % log_every == 0:
            pbar.set_postfix(R=f"{running_r:.3f}", B=f"{running_b:.3f}",
                             adv=f"{running_r - running_b:+.3f}",
                             accum=f"{(step % grad_accum)+1}/{grad_accum}")

        if step > 0 and step % eval_every == 0:
            eval_r = evaluate_rl(las, eval_set, cfg)
            improved = eval_r > best_reward + min_delta
            if improved:
                best_reward = eval_r
                torch.save(las.state_dict(), ckpt_path)
                evals_since_improvement = 0
                pbar.write(f"[rl] step {step}  eval_R={eval_r:.4f}  ↓ saved best")
            else:
                evals_since_improvement += 1
                pbar.write(f"[rl] step {step}  eval_R={eval_r:.4f}  best={best_reward:.4f}  "
                            f"(no improvement, {evals_since_improvement}/{patience})")
                if step >= min_steps and evals_since_improvement >= patience:
                    pbar.write(f"[rl] early stop at step {step}")
                    break
    return ckpt_path


@torch.no_grad()
def evaluate_rl(las: LAS, eval_set: SyntheticKGQA, cfg=CFG) -> float:
    las.eval()
    device = next(las.parameters()).device
    total = 0.0
    for i in range(len(eval_set)):
        sample = eval_set[i]
        sample.query_emb = sample.query_emb.to(device)
        sample.paths = sample.paths.to(device)
        sample.path_to_candidate = sample.path_to_candidate.to(device)
        sample.candidate_gnn_scores = sample.candidate_gnn_scores.to(device)
        sample.candidate_ids = sample.candidate_ids.to(device)
        sample.gold_entities = sample.gold_entities.to(device)

        trace = deterministic_select(las, sample, cfg)
        total += _reward(sample, trace.selected_path_idx, cfg)
    las.train()
    return total / max(1, len(eval_set))
