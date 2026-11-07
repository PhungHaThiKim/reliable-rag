"""Stage 1: distill LAS to mimic teacher's P(answer | q, S) over candidate sets.

Loss = KL(teacher || LAS) on multiple subgraph subsets per question, so the
LAS learns to predict not only the full-S distribution but also intermediate
subset distributions. This is what makes counterfactual scoring meaningful.
"""
import os
import random
from typing import List

import torch
from torch.optim import AdamW
from tqdm import tqdm

from .config import CFG
from .data import Sample, SyntheticKGQA, TeacherLLM, make_loader
from .las import LAS


def _sample_subset(n_paths: int) -> List[int]:
    """Random subset size in [0, n_paths]; covers empty / partial / full S."""
    k = random.randint(0, n_paths)
    return random.sample(range(n_paths), k)


def _stratified_subsets(n_paths: int) -> List[List[int]]:
    """Size-stratified subsets — Issue #1 fix.

    Returns ~6 subsets covering the size spectrum so the controller sees
    |S_t| values matching its inference-time queries.
    """
    targets = [0, 2, 5, 10, 20, n_paths]
    targets = sorted({min(n_paths, max(0, t)) for t in targets})
    out = []
    for k in targets:
        if k == 0:
            out.append([])
        elif k >= n_paths:
            out.append(list(range(n_paths)))
        else:
            out.append(sorted(random.sample(range(n_paths), k)))
    return out


def distill_step(las: LAS, sample: Sample, teacher: TeacherLLM,
                 cfg=CFG, stratified: bool = True) -> torch.Tensor:
    """Single-sample distill loss across multiple subsets to teach LAS the
    counterfactual structure (not just the full-S target).

    `stratified=True` (default, Issue #1 fix): size-stratified subsets
    {0, 2, 5, 10, 20, full} — covers controller's inference-time |S_t|.
    `stratified=False`: legacy 3-mode {empty, random, full}.
    """
    n_paths = sample.paths.size(0)
    if stratified:
        subsets = _stratified_subsets(n_paths)
    else:
        subsets = [
            [],
            _sample_subset(n_paths),
            list(range(n_paths)),
        ]
    losses = []
    for subset in subsets:
        student_probs, conf, _ = las(
            sample.query_emb, sample.paths, sample.path_to_candidate,
            sample.candidate_gnn_scores, selected_path_idx=subset,
        )
        teacher_probs = teacher.score_subgraph(sample, subset).to(student_probs.device)
        kl = las.kl_to_teacher(student_probs, teacher_probs)

        # ── Issue #3 fix: calibrate confidence against TEACHER's expected recall,
        # not student's (which would be self-consistency, not calibration).
        # Reviewer-flagged: see PRISM_review_response.md.
        gold_set = set(sample.gold_entities.tolist())
        gold_in_cand = torch.tensor(
            [int(e.item()) in gold_set for e in sample.candidate_ids],
            dtype=torch.float32, device=student_probs.device,
        )
        # Target = E_{a ~ P_M}[1[a in G]] — grounded in the LLM, not the student
        target_recall = float((teacher_probs * gold_in_cand).sum().item())
        calib = (conf - target_recall) ** 2
        losses.append(kl + 0.5 * calib)
    return torch.stack(losses).mean()


def train_distill(las: LAS, train_set: SyntheticKGQA, eval_set: SyntheticKGQA,
                   cfg=CFG, log_every: int = 50):
    teacher = TeacherLLM()
    opt = AdamW(las.parameters(), lr=cfg.distill_lr, weight_decay=1e-4)
    loader = make_loader(train_set, batch_size=cfg.distill_batch_size, shuffle=True)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    las.to(device)

    best_kl = float("inf")
    epochs_since_improvement = 0
    os.makedirs(cfg.out_dir, exist_ok=True)
    ckpt_path = os.path.join(cfg.out_dir, cfg.ckpt_distill)

    patience = getattr(cfg, "distill_patience", 3)
    min_epochs = getattr(cfg, "distill_min_epochs", 2)
    min_delta = getattr(cfg, "distill_min_delta", 1e-3)

    for epoch in range(cfg.distill_epochs):
        las.train()
        running = 0.0; n_steps = 0
        pbar = tqdm(loader, desc=f"distill ep{epoch}")
        for batch in pbar:
            opt.zero_grad()
            batch_loss = 0.0
            for sample in batch:
                sample.query_emb = sample.query_emb.to(device)
                sample.paths = sample.paths.to(device)
                sample.path_to_candidate = sample.path_to_candidate.to(device)
                sample.candidate_gnn_scores = sample.candidate_gnn_scores.to(device)
                sample.candidate_ids = sample.candidate_ids.to(device)
                sample.gold_entities = sample.gold_entities.to(device)
                sample.teacher_distribution = sample.teacher_distribution.to(device)

                loss = distill_step(las, sample, teacher, cfg)
                loss.backward()
                batch_loss += float(loss.item())
            torch.nn.utils.clip_grad_norm_(las.parameters(), 1.0)
            opt.step()

            running += batch_loss / max(1, len(batch))
            n_steps += 1
            if n_steps % log_every == 0:
                pbar.set_postfix(loss=running / n_steps)

        eval_kl = evaluate_distill(las, eval_set, teacher, cfg)
        improved = eval_kl < best_kl - min_delta
        marker = "  ↓ saved best" if improved else f"  (no improvement, {epochs_since_improvement+1}/{patience})"
        print(f"[distill] epoch {epoch}  train_loss={running/max(1,n_steps):.4f}  "
              f"eval_kl={eval_kl:.4f}  best={best_kl:.4f}{marker}")

        if improved:
            best_kl = eval_kl
            torch.save(las.state_dict(), ckpt_path)
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            if epoch + 1 >= min_epochs and epochs_since_improvement >= patience:
                print(f"[distill] early stop at epoch {epoch} (no improvement for {patience} epochs)")
                break
    return ckpt_path


@torch.no_grad()
def evaluate_distill(las: LAS, eval_set: SyntheticKGQA, teacher: TeacherLLM, cfg=CFG):
    las.eval()
    device = next(las.parameters()).device
    total = 0.0; n = 0
    for i in range(len(eval_set)):
        sample = eval_set[i]
        sample.query_emb = sample.query_emb.to(device)
        sample.paths = sample.paths.to(device)
        sample.path_to_candidate = sample.path_to_candidate.to(device)
        sample.candidate_gnn_scores = sample.candidate_gnn_scores.to(device)
        sample.candidate_ids = sample.candidate_ids.to(device)
        sample.gold_entities = sample.gold_entities.to(device)
        sample.teacher_distribution = sample.teacher_distribution.to(device)

        student_probs, _, _ = las(
            sample.query_emb, sample.paths, sample.path_to_candidate,
            sample.candidate_gnn_scores, selected_path_idx=None,
        )
        teacher_probs = teacher.score_subgraph(sample, list(range(sample.paths.size(0)))).to(device)
        total += float(las.kl_to_teacher(student_probs, teacher_probs).item())
        n += 1
    return total / max(1, n)
