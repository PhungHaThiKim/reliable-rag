"""End-to-end evaluation: runs deterministic controller on eval set and reports
Hit@1, F1, ECE, Brier, mean #paths, mean K, retrieval cost.
"""
import argparse
import os
from typing import Dict

import torch

from .config import CFG
from .controller import deterministic_select
from .data import SyntheticKGQA
from .las import LAS
from .metrics import (brier_score, expected_calibration_error, f1_score,
                       hit_at_k)


@torch.no_grad()
def evaluate(las: LAS, eval_set: SyntheticKGQA, cfg=CFG) -> Dict[str, float]:
    las.eval()
    device = next(las.parameters()).device

    hits, f1s, n_paths, k_chosen = [], [], [], []
    confidences, correctness = [], []

    for i in range(len(eval_set)):
        sample = eval_set[i]
        sample.query_emb = sample.query_emb.to(device)
        sample.paths = sample.paths.to(device)
        sample.path_to_candidate = sample.path_to_candidate.to(device)
        sample.candidate_gnn_scores = sample.candidate_gnn_scores.to(device)
        sample.candidate_ids = sample.candidate_ids.to(device)
        sample.gold_entities = sample.gold_entities.to(device)

        trace = deterministic_select(las, sample, cfg)

        hit = hit_at_k(trace.final_probs, sample.candidate_ids,
                        sample.gold_entities, k=1)
        f1 = f1_score(trace.final_probs, sample.candidate_ids,
                      sample.gold_entities, threshold=0.05)

        hits.append(hit); f1s.append(f1)
        n_paths.append(len(trace.selected_path_idx))
        k_chosen.append(trace.chosen_K)
        confidences.append(float(trace.confidence.item()))
        correctness.append(hit)

    return {
        "hit@1": sum(hits) / len(hits),
        "f1": sum(f1s) / len(f1s),
        "ece": expected_calibration_error(confidences, correctness),
        "brier": brier_score(confidences, correctness),
        "mean_paths": sum(n_paths) / len(n_paths),
        "mean_K": sum(k_chosen) / len(k_chosen),
        "n": len(hits),
    }


def baseline_full_subgraph(eval_set: SyntheticKGQA) -> Dict[str, float]:
    """Reference baseline: feed teacher distribution over the FULL subgraph,
    no selection. Mimics 'GNN-RAG full retrieval'."""
    from .data import TeacherLLM
    teacher = TeacherLLM()
    hits, f1s, n_paths = [], [], []
    for i in range(len(eval_set)):
        sample = eval_set[i]
        full_probs = teacher.score_subgraph(sample, list(range(sample.paths.size(0))))
        hits.append(hit_at_k(full_probs, sample.candidate_ids, sample.gold_entities, k=1))
        f1s.append(f1_score(full_probs, sample.candidate_ids, sample.gold_entities, 0.05))
        n_paths.append(sample.paths.size(0))
    return {
        "hit@1": sum(hits) / len(hits),
        "f1": sum(f1s) / len(f1s),
        "mean_paths": sum(n_paths) / len(n_paths),
        "n": len(hits),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n_eval", type=int, default=400)
    ap.add_argument("--seed", type=int, default=43)
    args = ap.parse_args()

    cfg = CFG
    eval_set = SyntheticKGQA(args.n_eval, cfg=cfg, seed=args.seed)
    las = LAS(cfg)
    state = torch.load(args.ckpt, map_location="cpu")
    las.load_state_dict(state)
    las.to(cfg.device if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("PRISM evaluation")
    print("=" * 60)
    prism_metrics = evaluate(las, eval_set, cfg)
    for k, v in prism_metrics.items():
        print(f"  {k:14s} = {v}")

    print("\nBaseline (full subgraph, no selection):")
    base_metrics = baseline_full_subgraph(eval_set)
    for k, v in base_metrics.items():
        print(f"  {k:14s} = {v}")

    print("\nDelta (PRISM vs baseline):")
    print(f"  hit@1     Δ = {prism_metrics['hit@1'] - base_metrics['hit@1']:+.4f}")
    print(f"  paths     Δ = {prism_metrics['mean_paths'] - base_metrics['mean_paths']:+.2f}")
    cost_save = 1 - prism_metrics["mean_paths"] / base_metrics["mean_paths"]
    print(f"  cost save = {cost_save * 100:.1f}%")


if __name__ == "__main__":
    main()
