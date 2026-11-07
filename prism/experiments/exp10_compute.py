"""Experiment 10 — Compute & latency profile (Paper §5.2 + Appendix).

Measures per-component latency:
    - GNN-RAG path extraction (offline, but measurable)
    - PRISM controller decision
    - Total end-to-end (assumes LLM call ~800ms separately)

Reports table for paper Appendix.
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

from ..config import CFG
from ..controller_batched import deterministic_select as prism_batched
from ..controller import deterministic_select as prism_orig
from ..baselines.gnn_rag_topk import gnnrag_topk_select
from ..las import LAS


@torch.no_grad()
def run_exp10(las: LAS, eval_set, cfg=CFG,
               out_dir: str = "./results", n_samples: int = 200) -> Dict:
    os.makedirs(out_dir, exist_ok=True)
    device = next(las.parameters()).device

    # Take subsample for timing
    indices = list(range(min(n_samples, len(eval_set))))

    timings = {
        "gnn_rag_K10": [],
        "prism_original": [],
        "prism_batched": [],
    }

    print(f"\n=== Exp 10 latency profile (n={len(indices)}) ===")

    for i in tqdm(indices, desc="profile"):
        sample = eval_set[i]
        for f in ["query_emb", "paths", "path_to_candidate",
                  "candidate_gnn_scores", "candidate_ids", "gold_entities"]:
            setattr(sample, f, getattr(sample, f).to(device))

        # GNN-RAG (fast baseline)
        if device.type == 'cuda': torch.cuda.synchronize()
        t0 = time.perf_counter()
        gnnrag_topk_select(sample, K=10)
        if device.type == 'cuda': torch.cuda.synchronize()
        timings["gnn_rag_K10"].append(time.perf_counter() - t0)

        # PRISM batched
        if device.type == 'cuda': torch.cuda.synchronize()
        t0 = time.perf_counter()
        prism_batched(las, sample, cfg)
        if device.type == 'cuda': torch.cuda.synchronize()
        timings["prism_batched"].append(time.perf_counter() - t0)

    # Optionally run original (slow) for comparison on small subset
    print("  → skipping prism_original (would take 100x longer)")

    summary = {}
    for method, times in timings.items():
        if not times: continue
        summary[method] = {
            "n": len(times),
            "mean_ms": float(np.mean(times) * 1000),
            "p50_ms": float(np.percentile(times, 50) * 1000),
            "p95_ms": float(np.percentile(times, 95) * 1000),
            "p99_ms": float(np.percentile(times, 99) * 1000),
            "min_ms": float(np.min(times) * 1000),
            "max_ms": float(np.max(times) * 1000),
        }

    # Compute PRISM overhead
    if "gnn_rag_K10" in summary and "prism_batched" in summary:
        overhead_ms = summary["prism_batched"]["mean_ms"] - summary["gnn_rag_K10"]["mean_ms"]
        # Assume LLM inference 800ms
        llm_ms = 800.0
        total_with = summary["prism_batched"]["mean_ms"] + llm_ms
        total_without = summary["gnn_rag_K10"]["mean_ms"] + llm_ms
        summary["analysis"] = {
            "prism_overhead_ms": overhead_ms,
            "assumed_llm_ms": llm_ms,
            "total_with_prism_ms": total_with,
            "total_without_prism_ms": total_without,
            "overhead_pct_of_total": overhead_ms / total_with * 100,
        }

    with open(os.path.join(out_dir, "exp10_compute.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Latency summary ===")
    print(f"{'Method':<20} {'mean(ms)':<10} {'p50':<10} {'p95':<10} {'p99':<10}")
    print("-" * 60)
    for method, s in summary.items():
        if method == "analysis": continue
        print(f"{method:<20} {s['mean_ms']:<10.2f} {s['p50_ms']:<10.2f} "
              f"{s['p95_ms']:<10.2f} {s['p99_ms']:<10.2f}")
    if "analysis" in summary:
        a = summary["analysis"]
        print(f"\nPRISM overhead: {a['prism_overhead_ms']:.2f}ms "
              f"({a['overhead_pct_of_total']:.2f}% of total with LLM)")

    return summary
