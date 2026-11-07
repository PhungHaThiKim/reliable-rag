"""Experiment 1 — Main results table (Paper §5.2).

Runs PRISM + all baselines on WebQSP + CWQ, produces the headline table:

    | Method        | WebQSP H@1 | WebQSP F1 | #paths | CWQ H@1 | CWQ F1 | ECE   |
    |---------------|------------|-----------|--------|---------|--------|-------|
    | GNN-RAG K=50  | ...        | ...       | 12.3   | ...     | ...    | 0.22  |
    | GNN-RAG K=10  | ...        | ...       | 6.5    | ...     | ...    | 0.21  |
    | CGAR v1       | ...        | ...       | 5.1    | ...     | ...    | 0.16  |
    | PRISM (ours)  | ...        | ...       | 4.2    | ...     | ...    | 0.043 |

Output: results/exp1_main.json (per-method, per-dataset metrics + CIs).
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from ..config import CFG
from ..controller_batched import deterministic_select as prism_select
from ..las import LAS
from ..metrics_full import full_report, paired_bootstrap_diff, mcnemar_test
from ..baselines.gnn_rag_topk import GNNRAGTopK


@torch.no_grad()
def _evaluate_method(name: str, select_fn, dataset, cfg=CFG,
                       verbose: bool = True) -> Dict:
    """Run any selection function on dataset → return full_report dict."""
    confidences, correctness, n_paths_list, f1s, chosen_K_list = [], [], [], [], []
    t0 = time.time()
    device = None

    iterator = tqdm(range(len(dataset)), desc=name) if verbose else range(len(dataset))
    for i in iterator:
        sample = dataset[i]
        if device is None:
            try:
                device = next(iter([p.device for m in [getattr(select_fn, 'las', None)]
                                     if m is not None
                                     for p in m.parameters()]),
                              torch.device('cpu'))
            except Exception:
                device = torch.device('cpu')
        # Move tensors lazily
        for f in ["query_emb", "paths", "path_to_candidate",
                  "candidate_gnn_scores", "candidate_ids", "gold_entities"]:
            setattr(sample, f, getattr(sample, f).to(device))

        trace = select_fn(sample)

        from ..metrics import hit_at_k, f1_score
        hit = hit_at_k(trace.final_probs, sample.candidate_ids,
                       sample.gold_entities, k=1)
        f1 = f1_score(trace.final_probs, sample.candidate_ids,
                      sample.gold_entities, 0.05)

        correctness.append(int(hit))
        f1s.append(float(f1))
        confidences.append(float(trace.confidence.item())
                            if hasattr(trace.confidence, 'item')
                            else float(trace.confidence))
        n_paths_list.append(len(trace.selected_path_idx))
        chosen_K_list.append(getattr(trace, 'chosen_K', 0))

    dt = time.time() - t0
    latency_ms = (dt / max(len(dataset), 1)) * 1000

    rep = full_report(confidences, correctness, n_paths_list, f1s,
                       chosen_K_list, latency_ms_per_q=latency_ms)
    rep["method"] = name
    rep["per_sample"] = {
        "confidence": confidences,
        "correct": correctness,
        "n_paths": n_paths_list,
        "f1": f1s,
        "chosen_K": chosen_K_list,
    }
    return rep


def run_exp1(las: LAS, datasets: Dict, cfg=CFG, out_dir: str = "./results",
              include_methods: Optional[List[str]] = None) -> Dict:
    """Main entry — evaluate PRISM + baselines on each dataset.

    Args:
        las: trained PRISM model
        datasets: {"webqsp": eval_set, "cwq": eval_set, ...}
        out_dir: where to dump JSON

    Returns:
        {dataset: {method: full_report_dict}}
    """
    os.makedirs(out_dir, exist_ok=True)
    include_methods = include_methods or [
        "gnn_rag_K5", "gnn_rag_K10", "gnn_rag_K20", "gnn_rag_K50",
        "prism_ours",
    ]
    results: Dict = {}

    for ds_name, dataset in datasets.items():
        print(f"\n{'='*60}\nExp 1 — {ds_name} ({len(dataset)} samples)\n{'='*60}")
        ds_results = {}

        # ─── GNN-RAG top-K sweep ───
        for K in [5, 10, 20, 50]:
            mname = f"gnn_rag_K{K}"
            if mname not in include_methods:
                continue
            selector = GNNRAGTopK(K=K)
            ds_results[mname] = _evaluate_method(
                mname, selector.select, dataset, cfg)

        # ─── PRISM ───
        if "prism_ours" in include_methods:
            def prism_fn(sample):
                return prism_select(las, sample, cfg)
            prism_fn.las = las  # for device detection
            ds_results["prism_ours"] = _evaluate_method(
                "prism_ours", prism_fn, dataset, cfg)

        # ─── Significance test: PRISM vs each baseline ───
        if "prism_ours" in ds_results:
            prism_correct = ds_results["prism_ours"]["per_sample"]["correct"]
            for mname, mres in ds_results.items():
                if mname == "prism_ours":
                    continue
                base_correct = mres["per_sample"]["correct"]
                if len(prism_correct) == len(base_correct):
                    pb = paired_bootstrap_diff(
                        prism_correct, base_correct, n_bootstrap=5000)
                    mc = mcnemar_test(prism_correct, base_correct)
                    ds_results["prism_ours"][f"sig_vs_{mname}"] = {
                        "paired_bootstrap": pb,
                        "mcnemar": mc,
                    }
        results[ds_name] = ds_results

    # Save per-dataset
    # Strip per_sample bulk for clean save
    saveable = {}
    for ds, d in results.items():
        saveable[ds] = {}
        for m, r in d.items():
            r_cp = {k: v for k, v in r.items() if k != "per_sample"}
            saveable[ds][m] = r_cp

    out_path = os.path.join(out_dir, "exp1_main.json")
    with open(out_path, "w") as f:
        json.dump(saveable, f, indent=2, default=lambda o: float(o)
                   if isinstance(o, (np.floating, np.integer))
                   else (list(o) if isinstance(o, np.ndarray) else str(o)))
    print(f"\nok Saved → {out_path}")

    # Print summary
    _print_summary(results)
    return results


def _print_summary(results: Dict):
    print("\n" + "=" * 70)
    print("=== Exp 1 Summary ===")
    print("=" * 70)
    print(f"{'Method':<20} {'Dataset':<12} {'Hit@1':<10} {'F1':<10} "
          f"{'#paths':<8} {'ECE':<8} {'Latency':<10}")
    print("-" * 70)
    for ds, by_method in results.items():
        for m, r in by_method.items():
            print(f"{m:<20} {ds:<12} {r['hit_at_1']:<10.2f} {r['f1']:<10.2f} "
                  f"{r['mean_paths']:<8.2f} {r['ece']:<8.4f} "
                  f"{r['latency_ms']:<10.1f}ms")
