"""Experiment 4 — Calibration analysis (Paper §5.3, novelty showcase).

Concern C response: compares PRISM's calibration head against FAIR baselines:
    - max-GNN-score (raw, uncalibrated)
    - GNN-RAG + temperature scaling
    - GNN-RAG + isotonic regression
    - LLM self-eval confidence (Tian et al. 2023)
    - PRISM (ours)

Sub-experiments:
    4a — In-distribution ECE/Brier/MCE
    4b — Distribution shift (WebQSP → CWQ)
    4c — Reliability diagrams (4-panel figure)
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from ..baselines.gnn_rag_topk import gnnrag_topk_select
from ..baselines.isotonic_gnn import IsotonicGNN
from ..baselines.temp_scaled_gnn import TempScaledGNN
from ..baselines.path_count import PathCountConfidence
from ..config import CFG
from ..controller_batched import deterministic_select as prism_select
from ..las import LAS
from ..metrics_full import calibration_report
from ..metrics import hit_at_k


@torch.no_grad()
def _collect_outputs(las: LAS, dataset, cfg=CFG, verbose=True):
    """Run BOTH PRISM and GNN-RAG-K=10 in parallel — collect per-method
    confidence + correctness."""
    out = {
        "prism": {"conf": [], "corr": []},
        "max_gnn":    {"conf": [], "corr": []},
        "path_count": {"conf": [], "corr": []},
    }
    iterator = tqdm(range(len(dataset)), desc="collect") if verbose else range(len(dataset))
    for i in iterator:
        sample = dataset[i]
        device = next(las.parameters()).device
        for f in ["query_emb", "paths", "path_to_candidate",
                  "candidate_gnn_scores", "candidate_ids", "gold_entities"]:
            setattr(sample, f, getattr(sample, f).to(device))

        # PRISM
        prism_trace = prism_select(las, sample, cfg)
        prism_hit = hit_at_k(prism_trace.final_probs, sample.candidate_ids,
                              sample.gold_entities, k=1)
        out["prism"]["conf"].append(float(prism_trace.confidence.item()))
        out["prism"]["corr"].append(int(prism_hit))

        # GNN-RAG K=10 (raw max-GNN-score as confidence)
        gnn_trace = gnnrag_topk_select(sample, K=10)
        gnn_hit = hit_at_k(gnn_trace.final_probs, sample.candidate_ids,
                            sample.gold_entities, k=1)
        out["max_gnn"]["conf"].append(float(gnn_trace.confidence.item()))
        out["max_gnn"]["corr"].append(int(gnn_hit))

        # Path-count
        pc_conf = PathCountConfidence.score(sample, prism_trace.selected_path_idx)
        out["path_count"]["conf"].append(pc_conf)
        out["path_count"]["corr"].append(int(prism_hit))

    return out


def run_exp4a_in_dist(las: LAS, eval_set, dev_set, cfg=CFG,
                       out_dir: str = "./results") -> Dict:
    """Sub-experiment 4a — in-distribution calibration."""
    os.makedirs(out_dir, exist_ok=True)
    print("\n=== Exp 4a — In-distribution calibration ===")

    # Collect dev outputs to fit temp/isotonic calibrators
    print("\n[1/3] Collecting dev outputs for calibrator fitting...")
    dev_outs = _collect_outputs(las, dev_set, cfg)

    # Fit calibrators on dev
    print("\n[2/3] Fitting temp + isotonic on dev...")
    ts = TempScaledGNN().fit(dev_outs["max_gnn"]["conf"],
                              dev_outs["max_gnn"]["corr"])
    try:
        iso = IsotonicGNN().fit(dev_outs["max_gnn"]["conf"],
                                 dev_outs["max_gnn"]["corr"])
    except RuntimeError:
        print("  (sklearn not available — isotonic skipped)")
        iso = None

    # Collect test outputs + apply calibrators
    print("\n[3/3] Evaluating on test set...")
    test_outs = _collect_outputs(las, eval_set, cfg)

    # Apply calibrators to test
    temp_conf = ts.calibrate_batch(test_outs["max_gnn"]["conf"])
    iso_conf = (iso.calibrate_batch(test_outs["max_gnn"]["conf"])
                 if iso else None)

    methods = {
        "PRISM (ours)":   (test_outs["prism"]["conf"],   test_outs["prism"]["corr"]),
        "Max GNN (raw)":  (test_outs["max_gnn"]["conf"], test_outs["max_gnn"]["corr"]),
        "Temp-scaled GNN":(temp_conf,                    test_outs["max_gnn"]["corr"]),
        "Path-count":     (test_outs["path_count"]["conf"], test_outs["path_count"]["corr"]),
    }
    if iso_conf is not None:
        methods["Isotonic GNN"] = (iso_conf, test_outs["max_gnn"]["corr"])

    reports = {name: calibration_report(c, y) for name, (c, y) in methods.items()}
    reports["_meta"] = {"temp_T": ts.T, "n_dev": len(dev_outs["max_gnn"]["conf"]),
                         "n_test": len(test_outs["max_gnn"]["conf"])}

    with open(os.path.join(out_dir, "exp4a_calibration_indist.json"), "w") as f:
        json.dump(reports, f, indent=2, default=str)

    _print_calib_table(reports, "Exp 4a — In-distribution")
    return reports


def run_exp4b_shift(las: LAS, train_dev_set, ood_test_set, cfg=CFG,
                      out_dir: str = "./results") -> Dict:
    """Sub-experiment 4b — distribution shift (train WebQSP, test CWQ).

    Fit calibrator on WebQSP dev; evaluate on CWQ test (out-of-distribution).
    PRISM's calibration should degrade less than post-hoc methods.
    """
    os.makedirs(out_dir, exist_ok=True)
    print("\n=== Exp 4b — Distribution shift ===")

    # Fit on train_dev (WebQSP dev)
    dev_outs = _collect_outputs(las, train_dev_set, cfg)
    ts = TempScaledGNN().fit(dev_outs["max_gnn"]["conf"],
                              dev_outs["max_gnn"]["corr"])

    # Evaluate on ood_test (CWQ)
    ood_outs = _collect_outputs(las, ood_test_set, cfg)
    temp_conf_ood = ts.calibrate_batch(ood_outs["max_gnn"]["conf"])

    methods = {
        "PRISM (ours)":     (ood_outs["prism"]["conf"],   ood_outs["prism"]["corr"]),
        "Max GNN (raw)":    (ood_outs["max_gnn"]["conf"], ood_outs["max_gnn"]["corr"]),
        "Temp-scaled GNN":  (temp_conf_ood,               ood_outs["max_gnn"]["corr"]),
    }
    reports = {name: calibration_report(c, y) for name, (c, y) in methods.items()}

    with open(os.path.join(out_dir, "exp4b_calibration_shift.json"), "w") as f:
        json.dump(reports, f, indent=2, default=str)
    _print_calib_table(reports, "Exp 4b — Distribution shift")
    return reports


def plot_reliability_diagrams(report_dict: Dict, dev_outs: Dict,
                                test_outs: Dict, ts: TempScaledGNN,
                                out_path: str = "exp4c_reliability.pdf"):
    """Sub-experiment 4c — 4-panel reliability diagram figure."""
    import matplotlib
    matplotlib.rcParams.update({'font.size': 12, 'figure.dpi': 150,
                                 'font.family': 'serif'})
    import matplotlib.pyplot as plt

    from ..metrics import reliability_diagram

    methods = {
        "Max GNN (raw)":   (test_outs["max_gnn"]["conf"], test_outs["max_gnn"]["corr"]),
        "Temp-scaled GNN": (ts.calibrate_batch(test_outs["max_gnn"]["conf"]),
                             test_outs["max_gnn"]["corr"]),
        "Path-count":      (test_outs["path_count"]["conf"], test_outs["path_count"]["corr"]),
        "PRISM (ours)":    (test_outs["prism"]["conf"], test_outs["prism"]["corr"]),
    }
    colors = ['#1f77b4', '#2ca02c', '#9467bd', '#d62728']

    fig, axes = plt.subplots(1, len(methods), figsize=(4.5 * len(methods), 4.5),
                              sharey=True)
    if len(methods) == 1: axes = [axes]
    for ax, (name, (c, y)), color in zip(axes, methods.items(), colors):
        centers, accs, counts = reliability_diagram(c, y, n_bins=10)
        valid = counts > 0
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax.bar(centers[valid], accs[valid], width=0.09, alpha=0.6,
                color=color, edgecolor='black')
        from ..metrics import expected_calibration_error
        ece = expected_calibration_error(c, y)
        ax.set_xlabel('Predicted confidence')
        ax.set_title(f"{name}\nECE={ece:.4f}", fontsize=10)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel('Empirical accuracy')
    plt.suptitle('Calibration reliability diagrams', fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.savefig(out_path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✓ Saved {out_path}")


def _print_calib_table(reports: Dict, title: str):
    print(f"\n{title}")
    print("=" * 70)
    print(f"{'Method':<22} {'ECE':<10} {'Brier':<10} {'MCE':<10} {'NLL':<10}")
    print("-" * 70)
    for name, r in reports.items():
        if name.startswith("_"):
            continue
        print(f"{name:<22} {r['ece']:<10.4f} {r['brier']:<10.4f} "
              f"{r['mce']:<10.4f} {r['nll']:<10.4f}")
