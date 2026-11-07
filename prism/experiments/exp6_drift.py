"""Experiment 6 — Graph Drift Robustness (Paper §5.6, THE killer).

Wrapper around graph_drift.full_drift_experiment that integrates with the
runner pipeline. Outputs `exp6_drift.json` and figure.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

from ..config import CFG
from ..graph_drift import (full_drift_experiment, DriftResult,
                            drift_summary_table)


def run_exp6(las, eval_set, cfg=CFG, out_dir: str = "./results",
              drift_pcts: Optional[List[float]] = None,
              seed: int = 42) -> Dict:
    drift_pcts = drift_pcts or [0.0, 0.1, 0.3, 0.5, 0.7]
    results = full_drift_experiment(las, eval_set, cfg=cfg,
                                      out_dir=os.path.join(out_dir, "drift"),
                                      drift_pcts=drift_pcts, seed=seed)
    # JSON dump for paper consumption
    out = {
        "prism_drift": [
            {"drift_pct": r.drift_pct,
             "hit_at_1": r.hit_at_1, "f1": r.f1,
             "ece": r.ece, "brier": r.brier,
             "mean_paths": r.mean_paths, "n": r.n}
            for r in results["prism"]
        ],
        "transductive_drift": [
            {"drift_pct": r.drift_pct,
             "hit_at_1": r.hit_at_1, "f1": r.f1,
             "ece": r.ece, "brier": r.brier, "n": r.n}
            for r in results["transductive"]
        ],
    }
    with open(os.path.join(out_dir, "exp6_drift.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    return out
