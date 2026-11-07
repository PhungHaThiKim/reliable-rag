"""Master orchestrator — runs the full PRISM experiment suite.

Designed so you can run individual experiments OR the whole pipeline.
Each experiment is independent and saves its own JSON.

Usage:

    from prism.experiments.runner import ExperimentRunner

    runner = ExperimentRunner(
        las=trained_las,
        datasets={"webqsp": eval_webqsp, "cwq": eval_cwq},
        out_dir="./results",
        cfg=CFG,
    )

    # Run everything
    runner.run_all()

    # OR pick specific experiments
    runner.run("exp1")
    runner.run("exp4")
    runner.run(["exp1", "exp3", "exp4", "exp5"])

Aggregates everything into `out_dir/all_results.json` for paper_fill.py.
"""
from __future__ import annotations

import json
import os
import time
from typing import Callable, Dict, List, Optional, Union

from ..config import CFG
from ..las import LAS


class ExperimentRunner:
    EXPERIMENTS = {
        "exp1":          ("Main results table",       "exp1_main_results"),
        "exp3":          ("Component ablation",       "exp3_ablation"),
        "exp4_indist":   ("Calibration in-dist",      "exp4_calibration"),
        "exp4_shift":    ("Calibration shift",        "exp4_calibration"),
        "exp5":          ("Pareto cost-quality",      "exp5_pareto"),
        "exp6":          ("Graph drift",              "exp6_drift"),
        "exp7":          ("Cross-LLM transfer",       "exp7_cross_llm"),
        "exp8":          ("Stratified by difficulty", "exp8_stratified"),
        "exp9":          ("Qualitative case study",   "exp9_qualitative"),
        "exp10":         ("Compute & latency",        "exp10_compute"),
        "extra_subset":  ("Subset-size ablation",     "exp_extra_subset"),
        "extra_scoring": ("Scoring-mode agreement",   "exp_extra_scoring"),
        "extra_seed":    ("Multi-seed",               "exp_extra_seed"),
    }

    def __init__(self, las: LAS, datasets: Dict, out_dir: str,
                 cfg=CFG,
                 las_distill_only: Optional[LAS] = None,
                 las_dict_for_cross_llm: Optional[Dict[str, LAS]] = None,
                 readers_for_cross_llm: Optional[Dict] = None,
                 lookups: Optional[Dict] = None,
                 train_set=None,
                 las_factory: Optional[Callable] = None,
                 teacher_factory: Optional[Callable] = None,
                 ):
        """Args:
            las: trained PRISM model (full)
            datasets: {"webqsp": eval, "cwq": eval, ...}
            out_dir: where to dump every experiment's JSON
            las_distill_only: optional Stage-1-only checkpoint for ablation C
            las_dict_for_cross_llm: {teacher_name: las} for exp7
            readers_for_cross_llm: {reader_name: reader_object} for exp7
            lookups: {"rel": rel_lookup, "ent": ent_lookup, "q": q_lookup}
            train_set: for extra experiments
            las_factory: callable() -> fresh LAS (for subset ablation)
            teacher_factory: callable() -> teacher object
        """
        self.las = las
        self.datasets = datasets
        self.out_dir = out_dir
        self.cfg = cfg
        self.las_distill_only = las_distill_only
        self.las_dict_for_cross_llm = las_dict_for_cross_llm or {}
        self.readers_for_cross_llm = readers_for_cross_llm or {}
        self.lookups = lookups or {}
        self.train_set = train_set
        self.las_factory = las_factory
        self.teacher_factory = teacher_factory

        os.makedirs(out_dir, exist_ok=True)
        self.results: Dict[str, Dict] = {}
        self.elapsed: Dict[str, float] = {}

    def run(self, experiments: Union[str, List[str]] = "exp1"):
        if isinstance(experiments, str):
            experiments = [experiments]
        for name in experiments:
            self._run_one(name)
        self._save_aggregate()

    def run_all(self, skip: Optional[List[str]] = None):
        skip = set(skip or [])
        order = ["exp1", "exp3", "exp4_indist", "exp4_shift", "exp5",
                 "exp6", "exp7", "exp8", "exp9", "exp10",
                 "extra_subset", "extra_scoring", "extra_seed"]
        for name in order:
            if name in skip:
                print(f"\n... skipping {name}")
                continue
            try:
                self._run_one(name)
            except Exception as e:
                print(f"  ⚠ {name} failed: {e}")
                self.results[name] = {"error": str(e)}
        self._save_aggregate()

    def _run_one(self, name: str):
        if name not in self.EXPERIMENTS:
            print(f"  ⚠ unknown experiment: {name}")
            return
        desc, mod_name = self.EXPERIMENTS[name]
        print(f"\n{'='*70}\nRunning {name} — {desc}\n{'='*70}")
        t0 = time.time()
        try:
            self.results[name] = self._dispatch(name, mod_name)
        finally:
            self.elapsed[name] = time.time() - t0
            print(f"  ⏱ {name} took {self.elapsed[name]:.1f}s")

    def _dispatch(self, name: str, mod_name: str) -> Dict:
        out_dir = self.out_dir
        ev = self.datasets

        if name == "exp1":
            from .exp1_main_results import run_exp1
            return run_exp1(self.las, ev, self.cfg, out_dir)
        if name == "exp3":
            from .exp3_ablation import run_exp3
            dataset = list(ev.values())[0]
            return run_exp3(self.las, self.las_distill_only, dataset,
                             self.cfg, out_dir)
        if name == "exp4_indist":
            from .exp4_calibration import run_exp4a_in_dist
            dataset = ev.get("webqsp", list(ev.values())[0])
            # Use last 10% as dev fold
            n = len(dataset); split = int(0.8 * n)
            dev = [dataset[i] for i in range(split, n)]
            test = [dataset[i] for i in range(split)]
            return run_exp4a_in_dist(self.las, test, dev, self.cfg, out_dir)
        if name == "exp4_shift":
            from .exp4_calibration import run_exp4b_shift
            if "cwq" not in ev or "webqsp" not in ev:
                raise RuntimeError("exp4_shift needs both webqsp+cwq")
            return run_exp4b_shift(self.las, ev["webqsp"], ev["cwq"],
                                     self.cfg, out_dir)
        if name == "exp5":
            from .exp5_pareto import run_exp5
            dataset = list(ev.values())[0]
            return run_exp5(self.las, dataset, self.cfg, out_dir)
        if name == "exp6":
            from .exp6_drift import run_exp6
            dataset = list(ev.values())[0]
            return run_exp6(self.las, dataset, self.cfg, out_dir)
        if name == "exp7":
            from .exp7_cross_llm import run_exp7
            dataset = list(ev.values())[0]
            las_dict = self.las_dict_for_cross_llm or {"prism": self.las}
            readers = self.readers_for_cross_llm or {"self": None}
            return run_exp7(las_dict, readers, dataset, self.cfg, out_dir)
        if name == "exp8":
            from .exp8_stratified import run_exp8
            dataset = list(ev.values())[0]
            return run_exp8(self.las, dataset, self.cfg, out_dir)
        if name == "exp9":
            from .exp9_qualitative import run_exp9
            dataset = list(ev.values())[0]
            return run_exp9(self.las, dataset, self.cfg, out_dir)
        if name == "exp10":
            from .exp10_compute import run_exp10
            dataset = list(ev.values())[0]
            return run_exp10(self.las, dataset, self.cfg, out_dir)
        if name == "extra_subset":
            from .exp_extra_subset import run_subset_sweep
            if self.train_set is None or self.las_factory is None:
                raise RuntimeError("extra_subset needs train_set + las_factory")
            return run_subset_sweep(self.las_factory, self.train_set,
                                      list(ev.values())[0],
                                      self.teacher_factory,
                                      out_dir=out_dir, cfg=self.cfg)
        if name == "extra_scoring":
            from .exp_extra_scoring import run_scoring_agreement
            return run_scoring_agreement(
                list(ev.values())[0],
                out_dir=out_dir,
                rel_lookup=self.lookups.get("rel"),
                ent_lookup=self.lookups.get("ent"),
                question_lookup=self.lookups.get("q"),
            )
        if name == "extra_seed":
            from .exp_extra_seed import run_multi_seed
            if self.train_set is None:
                raise RuntimeError("extra_seed needs train_set")
            return run_multi_seed(self.las.state_dict(), self.train_set,
                                    list(ev.values())[0], self.cfg, out_dir=out_dir)
        return {}

    def _save_aggregate(self):
        agg = {
            "experiments": self.results,
            "elapsed_seconds": self.elapsed,
            "total_elapsed_seconds": sum(self.elapsed.values()),
        }
        path = os.path.join(self.out_dir, "all_results.json")
        with open(path, "w") as f:
            json.dump(agg, f, indent=2, default=str)
        print(f"\n✓ Aggregated all results → {path}")
        print(f"  Total compute: {agg['total_elapsed_seconds']/60:.1f} min")
