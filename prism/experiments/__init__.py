"""PRISM experiment module — every paper-required measurement.

Public API:
    exp1_main_results      — main Hit/F1/ECE table
    exp3_ablation          — 8-row component ablation
    exp4_calibration       — PRISM vs 4 calibration baselines
    exp5_pareto            — cost-quality sweep
    exp6_drift             — graph drift robustness (wraps graph_drift.py)
    exp7_cross_llm         — 2x2 teacher × reader transfer matrix
    exp8_stratified        — per-difficulty stratification
    exp9_qualitative       — case study generator
    exp10_compute          — latency profiling
    exp_extra_subset       — subset-size ablation (Issue #1 defense)
    exp_extra_scoring      — scoring-mode agreement (Issue #2 defense)
    exp_extra_seed         — multi-seed wrapper
    runner                 — master orchestrator
"""
from . import (
    exp1_main_results, exp3_ablation, exp4_calibration,
    exp5_pareto, exp6_drift, exp7_cross_llm,
    exp8_stratified, exp9_qualitative, exp10_compute,
    exp_extra_subset, exp_extra_scoring, exp_extra_seed,
    runner,
)

__all__ = [
    "exp1_main_results", "exp3_ablation", "exp4_calibration",
    "exp5_pareto", "exp6_drift", "exp7_cross_llm",
    "exp8_stratified", "exp9_qualitative", "exp10_compute",
    "exp_extra_subset", "exp_extra_scoring", "exp_extra_seed",
    "runner",
]
