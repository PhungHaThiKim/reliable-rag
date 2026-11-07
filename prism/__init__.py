"""PRISM — Predictive Retrieval with Inductive Subgraph Memory."""
from .config import CFG, Config
from .las import LAS

# Batched controller
try:
    from .controller_batched import (
        deterministic_select as deterministic_select_batched,
        stochastic_select as stochastic_select_batched,
    )
except Exception:
    deterministic_select_batched = None
    stochastic_select_batched = None

# Teacher LLM
try:
    from .teacher_llm import CachedTeacherLLM, LiveTeacherLLM, TeacherConfig
except Exception:
    CachedTeacherLLM = None
    LiveTeacherLLM = None
    TeacherConfig = None

# Improvements
try:
    from .improvements import (
        calib_shaped_reward, difficulty_aware_reward,
        rl_step_entropy_reg, train_rl_improved,
        OnlineAdapter, TemperatureScaler,
        risk_coverage_curve, selective_abstention_eval,
    )
except Exception:
    pass

# Extended metrics
try:
    from .metrics_full import (
        full_report, calibration_report,
        paired_bootstrap_diff, mcnemar_test,
        bootstrap_ci, aggregate_seeds,
        maximum_calibration_error, area_under_risk_coverage,
    )
except Exception:
    pass

# Baselines
try:
    from .baselines import (
        TempScaledGNN, IsotonicGNN,
        LLMSelfEvalConfidence, PathCountConfidence, GNNRAGTopK,
    )
except Exception:
    pass

# Graph drift
try:
    from .graph_drift import (
        run_drift_experiment, full_drift_experiment,
        DriftInjection, TransductiveBaseline,
    )
except Exception:
    pass

# Experiment runner
try:
    from .experiments.runner import ExperimentRunner
except Exception:
    ExperimentRunner = None

# Paper fill utility
try:
    from .paper_fill import fill_paper, fill_text
except Exception:
    pass

__all__ = [
    "CFG", "Config", "LAS",
    "deterministic_select_batched", "stochastic_select_batched",
    "CachedTeacherLLM", "LiveTeacherLLM", "TeacherConfig",
    "calib_shaped_reward", "difficulty_aware_reward",
    "rl_step_entropy_reg", "train_rl_improved",
    "OnlineAdapter", "TemperatureScaler",
    "risk_coverage_curve", "selective_abstention_eval",
    "full_report", "calibration_report",
    "paired_bootstrap_diff", "mcnemar_test", "bootstrap_ci",
    "aggregate_seeds", "maximum_calibration_error",
    "area_under_risk_coverage",
    "TempScaledGNN", "IsotonicGNN",
    "LLMSelfEvalConfidence", "PathCountConfidence", "GNNRAGTopK",
    "run_drift_experiment", "full_drift_experiment",
    "DriftInjection", "TransductiveBaseline",
    "ExperimentRunner",
    "fill_paper", "fill_text",
]
