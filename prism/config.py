"""PRISM hyperparameters. Single source of truth."""
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    seed: int = 42
    device: str = "cuda"

    data_mode: str = "synthetic"
    data_path: str = ""
    n_train: int = 2000
    n_eval: int = 400
    n_relations: int = 64
    n_entities: int = 4096
    max_paths: int = 50
    max_path_len: int = 3
    n_candidates: int = 32

    d_model: int = 256
    d_query: int = 384
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.1

    k_options: List[int] = field(default_factory=lambda: [5, 10, 20, 50])
    max_select_paths: int = 10
    stopping_eps: float = 0.05
    diversity_lambda: float = 0.3

    distill_epochs: int = 5
    distill_lr: float = 3e-4
    distill_batch_size: int = 16
    distill_temp: float = 1.0
    distill_patience: int = 3        # early-stop after N epochs without eval_kl improvement
    distill_min_epochs: int = 2      # don't stop before this (warmup)
    distill_min_delta: float = 1e-3  # minimum eval_kl improvement to count as progress

    rl_steps: int = 500
    rl_lr: float = 5e-5
    rl_n_samples: int = 4
    rl_clip_norm: float = 0.5
    rl_w_coverage: float = 0.6
    rl_w_efficiency: float = 0.2
    rl_w_diversity: float = 0.2
    rl_calib_weight: float = 0.1
    rl_eval_every: int = 100         # evaluate eval-reward every N steps
    rl_patience: int = 3             # early-stop after N evals without reward improvement
    rl_min_steps: int = 200          # warmup before early-stop kicks in
    rl_min_delta: float = 1e-3       # minimum reward improvement to count as progress
    rl_grad_accum: int = 1           # accumulate gradients over N queries before opt.step()

    out_dir: str = "outputs"
    ckpt_distill: str = "prism_distill_best.pt"
    ckpt_rl: str = "prism_rl_best.pt"


CFG = Config()
