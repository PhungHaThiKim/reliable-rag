"""Main training entrypoint. Run distill or rl stage.

Usage:
  python -m colab.graph_rag.prism.train --stage distill
  python -m colab.graph_rag.prism.train --stage rl --init outputs/prism_distill_best.pt
  python -m colab.graph_rag.prism.train --stage all
"""
import argparse
import os

import torch

from .config import CFG
from .data import SyntheticKGQA
from .distill import train_distill
from .evaluate import baseline_full_subgraph, evaluate
from .las import LAS
from .rl import train_rl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["distill", "rl", "all"], default="all")
    ap.add_argument("--data", choices=["synthetic", "webqsp"], default="synthetic")
    ap.add_argument("--n_train", type=int, default=None)
    ap.add_argument("--n_eval", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None, help="distill epochs")
    ap.add_argument("--steps", type=int, default=None, help="rl steps")
    ap.add_argument("--init", type=str, default=None, help="init ckpt for rl stage")
    args = ap.parse_args()

    cfg = CFG
    if args.n_train is not None: cfg.n_train = args.n_train
    if args.n_eval is not None: cfg.n_eval = args.n_eval
    if args.epochs is not None: cfg.distill_epochs = args.epochs
    if args.steps is not None: cfg.rl_steps = args.steps

    if args.data == "synthetic":
        train_set = SyntheticKGQA(cfg.n_train, cfg=cfg, seed=cfg.seed)
        eval_set = SyntheticKGQA(cfg.n_eval, cfg=cfg, seed=cfg.seed + 100)
    else:
        raise NotImplementedError("Wire up WebQSPDataset in data.py first.")

    print(f"train: {len(train_set)} samples, eval: {len(eval_set)} samples")
    print(f"#relations={cfg.n_relations}  #candidates={cfg.n_candidates}  "
          f"max_paths={cfg.max_paths}  d_model={cfg.d_model}")

    las = LAS(cfg)
    n_params = sum(p.numel() for p in las.parameters() if p.requires_grad)
    print(f"LAS trainable params: {n_params:,}")

    device = cfg.device if torch.cuda.is_available() else "cpu"
    las.to(device)

    if args.init:
        print(f"loading init from {args.init}")
        las.load_state_dict(torch.load(args.init, map_location="cpu"))

    if args.stage in ("distill", "all"):
        print("\n" + "=" * 60 + "\nSTAGE 1: distillation\n" + "=" * 60)
        ckpt = train_distill(las, train_set, eval_set, cfg)
        las.load_state_dict(torch.load(ckpt, map_location="cpu"))

    if args.stage in ("rl", "all"):
        print("\n" + "=" * 60 + "\nSTAGE 2: REINFORCE fine-tune\n" + "=" * 60)
        train_rl(las, train_set, eval_set, cfg)

    print("\n" + "=" * 60 + "\nFINAL EVALUATION\n" + "=" * 60)
    prism_metrics = evaluate(las, eval_set, cfg)
    base = baseline_full_subgraph(eval_set)
    for k, v in prism_metrics.items():
        print(f"  PRISM {k:14s} = {v}")
    print()
    for k, v in base.items():
        print(f"  Base  {k:14s} = {v}")


if __name__ == "__main__":
    main()
