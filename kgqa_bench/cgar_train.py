"""CGAR training extracted from P7v3 cells 7 & 9 so it can run from a script
on any BenchDataset (and with several seeds).

Stage 1: supervised (confidence MSE vs. actual recall + weighted BCE on path
         relevance), best checkpoint by eval F1.
Stage 2: SCST REINFORCE (adagraph_cgar_v2.CGARSCSTTrainer), best checkpoint by
         eval F1 (NOT by train reward as the notebook did).
"""
from __future__ import annotations

import os
import random
import time
from typing import Dict, List, Optional

import numpy as np
import torch

from adagraph_cgar_v2 import (CGARController, CGARLoss, CGARRetriever,
                              CGARSCSTTrainer, evaluate_cgar_records, path_entities)
from grl_rag import extract_candidate_paths_by_k

from .metrics import summarize_records


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _eval(controller, gnn, qids, embedder, max_paths_per_pair, device) -> Dict:
    controller.eval()
    ret = CGARRetriever(gnn, controller, embedder, max_paths_per_pair=max_paths_per_pair,
                        device=device)
    recs = evaluate_cgar_records(ret, gnn, qids)
    controller.train()
    return summarize_records(recs, n_bootstrap=50)


def train_cgar_stage1(controller: CGARController, gnn_train, train_qids: List[str],
                      embedder, gnn_eval=None, eval_qids: Optional[List[str]] = None,
                      epochs: int = 10, lr: float = 3e-4, w_conf: float = 0.4,
                      w_rel: float = 0.6, max_paths_per_pair: int = 3,
                      device: str = "cpu", ckpt_path: Optional[str] = None,
                      log_every: int = 300, eval_max: Optional[int] = None,
                      verbose: bool = True) -> Dict:
    loss_fn = CGARLoss(w_conf=w_conf, w_rel=w_rel)
    opt = torch.optim.AdamW(controller.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    history, best_f1, best_state = [], -1.0, None
    train_qids = list(train_qids)
    eval_qids = list(eval_qids or [])[: eval_max] if eval_qids else []

    for epoch in range(epochs):
        controller.train()
        random.shuffle(train_qids)
        ep_loss = ep_conf = ep_rel = 0.0; n = 0; t0 = time.time()
        for qi, qid in enumerate(train_qids):
            qdata = gnn_train.get(qid)
            if qdata is None:
                continue
            gold = set(str(a) for a in gnn_train.get_answers(qid))
            if not gold:
                continue
            by_k = extract_candidate_paths_by_k(qid, gnn_train, k_options=controller.K_OPTIONS,
                                                max_paths_per_pair=max_paths_per_pair)
            if not any(by_k.values()):
                continue
            q_emb = embedder.encode(qdata["question"]).clone().unsqueeze(0).to(device)
            selected, details = controller(q_emb, by_k, embedder, deterministic=True,
                                           return_details=True)
            actual_recall = len(path_entities(selected) & gold) / max(len(gold), 1)
            losses = loss_fn(details, gold, by_k, actual_recall=actual_recall)
            opt.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(controller.parameters(), 1.0)
            opt.step()
            ep_loss += losses["total"].item(); ep_conf += losses["confidence"]; ep_rel += losses["relevance"]; n += 1
            if verbose and log_every and (qi + 1) % log_every == 0:
                print(f"    [{qi+1}/{len(train_qids)}] loss={ep_loss/n:.4f} "
                      f"(C:{ep_conf/n:.3f} R:{ep_rel/n:.3f})")
        sched.step()
        rec = {"epoch": epoch + 1, "loss": ep_loss / max(n, 1), "conf_loss": ep_conf / max(n, 1),
               "rel_loss": ep_rel / max(n, 1), "sec": time.time() - t0}
        if gnn_eval is not None and eval_qids:
            ev = _eval(controller, gnn_eval, eval_qids, embedder, max_paths_per_pair, device)
            rec.update({"eval_f1": ev["f1"], "eval_cov": ev["cov"], "eval_cost": ev["cost"],
                        "eval_ece": ev.get("ece")})
            if ev["f1"] > best_f1:
                best_f1 = ev["f1"]
                best_state = {k: v.detach().cpu().clone() for k, v in controller.state_dict().items()}
                if ckpt_path:
                    os.makedirs(os.path.dirname(os.path.abspath(ckpt_path)), exist_ok=True)
                    torch.save({"epoch": epoch + 1, "model_state": best_state, "eval_f1": best_f1}, ckpt_path)
        history.append(rec)
        if verbose:
            print(f"  S1 epoch {epoch+1}/{epochs} loss={rec['loss']:.4f} "
                  + (f"eval_f1={rec['eval_f1']*100:.1f} cost={rec['eval_cost']:.2f}" if "eval_f1" in rec else ""))
    if best_state is not None:
        controller.load_state_dict(best_state)
    return {"history": history, "best_eval_f1": best_f1}


def train_cgar_stage2(controller: CGARController, gnn_train, train_qids: List[str],
                      embedder, gnn_eval=None, eval_qids: Optional[List[str]] = None,
                      steps: int = 300, lr: float = 5e-5, n_samples: int = 8,
                      alpha: float = 0.6, beta: float = 0.2, gamma: float = 0.2,
                      w_calib: float = 0.1, max_paths_per_pair: int = 3,
                      device: str = "cpu", ckpt_path: Optional[str] = None,
                      eval_every: int = 50, eval_max: Optional[int] = None,
                      patience: int = 3, verbose: bool = True) -> Dict:
    trainer = CGARSCSTTrainer(controller, gnn_train, embedder, alpha=alpha, beta=beta,
                              gamma=gamma, n_samples=n_samples, lr=lr, device=device,
                              w_calib=w_calib, max_paths_per_pair=max_paths_per_pair)
    train_qids = list(train_qids); random.shuffle(train_qids)
    eval_qids = list(eval_qids or [])[: eval_max] if eval_qids else []
    history, best_f1, best_state, since = [], -1.0, None, 0
    if gnn_eval is not None and eval_qids:  # Stage-1 checkpoint is a candidate
        ev = _eval(controller, gnn_eval, eval_qids, embedder, max_paths_per_pair, device)
        best_f1 = ev["f1"]; best_state = {k: v.detach().cpu().clone() for k, v in controller.state_dict().items()}
        history.append({"step": 0, "eval_f1": ev["f1"], "eval_cost": ev["cost"]})
    adv_nz_total = 0
    for step in range(1, steps + 1):
        qid = train_qids[(step - 1) % len(train_qids)]
        r = trainer.train_step(qid)
        if r is None:
            continue
        adv_nz_total += r.get("advantage_nonzero", 0)
        if verbose and (step == 1 or step % 20 == 0):
            print(f"  SCST {step:4d}/{steps} R={r['avg_reward']:.4f} B={r['baseline']:.4f} "
                  f"loss={r['loss']:.4f} adv!=0: {r['advantage_nonzero']}/{n_samples}")
        if gnn_eval is not None and eval_qids and step % eval_every == 0:
            ev = _eval(controller, gnn_eval, eval_qids, embedder, max_paths_per_pair, device)
            rec = {"step": step, "eval_f1": ev["f1"], "eval_cost": ev["cost"], "train_reward": r["avg_reward"]}
            history.append(rec)
            if ev["f1"] > best_f1 + 1e-4:
                best_f1 = ev["f1"]; since = 0
                best_state = {k: v.detach().cpu().clone() for k, v in controller.state_dict().items()}
                if ckpt_path:
                    torch.save({"step": step, "model_state": best_state, "eval_f1": best_f1}, ckpt_path)
                if verbose: print(f"    eval_f1={ev['f1']*100:.2f} ↓ saved best")
            else:
                since += 1
                if verbose: print(f"    eval_f1={ev['f1']*100:.2f} (no improvement {since}/{patience})")
                if since >= patience:
                    if verbose: print(f"  early stop at step {step}")
                    break
    if best_state is not None:
        controller.load_state_dict(best_state)
    return {"history": history, "best_eval_f1": best_f1,
            "advantage_nonzero_frac": adv_nz_total / max(steps * n_samples, 1)}
