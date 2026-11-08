"""Method runners. Every runner maps a BenchDataset → list of per-question
records with the SAME keys, so `metrics.summarize_records` can score any of
them identically.

Record keys produced here
-------------------------
qid, method, dataset, coverage, precision, f1, hit_ctx, cost, n_triples,
n_tokens, latency_ms, selected_paths (texts), chosen_k, confidence,
hops, n_gold, n_candidates, answer_in_subgraph, candidates

Methods
-------
GNNRAGMethod(top_k=None)       GNN-RAG all paths / top-K by candidate score
CGARMethod(controller, ...)    CGAR v2.1 controller (+ .fit() to train)
CGAREscalateMethod(cgar, ...)  CGAR-G: CGAR budget K, escalate hard / low-confidence questions to all paths
SBERTTopKMethod(embedder, k)   non-learned baseline: top-K paths by cosine(question, path) — isolates
                               the controller's contribution from the embedding's
All path-dict baselines take `max_candidates` (default 20 = CGAR's reachable
pool: K_OPTIONS up to 50 but datasets carry n_top=20 candidates). Pass
`--baseline-candidates 10` to reproduce the Seminar-1 "GNN-RAG-all" (10 cand).
"""
from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from .adapters import to_gnn_features
from .data import BenchDataset, BenchQuestion


# ─────────────────────────────────────────────────────────────────────
# shared helpers
# ─────────────────────────────────────────────────────────────────────

def _ents_of_triples(triples) -> set:
    ents = set()
    for h, _r, t in triples:
        ents.add(str(h)); ents.add(str(t))
    return ents


def _approx_tokens(texts: Sequence[str]) -> int:
    return int(sum(len(t.split()) for t in texts) * 1.3)


def make_record(q: BenchQuestion, method: str, selected_triples, selected_texts: List[str],
                cost: int, latency_ms: float, confidence: Optional[float] = None,
                chosen_k: Optional[int] = None, extra: Optional[Dict] = None) -> Dict:
    gold = set(q.answers)
    collected = _ents_of_triples(selected_triples)
    overlap = len(collected & gold)
    cov = overlap / max(len(gold), 1)
    prec = overlap / max(len(collected), 1)
    f1 = 2 * prec * cov / max(prec + cov, 1e-8)
    rec = {
        "qid": q.qid, "method": method, "dataset": q.meta.get("dataset"),
        "coverage": cov, "precision": prec, "f1": f1, "hit_ctx": int(overlap > 0),
        "cost": int(cost), "n_triples": len(selected_triples),
        "n_tokens": _approx_tokens(selected_texts), "latency_ms": float(latency_ms),
        "selected_paths": list(selected_texts),
        "chosen_k": chosen_k, "confidence": confidence,
        "hops": q.meta.get("hops"), "n_gold": len(q.answers),
        "n_candidates": len(q.top_candidates),
        "answer_in_subgraph": q.meta.get("answer_in_subgraph"),
        # answer candidates = entities the selected paths lead to, in path order (CGAR-G reader prompt)
        "candidates": _tails(selected_texts),
    }
    if extra:
        rec.update(extra)
    return rec


def _tails(texts: List[str]) -> List[str]:
    out = []
    for t in texts:
        parts = [x.strip() for x in str(t).split(" -> ")]
        if len(parts) >= 3 and parts[-1] not in out:
            out.append(parts[-1])
    return out


class Method:
    name: str = "method"

    def run(self, dataset: BenchDataset, verbose: bool = True) -> List[Dict]:
        raise NotImplementedError

    def _iter(self, dataset, verbose):
        if verbose:
            try:
                from tqdm import tqdm
                return tqdm(dataset.questions, desc=self.name, leave=False)
            except ImportError:
                pass
        return dataset.questions


# ─────────────────────────────────────────────────────────────────────
# GNN-RAG (all / top-K) and SBERT top-K — path-dict based baselines
# ─────────────────────────────────────────────────────────────────────

class GNNRAGMethod(Method):
    def __init__(self, top_k: Optional[int] = None, max_candidates: int = 20,
                 max_paths_per_pair: int = 3):
        self.top_k, self.max_candidates, self.max_paths_per_pair = top_k, max_candidates, max_paths_per_pair
        self.name = f"gnnrag_top{top_k}" if top_k else "gnnrag_all"

    def run(self, dataset, verbose=True):
        from grl_rag import extract_candidate_paths
        gnn = to_gnn_features(dataset)
        recs = []
        for q in self._iter(dataset, verbose):
            if not q.answers:
                continue
            t0 = time.perf_counter()
            paths = extract_candidate_paths(q.qid, gnn, max_candidates=self.max_candidates,
                                            max_paths_per_pair=self.max_paths_per_pair)
            if self.top_k:
                paths.sort(key=lambda p: -p["gnn_score"]); paths = paths[: self.top_k]
            dt = (time.perf_counter() - t0) * 1000
            triples = [t for p in paths for t in p["path_triples"]]
            conf = max([p["gnn_score"] for p in paths], default=0.0)
            recs.append(make_record(q, self.name, triples, [p["path_text"] for p in paths],
                                    len(paths), dt, confidence=float(conf)))
        return recs




class SBERTTopKMethod(Method):
    """Non-learned relevance baseline: rank every candidate path by
    cosine(question, path_text) with the SAME embedder CGAR uses, keep top-K.
    If CGAR only beats this by a hair, its gain comes from the embedding, not
    from the controller."""

    def __init__(self, embedder, k: int = 10, max_candidates: int = 20, max_paths_per_pair: int = 3):
        self.embedder, self.k = embedder, k
        self.max_candidates, self.max_paths_per_pair = max_candidates, max_paths_per_pair
        self.name = f"sbert_top{k}"

    def run(self, dataset, verbose=True):
        from grl_rag import extract_candidate_paths
        gnn = to_gnn_features(dataset)
        recs = []
        for q in self._iter(dataset, verbose):
            if not q.answers:
                continue
            t0 = time.perf_counter()
            paths = extract_candidate_paths(q.qid, gnn, max_candidates=self.max_candidates,
                                            max_paths_per_pair=self.max_paths_per_pair)
            sims: List[float] = []
            if paths:
                qe = self.embedder.encode(q.question).float()
                pe = self.embedder.encode_batch([p["path_text"] for p in paths]).float().to(qe.device)
                sims = torch.nn.functional.cosine_similarity(pe, qe.unsqueeze(0).expand_as(pe), dim=-1).tolist()
                order = sorted(range(len(paths)), key=lambda i: -sims[i])[: self.k]
                paths = [paths[i] for i in order]; sims = [sims[i] for i in order]
            dt = (time.perf_counter() - t0) * 1000
            triples = [t for p in paths for t in p["path_triples"]]
            conf = max(sims, default=0.0)
            recs.append(make_record(q, self.name, triples, [p["path_text"] for p in paths],
                                    len(paths), dt, confidence=float(conf)))
        return recs




# ─────────────────────────────────────────────────────────────────────
# CGAR
# ─────────────────────────────────────────────────────────────────────

class CGARMethod(Method):
    """CGAR v2.1. Construct with a trained controller or call .fit()."""

    def __init__(self, embedder, controller=None, max_paths_per_pair: int = 3,
                 device: str = "cpu", name: str = "cgar", **controller_kw):
        from adagraph_cgar_v2 import CGARController
        self.embedder = embedder
        self.device = device
        self.max_paths_per_pair = max_paths_per_pair
        self.name = name
        if controller is None:
            kw = dict(emb_dim=embedder.emb_dim, hidden_dim=256, max_paths=10, sprt_beta=0.85,
                      redundancy_lambda=0.3, epsilon=0.3)
            kw.update(controller_kw)
            controller = CGARController(**kw)
        self.controller = controller.to(device)
        self.train_info: Dict = {}

    # --- training --------------------------------------------------------
    def fit(self, train_ds: BenchDataset, eval_ds: Optional[BenchDataset] = None,
            s1_epochs: int = 10, s1_lr: float = 3e-4, s2_steps: int = 300, s2_lr: float = 5e-5,
            s2_samples: int = 8, eval_max: Optional[int] = 400, ckpt_dir: Optional[str] = None,
            seed: int = 42, verbose: bool = True, loss_kw: Optional[Dict] = None) -> Dict:
        """loss_kw (v3): conf_target / rel_label / w_stop — see cgar_train.train_cgar_stage1."""
        from .cgar_train import set_seed, train_cgar_stage1, train_cgar_stage2
        set_seed(seed)
        gnn_tr = to_gnn_features(train_ds)
        gnn_ev = to_gnn_features(eval_ds) if eval_ds is not None else None
        ev_qids = eval_ds.qids() if eval_ds is not None else None
        c1 = os.path.join(ckpt_dir, f"cgar_s1_seed{seed}.pt") if ckpt_dir else None
        c2 = os.path.join(ckpt_dir, f"cgar_s2_seed{seed}.pt") if ckpt_dir else None
        info1 = train_cgar_stage1(self.controller, gnn_tr, train_ds.qids(), self.embedder, gnn_ev, ev_qids,
                                  epochs=s1_epochs, lr=s1_lr, max_paths_per_pair=self.max_paths_per_pair,
                                  device=self.device, ckpt_path=c1, eval_max=eval_max, verbose=verbose,
                                  loss_kw=loss_kw)
        info2 = {}
        if s2_steps > 0:
            info2 = train_cgar_stage2(self.controller, gnn_tr, train_ds.qids(), self.embedder, gnn_ev, ev_qids,
                                      steps=s2_steps, lr=s2_lr, n_samples=s2_samples,
                                      max_paths_per_pair=self.max_paths_per_pair, device=self.device,
                                      ckpt_path=c2, eval_max=eval_max, verbose=verbose)
        self.train_info = {"stage1": info1, "stage2": info2, "seed": seed,
                           "controller": {"learned_stop": self.controller.learned_stop,
                                          "stop_threshold": self.controller.stop_threshold,
                                          "max_paths": self.controller.max_paths},
                           "loss_kw": dict(loss_kw or {})}
        return self.train_info

    def load(self, path: str):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        sd = ck["model_state"] if "model_state" in ck else ck
        # strict=False: pre-v3 checkpoints have no stop_head (fine unless learned_stop=True)
        missing, unexpected = self.controller.load_state_dict(sd, strict=False)
        missing = [k for k in missing if not k.startswith("stop_head.")] + \
                  ([k for k in missing if k.startswith("stop_head.")] if self.controller.learned_stop else [])
        if missing or unexpected:
            raise RuntimeError(f"checkpoint {path} incompatible: missing={missing} unexpected={unexpected}")
        self.controller.to(self.device)
        return self

    # --- inference -------------------------------------------------------
    def run(self, dataset, verbose=True):
        from adagraph_cgar_v2 import CGARRetriever
        gnn = to_gnn_features(dataset)
        ret = CGARRetriever(gnn, self.controller, self.embedder,
                            max_paths_per_pair=self.max_paths_per_pair, device=self.device)
        self.controller.eval()
        recs = []
        for q in self._iter(dataset, verbose):
            if not q.answers:
                continue
            t0 = time.perf_counter()
            with torch.no_grad():
                r = ret.retrieve(q.qid, deterministic=True, return_details=True)
            dt = (time.perf_counter() - t0) * 1000
            det = r.get("details") or {}
            conf = det.get("final_confidence")
            sel = r["selected_paths"]
            recs.append(make_record(q, self.name, r["triples"], [p["path_text"] for p in sel], r["cost"], dt,
                                    confidence=float(conf.item()) if conf is not None else None,
                                    chosen_k=int(det.get("chosen_k", 0) or 0),
                                    extra={"n_total_paths": r.get("n_total_paths", 0)}))
        return recs


class CGAREscalateMethod(Method):
    """CGAR-G retrieval (Nov 2026): CGAR with budget B, escalated per question.

    Run 3 showed that CGAR's whole loss vs GNN-RAG-all sits in questions with > 10
    candidates (F1 20.7 vs 33.4 on 17% of WebQSP) while it matches or beats all on
    the rest, and that gating on its confidence recovers all's F1 at 75% of the cost
    (offline escalation analysis in the research repo). This method makes that policy a first-class retriever:

        selection = CGAR(B)                          if n_candidates <= n_cand and conf >= tau
                  = fallback                         otherwise
        fallback  = 'all'      every path of the top-20 pool (= gnnrag_all's record)
                  = 'budget:K' CGAR with max_paths K (same controller)

    confidence stays CGAR's (so AURC / selective answering are unchanged), `escalated`
    is stored per record, cost = paths actually sent to the reader."""

    def __init__(self, cgar: "CGARMethod", n_cand: Optional[int] = 10, tau: Optional[float] = None,
                 fallback: str = "all", max_candidates: int = 20, name: str = "cgar_esc"):
        self.cgar, self.n_cand, self.tau, self.fallback = cgar, n_cand, tau, fallback
        self.max_candidates = max_candidates
        self.name = name
        if not (fallback == "all" or fallback.startswith("budget:")):
            raise ValueError("fallback must be 'all' or 'budget:K'")

    def should_escalate(self, n_candidates: int, confidence: Optional[float]) -> bool:
        if self.n_cand is not None and n_candidates > self.n_cand:
            return True
        return self.tau is not None and confidence is not None and confidence < self.tau

    def run(self, dataset, verbose=True):
        from adagraph_cgar_v2 import CGARRetriever
        from grl_rag import extract_candidate_paths
        gnn = to_gnn_features(dataset)
        ctrl = self.cgar.controller
        ret = CGARRetriever(gnn, ctrl, self.cgar.embedder,
                            max_paths_per_pair=self.cgar.max_paths_per_pair, device=self.cgar.device)
        ctrl.eval()
        base_budget = ctrl.max_paths
        recs = []
        for q in self._iter(dataset, verbose):
            if not q.answers:
                continue
            t0 = time.perf_counter()
            with torch.no_grad():
                r = ret.retrieve(q.qid, deterministic=True, return_details=True)
            det = r.get("details") or {}
            conf = det.get("final_confidence")
            conf = float(conf.item()) if conf is not None else None
            sel = r["selected_paths"]
            esc = self.should_escalate(len(q.top_candidates), conf)
            if esc:
                if self.fallback == "all":
                    sel = extract_candidate_paths(q.qid, gnn, max_candidates=self.max_candidates,
                                                  max_paths_per_pair=self.cgar.max_paths_per_pair)
                else:
                    ctrl.max_paths = int(self.fallback.split(":")[1])
                    try:
                        with torch.no_grad():
                            sel = ret.retrieve(q.qid, deterministic=True)["selected_paths"]
                    finally:
                        ctrl.max_paths = base_budget
            dt = (time.perf_counter() - t0) * 1000
            triples = [t for p in sel for t in p["path_triples"]]
            recs.append(make_record(q, self.name, triples, [p["path_text"] for p in sel], len(sel), dt,
                                    confidence=conf, chosen_k=int(det.get("chosen_k", 0) or 0),
                                    extra={"n_total_paths": r.get("n_total_paths", 0), "escalated": int(esc),
                                           "cost_primary": len(r["selected_paths"])}))
        return recs


# ─────────────────────────────────────────────────────────────────────
# registry for the CLI
# ─────────────────────────────────────────────────────────────────────

def baseline_from_name(name: str, max_candidates: Optional[int] = None, embedder=None,
                       max_paths_per_pair: int = 3) -> Method:
    """`max_candidates=None` keeps each class default (20)."""
    kw = dict(max_paths_per_pair=max_paths_per_pair)
    if max_candidates is not None:
        kw["max_candidates"] = max_candidates
    if name == "gnnrag_all":
        return GNNRAGMethod(None, **kw)
    if name.startswith("gnnrag_top"):
        return GNNRAGMethod(int(name[len("gnnrag_top"):]), **kw)
    if name.startswith("sbert_top"):
        if embedder is None:
            raise ValueError("sbert_topK needs an embedder")
        return SBERTTopKMethod(embedder, int(name[len("sbert_top"):]), **kw)
    raise ValueError(f"unknown baseline: {name}")
