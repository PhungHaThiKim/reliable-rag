"""Method runners. Every runner maps a BenchDataset → list of per-question
records with the SAME keys, so `metrics.summarize_records` can score any of
them identically.

Record keys produced here
-------------------------
qid, method, dataset, coverage, precision, f1, hit_ctx, cost, n_triples,
n_tokens, latency_ms, selected_paths (texts), chosen_k, confidence,
hops, n_gold, n_candidates, answer_in_subgraph, (PRISM only) las_hit1

Methods
-------
GNNRAGMethod(top_k=None)       GNN-RAG all paths / top-K by candidate score
GraphRAGFIMethod(tau, k_keep)  GraphRAG-FI two-stage filter (re-impl)
CGARMethod(controller, ...)    CGAR v2.1 controller (+ .fit() to train)
PRISMMethod(las, vocab, ...)   PRISM LAS + batched information-theoretic controller
RandomKMethod(k)               sanity baseline: random K paths
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

from .adapters import (PrismBuildConfig, PrismSampleDataset, PrismVocab,
                       build_prism_samples, move_sample, to_gnn_features)
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
    }
    if extra:
        rec.update(extra)
    return rec


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
# GNN-RAG (all / top-K)  and GraphRAG-FI  — path-dict based baselines
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


class RandomKMethod(Method):
    def __init__(self, k: int = 5, max_candidates: int = 20, max_paths_per_pair: int = 3, seed: int = 0):
        self.k, self.max_candidates, self.max_paths_per_pair, self.seed = k, max_candidates, max_paths_per_pair, seed
        self.name = f"random_k{k}"

    def run(self, dataset, verbose=True):
        import random
        from grl_rag import extract_candidate_paths
        rng = random.Random(self.seed)
        gnn = to_gnn_features(dataset)
        recs = []
        for q in self._iter(dataset, verbose):
            if not q.answers:
                continue
            t0 = time.perf_counter()
            paths = extract_candidate_paths(q.qid, gnn, max_candidates=self.max_candidates,
                                            max_paths_per_pair=self.max_paths_per_pair)
            rng.shuffle(paths); paths = paths[: self.k]
            dt = (time.perf_counter() - t0) * 1000
            triples = [t for p in paths for t in p["path_triples"]]
            recs.append(make_record(q, self.name, triples, [p["path_text"] for p in paths], len(paths), dt))
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


class GraphRAGFIMethod(Method):
    def __init__(self, tau_drop: float = 0.3, k_keep: int = 8, max_candidates: int = 20,
                 max_paths_per_pair: int = 3):
        self.kw = dict(tau_drop=tau_drop, k_keep=k_keep, max_candidates=max_candidates,
                       max_paths_per_pair=max_paths_per_pair)
        self.name = f"graphrag_fi_k{k_keep}"

    def run(self, dataset, verbose=True):
        from baseline_graphrag_fi import GraphRAGFIRetriever
        gnn = to_gnn_features(dataset)
        ret = GraphRAGFIRetriever(gnn, **self.kw)
        recs = []
        for q in self._iter(dataset, verbose):
            if not q.answers:
                continue
            t0 = time.perf_counter()
            r = ret.retrieve(q.qid)
            dt = (time.perf_counter() - t0) * 1000
            sel = r.get("selected_paths", [])
            conf = max([p["gnn_score"] for p in sel], default=0.0)
            recs.append(make_record(q, self.name, r["triples"], [p["path_text"] for p in sel],
                                    r["cost"], dt, confidence=float(conf)))
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
            seed: int = 42, verbose: bool = True) -> Dict:
        from .cgar_train import set_seed, train_cgar_stage1, train_cgar_stage2
        set_seed(seed)
        gnn_tr = to_gnn_features(train_ds)
        gnn_ev = to_gnn_features(eval_ds) if eval_ds is not None else None
        ev_qids = eval_ds.qids() if eval_ds is not None else None
        c1 = os.path.join(ckpt_dir, f"cgar_s1_seed{seed}.pt") if ckpt_dir else None
        c2 = os.path.join(ckpt_dir, f"cgar_s2_seed{seed}.pt") if ckpt_dir else None
        info1 = train_cgar_stage1(self.controller, gnn_tr, train_ds.qids(), self.embedder, gnn_ev, ev_qids,
                                  epochs=s1_epochs, lr=s1_lr, max_paths_per_pair=self.max_paths_per_pair,
                                  device=self.device, ckpt_path=c1, eval_max=eval_max, verbose=verbose)
        info2 = {}
        if s2_steps > 0:
            info2 = train_cgar_stage2(self.controller, gnn_tr, train_ds.qids(), self.embedder, gnn_ev, ev_qids,
                                      steps=s2_steps, lr=s2_lr, n_samples=s2_samples,
                                      max_paths_per_pair=self.max_paths_per_pair, device=self.device,
                                      ckpt_path=c2, eval_max=eval_max, verbose=verbose)
        self.train_info = {"stage1": info1, "stage2": info2, "seed": seed}
        return self.train_info

    def load(self, path: str):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.controller.load_state_dict(ck["model_state"] if "model_state" in ck else ck)
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


# ─────────────────────────────────────────────────────────────────────
# PRISM
# ─────────────────────────────────────────────────────────────────────

class PRISMMethod(Method):
    """PRISM = LAS (answer simulator) + batched information-theoretic controller.

    Samples are built from the BenchDataset with the SAME candidate ranking
    and shortest paths CGAR sees (see adapters.build_prism_samples).
    """

    def __init__(self, embedder, vocab: Optional[PrismVocab] = None, las=None, cfg=None,
                 build_cfg: Optional[PrismBuildConfig] = None, device: str = "cpu",
                 name: str = "prism"):
        from prism.config import CFG
        self.embedder = embedder
        self.cfg = cfg or CFG
        self.cfg.device = device
        self.device = device
        self.vocab = vocab or PrismVocab()
        self.build_cfg = build_cfg or PrismBuildConfig(n_candidates=self.cfg.n_candidates,
                                                       max_paths=self.cfg.max_paths,
                                                       max_path_len=self.cfg.max_path_len)
        self.las = las
        self.name = name
        self.train_info: Dict = {}
        self._sample_cache: Dict[str, List] = {}

    def samples_for(self, dataset: BenchDataset, force_gold: bool = False) -> List:
        key = f"{dataset.key}|n={len(dataset)}|id={id(dataset)}|gold={force_gold}"
        if key not in self._sample_cache:
            bc = PrismBuildConfig(**{**self.build_cfg.__dict__, "force_gold_in_candidates": force_gold})
            self._sample_cache[key] = build_prism_samples(dataset, self.vocab, self.embedder, bc)
        return self._sample_cache[key]

    def _ensure_las(self):
        if self.las is None:
            from prism.las import LAS
            self.cfg.n_relations = max(self.cfg.n_relations, len(self.vocab) + 8)
            self.cfg.d_query = int(self.embedder.emb_dim)
            self.las = LAS(self.cfg).to(self.device)
        return self.las

    # --- training --------------------------------------------------------
    def fit(self, train_ds: BenchDataset, eval_ds: Optional[BenchDataset] = None,
            distill_epochs: Optional[int] = None, rl_steps: Optional[int] = None,
            teacher=None, rl_mode: str = "calib_shaped", entropy_coef: float = 0.02,
            force_gold_train: bool = True, ckpt_dir: Optional[str] = None,
            seed: int = 42, verbose: bool = True) -> Dict:
        """Stage 1 distillation (+ Stage 2 RL). `teacher`: CachedTeacherLLM
        (real LLM logits) — if None the ORACLE teacher built into the samples
        is used; the summary records which one, because the paper must say
        "oracle-supervised" vs "LLM-distilled"."""
        import prism.distill as distill_mod
        from prism.distill import train_distill
        from prism.improvements import train_rl_improved
        from .cgar_train import set_seed
        set_seed(seed)
        if self.vocab is None or len(self.vocab) <= 2:
            self.vocab = PrismVocab().fit(train_ds)
        las = self._ensure_las()
        if distill_epochs is not None: self.cfg.distill_epochs = distill_epochs
        if rl_steps is not None: self.cfg.rl_steps = rl_steps
        if ckpt_dir: self.cfg.out_dir = ckpt_dir
        tr = PrismSampleDataset(self.samples_for(train_ds, force_gold=force_gold_train), "train")
        ev = PrismSampleDataset(self.samples_for(eval_ds, force_gold=False), "eval") if eval_ds is not None \
            else PrismSampleDataset(tr.samples[: max(8, len(tr) // 10)], "eval")
        orig_teacher = distill_mod.TeacherLLM
        teacher_name = "oracle_heuristic"
        if teacher is not None:
            distill_mod.TeacherLLM = lambda: teacher
            teacher_name = type(teacher).__name__
        try:
            ck1 = train_distill(las, tr, ev, self.cfg)
        finally:
            distill_mod.TeacherLLM = orig_teacher
        las.load_state_dict(torch.load(ck1, map_location="cpu"))
        info = {"teacher": teacher_name, "distill_ckpt": ck1, "seed": seed, "vocab": len(self.vocab)}
        if self.cfg.rl_steps > 0:
            ck2 = train_rl_improved(las, tr, ev, cfg=self.cfg, reward_mode=rl_mode, entropy_coef=entropy_coef,
                                    use_batched=True, eval_every=max(10, self.cfg.rl_eval_every),
                                    log_every=25, eval_max_items=200)
            las.load_state_dict(torch.load(ck2, map_location="cpu"))
            info["rl_ckpt"] = ck2
            info["rl_history"] = getattr(train_rl_improved, "last_history", [])
        self.train_info = info
        return info

    def load(self, path: str, vocab_path: Optional[str] = None):
        import json
        if vocab_path:
            with open(vocab_path) as f:
                self.vocab = PrismVocab.from_state_dict(json.load(f))
        las = self._ensure_las()
        las.load_state_dict(torch.load(path, map_location="cpu"))
        las.to(self.device)
        return self

    def save(self, path: str):
        import json
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(self._ensure_las().state_dict(), path)
        with open(path + ".vocab.json", "w") as f:
            json.dump(self.vocab.state_dict(), f)

    # --- inference -------------------------------------------------------
    def run(self, dataset, verbose=True):
        from prism.controller_batched import deterministic_select
        las = self._ensure_las(); las.eval()
        samples = self.samples_for(dataset, force_gold=False)
        by_qid = {q.qid: q for q in dataset}
        recs = []
        it = samples
        if verbose:
            try:
                from tqdm import tqdm; it = tqdm(samples, desc=self.name, leave=False)
            except ImportError:
                pass
        for s in it:
            q = by_qid[s.qid]
            if not q.answers:
                continue
            move_sample(s, self.device)
            t0 = time.perf_counter()
            with torch.no_grad():
                trace = deterministic_select(las, s, self.cfg)
            dt = (time.perf_counter() - t0) * 1000
            sel = trace.selected_path_idx
            triples = [t for i in sel for t in s.path_triples[i]]
            texts = [s.path_texts[i] for i in sel]
            top1 = int(trace.final_probs.argmax().item())
            las_hit1 = int(s.candidate_names[top1] in set(q.answers))
            recs.append(make_record(q, self.name, triples, texts, len(sel), dt,
                                    confidence=float(trace.confidence.item()),
                                    chosen_k=int(trace.chosen_K),
                                    extra={"las_hit1": las_hit1, "stopped_reason": trace.stopped_reason,
                                           "n_total_paths": int(s.paths.size(0)),
                                           "n_steps": len(trace.step_kl),
                                           "mean_step_kl": float(np.mean(trace.step_kl)) if trace.step_kl else 0.0}))
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
    if name.startswith("graphrag_fi"):
        k = int(name.split("_k")[1]) if "_k" in name else 8
        return GraphRAGFIMethod(k_keep=k, **kw)
    if name.startswith("random_k"):
        return RandomKMethod(int(name[len("random_k"):]), **kw)
    if name.startswith("sbert_top"):
        if embedder is None:
            raise ValueError("sbert_topK needs an embedder")
        return SBERTTopKMethod(embedder, int(name[len("sbert_top"):]), **kw)
    raise ValueError(f"unknown baseline: {name}")
