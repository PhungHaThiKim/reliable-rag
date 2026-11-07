"""Command-line entry point.

    # smoke (CPU, no external deps):
    python -m kgqa_bench.cli run --train synthetic:60:0:train --eval synthetic:30:1 \
        --methods gnnrag_all gnnrag_top5 graphrag_fi cgar prism --embedder hash \
        --cgar-s1-epochs 2 --cgar-s2-steps 20 --prism-distill-epochs 2 --prism-rl-steps 20 \
        --reader dummy --out /tmp/bench_smoke

    # Colab (A100):
    python -m kgqa_bench.cli run \
        --train rog:rmanluo/RoG-webqsp:train --eval rog:rmanluo/RoG-webqsp:test rog:rmanluo/RoG-cwq:test \
        --methods gnnrag_all gnnrag_top10 gnnrag_top5 graphrag_fi cgar prism \
        --seeds 42 1337 2024 --reader hf:Qwen/Qwen2.5-7B-Instruct --out results/bench_rog

    # with GNN-RAG features (true GNN scores):
    python -m kgqa_bench.cli run --train gnnrag:ckpt/gnn_features_train.pkl:webqsp:train \
        --eval gnnrag:ckpt/gnn_features_webqsp_test.pkl:webqsp:test gnnrag:ckpt/gnn_features_cwq_test.pkl:cwq:test ...

    python -m kgqa_bench.cli report --results results/bench_rog
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np

from . import metrics as M
from .data import BenchDataset, load_spec
from .methods import CGARMethod, PRISMMethod, baseline_from_name
from .llm_reader import NoRetrievalReader, attach_llm_answers, reader_from_spec
from .report import reliability_plot, write_report


def _embedder(kind: str):
    from grl_rag import Embedder
    if kind == "hash":
        return Embedder(fallback="hash")
    return Embedder(model_name=kind if kind != "sbert" else "all-MiniLM-L6-v2")


def _dump_records(path: str, records: List[Dict]):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, default=float) + "\n")


def cmd_run(a):
    os.makedirs(a.out, exist_ok=True)
    t_start = time.time()
    emb = _embedder(a.embedder)
    device = a.device

    load_kw = dict(max_items=a.max_items, max_hops=a.max_hops, cache_dir=os.path.join(a.out, "data_cache"))
    if a.subset_seed is not None:
        load_kw["subset_seed"] = a.subset_seed
    if a.score_sim > 0:
        load_kw.update(embedder=emb, w_sim=a.score_sim)
    train_ds = load_spec(a.train, **load_kw) if a.train else None
    eval_dss = [load_spec(s, **load_kw) for s in a.eval]
    for ds in ([train_ds] if train_ds else []) + eval_dss:
        ds.annotate_hops()
        print(f"[data] {ds.key}: {len(ds)} questions  ({ds.source})")
    dataset_stats = {ds.key: ds.stats() for ds in eval_dss}
    dev_ds = eval_dss[0] if a.dev_from_first_eval else None   # ONLY for model selection; flagged in config
    if a.dev_split and train_ds is not None:
        n_dev = int(len(train_ds) * a.dev_split)
        dev_ds = BenchDataset(train_ds.name, "dev", train_ds.questions[-n_dev:], train_ds.source)
        train_ds = BenchDataset(train_ds.name, train_ds.split, train_ds.questions[:-n_dev], train_ds.source)
        dev_ds.annotate_hops()
        print(f"[data] dev split carved from train: {len(dev_ds)} questions")

    seeds = a.seeds or [42]
    summaries: Dict[str, Dict[str, Dict]] = {ds.key: {} for ds in eval_dss}
    strat: Dict[str, Dict[str, Dict]] = {ds.key: {} for ds in eval_dss}
    per_seed: Dict[str, Dict[str, Dict[int, Dict]]] = {ds.key: {} for ds in eval_dss}
    records_all: Dict[str, Dict[str, List[Dict]]] = {ds.key: {} for ds in eval_dss}
    train_logs: Dict = {}
    reader = reader_from_spec(a.reader)

    def evaluate(method, seed=None):
        for ds in eval_dss:
            recs = method.run(ds, verbose=not a.quiet)
            if reader is not None:
                attach_llm_answers(recs, ds, reader, cache_path=os.path.join(a.out, "llm_cache.jsonl"),
                                   verbose=not a.quiet)
            tag = method.name if seed is None else f"{method.name}@seed{seed}"
            _dump_records(os.path.join(a.out, f"records_{ds.key.replace(':', '_')}_{tag}.jsonl"), recs)
            summ = M.summarize_records(recs)
            if seed is None or method.name not in summaries[ds.key]:
                summaries[ds.key][method.name] = summ
                records_all[ds.key][method.name] = recs
                strat[ds.key][method.name] = {
                    "hops": M.stratify_records(recs, "hops", bins={4: "4+", 5: "4+", 6: "4+"}),
                    "n_gold": M.stratify_records(recs, "n_gold", bins={**{k: "3+" for k in range(3, 50)}}),
                }
            if seed is not None:
                per_seed[ds.key].setdefault(method.name, {})[seed] = summ
            print(f"  [{ds.key}] {tag}: F1={summ['f1']*100:.1f} Rec={summ['cov']*100:.1f} "
                  f"cost={summ['cost']:.2f} ECE={summ.get('ece', float('nan')):.3f} "
                  + (f"LLM-Hit={summ['llm_hit']*100:.1f}" if 'llm_hit' in summ else ""))

    # ---- baselines ----
    for name in a.methods:
        if name in ("cgar", "prism"):
            continue
        print(f"\n=== {name} ===")
        evaluate(baseline_from_name(name, max_candidates=a.baseline_candidates, embedder=emb,
                                    max_paths_per_pair=a.max_paths_per_pair))

    # ---- no-retrieval reader baseline (hallucination reference) ----
    if reader is not None and a.no_retrieval_baseline:
        print("\n=== no_retrieval (reader without context) ===")
        from .methods import GNNRAGMethod
        base = GNNRAGMethod(None); base.name = "no_retrieval"
        nr = NoRetrievalReader(reader)
        for ds in eval_dss:
            recs = base.run(ds, verbose=False)
            for r in recs:
                r["selected_paths"] = []; r["cost"] = 0; r["n_triples"] = 0; r["n_tokens"] = 0
                r["coverage"] = r["precision"] = r["f1"] = 0.0; r["hit_ctx"] = 0; r["confidence"] = None
            attach_llm_answers(recs, ds, nr, cache_path=os.path.join(a.out, "llm_cache.jsonl"), verbose=not a.quiet)
            summaries[ds.key]["no_retrieval"] = M.summarize_records(recs)
            records_all[ds.key]["no_retrieval"] = recs
            _dump_records(os.path.join(a.out, f"records_{ds.key.replace(':', '_')}_no_retrieval.jsonl"), recs)

    # ---- CGAR ----
    if "cgar" in a.methods:
        print("\n=== cgar ===")
        for seed in seeds:
            m = CGARMethod(emb, device=device, max_paths_per_pair=a.max_paths_per_pair,
                           rel_threshold=a.cgar_rel_threshold, sprt_beta=a.cgar_beta,
                           max_paths=a.cgar_max_paths, learned_stop=a.cgar_learned_stop,
                           stop_threshold=a.cgar_stop_threshold)
            loss_kw = dict(conf_target=a.cgar_conf_target, rel_label=a.cgar_rel_label,
                           w_stop=(a.cgar_w_stop if a.cgar_learned_stop else 0.0))
            if a.cgar_ckpt:
                m.load(a.cgar_ckpt)
            elif train_ds is not None:
                info = m.fit(train_ds, dev_ds, s1_epochs=a.cgar_s1_epochs, s2_steps=a.cgar_s2_steps,
                             s2_samples=a.cgar_s2_samples, eval_max=a.dev_max, seed=seed,
                             ckpt_dir=os.path.join(a.out, "ckpt"), verbose=not a.quiet, loss_kw=loss_kw)
                train_logs[f"cgar@seed{seed}"] = info
            else:
                print("  (no --train and no --cgar-ckpt: evaluating an UNTRAINED controller)")
            evaluate(m, seed=seed if len(seeds) > 1 else None)
            # Sweeps for the Pareto curve (same checkpoint, no retraining, on a fixed random subset).
            # cost knobs = max_paths (budget) x rel_threshold (adaptive stop); β only changes the K
            # level, NOT the number of selected paths, so it is logged separately as evidence.
            if a.sweep and seed == seeds[0]:
                sub = [ds.subset(a.sweep_max, seed=0) if a.sweep_max and len(ds) > a.sweep_max else ds for ds in eval_dss]
                pts = []
                for mp, thr in [(3, None), (5, None), (10, None), (20, None), (10, 0.0), (20, 0.0), (20, -1.0)]:
                    m.controller.max_paths, m.controller.rel_threshold = mp, thr
                    for d, ds in zip(sub, eval_dss):
                        s = M.summarize_records(m.run(d, verbose=False), n_bootstrap=50)
                        pts.append({"dataset": ds.key, "max_paths": mp, "rel_threshold": thr, "cost": s["cost"],
                                    "f1": s["f1"], "cov": s["cov"], "hit_ctx": s["hit_ctx"], "n": s["n"]})
                m.controller.max_paths, m.controller.rel_threshold = a.cgar_max_paths, a.cgar_rel_threshold
                train_logs["cgar_cost_sweep"] = pts
                pts = []
                for beta in [0.5, 0.99]:
                    m.controller.sprt_beta = beta
                    for d, ds in zip(sub, eval_dss):
                        s = M.summarize_records(m.run(d, verbose=False), n_bootstrap=50)
                        pts.append({"dataset": ds.key, "beta": beta, "cost": s["cost"], "f1": s["f1"], "cov": s["cov"]})
                m.controller.sprt_beta = a.cgar_beta
                train_logs["cgar_beta_sweep"] = pts

    # ---- PRISM ----
    if "prism" in a.methods:
        print("\n=== prism ===")
        from prism.config import CFG
        CFG.device = device
        if a.prism_d_model: CFG.d_model = a.prism_d_model
        if a.prism_layers: CFG.n_layers = a.prism_layers
        CFG.max_paths = a.prism_max_paths
        CFG.n_candidates = a.prism_candidates
        CFG.stopping_eps = a.prism_eps
        from .adapters import PrismVocab
        vocab = PrismVocab().fit(*( [train_ds] if train_ds else eval_dss ))
        teacher = None
        if a.prism_teacher_cache:
            from prism.teacher_llm import CachedTeacherLLM
            teacher = CachedTeacherLLM(cache_path=a.prism_teacher_cache, strict=False, verbose=True)
        for seed in seeds:
            m = PRISMMethod(emb, vocab=vocab, cfg=CFG, device=device)
            if a.prism_ckpt:
                m.load(a.prism_ckpt, vocab_path=a.prism_ckpt + ".vocab.json" if os.path.exists(a.prism_ckpt + ".vocab.json") else None)
            elif train_ds is not None:
                info = m.fit(train_ds, dev_ds, distill_epochs=a.prism_distill_epochs, rl_steps=a.prism_rl_steps,
                             teacher=teacher, ckpt_dir=os.path.join(a.out, "ckpt"), seed=seed, verbose=not a.quiet)
                train_logs[f"prism@seed{seed}"] = {k: v for k, v in info.items() if k != "rl_history"}
                m.save(os.path.join(a.out, "ckpt", f"prism_seed{seed}.pt"))
            else:
                print("  (no --train and no --prism-ckpt: evaluating an UNTRAINED LAS)")
            evaluate(m, seed=seed if len(seeds) > 1 else None)
            if a.sweep and seed == seeds[0]:
                # eps=0 never stops early -> budget-only curve (upper bound of the controller);
                # eps>0 at the full budget shows whether the saturation stop actually moves cost.
                sub = [ds.subset(a.sweep_max, seed=0) if a.sweep_max and len(ds) > a.sweep_max else ds for ds in eval_dss]
                pts = []
                base_sel = CFG.max_select_paths
                for eps, msel in [(0.0, 3), (0.0, 5), (0.0, 10), (0.005, 10), (0.02, 10), (0.05, 10), (0.2, 10)]:
                    CFG.stopping_eps, CFG.max_select_paths = eps, msel
                    for d, ds in zip(sub, eval_dss):
                        s = M.summarize_records(m.run(d, verbose=False), n_bootstrap=50)
                        pts.append({"dataset": ds.key, "eps": eps, "max_select": msel, "cost": s["cost"], "f1": s["f1"],
                                    "cov": s["cov"], "hit_ctx": s["hit_ctx"], "las_hit1": s.get("las_hit1"),
                                    "mean_step_kl": s.get("mean_step_kl"), "n": s["n"]})
                CFG.stopping_eps, CFG.max_select_paths = a.prism_eps, base_sel
                train_logs["prism_eps_sweep"] = pts

    # ---- comparisons ----
    comparisons: Dict[str, Dict] = {}
    for ds in eval_dss:
        comparisons[ds.key] = {}
        methods_here = list(records_all[ds.key])
        ref = a.compare_ref if a.compare_ref in methods_here else (methods_here[0] if methods_here else None)
        for m in methods_here:
            if m == ref:
                continue
            for metric in (["f1", "hit_ctx"] + (["llm_hit"] if reader is not None else [])):
                comparisons[ds.key][f"{m}_vs_{ref}:{metric}"] = M.compare_records(
                    records_all[ds.key][m], records_all[ds.key][ref], metric, n_bootstrap=a.n_bootstrap)
        reliability_plot(records_all[ds.key], os.path.join(a.out, f"reliability_{ds.key.replace(':', '_')}.png"))

    seeds_agg = ({ds.key: {m: M.aggregate_seeds(ps) for m, ps in per_seed[ds.key].items()}
                  for ds in eval_dss} if len(seeds) > 1 else {})

    out = {"summaries": summaries, "stratified": strat, "comparisons": comparisons, "seeds": seeds_agg,
           "dataset_stats": dataset_stats, "train_logs": train_logs,
           "config": {**vars(a), "elapsed_sec": time.time() - t_start,
                      "model_selection_on_first_eval_split": bool(a.dev_from_first_eval and not a.dev_split)}}
    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump(out, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    print("\n" + write_report(a.out))
    print(f"\n✓ results → {a.out}/results.json, report.md, pareto.png   ({(time.time()-t_start)/60:.1f} min)")


def cmd_report(a):
    print(write_report(a.results))


def build_parser():
    p = argparse.ArgumentParser(prog="kgqa_bench", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--train", default=None, help="dataset spec for training (see data.load_spec)")
    r.add_argument("--eval", nargs="+", required=True, help="one or more eval dataset specs")
    r.add_argument("--methods", nargs="+", default=["gnnrag_all", "gnnrag_top10", "gnnrag_top5", "sbert_top10", "graphrag_fi", "cgar", "prism"],
                   help="gnnrag_all | gnnrag_topK | sbert_topK | graphrag_fi[_kK] | random_kK | cgar | prism")
    r.add_argument("--embedder", default="sbert", help="sbert | hash | <sentence-transformers model>")
    r.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    r.add_argument("--out", required=True)
    r.add_argument("--max-items", type=int, default=None)
    r.add_argument("--max-hops", type=int, default=3)
    r.add_argument("--score-sim", type=float, default=0.0, help="blend SBERT q-entity sim into heuristic scores (RoG/JSONL only)")
    r.add_argument("--max-paths-per-pair", type=int, default=3)
    r.add_argument("--baseline-candidates", type=int, default=20,
                   help="candidate pool for gnnrag_*/random_k/graphrag_fi/sbert_top* (20 = same reachable pool as CGAR; 10 = Seminar-1 GNN-RAG-all)")
    r.add_argument("--seeds", nargs="*", type=int, default=None)
    r.add_argument("--dev-split", type=float, default=0.1, help="fraction of TRAIN carved out for model selection (0 = none)")
    r.add_argument("--dev-from-first-eval", action="store_true", help="legacy notebook behaviour: select on the first eval split (reported in config)")
    r.add_argument("--dev-max", type=int, default=400)
    r.add_argument("--reader", default="none", help="none | dummy | hf:<model>[:chat|grounded|llama2|general][:seq2seq]  (grounded = strict answer-from-paths prompt with abstention)")
    r.add_argument("--no-retrieval-baseline", action="store_true")
    r.add_argument("--compare-ref", default="gnnrag_all")
    r.add_argument("--n-bootstrap", type=int, default=5000)
    r.add_argument("--sweep", action="store_true", help="cost sweeps (CGAR max_paths×rel_threshold, PRISM ε×max_select) for Pareto curves; same checkpoint")
    r.add_argument("--sweep-max", type=int, default=600, help="random subset size per eval dataset used for sweeps (0 = full)")
    r.add_argument("--subset-seed", type=int, default=None, help="with --max-items: take a RANDOM subset (seeded) instead of the first N")
    r.add_argument("--quiet", action="store_true")
    # CGAR
    r.add_argument("--cgar-ckpt", default=None)
    r.add_argument("--cgar-s1-epochs", type=int, default=10)
    r.add_argument("--cgar-s2-steps", type=int, default=300)
    r.add_argument("--cgar-s2-samples", type=int, default=8)
    r.add_argument("--cgar-beta", type=float, default=0.85)
    r.add_argument("--cgar-max-paths", type=int, default=10)
    r.add_argument("--cgar-rel-threshold", type=float, default=None, help="adaptive path count (logit threshold, e.g. 0.0)")
    # v3 variants (Nov 2026) — all default OFF so `cgar` stays the legacy v2.1 controller
    r.add_argument("--cgar-learned-stop", action="store_true", help="B1: learned per-path stop head replaces the K-level SPRT as cost mechanism")
    r.add_argument("--cgar-stop-threshold", type=float, default=0.5, help="P(stop) above which the learned stop fires")
    r.add_argument("--cgar-w-stop", type=float, default=0.3, help="weight of the stop BCE in Stage 1")
    r.add_argument("--cgar-rel-label", choices=["entity", "proof"], default="entity", help="B2: relevance label = path touches gold (entity) | path relation sequence equals a reference proof (proof)")
    r.add_argument("--cgar-conf-target", choices=["recall", "hit"], default="recall", help="B3: confidence head target = answer recall (MSE) | hit (BCE; what ECE/AURC measure)")
    # PRISM
    r.add_argument("--prism-ckpt", default=None)
    r.add_argument("--prism-teacher-cache", default=None, help="CachedTeacherLLM .pt (real LLM logits); else oracle teacher")
    r.add_argument("--prism-distill-epochs", type=int, default=10)
    r.add_argument("--prism-rl-steps", type=int, default=500)
    r.add_argument("--prism-d-model", type=int, default=None)
    r.add_argument("--prism-layers", type=int, default=None)
    r.add_argument("--prism-max-paths", type=int, default=50)
    r.add_argument("--prism-candidates", type=int, default=32)
    r.add_argument("--prism-eps", type=float, default=0.05)
    r.set_defaults(func=cmd_run)
    rp = sub.add_parser("report")
    rp.add_argument("--results", required=True)
    rp.set_defaults(func=cmd_report)
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
