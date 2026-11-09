"""Command-line entry point. Every subcommand accepts `--config <yaml> [...]`: leaf keys of the YAML that
match a flag name (dashes -> underscores) become defaults; explicit flags still win.

    # CPU smoke (no GPU, no downloads):
    python -m pypi_rag.cli prepare --spec synthetic:120:0 --out /tmp/prd/data
    python -m pypi_rag.cli build-kb --data /tmp/prd/data --out /tmp/prd/kb.jsonl
    python -m pypi_rag.cli run --data /tmp/prd/data --split test --kb /tmp/prd/kb.jsonl \
        --methods e0 e1 e2 e3 e3a e4 --llm dummy --embedder hash --out /tmp/prd/run

    # Colab (A100):
    python -m pypi_rag.cli prepare --spec malllm:external/mal-LLM --strategy upstream --out work/data/malllm
    python -m pypi_rag.cli run --data work/data/malllm --split test --kb work/kb/malllm.jsonl \
        --methods e0 e1 e2 e3 e3a e4 --llm hf:Qwen/Qwen2.5-Coder-7B-Instruct --embedder BAAI/bge-small-en-v1.5 \
        --out work/results/malllm_qwen
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Dict, List

from .schema import PredictionResult, Sample, load_samples, read_jsonl, save_samples, write_jsonl


def _p(*a):
    print(*a, flush=True)


def _load_data(d: str) -> List[Sample]:
    p = os.path.join(d, "samples.jsonl.gz") if os.path.isdir(d) else d
    return load_samples(p)


def _facts_cache_dir(a) -> str:
    if getattr(a, "facts_cache", None):
        return a.facts_cache
    d = a.data if os.path.isdir(a.data) else os.path.dirname(os.path.abspath(a.data))
    return os.path.join(d, "facts_cache")


def _facts_for(samples: List[Sample], a) -> Dict:
    """Static facts for these samples: parallel workers + on-disk cache (analysis.parallel)."""
    from .analysis.parallel import analyze_many
    return analyze_many([s.public_view() for s in samples], workers=getattr(a, "workers", 0),
                        cache_dir=_facts_cache_dir(a), timeout=getattr(a, "analysis_timeout", 180), log=_p)


def _code_hash() -> str:
    import glob
    import hashlib
    h = hashlib.sha256()
    for f in sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "**", "*.py"), recursive=True)):
        with open(f, "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()[:16]


def _subset(samples: List[Sample], n: int, seed: int) -> List[Sample]:
    """Label-stratified seeded subset: the same ids for every method and every run with this seed."""
    if not n or n >= len(samples):
        return samples
    rng = random.Random(seed)
    by = {}
    for s in sorted(samples, key=lambda s: s.sample_id):
        by.setdefault(s.label, []).append(s)
    out = []
    for lab, xs in sorted(by.items()):
        k = round(n * len(xs) / len(samples))
        out += rng.sample(xs, min(k, len(xs)))
    return sorted(out, key=lambda s: s.sample_id)


# ------------------------------------------------------------------------------------------ prepare
def cmd_prepare(a):
    from .data.leakage_check import split_leakage
    from .data.loaders import load_spec
    from .data.splitter import assign_splits
    os.makedirs(a.out, exist_ok=True)
    samples: List[Sample] = []
    for spec in a.spec:
        t0 = time.time()
        xs = load_spec(spec, max_items=a.max_items)
        _p(f"[data] {spec}: {len(xs)} samples ({time.time() - t0:.0f}s)")
        samples += xs
    ids = set()
    for s in samples:                                        # unique ids across specs
        while s.sample_id in ids:
            s.sample_id += "_dup"
        ids.add(s.sample_id)
    rep = assign_splits(samples, a.strategy, tuple(a.ratios), a.seed, a.near_dup if a.near_dup > 0 else None)
    leak = split_leakage(samples, a.near_dup if a.near_dup > 0 else 0.9)
    save_samples(os.path.join(a.out, "samples.jsonl.gz"), samples)
    write_jsonl(os.path.join(a.out, "manifest.jsonl"), (s.manifest_record() for s in samples))
    stats = {"n": len(samples), "specs": a.spec, "split_report": rep, "leakage": leak,
             "no_source": sum(1 for s in samples if not s.files),
             "parse_files": sum(len(s.files) for s in samples)}
    json.dump(stats, open(os.path.join(a.out, "stats.json"), "w"), indent=2)
    _p(json.dumps(rep["counts"], indent=1)); _p("leakage:", json.dumps({k: v for k, v in leak.items() if k != "counts"}))


# ------------------------------------------------------------------------------------------ PyGuard
def cmd_pyguard_bundle(a):
    from .knowledge import pyguard_adapter as PG
    origin = f"pyguard@{PG.repo_sha(a.repo)}"
    tax = PG.load_taxonomy(a.repo)
    lex = PG.harvest_api_lexicon(a.repo, max_files=a.max_files)
    pats = PG.load_patterns(a.repo)
    PG.save_bundle(a.out, lex, pats, tax, origin)
    kinds = {}
    for p in pats:
        k = PG._kind(p["type"]); kinds[k] = kinds.get(k, 0) + 1
    _p(f"[pyguard] {origin}: taxonomy {len(tax)} behaviors, lexicon {len(lex)} APIs "
       f"({sum(1 for v in lex.values() if v['behaviors'])} mapped), patterns {len(pats)} {kinds} -> {a.out}")


def _pyguard(a):
    if not getattr(a, "pyguard_bundle", None):
        return None, None
    from .analysis import taxonomy as T
    from .knowledge import pyguard_adapter as PG
    b = PG.load_bundle(a.pyguard_bundle)
    if getattr(a, "pyguard_taxonomy", False):   # opt-in: changes the static analysis shared by ALL methods
        n = T.extend_rules(PG.lexicon_rules(b["lexicon"], b["origin"], a.pyguard_min_count))
        _p(f"[pyguard] +{n} API rules from {b['origin']}")
    return b, PG.PyGuardMatcher(b["lexicon"], b["patterns"])


# ------------------------------------------------------------------------------------------ KB
def cmd_build_kb(a):
    from .analysis import analyze_package
    from .knowledge import build_kb, save_kb
    samples = _load_data(a.data)
    bundle, _ = _pyguard(a)
    t0 = time.time()
    facts = _facts_for([s for s in samples if s.split == "train"], a)
    facts_of = lambda s: facts.get(s.sample_id) or analyze_package(s)  # noqa: E731
    docs, rep = build_kb(samples, facts_of, a.sources, bundle, a.yara, a.advisories, a.max_examples_per_label, a.seed)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    save_kb(a.out, docs)
    json.dump(rep, open(a.out + ".report.json", "w"), indent=2)
    _p(f"[kb] {len(docs)} docs {rep['by_source_type']} | dropped by leakage audit: {rep['n_dropped']} -> {a.out} "
       f"({time.time() - t0:.0f}s)")


# ------------------------------------------------------------------------------------------ graph
def cmd_build_graph(a):
    from .analysis import analyze_package
    from .graph import build_behavior_graph, graph_evidence, graph_stats
    _, matcher = _pyguard(a)
    samples = [s for s in _load_data(a.data) if not a.split or s.split == a.split]
    samples = _subset(samples, a.max_items, a.subset_seed)
    os.makedirs(a.out, exist_ok=True)
    facts = _facts_for(samples, a)
    rows = []
    for s in samples:
        pub = s.public_view()
        f = facts.get(s.sample_id) or analyze_package(pub)
        G = build_behavior_graph(pub, f, matcher)
        items, diag = graph_evidence(G, pub)
        rows.append({"sample_id": s.sample_id, "label": s.label, **graph_stats(G), **diag,
                     "paths": [{"apis": i.meta.get("ops_api"), "patterns": i.meta.get("patterns"), "spans": i.meta.get("spans")}
                               for i in items if i.source_type == "graph_path"],
                     "gold_paths": s.gold.get("gold_paths", [])})
    write_jsonl(os.path.join(a.out, f"graphs_{a.split or 'all'}.jsonl"), rows)
    import numpy as np
    summ = {"n": len(rows), "edges_mean": float(np.mean([r["n_edges"] for r in rows])) if rows else 0,
            "paths_mean": float(np.mean([r["n_paths"] for r in rows])) if rows else 0,
            "missing_provenance_edges": int(sum(r["edges_missing_provenance"] for r in rows)),
            "pattern_hit_rate_malicious": float(np.mean([bool(r["pattern_ids"]) for r in rows if r["label"] == "malicious"] or [0])),
            "pattern_hit_rate_benign": float(np.mean([bool(r["pattern_ids"]) for r in rows if r["label"] == "benign"] or [0]))}
    json.dump(summ, open(os.path.join(a.out, f"graphs_{a.split or 'all'}.summary.json"), "w"), indent=2)
    _p("[graph]", json.dumps(summ))


# ------------------------------------------------------------------------------------------ run
def _ctx(a, kb_docs):
    from .generation.llm_client import CachedLLM, llm_from_spec
    from .pipelines import Context, RunConfig
    from .retrieval import CrossEncoderReranker, Embedder
    _, matcher = _pyguard(a)
    kw = dict(max_new_tokens=a.max_new_tokens, max_input_tokens=a.max_input_tokens, batch_size=a.batch_size,
              adapter=a.adapter, gpu_memory_utilization=a.gpu_mem, concurrency=a.concurrency, tokenizer=a.tokenizer)
    base = llm_from_spec(a.llm, **kw)
    cache_path = a.llm_cache or os.path.join(a.out, "llm_cache.jsonl")
    llm = CachedLLM(base, cache_path)
    cfg = RunConfig(code_tokens=a.code_tokens, evidence_tokens=a.evidence_tokens, vector_k=a.vector_k, bm25_k=a.bm25_k,
                    final_k=a.final_k, rrf_k=a.rrf_k, max_per_source=a.max_per_source, n_queries=a.n_queries,
                    kb_source_types=a.kb_source_types, max_hops=a.max_hops, max_paths=a.max_paths, seed=a.seed,
                    dispatch_caveat=not a.no_dispatch_caveat, use_sample_question=a.use_sample_question)
    if a.policy:
        pol = json.load(open(a.policy))
        cfg.policy_min_critical = pol.get("min_critical", 1); cfg.policy_conflict_severity = pol.get("conflict_severity")
        cfg.policy_abstain_on_dispatch = bool(pol.get("abstain_on_dispatch", False))
    if a.abstain_on_dispatch:
        cfg.policy_abstain_on_dispatch = True
    emb = Embedder(a.embedder, device=None if a.embedder == "hash" else a.device)
    rr = CrossEncoderReranker(a.reranker, device=a.device) if a.reranker and any(m.endswith("rerank") for m in a.methods) else None
    return Context(llm, cfg, kb_docs, emb, rr, matcher, cache_dir=a.out), cfg


def cmd_run(a):
    from .evaluation import case_table, comparisons, cost_estimate, grounding_vs_base, stratify, summarize_method
    from .evaluation.report import write_report
    from .knowledge import load_kb
    from .pipelines import METHODS, predict_all
    os.makedirs(a.out, exist_ok=True)
    t_start = time.time()
    all_samples = _load_data(a.data)
    samples = [s for s in all_samples if s.split == a.split]
    samples = _subset(samples, a.max_items, a.subset_seed)       # subset FIRST: filtered runs reuse the LLM cache
    if a.drop_cross_dups:
        samples = [s for s in samples if not s.meta.get("cross_split_dup")]
    if a.require_source:
        samples = [s for s in samples if s.files]
    kb_docs = load_kb(a.kb) if a.kb else []
    if not kb_docs and any(METHODS[m]["kind"] in ("text_rag",) or METHODS[m].get("ctx") for m in a.methods):
        _p("[kb] no --kb given: building builtin + train examples on the fly")
        from .analysis import analyze_package
        from .knowledge import build_kb
        tr_facts = _facts_for([s for s in all_samples if s.split == "train"], a)
        kb_docs, _ = build_kb(all_samples, lambda s: tr_facts.get(s.sample_id) or analyze_package(s), seed=a.seed)
    facts = _facts_for(samples, a)
    ctx, cfg = _ctx(a, kb_docs)
    ctx._facts.update(facts)
    _p(f"[run] {len(samples)} samples ({a.split}) | methods {a.methods} | llm {a.llm} | kb {len(kb_docs)} docs")
    by_id = {s.sample_id: s for s in samples}
    preds_by, traces_by, summaries, strata = {}, {}, {}, {}
    run_id = f"{os.path.basename(a.out.rstrip('/'))}-{int(t_start)}"
    t0 = time.time()
    results = predict_all(a.methods, samples, ctx, run_id, progress=_p)
    for m in a.methods:
        preds, traces = results[m]
        write_jsonl(os.path.join(a.out, f"predictions_{a.split}_{m}.jsonl"), (p.to_dict() for p in preds))
        write_jsonl(os.path.join(a.out, f"traces_{a.split}_{m}.jsonl"), traces)
        preds_by[m], traces_by[m] = preds, traces
        summaries[m] = summarize_method(preds, by_id, traces, n_bootstrap=a.n_bootstrap)
        strata[m] = stratify(preds, by_id, a.strata_key)
        s = summaries[m]
        _p(f"[{m}] strict F1 {s['strict/f1_malicious']:.3f} | FPR {s['strict/fpr']:.3f} | coverage {s['coverage']:.3f} | "
           f"unsupported {s['evidence/unsupported_claim_rate']:.3f} | parse-fail {s['parse_failure_rate']:.3f}")
    comps = comparisons(preds_by, by_id, a.compare_ref, a.n_bootstrap)
    labels = {}
    for s in samples:
        labels[s.label] = labels.get(s.label, 0) + 1
    kb_rep = json.load(open(a.kb + ".report.json")) if a.kb and os.path.exists(a.kb + ".report.json") else None
    R = {"config": {**{k: v for k, v in vars(a).items() if k != "func"}, "run_config": cfg.__dict__,
                    "llm_config": ctx.llm.config(), "code_hash": _code_hash()},
         "n_samples": len(samples), "labels": labels, "sample_ids": [s.sample_id for s in samples],
         "summaries": summaries, "comparisons": comps, "strata": strata, "kb_report": kb_rep,
         "cases": case_table(preds_by, by_id, traces_by), "grounding_vs_e3": grounding_vs_base(preds_by),
         "cost": cost_estimate(preds_by, a.gpu_price_hour), "elapsed_min": (time.time() - t_start) / 60}
    json.dump(R, open(os.path.join(a.out, "results.json"), "w"), indent=1, default=float)
    write_report(a.out)
    _p(f"[done] {a.out}/results.json ({R['elapsed_min']:.1f} min)")


# ------------------------------------------------------------------------------------------ offline
def _load_preds(results: str, split: str, method: str) -> List[PredictionResult]:
    return [PredictionResult.from_dict(d) for d in read_jsonl(os.path.join(results, f"predictions_{split}_{method}.jsonl"))]


def _reverify(preds: List[PredictionResult], by_id: Dict[str, Sample], policy: str, min_critical: int, sev, facts_cache,
              new_method: str, abstain_dispatch: bool = False) -> List[PredictionResult]:
    from .analysis import analyze_package
    from .generation.output_schema import parse_json_response
    from .verification import apply_policy, check_claims, restrict_check
    out = []
    for p in preds:
        s = by_id[p.sample_id].public_view()
        f = facts_cache.setdefault(s.sample_id, analyze_package(s))
        vis = {e.evidence_id: e for e in p.evidence}
        parsed = parse_json_response(p.raw_response, vis)
        chk = check_claims(parsed["claims"], vis, f, s)
        pol = apply_policy(parsed["verdict"], chk, vis, policy, min_critical, sev, abstain_dispatch)
        q = PredictionResult.from_dict(p.to_dict())
        q.method, q.verdict, q.raw_verdict = new_method, pol["verdict"], parsed["verdict"]
        keep_ids = {id(l) for l in pol["kept_claims"]}
        q.claims = [c for c, l in zip(parsed["claims"], chk["claims"]) if id(l) in keep_ids]
        q.verification = {**restrict_check(chk, pol["kept_claims"]), "policy": policy, "policy_reason": pol["reason"]}
        out.append(q)
    return out


def cmd_tune_policy(a):
    from .analysis import analyze_package
    from .generation.output_schema import parse_json_response
    from .verification import check_claims, tune_policy
    by_id = {s.sample_id: s for s in _load_data(a.data)}
    preds = _load_preds(a.results, a.split, a.method)
    rows, cache = [], {}
    for p in preds:
        s = by_id[p.sample_id].public_view()
        f = cache.setdefault(s.sample_id, analyze_package(s))
        vis = {e.evidence_id: e for e in p.evidence}
        parsed = parse_json_response(p.raw_response, vis)
        rows.append({"sample_id": p.sample_id, "verdict": parsed["verdict"], "visible": vis,
                     "check": check_claims(parsed["claims"], vis, f, s)})
    pol = tune_policy(rows, {i: s.label for i, s in by_id.items()})
    pol.update({"tuned_on": f"{a.results}:{a.split}:{a.method}", "n": len(rows)})
    json.dump(pol, open(a.out, "w"), indent=2)
    _p("[policy]", pol)


def cmd_rescore(a):
    """Recompute summaries from saved predictions; with --derive-e4 also build e4/e4_filter/e4_cite from e3
    offline (no LLM call)."""
    from .evaluation import case_table, comparisons, cost_estimate, grounding_vs_base, stratify, summarize_method
    from .evaluation.report import write_report
    R = json.load(open(os.path.join(a.results, "results.json")))
    split = R["config"]["split"]
    by_id = {s.sample_id: s for s in _load_data(a.data or R["config"]["data"])}
    by_id = {i: by_id[i] for i in R["sample_ids"] if i in by_id}
    methods = list(R["summaries"])
    preds_by = {m: _load_preds(a.results, split, m) for m in methods}
    if a.derive_e4 and "e3" in preds_by:
        pol = json.load(open(a.policy)) if a.policy else {"min_critical": 1, "conflict_severity": 0.9}
        cache = {}
        for name, mode in (("e4", "abstain"), ("e4_filter", "filter"), ("e4_cite", "cite")):
            preds_by[name] = _reverify(preds_by["e3"], by_id, mode, pol["min_critical"], pol.get("conflict_severity"),
                                       cache, name, bool(pol.get("abstain_on_dispatch", False)))
            write_jsonl(os.path.join(a.results, f"predictions_{split}_{name}.jsonl"), (p.to_dict() for p in preds_by[name]))
        R["policy"] = pol
    traces = {}
    for m in preds_by:
        p = os.path.join(a.results, f"traces_{split}_{m}.jsonl")
        if not os.path.exists(p) and m.startswith("e4"):
            p = os.path.join(a.results, f"traces_{split}_e3.jsonl")
        traces[m] = list(read_jsonl(p)) if os.path.exists(p) else []
    R["summaries"] = {m: summarize_method(ps, by_id, traces[m], a.n_bootstrap) for m, ps in preds_by.items()}
    R["strata"] = {m: stratify(ps, by_id, R["config"].get("strata_key", "difficulty")) for m, ps in preds_by.items()}
    R["comparisons"] = comparisons(preds_by, by_id, R["config"].get("compare_ref"), a.n_bootstrap)
    R["cases"] = case_table(preds_by, by_id, traces)
    R["grounding_vs_e3"] = grounding_vs_base(preds_by)
    R["cost"] = cost_estimate(preds_by, a.gpu_price_hour)
    json.dump(R, open(os.path.join(a.results, "results.json"), "w"), indent=1, default=float)
    write_report(a.results)
    _p("[rescore]", a.results, list(R["summaries"]))


def cmd_report(a):
    from .evaluation.report import write_report
    print(write_report(a.results))


# ------------------------------------------------------------------------------------------ parser
def build_parser():
    ap = argparse.ArgumentParser(prog="pypi_rag")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--config", nargs="*", default=[], help="YAML file(s) with defaults")
        p.add_argument("--seed", type=int, default=42)
        return p

    p = common(sub.add_parser("prepare", help="load specs, grouped split, leakage audit"))
    p.add_argument("--spec", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--strategy", default="grouped", choices=["grouped", "time", "upstream", "keep"])
    p.add_argument("--ratios", nargs=3, type=float, default=[0.7, 0.15, 0.15])
    p.add_argument("--near-dup", type=float, default=0.9, help="MinHash Jaccard for near-duplicate grouping (0 = off)")
    p.add_argument("--max-items", type=int, default=None)
    p.set_defaults(func=cmd_prepare)

    p = common(sub.add_parser("pyguard-bundle", help="extract lexicon + patterns from a PyGuard checkout"))
    p.add_argument("--repo", required=True); p.add_argument("--out", required=True)
    p.add_argument("--max-files", type=int, default=None)
    p.set_defaults(func=cmd_pyguard_bundle)

    def anal(p):
        p.add_argument("--workers", type=int, default=0, help="analysis worker processes (0 = cpu_count - 2, max 64)")
        p.add_argument("--facts-cache", default=None, help="facts cache dir (default <data>/facts_cache)")
        p.add_argument("--analysis-timeout", type=int, default=180, help="seconds per package before regex-only scan")

    def pyg(p):
        p.add_argument("--pyguard-bundle", default=None)
        p.add_argument("--pyguard-min-count", type=int, default=3)
        p.add_argument("--pyguard-taxonomy", action="store_true",
                       help="also add PyGuard API->behavior rules to the static analyzer (ablation; affects every method)")

    p = common(sub.add_parser("build-kb", help="text knowledge base from train split + external knowledge"))
    p.add_argument("--data", required=True); p.add_argument("--out", required=True)
    p.add_argument("--sources", nargs="+", default=["builtin", "examples", "pyguard"])
    p.add_argument("--yara", default=None); p.add_argument("--advisories", default=None)
    p.add_argument("--max-examples-per-label", type=int, default=1500)
    pyg(p); anal(p); p.set_defaults(func=cmd_build_kb)

    p = common(sub.add_parser("build-graph", help="graph diagnostics without an LLM"))
    p.add_argument("--data", required=True); p.add_argument("--out", required=True)
    p.add_argument("--split", default="test"); p.add_argument("--max-items", type=int, default=None)
    p.add_argument("--subset-seed", type=int, default=0)
    pyg(p); anal(p); p.set_defaults(func=cmd_build_graph)

    p = common(sub.add_parser("run", help="run methods and evaluate"))
    p.add_argument("--data", required=True); p.add_argument("--split", default="test")
    p.add_argument("--kb", default=None); p.add_argument("--out", required=True)
    p.add_argument("--methods", nargs="+", default=["e0", "e1", "e2", "e3", "e3a", "e4"])
    p.add_argument("--llm", default="dummy", help="dummy | hf:<model> | vllm:<model> | openai:<model>")
    p.add_argument("--gpu-mem", type=float, default=0.85, help="vLLM gpu_memory_utilization")
    p.add_argument("--concurrency", type=int, default=64, help="parallel requests for openai: (vLLM server)")
    p.add_argument("--tokenizer", default=None, help="HF tokenizer id for openai: models (prompt budgets)")
    p.add_argument("--adapter", default=None, help="LoRA adapter for hf models (optional)")
    p.add_argument("--llm-cache", default=None, help="shared JSONL response cache (default <out>/llm_cache.jsonl)")
    p.add_argument("--embedder", default="hash", help="hash | sentence-transformers model id")
    p.add_argument("--reranker", default=None, help="cross-encoder id for e2_rerank (e.g. BAAI/bge-reranker-base)")
    p.add_argument("--device", default=None)
    p.add_argument("--max-items", type=int, default=None); p.add_argument("--subset-seed", type=int, default=0)
    p.add_argument("--drop-cross-dups", action="store_true", help="exclude test samples sharing a group with train")
    p.add_argument("--require-source", action="store_true", help="exclude samples without any source file")
    p.add_argument("--code-tokens", type=int, default=3000); p.add_argument("--evidence-tokens", type=int, default=1500)
    p.add_argument("--max-new-tokens", type=int, default=512); p.add_argument("--max-input-tokens", type=int, default=8192)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--vector-k", type=int, default=10); p.add_argument("--bm25-k", type=int, default=10)
    p.add_argument("--final-k", type=int, default=5); p.add_argument("--rrf-k", type=int, default=60)
    p.add_argument("--max-per-source", type=int, default=3); p.add_argument("--n-queries", type=int, default=4)
    p.add_argument("--kb-source-types", nargs="*", default=None)
    p.add_argument("--max-hops", type=int, default=3); p.add_argument("--max-paths", type=int, default=6)
    p.add_argument("--policy", default=None, help="policy.json from tune-policy (E4)")
    p.add_argument("--compare-ref", default="e0"); p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--gpu-price-hour", type=float, default=None, help="USD per GPU hour, for the cost estimate")
    p.add_argument("--strata-key", default="difficulty")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--use-sample-question", action="store_true", help="add the sample's own question (cases) to the prompt")
    p.add_argument("--abstain-on-dispatch", action="store_true", help="E4: benign verdict with unresolved dynamic dispatch -> uncertain")
    p.add_argument("--no-dispatch-caveat", action="store_true", help="do not tell the reader that dynamic dispatch hides targets")
    pyg(p); anal(p); p.set_defaults(func=cmd_run)

    p = common(sub.add_parser("tune-policy", help="tune the E4 abstention policy on validation predictions"))
    p.add_argument("--results", required=True); p.add_argument("--data", required=True)
    p.add_argument("--split", default="val"); p.add_argument("--method", default="e3"); p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_tune_policy)

    p = common(sub.add_parser("rescore", help="recompute metrics offline (optionally derive E4 variants from E3)"))
    p.add_argument("--results", required=True); p.add_argument("--data", default=None)
    p.add_argument("--derive-e4", action="store_true"); p.add_argument("--policy", default=None)
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--gpu-price-hour", type=float, default=None, help="USD per GPU hour, for the cost estimate")
    p.set_defaults(func=cmd_rescore)

    p = common(sub.add_parser("report", help="rewrite report.md")); p.add_argument("--results", required=True)
    p.set_defaults(func=cmd_report)

    def llm_args(p):
        p.add_argument("--llm", default="dummy", help="dummy | hf:<model> | vllm:<model> | openai:<model>")
        p.add_argument("--gpu-mem", type=float, default=0.85); p.add_argument("--concurrency", type=int, default=64)
        p.add_argument("--tokenizer", default=None); p.add_argument("--llm-cache", default=None)
        p.add_argument("--embedder", default="hash"); p.add_argument("--device", default=None)
        p.add_argument("--max-new-tokens", type=int, default=512); p.add_argument("--max-input-tokens", type=int, default=8192)
        p.add_argument("--batch-size", type=int, default=4)

    from .supplychain.pipeline import add_parsers          # Task B: sc-generate, sc-run, sc-tune
    add_parsers(sub, common, llm_args)
    return ap


def _flatten(d, out=None):
    out = {} if out is None else out
    for k, v in (d or {}).items():
        if isinstance(v, dict):
            _flatten(v, out)
        else:
            out[k.replace("-", "_")] = v
    return out


def main(argv=None):
    ap = build_parser()
    argv = list(sys.argv[1:] if argv is None else argv)
    a = ap.parse_args(argv)
    if a.config:
        import yaml
        flat = {}
        for c in a.config:
            with open(c, encoding="utf-8") as f:
                _flatten(yaml.safe_load(f), flat)
        sp = ap._subparsers._group_actions[0].choices[a.cmd]
        known = {act.dest for act in sp._actions}
        sp.set_defaults(**{k: v for k, v in flat.items() if k in known})
        a = ap.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
