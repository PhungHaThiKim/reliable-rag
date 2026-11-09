"""CLI steps of Task B: sc-generate (org -> corpus + questions), sc-run (all methods, one pass), sc-tune (E4 on dev)."""
from __future__ import annotations

import dataclasses
import json
import os
import time
from collections import Counter
from typing import Dict, List

from ..knowledge.documents import load_kb, save_kb
from ..schema import read_jsonl, write_jsonl
from .corpus import render_corpus
from .evaluate import ANSWERABLE, GROUPS, compare, cost_estimate, policy_sweep, summarize, tune_policy
from .kg import KGIndex, build_kg, kg_stats
from .org import make_org
from .qa import QA_METHODS, QAContext, predict_qa, reapply_policy
from .questions import make_questions


def _p(*a):
    print(*a, flush=True)


def generate(seed: int, out: str, n_per_group: int = 100) -> Dict:
    os.makedirs(out, exist_ok=True)
    org = make_org(seed)
    docs = render_corpus(org)
    qs = make_questions(org, n_per_group)
    G = build_kg(docs)
    save_kb(os.path.join(out, "corpus.jsonl"), docs)
    write_jsonl(os.path.join(out, "questions.jsonl"), qs)
    json.dump(dataclasses.asdict(org), open(os.path.join(out, "org.json"), "w"), indent=1)
    stats = {"seed": seed, "today": org.today, "n_docs": len(docs), "doc_types": dict(Counter(d.source_type for d in docs)),
             "n_questions": len(qs), "by_group": dict(Counter(q["group"] for q in qs)),
             "by_type": dict(Counter(f"{q['group']}/{q['qtype']}" for q in qs)),
             "by_trap": dict(Counter(q["trap"] for q in qs)),
             "answerable": dict(Counter(f"{q['group']}/{q['answerable']}" for q in qs)), "kg": kg_stats(G),
             "advisories": {a.adv_id: {"package": a.package, "spec": a.spec, "kind": a.kind,
                                       "affected": org.affected_services(a.adv_id)} for a in org.advisories}}
    json.dump(stats, open(os.path.join(out, "stats.json"), "w"), indent=1)
    return stats


def _known(G):
    kinds = {"svc": set(), "pkg": set(), "team": set()}
    known = set()
    for _n, a in G.nodes(data=True):
        k, name = a.get("kind"), a.get("name")
        if not name:
            continue
        known.add(name.rsplit("-v", 1)[0] if k == "pol" and name.count("-") > 2 else name)
        if k in kinds:
            kinds[k].add(name.lower())
    return sorted(known), kinds


def run(a) -> Dict:
    from ..generation.llm_client import CachedLLM, llm_from_spec
    from ..retrieval import Embedder
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    docs = load_kb(os.path.join(a.data, "corpus.jsonl"))
    qs = list(read_jsonl(os.path.join(a.data, "questions.jsonl")))
    if a.groups:
        qs = [q for q in qs if q["group"] in a.groups]
    if a.max_items:
        per = {}
        keep = []
        for q in qs:
            per[q["group"]] = per.get(q["group"], 0) + 1
            if per[q["group"]] <= a.max_items:
                keep.append(q)
        qs = keep
    today = json.load(open(os.path.join(a.data, "stats.json")))["today"]
    G = build_kg(docs)
    known, kinds = _known(G)
    kw = dict(max_new_tokens=a.max_new_tokens, max_input_tokens=a.max_input_tokens, batch_size=a.batch_size,
              gpu_memory_utilization=a.gpu_mem, concurrency=a.concurrency, tokenizer=a.tokenizer)
    llm = CachedLLM(llm_from_spec(a.llm, **kw), a.llm_cache or os.path.join(a.out, "llm_cache.jsonl"))
    emb = Embedder(a.embedder, device=None if a.embedder == "hash" else a.device)
    ctx = QAContext(llm, docs, KGIndex(G), emb, today, known, kinds, k_text=a.k_text, k_facts=a.k_facts,
                    max_hops=a.max_hops, max_items=a.max_paths, evidence_tokens=a.evidence_tokens, cache_dir=a.out,
                    seed=a.seed)
    policy = json.load(open(a.policy)) if a.policy else {"tau": 0.5, "filter_elements": True, "untuned": True}
    _p(f"[sc] {len(qs)} questions {dict(Counter(q['group'] for q in qs))} | methods {a.methods} | llm {a.llm} | "
       f"{len(docs)} docs | KG {G.number_of_nodes()} nodes {G.number_of_edges()} edges | policy {policy}")
    R0 = {"config": {k: v for k, v in vars(a).items() if k != "func"}, "llm_config": llm.config(),
          "n_questions": len(qs), "question_ids": [q["qid"] for q in qs], "kg": kg_stats(G)}
    res = predict_qa(a.methods, qs, ctx, policy, progress=_p)
    by_id = {q["qid"]: q for q in qs}
    summaries = {}
    for m, rows in res.items():
        write_jsonl(os.path.join(a.out, f"predictions_{m}.jsonl"), rows)
        summaries[m] = summarize(rows, by_id)
        _log_line(m, summaries[m])
    return _finish(a.out, R0, res, by_id, policy, a.n_bootstrap, getattr(a, "gpu_price_hour", None), t0)


def _log_line(m: str, s: Dict):
    _p(f"[{m}] exact " + " ".join(f"{g} {s[g].get('exact_accuracy', float('nan')):.3f}" for g in ANSWERABLE)
       + f" | G4 correct-abstention {s['G4'].get('correct_abstention_rate', float('nan')):.3f}"
       + f" | answer coverage {s['answerable'].get('answer_coverage', float('nan')):.3f}"
       + f" | unsupported/original {s['grounding_all'].get('unsupported_over_original_claims', float('nan')):.3f}"
       + f" | fabricated {s['grounding_all'].get('fabricated_answer_rate', float('nan')):.3f}")


def _finish(out: str, R0: Dict, res: Dict, by_id: Dict, policy: Dict, n_bootstrap: int, price, t0) -> Dict:
    summaries = {m: summarize(rows, by_id) for m, rows in res.items()}
    R = {**R0, "policy": policy, "summaries": summaries, "comparisons": compare(res, by_id, n_bootstrap),
         "cost": cost_estimate(res, price), "elapsed_min": (time.time() - t0) / 60}
    if "e3" in res:
        R["policy_sweep"] = policy_sweep(res["e3"], by_id, filter_elements=policy.get("filter_elements", True))
    json.dump(R, open(os.path.join(out, "results.json"), "w"), indent=1, default=float)
    write_report(out, R)
    _p(f"[done] {out}/results.json ({R['elapsed_min']:.1f} min)")
    return R


def rescore(a) -> Dict:
    """Recompute every metric from saved predictions (no LLM call)."""
    t0 = time.time()
    R0 = json.load(open(os.path.join(a.results, "results.json")))
    qs = {q["qid"]: q for q in read_jsonl(os.path.join(a.data, "questions.jsonl"))}
    by_id = {i: qs[i] for i in R0["question_ids"] if i in qs}
    res = {m: list(read_jsonl(os.path.join(a.results, f"predictions_{m}.jsonl"))) for m in R0["summaries"]}
    for m, rows in res.items():
        for r in rows:
            r.setdefault("method", m)
    keep = {k: v for k, v in R0.items() if k in ("config", "llm_config", "n_questions", "question_ids", "kg")}
    policy = R0.get("policy", {})
    if getattr(a, "policy", None) and "e3" in res:      # re-derive E4 from E3's answers with another (re-tuned) policy
        policy = json.load(open(a.policy))
        res["e4"] = reapply_policy(res["e3"], by_id, policy["tau"], policy.get("filter_elements", True))
        write_jsonl(os.path.join(a.results, "predictions_e4.jsonl"), res["e4"])
    R = _finish(a.results, keep, res, by_id, policy, a.n_bootstrap, a.gpu_price_hour, t0)
    for m, s in R["summaries"].items():
        _log_line(m, s)
    return R


def tune(a) -> Dict:
    qs = {q["qid"]: q for q in read_jsonl(os.path.join(a.data, "questions.jsonl"))}
    rows = list(read_jsonl(os.path.join(a.results, "predictions_e3.jsonl")))
    pol = tune_policy(rows, qs)
    pol.update({"tuned_on": f"{a.results}:e3", "n": len(rows)})
    json.dump(pol, open(a.out, "w"), indent=2)
    _p("[sc-policy]", pol)
    return pol


def write_report(out: str, R: Dict) -> str:
    """Per group, never pooled: answer quality (G1-G3), correct abstention (G4), grounding, evidence, cost."""
    def f(x):
        try:
            return f"{float(x):.3f}"
        except (TypeError, ValueError):
            return "-"
    lines = [f"# Task B supply-chain QA ({R['n_questions']} questions)", "", f"LLM: {R['llm_config'].get('name')}",
             f"E4 policy: {R['policy']}", ""]
    q_cols = ["exact_accuracy", "set_f1", "answer_coverage", "answered_accuracy", "aurc", "unsupported_over_original_claims",
              "fabricated_answer_rate", "ev_recall", "tokens_in_mean"]
    for g in ANSWERABLE + ("answerable",):
        lines += [f"## {g} (answer quality)", "", "| method | n | " + " | ".join(q_cols) + " |", "|" + "---|" * (len(q_cols) + 2)]
        for m, s in R["summaries"].items():
            b = s[g]
            lines.append(f"| {m} | {b.get('n')} | " + " | ".join(f(b.get(c)) for c in q_cols) + " |")
        lines.append("")
    g_cols = ["correct_abstention_rate", "unsupported_over_original_claims", "fabricated_answer_rate", "tokens_in_mean"]
    lines += ["## G4 (insufficient evidence)", "", "| method | n | " + " | ".join(g_cols) + " |", "|" + "---|" * (len(g_cols) + 2)]
    for m, s in R["summaries"].items():
        lines.append(f"| {m} | {s['G4'].get('n')} | " + " | ".join(f(s["G4"].get(c)) for c in g_cols) + " |")
    a_cols = ["unsupported_claim_rate", "unsupported_over_original_claims", "questions_with_unsupported", "claims_retained",
              "fabricated_answer_rate", "ev_recall", "ev_recall_visible", "tokens_in_mean", "tokens_out_mean", "latency_p50_ms"]
    lines += ["", "## Grounding and cost over all questions (same denominators for every method)", "",
              "| method | " + " | ".join(a_cols) + " |", "|" + "---|" * (len(a_cols) + 1)]
    for m, s in R["summaries"].items():
        lines.append(f"| {m} | " + " | ".join(f(s["grounding_all"].get(c)) for c in a_cols) + " |")
    if R.get("cost"):
        c = R["cost"]
        lines += ["", f"Measured GPU seconds {f(c['gpu_seconds_total'])}, {c['unique_prompts']} unique prompts, "
                      f"{f(c['seconds_per_1k_tokens'])} s per 1k tokens; price/h {c.get('price_per_hour')}", "",
                  "| method | tokens/item | GPU s/item | USD/1000 | reuses E3 |", "|---|---|---|---|---|"]
        for m, x in c["methods"].items():
            lines.append(f"| {m} | {f(x['tokens_per_item'])} | {f(x['gpu_seconds_per_item'])} | {f(x['usd_per_1000_items'])} | "
                         f"{x['reuses_e3_generation']} |")
    if R.get("policy_sweep"):
        lines += ["", "## E4 operating points (tau sweep on E3 answers, descriptive)", "",
                  "| tau | answer coverage | answered accuracy | exact accuracy | correct abstention | unsupported/original |",
                  "|---|---|---|---|---|---|"]
        for x in R["policy_sweep"]:
            lines.append(f"| {x['tau']} | {f(x['answer_coverage'])} | {f(x['answered_accuracy'])} | {f(x['exact_accuracy'])} | "
                         f"{f(x['correct_abstention_rate'])} | {f(x['unsupported_over_original_claims'])} |")
    lines += ["", "## Paired comparisons per group (cluster bootstrap over entities, McNemar)", "",
              "| pair | group | metric | n | delta | 95% CI | McNemar p |", "|---|---|---|---|---|---|---|"]
    for k, c in R["comparisons"].items():
        lines.append(f"| {k.split('/')[0]} | {c['group']} | {c.get('metric', '')} | {c['n']} | {c['delta']:+.3f} | "
                     f"[{c['ci_low']:+.3f}, {c['ci_high']:+.3f}] | {c['mcnemar_p']:.4f} |")
    txt = "\n".join(lines) + "\n"
    open(os.path.join(out, "report.md"), "w", encoding="utf-8").write(txt)
    return txt


def add_parsers(sub, common, llm_args):
    p = common(sub.add_parser("sc-generate", help="Task B: synthetic org -> corpus + KG stats + questions"))
    p.add_argument("--out", required=True); p.add_argument("--n-per-group", type=int, default=100)
    p.set_defaults(func=lambda a: _p(json.dumps({k: v for k, v in generate(a.seed, a.out, a.n_per_group).items()
                                                 if k in ("n_docs", "n_questions", "by_group", "kg")}, indent=1)))
    p = common(sub.add_parser("sc-run", help="Task B: run every method on the same questions and evaluate"))
    p.add_argument("--data", required=True); p.add_argument("--out", required=True)
    p.add_argument("--methods", nargs="+", default=list(QA_METHODS))
    p.add_argument("--groups", nargs="*", default=None); p.add_argument("--max-items", type=int, default=None,
                                                                          help="max questions per group")
    p.add_argument("--k-text", type=int, default=8); p.add_argument("--k-facts", type=int, default=16)
    p.add_argument("--max-hops", type=int, default=4); p.add_argument("--max-paths", type=int, default=20)
    p.add_argument("--evidence-tokens", type=int, default=2000)
    p.add_argument("--policy", default=None); p.add_argument("--n-bootstrap", type=int, default=2000)
    p.add_argument("--gpu-price-hour", type=float, default=None, help="USD per GPU hour, for the cost estimate")
    llm_args(p)
    p.set_defaults(func=run)
    p = common(sub.add_parser("sc-rescore", help="Task B: recompute every metric from saved predictions"))
    p.add_argument("--results", required=True); p.add_argument("--data", required=True)
    p.add_argument("--n-bootstrap", type=int, default=2000)
    p.add_argument("--gpu-price-hour", type=float, default=None)
    p.add_argument("--policy", default=None, help="re-derive E4 from the saved E3 answers with this policy.json")
    p.set_defaults(func=rescore)
    p = common(sub.add_parser("sc-tune", help="Task B: tune the E4 policy on the dev org (from E3 predictions)"))
    p.add_argument("--results", required=True); p.add_argument("--data", required=True); p.add_argument("--out", required=True)
    p.set_defaults(func=tune)
