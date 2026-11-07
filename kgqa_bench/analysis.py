"""Offline analyses on bench records (no training, no LLM) — items A2–A4 of the
Nov-2026 improvement list.

A2  adaptivity breakdown   — is a low cost the controller's decision or pool exhaustion?
                             per question: n_paths available (recomputed from the dataset
                             with the same extraction CGAR uses), budget, cost →
                             controller_stopped_early = cost < min(n_paths, budget)
                             + F1 / Hit-ctx by #candidates bin for every method
A3  pairwise CIs           — paired bootstrap + McNemar for arbitrary method pairs
                             (e.g. cgar vs sbert_top5, cgar vs gnnrag_top5) on common qids
A4  risk–coverage (hit_ctx) — selective-answering curves ranked by each method's confidence,
                             sel_acc at several coverages, AURC; figure for the report

Usage
    python -m kgqa_bench.analysis --results results_gnn/b06_main --dataset webqsp:test \
        --eval gnnrag:ckpt/gnn_features_webqsp_test.pkl:webqsp:test --budget 5 \
        --primary cgar --pairs cgar:sbert_top5 cgar:gnnrag_top5 cgar:gnnrag_all --plot
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np

from .metrics import compare_records

BINS = [(1, 1, "1"), (2, 3, "2-3"), (4, 5, "4-5"), (6, 10, "6-10"), (11, 10 ** 9, ">10")]


def _bin(n: int) -> str:
    for lo, hi, lab in BINS:
        if lo <= n <= hi:
            return lab
    return "0"


def _load(path: str) -> List[Dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def find_records(results_dir: str, ds_key: str, method: str) -> Optional[str]:
    base = os.path.join(results_dir, f"records_{ds_key.replace(':', '_')}_")
    for cand in (method, f"{method}@seed42"):
        if os.path.exists(base + cand + ".jsonl"):
            return base + cand + ".jsonl"
    hits = sorted(glob.glob(base + method + "@seed*.jsonl"))
    return hits[0] if hits else None


def discover(results_dir: str, ds_key: str) -> List[str]:
    pre = f"records_{ds_key.replace(':', '_')}_"
    out = []
    for p in sorted(glob.glob(os.path.join(results_dir, pre + "*.jsonl"))):
        m = os.path.basename(p)[len(pre):-6]
        base = m.split("@seed")[0]
        if base not in out:
            out.append(base)
    return out


# ─────────────────────────────────────────────────────────────────────
# A2
# ─────────────────────────────────────────────────────────────────────

def paths_available(dataset, max_candidates: int = 20, max_paths_per_pair: int = 3) -> Dict[str, int]:
    """qid → number of candidate paths the controller could choose from (same
    extraction as every method: all shortest paths topic→top-20 candidates)."""
    from grl_rag import extract_candidate_paths
    from .adapters import to_gnn_features
    gnn = to_gnn_features(dataset)
    return {q.qid: len(extract_candidate_paths(q.qid, gnn, max_candidates=max_candidates,
                                                max_paths_per_pair=max_paths_per_pair)) for q in dataset}


def adaptivity_breakdown(records: Sequence[Dict], budget: int, n_paths: Optional[Dict[str, int]] = None) -> Dict:
    """Split questions into: pool_limited (n_paths ≤ budget → cost can't reach budget),
    controller_stopped_early (n_paths > budget and cost < budget), full_budget."""
    out = {"n": len(records), "budget": budget}
    costs = [int(r["cost"]) for r in records]
    out["cost_hist"] = dict(sorted(Counter(costs).items()))
    out["chosen_k_hist"] = dict(sorted(Counter(r.get("chosen_k") for r in records).items(), key=lambda x: str(x[0])))
    if n_paths:
        cats = Counter()
        early_costs = []
        for r in records:
            n = n_paths.get(r["qid"])
            if n is None:
                cats["unknown"] += 1; continue
            c = int(r["cost"])
            if n <= budget:
                cats["pool_limited"] += 1
            elif c < budget:
                cats["controller_stopped_early"] += 1; early_costs.append(c)
            else:
                cats["full_budget"] += 1
        tot = max(sum(cats.values()), 1)
        out["categories"] = {k: {"n": v, "frac": v / tot} for k, v in cats.items()}
        out["mean_cost_when_stopped_early"] = float(np.mean(early_costs)) if early_costs else None
        out["frac_questions_with_more_paths_than_budget"] = float(np.mean([n_paths.get(r["qid"], 0) > budget for r in records]))
    return out


def by_candidates(records_by_method: Dict[str, Sequence[Dict]]) -> Dict[str, Dict[str, Dict]]:
    """method → bin(#candidates) → {n, f1, hit_ctx, cost}."""
    out: Dict[str, Dict[str, Dict]] = {}
    for m, recs in records_by_method.items():
        g = defaultdict(list)
        for r in recs:
            g[_bin(int(r.get("n_candidates") or 0))].append(r)
        out[m] = {b: {"n": len(rs), "f1": float(np.mean([r["f1"] for r in rs])),
                      "hit_ctx": float(np.mean([r["hit_ctx"] for r in rs])),
                      "cost": float(np.mean([r["cost"] for r in rs]))}
                  for b, rs in sorted(g.items(), key=lambda x: [l for _a, _b, l in BINS].index(x[0]) if x[0] in [l for _a, _b, l in BINS] else -1)}
    return out


# ─────────────────────────────────────────────────────────────────────
# A3
# ─────────────────────────────────────────────────────────────────────

def pairwise(records_by_method: Dict[str, Sequence[Dict]], pairs: Sequence[str],
             metrics: Sequence[str] = ("f1", "hit_ctx"), n_bootstrap: int = 10000) -> Dict[str, Dict]:
    out = {}
    for pr in pairs:
        a, b = pr.split(":")
        if a not in records_by_method or b not in records_by_method:
            out[pr] = {"error": "missing records"}; continue
        out[pr] = {m: compare_records(records_by_method[a], records_by_method[b], m, n_bootstrap=n_bootstrap) for m in metrics}
    return out


# ─────────────────────────────────────────────────────────────────────
# A4
# ─────────────────────────────────────────────────────────────────────

def selective_table(records_by_method: Dict[str, Sequence[Dict]],
                    coverages: Sequence[float] = (0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0)) -> Dict[str, Dict]:
    out = {}
    for m, recs in records_by_method.items():
        rs = [r for r in recs if r.get("confidence") is not None]
        if len(rs) < len(recs) or not rs:
            continue
        pairs = sorted(((float(r["confidence"]), int(r["hit_ctx"]), float(r["f1"])) for r in rs), key=lambda x: -x[0])
        n = len(pairs)
        row = {"n": n}
        cum_wrong, auc, curve = 0, 0.0, []
        for i, (_c, ok, _f) in enumerate(pairs, 1):
            cum_wrong += 1 - ok
            auc += (cum_wrong / i) / n
            curve.append((i / n, 1 - cum_wrong / i))
        row["aurc"] = auc
        for c in coverages:
            k = max(1, int(round(c * n)))
            row[f"sel_hit@{int(c*100)}"] = float(np.mean([ok for _c, ok, _f in pairs[:k]]))
            row[f"sel_f1@{int(c*100)}"] = float(np.mean([f for _c, _ok, f in pairs[:k]]))
        row["curve"] = curve[:: max(1, n // 200)]
        out[m] = row
    return out


def plot_selective(table: Dict[str, Dict], out_path: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for m, row in table.items():
        cur = row.get("curve") or []
        ax.plot([c for c, _a in cur], [a for _c, a in cur], label=f"{m} (AURC {row['aurc']:.3f})")
    ax.set_xlabel("coverage (fraction answered, ranked by confidence)"); ax.set_ylabel("Hit-ctx among answered")
    ax.set_title("Selective answering: accuracy vs coverage"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# report
# ─────────────────────────────────────────────────────────────────────

def markdown(res: Dict) -> str:
    L = [f"# Offline analysis — `{res['dataset']}` ({res['results']})", ""]
    ad = res.get("adaptivity")
    if ad:
        L += [f"## A2. Adaptivity of `{res['primary']}` (budget {ad['budget']})", "",
              f"cost histogram: {ad['cost_hist']}", f"chosen_k histogram: {ad['chosen_k_hist']}", ""]
        if "categories" in ad:
            L += ["| category | n | frac |", "|---|---|---|"]
            for k, v in ad["categories"].items():
                L.append(f"| {k} | {v['n']} | {v['frac']:.3f} |")
            L += ["", f"questions with more paths than budget: {ad['frac_questions_with_more_paths_than_budget']:.3f}; "
                  f"mean cost when the controller stopped early: {ad['mean_cost_when_stopped_early']}", "",
                  "Reading: only `controller_stopped_early` is adaptivity attributable to the controller; "
                  "`pool_limited` questions would have the same cost under any method.", ""]
    bc = res.get("by_candidates")
    if bc:
        bins = [l for _a, _b, l in BINS]
        L += ["## A2b. F1 / Hit-ctx by #candidates in the pool", "",
              "| method | " + " | ".join(f"{b} (n)" for b in bins) + " |", "|---|" + "---|" * len(bins)]
        for m, d in bc.items():
            L.append(f"| {m} | " + " | ".join(f"{d[b]['f1']*100:.1f} / {d[b]['hit_ctx']*100:.1f} ({d[b]['n']})" if b in d else "—" for b in bins) + " |")
        L.append("")
    pw = res.get("pairwise")
    if pw:
        L += ["## A3. Paired comparisons (A − B, 95% bootstrap CI, McNemar on hit_ctx)", "",
              "| pair | metric | n | Δ (pp) | 95% CI | p | McNemar p | sig |", "|---|---|---|---|---|---|---|---|"]
        for pr, d in pw.items():
            if "error" in d:
                L.append(f"| {pr} | — | — | {d['error']} | | | | |"); continue
            for met, c in d.items():
                if not c.get("n"):
                    continue
                b = c["bootstrap"]; mc = c.get("mcnemar", {}).get("p_value")
                L.append(f"| {pr} | {met} | {c['n']} | {b['delta']*100:+.2f} | [{b['ci_low']*100:+.2f}, {b['ci_high']*100:+.2f}] | "
                         f"{b['p_value']:.4f} | {'—' if mc is None else f'{mc:.4f}'} | {'yes' if c['significant_0.05'] else 'no'} |")
        L.append("")
    st = res.get("selective")
    if st:
        covs = [k for k in next(iter(st.values())) if k.startswith("sel_hit@")]
        L += ["## A4. Selective answering ranked by confidence (Hit-ctx among answered)", "",
              "| method | AURC | " + " | ".join(c.replace("sel_hit@", "hit@") + "%" for c in covs) + " |", "|---|---|" + "---|" * len(covs)]
        for m, row in st.items():
            L.append(f"| {m} | {row['aurc']:.3f} | " + " | ".join(f"{row[c]*100:.1f}" for c in covs) + " |")
        L.append("")
    return "\n".join(L)


def run(results_dir: str, ds_key: str, eval_spec: Optional[str] = None, budget: int = 5, primary: str = "cgar",
        pairs: Optional[Sequence[str]] = None, methods: Optional[Sequence[str]] = None, out_dir: Optional[str] = None,
        n_bootstrap: int = 10000, do_plot: bool = True, cache_dir: Optional[str] = None, verbose: bool = True) -> Dict:
    out_dir = out_dir or os.path.join(results_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)
    ms = list(methods) if methods else discover(results_dir, ds_key)
    recs = {}
    for m in ms:
        p = find_records(results_dir, ds_key, m)
        if p:
            recs[m] = _load(p)
    res: Dict = {"results": results_dir, "dataset": ds_key, "primary": primary, "methods": list(recs)}
    n_paths = None
    if eval_spec:
        from .data import load_spec
        ds = load_spec(eval_spec, cache_dir=cache_dir or os.path.join(results_dir, "data_cache"))
        n_paths = paths_available(ds)
        res["n_paths_stats"] = {"mean": float(np.mean(list(n_paths.values()))),
                                "frac_le_budget": float(np.mean([v <= budget for v in n_paths.values()]))}
    if primary in recs:
        res["adaptivity"] = adaptivity_breakdown(recs[primary], budget, n_paths)
    res["by_candidates"] = by_candidates(recs)
    pairs = list(pairs) if pairs else [f"{primary}:{m}" for m in ("sbert_top5", "gnnrag_top5", "gnnrag_all") if m in recs and primary in recs]
    res["pairwise"] = pairwise(recs, pairs, n_bootstrap=n_bootstrap)
    res["selective"] = selective_table(recs)
    tag = ds_key.replace(":", "_")
    if do_plot and res["selective"]:
        plot_selective(res["selective"], os.path.join(out_dir, f"selective_{tag}.png"))
    slim = dict(res); slim["selective"] = {m: {k: v for k, v in r.items() if k != "curve"} for m, r in res["selective"].items()}
    with open(os.path.join(out_dir, f"analysis_{tag}.json"), "w") as f:
        json.dump(slim, f, indent=1)
    md = markdown(slim)
    with open(os.path.join(out_dir, f"analysis_{tag}.md"), "w") as f:
        f.write(md)
    if verbose:
        print(md)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description="Offline adaptivity / pairwise / selective analyses on bench records")
    ap.add_argument("--results", required=True)
    ap.add_argument("--dataset", required=True, help="dataset key, e.g. webqsp:test")
    ap.add_argument("--eval", default=None, help="dataset spec (enables the exact n_paths-based adaptivity split)")
    ap.add_argument("--budget", type=int, default=5)
    ap.add_argument("--primary", default="cgar")
    ap.add_argument("--pairs", nargs="*", default=None, help="A:B pairs, e.g. cgar:sbert_top5")
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-bootstrap", type=int, default=10000)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    run(a.results, a.dataset, eval_spec=a.eval, budget=a.budget, primary=a.primary, pairs=a.pairs, methods=a.methods,
        out_dir=a.out, n_bootstrap=a.n_bootstrap, do_plot=a.plot, cache_dir=a.cache_dir, verbose=not a.quiet)


if __name__ == "__main__":
    main()
