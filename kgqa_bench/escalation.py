"""Confidence-gated evidence escalation (RA-style augmentation), evaluated OFFLINE.

GNN-RAG+RA (Mavromatis & Karypis, ACL 2025) improves answers by ADDING evidence from
a second retriever. CGAR currently only REMOVES evidence (budget 5 paths) and loses
~2 pp F1 vs GNN-RAG-all. Its confidence head, however, has the best AURC in the
bench, so the natural fix is a two-tier policy:

    answer with CGAR's selection            if confidence ≥ τ
    escalate to the fallback method's paths  otherwise   (e.g. gnnrag_all = every path)

Because every path CGAR selects is also in gnnrag_all's set, the escalated question
is exactly the fallback's record for that qid — so the whole τ-sweep (cost, F1, Rec,
Hit-ctx, and llm_hit / llm_f1 / unsupported when the records carry reader outputs)
can be computed from two existing records files with no new inference.

The oracle row (escalate exactly the questions where the fallback beats the
primary) upper-bounds what any gate on these two methods could reach.

Usage
    python -m kgqa_bench.escalation --results results_gnn/b06_main --dataset webqsp:test \
        --primary cgar@seed42 --fallback gnnrag_all --plot
    python -m kgqa_bench.escalation --results results_gnn/b08_reader_Qwen2.5-7B-Instruct \
        --dataset webqsp:test --primary cgar --fallback gnnrag_all     # adds llm_* columns
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np

METRICS = ("f1", "coverage", "precision", "hit_ctx", "cost", "n_tokens", "llm_hit", "llm_f1",
           "llm_unsupported_rate", "n_tokens_prompt")


def _index(recs: Sequence[Dict]) -> Dict[str, Dict]:
    return {r["qid"]: r for r in recs}


def _mean(rows: List[Dict], k: str) -> Optional[float]:
    v = [r[k] for r in rows if r.get(k) is not None]
    return float(np.mean(v)) if v else None


def escalation_curve(primary: Sequence[Dict], fallback: Sequence[Dict],
                     taus: Optional[Sequence[float]] = None, quality: str = "f1") -> List[Dict]:
    """One row per τ: metrics of the mixed policy over the common qids.
    τ = -inf → always primary; τ = +inf → always fallback. Also returns the
    oracle row and the quantile-based τ grid used."""
    P, F = _index(primary), _index(fallback)
    qids = sorted(set(P) & set(F))
    if not qids:
        return []
    conf = np.array([float(P[q].get("confidence") if P[q].get("confidence") is not None else 0.0) for q in qids])
    if taus is None:
        qs = np.quantile(conf, np.linspace(0, 1, 21))
        taus = [-np.inf] + sorted(set(np.round(qs, 4).tolist())) + [np.inf]
    rows = []
    for tau in taus:
        use_fb = conf < tau
        mixed = [F[q] if fb else P[q] for q, fb in zip(qids, use_fb)]
        row = {"tau": float(tau) if np.isfinite(tau) else ("-inf" if tau < 0 else "+inf"),
               "escalated_frac": float(use_fb.mean()), "n": len(qids)}
        for k in METRICS:
            m = _mean(mixed, k)
            if m is not None:
                row[k] = m
        rows.append(row)
    # oracle: escalate exactly where the fallback is strictly better on `quality`
    better = [F[q].get(quality, 0) > P[q].get(quality, 0) for q in qids]
    mixed = [F[q] if b else P[q] for q, b in zip(qids, better)]
    row = {"tau": "oracle", "escalated_frac": float(np.mean(better)), "n": len(qids)}
    for k in METRICS:
        m = _mean(mixed, k)
        if m is not None:
            row[k] = m
    rows.append(row)
    return rows


def pick_operating_points(rows: List[Dict], primary_row: Dict, fallback_row: Dict,
                          quality: str = "f1") -> Dict[str, Dict]:
    """Interpretable points on the curve: (a) cheapest τ that matches the
    fallback's quality within 0.5 pp, (b) best quality at ≤ 60 % of fallback cost,
    (c) the knee (max quality gain per unit cost over the primary)."""
    real = [r for r in rows if r["tau"] not in ("oracle",)]
    out: Dict[str, Dict] = {}
    tgt = fallback_row.get(quality, 0) - 0.005
    ok = [r for r in real if r.get(quality, 0) >= tgt]
    if ok:
        out["match_fallback_quality"] = min(ok, key=lambda r: r["cost"])
    cap = 0.6 * fallback_row.get("cost", 0)
    ok = [r for r in real if r.get("cost", 0) <= cap]
    if ok:
        out["best_at_60pct_cost"] = max(ok, key=lambda r: r.get(quality, 0))
    p_q, p_c = primary_row.get(quality, 0), primary_row.get("cost", 0)
    cand = [(r, (r.get(quality, 0) - p_q) / max(r.get("cost", 0) - p_c, 1e-6)) for r in real if r.get("cost", 0) > p_c + 1e-6]
    if cand:
        out["knee"] = max(cand, key=lambda x: x[1])[0]
    return out


def topk_reference(records_by_method: Dict[str, Sequence[Dict]], qids: Sequence[str]) -> List[Dict]:
    """(cost, f1, ...) of every other method on the SAME qids — the Pareto reference
    the escalation curve must beat (otherwise 'just use top-K')."""
    out = []
    qset = set(qids)
    for m, recs in records_by_method.items():
        rows = [r for r in recs if r["qid"] in qset]
        if len(rows) != len(qset):
            continue
        d = {"method": m, "n": len(rows)}
        for k in METRICS:
            v = _mean(rows, k)
            if v is not None:
                d[k] = v
        out.append(d)
    return out


def markdown(rows: List[Dict], points: Dict[str, Dict], refs: List[Dict], primary: str, fallback: str,
             quality: str = "f1") -> str:
    has_llm = any("llm_hit" in r for r in rows)
    cols = ["tau", "escalated_frac", "cost", "f1", "coverage", "hit_ctx"] + (["llm_hit", "llm_f1", "llm_unsupported_rate", "n_tokens_prompt"] if has_llm else [])
    f = lambda r, k: ("—" if r.get(k) is None else (r[k] if isinstance(r[k], str) else f"{r[k]:.4f}" if k != "cost" and k != "n_tokens_prompt" else f"{r[k]:.2f}"))
    L = [f"# Confidence-gated escalation: `{primary}` → `{fallback}` when confidence < τ", "",
         f"Common questions: {rows[0]['n']}. τ grid = confidence quantiles of `{primary}`. "
         "`-inf` = always primary, `+inf` = always fallback, `oracle` = escalate exactly where the fallback is better "
         f"on `{quality}` (upper bound).", "",
         "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        L.append("| " + " | ".join(f(r, k) for k in cols) + " |")
    L += ["", "## Operating points", ""]
    for name, r in points.items():
        L.append(f"- **{name}**: τ={r['tau']}, escalate {r['escalated_frac']*100:.0f}% → cost {r['cost']:.2f}, "
                 f"{quality} {r.get(quality, 0)*100:.2f}" + (f", llm_hit {r['llm_hit']*100:.1f}" if "llm_hit" in r else ""))
    if refs:
        L += ["", "## Same-qid reference points (must be beaten at equal cost)", "",
              "| method | cost | f1 | hit_ctx" + (" | llm_hit" if has_llm else "") + " |", "|---|---|---|---|" + ("---|" if has_llm else "")]
        for d in sorted(refs, key=lambda d: d.get("cost", 0)):
            L.append(f"| {d['method']} | {d.get('cost', 0):.2f} | {d.get('f1', 0)*100:.2f} | {d.get('hit_ctx', 0)*100:.2f}"
                     + (f" | {d['llm_hit']*100:.1f}" if has_llm and "llm_hit" in d else (" | —" if has_llm else "")) + " |")
    L += ["", "Reading: the curve is useful only where it lies ABOVE the top-K reference points at the same cost; "
          "compare against `oracle` to see how much of the attainable gain the confidence head captures."]
    return "\n".join(L)


def plot(rows: List[Dict], refs: List[Dict], out_path: str, quality: str = "f1"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    real = [r for r in rows if r["tau"] != "oracle"]
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot([r["cost"] for r in real], [r[quality] for r in real], "-o", ms=3, label="escalation (τ sweep)")
    orc = next((r for r in rows if r["tau"] == "oracle"), None)
    if orc:
        ax.scatter([orc["cost"]], [orc[quality]], marker="*", s=120, c="k", label="oracle gate")
    for d in refs:
        ax.scatter([d["cost"]], [d[quality]], marker="s", s=28)
        ax.annotate(d["method"], (d["cost"], d[quality]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("cost (#paths)"); ax.set_ylabel(quality); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title("Confidence-gated escalation vs fixed-K references")
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)


def _load(path: str) -> List[Dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def _find_records(results_dir: str, ds_key: str, method: str) -> Optional[str]:
    base = os.path.join(results_dir, f"records_{ds_key.replace(':', '_')}_")
    for cand in (method, f"{method}@seed42"):
        if os.path.exists(base + cand + ".jsonl"):
            return base + cand + ".jsonl"
    import glob
    hits = sorted(glob.glob(base + method + "@seed*.jsonl"))
    return hits[0] if hits else None


def run(results_dir: str, ds_key: str, primary: str, fallback: str, out_dir: Optional[str] = None,
        quality: str = "f1", refs: Optional[Sequence[str]] = None, do_plot: bool = True, verbose: bool = True) -> Dict:
    out_dir = out_dir or os.path.join(results_dir, "escalation")
    os.makedirs(out_dir, exist_ok=True)
    pp, fp = _find_records(results_dir, ds_key, primary), _find_records(results_dir, ds_key, fallback)
    if not pp or not fp:
        raise FileNotFoundError(f"records not found for {primary!r} ({pp}) / {fallback!r} ({fp}) in {results_dir}")
    P, F = _load(pp), _load(fp)
    rows = escalation_curve(P, F, quality=quality)
    prim_row, fb_row = rows[0], [r for r in rows if r["tau"] == "+inf"][0]
    points = pick_operating_points(rows, prim_row, fb_row, quality=quality)
    qids = sorted(set(_index(P)) & set(_index(F)))
    ref_methods = list(refs) if refs else [m for m in ("gnnrag_top3", "gnnrag_top5", "gnnrag_top10", "sbert_top5", "sbert_top10", "graphrag_fi_k8")]
    recs_by_m = {primary: P, fallback: F}
    for m in ref_methods:
        p = _find_records(results_dir, ds_key, m)
        if p:
            recs_by_m[m] = _load(p)
    refs_rows = topk_reference(recs_by_m, qids)
    tag = f"{ds_key.replace(':', '_')}_{primary.split('@')[0]}_to_{fallback}"
    res = {"dataset": ds_key, "primary": primary, "fallback": fallback, "quality": quality,
           "rows": rows, "operating_points": points, "references": refs_rows}
    with open(os.path.join(out_dir, f"escalation_{tag}.json"), "w") as f:
        json.dump(res, f, indent=1)
    md = markdown(rows, points, refs_rows, primary, fallback, quality)
    with open(os.path.join(out_dir, f"escalation_{tag}.md"), "w") as f:
        f.write(md)
    if do_plot:
        plot(rows, refs_rows, os.path.join(out_dir, f"escalation_{tag}.png"), quality)
    if verbose:
        print(md)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description="Offline confidence-gated escalation sweep from two records files")
    ap.add_argument("--results", required=True)
    ap.add_argument("--dataset", required=True, help="dataset key, e.g. webqsp:test")
    ap.add_argument("--primary", default="cgar", help="method whose confidence gates the escalation (cgar or cgar@seed42)")
    ap.add_argument("--fallback", default="gnnrag_all")
    ap.add_argument("--quality", default="f1", help="metric used for the oracle gate and operating points (f1 | hit_ctx | llm_hit)")
    ap.add_argument("--refs", nargs="*", default=None, help="other methods to draw as same-qid reference points")
    ap.add_argument("--out", default=None)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    run(a.results, a.dataset, a.primary, a.fallback, out_dir=a.out, quality=a.quality, refs=a.refs,
        do_plot=a.plot, verbose=not a.quiet)


if __name__ == "__main__":
    main()
