"""Tables and figures from bench results.

results.json layout (written by cli.run):
    {"summaries": {dataset_key: {method: summary_dict}},
     "stratified": {dataset_key: {method: {"hops": {...}, "n_gold": {...}}}},
     "comparisons": {dataset_key: {"<A>_vs_<B>": {...}}},
     "seeds": {dataset_key: {method: {metric: {mean, std, n}}}},
     "dataset_stats": {dataset_key: {...}}, "config": {...}}
"""
from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Optional

MAIN_COLS = [("f1", "F1", 100, 1), ("cov", "Rec", 100, 1), ("prec", "Prec", 100, 1),
             ("hit_ctx", "Hit-ctx", 100, 1), ("cost", "#paths", 1, 2), ("n_tokens", "tok", 1, 0),
             ("latency_ms", "ms/q", 1, 1), ("ece", "ECE", 1, 3), ("aurc", "AURC", 1, 3),
             ("sel_acc@80", "SelAcc80", 100, 1), ("las_hit1", "LAS-H@1", 100, 1),
             ("llm_hit", "LLM-Hit", 100, 1), ("llm_f1", "LLM-F1", 100, 1)]


def _fmt(v, scale, nd):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v * scale:.{nd}f}"


def _ci(s, key, scale=100):
    lo, hi = s.get(f"{key}_ci_low"), s.get(f"{key}_ci_high")
    if lo is None or hi is None:
        return ""
    return f" [{lo*scale:.1f},{hi*scale:.1f}]"


def markdown_main_table(summaries: Dict[str, Dict[str, Dict]], seeds: Optional[Dict] = None) -> str:
    lines = []
    for ds, by_m in summaries.items():
        cols = [c for c in MAIN_COLS if any(m.get(c[0]) is not None and not (isinstance(m.get(c[0]), float) and math.isnan(m[c[0]])) for m in by_m.values())]
        lines.append(f"\n### {ds}\n")
        lines.append("| Method | n | " + " | ".join(c[1] for c in cols) + " |")
        lines.append("|---|---|" + "---|" * len(cols))
        for m, s in by_m.items():
            cells = []
            for k, _lab, sc, nd in cols:
                cell = _fmt(s.get(k), sc, nd)
                if seeds and ds in seeds and m in seeds[ds] and k in seeds[ds][m]:
                    sd = seeds[ds][m][k]
                    cell = f"{sd['mean']*sc:.{nd}f} ± {sd['std']*sc:.{nd}f}"
                elif k in ("f1", "hit_ctx", "llm_hit"):
                    cell += _ci(s, k, sc)
                cells.append(cell)
            lines.append(f"| {m} | {s.get('n', 0)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def markdown_stratified(strat: Dict[str, Dict[str, Dict]], key: str = "hops") -> str:
    lines = []
    for ds, by_m in strat.items():
        lines.append(f"\n### {ds} — by {key}\n")
        groups = sorted({g for m in by_m.values() for g in m.get(key, {})},
                        key=lambda x: (x == "None", x))
        lines.append("| Method | " + " | ".join(f"{key}={g} (F1 / #paths / n)" for g in groups) + " |")
        lines.append("|---|" + "---|" * len(groups))
        for m, d in by_m.items():
            row = []
            for g in groups:
                s = d.get(key, {}).get(g)
                row.append(f"{s['f1']*100:.1f} / {s['cost']:.1f} / {s['n']}" if s else "—")
            lines.append(f"| {m} | " + " | ".join(row) + " |")
    return "\n".join(lines)


def markdown_comparisons(comps: Dict[str, Dict[str, Dict]]) -> str:
    lines = ["\n### Paired significance (A − B)\n",
             "| Dataset | Pair | metric | n | Δ | 95% CI | p (bootstrap) | p (McNemar) | sig@0.05 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for ds, pairs in comps.items():
        for name, c in pairs.items():
            if not c or c.get("n", 0) == 0:
                continue
            b = c["bootstrap"]
            mc = c.get("mcnemar", {}).get("p_value")
            lines.append(f"| {ds} | {name} | {c['metric']} | {c['n']} | {b['delta']*100:+.2f} | "
                         f"[{b['ci_low']*100:+.2f}, {b['ci_high']*100:+.2f}] | {b['p_value']:.4f} | "
                         f"{'—' if mc is None else f'{mc:.4f}'} | {'yes' if c['significant_0.05'] else 'no'} |")
    return "\n".join(lines)


def latex_main_table(summaries: Dict[str, Dict[str, Dict]], cols=None, caption: str = "",
                     label: str = "tab:main") -> str:
    cols = cols or [c for c in MAIN_COLS if c[0] in ("f1", "cov", "prec", "cost", "ece", "aurc", "llm_hit")]
    out = ["\\begin{table}[t]\\centering\\small", "\\begin{tabular}{ll" + "r" * len(cols) + "}", "\\toprule",
           "Dataset & Method & " + " & ".join(c[1] for c in cols) + " \\\\", "\\midrule"]
    for ds, by_m in summaries.items():
        for m, s in by_m.items():
            m_tex = m.replace("_", "\\_")
            out.append(f"{ds} & {m_tex} & " +
                       " & ".join(_fmt(s.get(k), sc, nd) for k, _l, sc, nd in cols) + " \\\\")
        out.append("\\midrule")
    out[-1] = "\\bottomrule"
    out += ["\\end{tabular}", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\end{table}"]
    return "\n".join(out)


def pareto_plot(summaries: Dict[str, Dict[str, Dict]], out_path: str, quality: str = "f1",
                sweeps: Optional[Dict[str, List[Dict]]] = None):
    """Cost (#paths) vs quality for every method; optional per-method sweep
    curves (e.g. CGAR β-sweep, PRISM ε-sweep, GNN-RAG K-sweep)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    n = len(summaries)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.2), squeeze=False)
    for ax, (ds, by_m) in zip(axes[0], summaries.items()):
        for m, s in by_m.items():
            if s.get("cost") is None or s.get(quality) is None:
                continue
            ax.scatter(s["cost"], s[quality] * 100, s=60, label=m, zorder=3)
            ax.annotate(m, (s["cost"], s[quality] * 100), xytext=(4, 4), textcoords="offset points", fontsize=8)
        for name, pts in (sweeps or {}).get(ds, {}).items() if isinstance((sweeps or {}).get(ds), dict) else []:
            ax.plot([p["cost"] for p in pts], [p[quality] * 100 for p in pts], "--", alpha=0.7, label=f"{name} sweep")
        ax.set_xlabel("mean #paths (cost)"); ax.set_ylabel(f"{quality} (%)"); ax.set_title(ds); ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="lower right")
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def reliability_plot(records_by_method: Dict[str, List[Dict]], out_path: str, n_bins: int = 10):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return None
    from prism.metrics import reliability_diagram, expected_calibration_error
    ms = [(m, r) for m, r in records_by_method.items() if r and all(x.get("confidence") is not None for x in r)]
    if not ms:
        return None
    fig, axes = plt.subplots(1, len(ms), figsize=(4 * len(ms), 4), squeeze=False)
    for ax, (m, recs) in zip(axes[0], ms):
        confs = [x["confidence"] for x in recs]; corr = [x["hit_ctx"] for x in recs]
        centers, accs, counts = reliability_diagram(confs, corr, n_bins)
        valid = counts > 0
        ax.bar(centers[valid], accs[valid], width=1.0 / n_bins * 0.9, alpha=0.6, edgecolor="black")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax.set_title(f"{m}\nECE={expected_calibration_error(confs, corr):.3f}", fontsize=9)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xlabel("confidence"); ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("empirical hit-ctx rate")
    plt.tight_layout(); plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return out_path


def _pool_note(R: Dict) -> str:
    """Say which candidate scorer produced each dataset's pool, so top-K rows
    are labelled honestly (heuristic-topK is NOT GNN-RAG top-K)."""
    lines = ["\nCandidate pool per dataset (all path-dict baselines share `baseline_candidates` = "
             f"{R.get('config', {}).get('baseline_candidates', '?')} candidates; CGAR/PRISM see the same pool):"]
    for k, s in R["dataset_stats"].items():
        src = str(s.get("source", ""))
        if src.startswith("gnnrag_pkl"):
            lines.append(f"- `{k}`: **GNN-RAG (ReaRev) scores** → `gnnrag_topK` rows are genuine GNN-RAG top-K.")
        else:
            lines.append(f"- `{k}`: **heuristic path-frequency scores** (no GNN) → read `gnnrag_topK` as "
                         "*heuristic-topK*; the raw score is not a probability, so ECE/Brier of these rows is "
                         "not meaningful (use AURC / sel_acc).")
    return "\n".join(lines)


def adaptivity_section(results_dir: str, R: Dict) -> str:
    """Histogram of cost and chosen_k per dataset for cgar/prism records — the
    evidence that the controller actually adapts per question."""
    from collections import Counter
    out = []
    for ds in R["summaries"]:
        for m in ("cgar", "prism"):
            for tag in (m, f"{m}@seed42"):
                p = os.path.join(results_dir, f"records_{ds.replace(':', '_')}_{tag}.jsonl")
                if not os.path.exists(p):
                    continue
                recs = [json.loads(l) for l in open(p)]
                if not recs:
                    continue
                cost = Counter(int(round(r.get("cost", 0))) for r in recs)
                ks = Counter(r.get("chosen_k") for r in recs if r.get("chosen_k") is not None)
                stop = Counter(r.get("stopped_reason") for r in recs if r.get("stopped_reason"))
                n = len(recs)
                row = f"- `{ds}` / `{tag}`: cost " + ", ".join(f"{c}:{v/n*100:.0f}%" for c, v in sorted(cost.items()))
                if ks:
                    row += " | chosen_k " + ", ".join(f"{k}:{v/n*100:.0f}%" for k, v in sorted(ks.items()))
                if stop:
                    row += " | stop " + ", ".join(f"{k}:{v/n*100:.0f}%" for k, v in sorted(stop.items()))
                adaptive = len(cost) > 1 or len(ks) > 1
                row += "  → " + ("adaptive" if adaptive else "**constant (budget-bound, not adaptive)**")
                out.append(row)
                break
    return ("\n## Adaptivity (distribution of cost / chosen_k per question)\n\n" + "\n".join(out)) if out else ""


def _sweep_curves(R: Dict) -> Dict[str, Dict[str, List[Dict]]]:
    """{dataset: {name: [points sorted by cost]}} from train_logs sweeps."""
    out: Dict[str, Dict[str, List[Dict]]] = {}
    tl = R.get("train_logs", {}) or {}
    for name in ("cgar_cost_sweep", "prism_eps_sweep"):
        for p in tl.get(name, []) or []:
            out.setdefault(p["dataset"], {}).setdefault(name, []).append(p)
    for ds in out:
        for name in out[ds]:
            out[ds][name].sort(key=lambda p: p["cost"])
    return out


def sweep_section(R: Dict) -> str:
    tl = R.get("train_logs", {}) or {}
    md = []
    for name, knobs in (("cgar_cost_sweep", ("max_paths", "rel_threshold")), ("cgar_beta_sweep", ("beta",)),
                        ("prism_eps_sweep", ("eps", "max_select"))):
        pts = tl.get(name) or []
        if not pts:
            continue
        md.append(f"\n### {name} (n={pts[0].get('n', '?')} per dataset, same checkpoint)\n")
        md.append("| dataset | " + " | ".join(knobs) + " | cost | F1 | Rec | Hit-ctx |")
        md.append("|---|" + "---|" * (len(knobs) + 4))
        for p in pts:
            md.append(f"| {p['dataset']} | " + " | ".join(str(p.get(k)) for k in knobs)
                      + f" | {p['cost']:.2f} | {p['f1']*100:.1f} | {p['cov']*100:.1f} | {p.get('hit_ctx', float('nan'))*100:.1f} |")
        costs = {}
        for p in pts:
            costs.setdefault(p["dataset"], set()).add(round(p["cost"], 1))
        flat = [ds for ds, c in costs.items() if len(c) <= 1]
        if flat:
            md.append(f"\n**Flat sweep on {flat}: this knob does not move cost → no Pareto curve from it.**")
    return ("\n## Sweeps" + "\n".join(md)) if md else ""


def write_report(results_dir: str) -> str:
    with open(os.path.join(results_dir, "results.json")) as f:
        R = json.load(f)
    md = ["# KGQA bench report\n", f"Config: `{json.dumps(R.get('config', {}))}`\n"]
    if R.get("dataset_stats"):
        md.append("\n## Datasets\n\n| dataset | n | source | answer-in-subgraph | mean hops | mean triples | mean #gold |\n|---|---|---|---|---|---|---|")
        for k, s in R["dataset_stats"].items():
            md.append(f"| {k} | {s['n']} | {s['source']} | {s['answer_in_subgraph_rate']*100:.1f}% | "
                      f"{s['mean_hops']:.2f} | {s['mean_triples']:.0f} | {s['mean_n_gold']:.2f} |")
        md.append(_pool_note(R))
    md.append("\n## Main results (mean; 95% bootstrap CI for F1 / Hit-ctx / LLM-Hit; ± = std over seeds)")
    md.append(markdown_main_table(R["summaries"], R.get("seeds")))
    if R.get("stratified"):
        md.append("\n## Stratified")
        md.append(markdown_stratified(R["stratified"], "hops"))
        md.append(markdown_stratified(R["stratified"], "n_gold"))
    if R.get("comparisons"):
        md.append(markdown_comparisons(R["comparisons"]))
    adapt = adaptivity_section(results_dir, R)
    if adapt:
        md.append(adapt)
    sw = sweep_section(R)
    if sw:
        md.append(sw)
    md.append("\n## LaTeX\n\n```latex\n" + latex_main_table(R["summaries"]) + "\n```\n")
    text = "\n".join(md)
    with open(os.path.join(results_dir, "report.md"), "w") as f:
        f.write(text)
    pareto_plot(R["summaries"], os.path.join(results_dir, "pareto.png"), sweeps=_sweep_curves(R))
    return text
