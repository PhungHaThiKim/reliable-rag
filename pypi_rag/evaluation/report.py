"""report.md + figures from results.json."""
from __future__ import annotations

import json
import os
from typing import Dict, List

# one column group per question: detection quality | coverage and selective risk | grounding | evidence recall | cost
HEAD = ["strict/f1_malicious", "strict/macro_f1", "strict/fpr", "strict/fnr", "answered/f1_malicious",
        "coverage", "aurc", "rc_max_coverage",
        "evidence/unsupported_claim_rate", "evidence/unsupported_over_original_claims", "evidence/items_with_unsupported_all",
        "evidence/citation_accuracy", "gold/evidence_coverage", "gold/path_recall", "case/required_evidence_recall_cited",
        "case/abstention_accuracy", "tokens_in_mean", "tokens_out_mean", "latency_p50_ms", "parse_failure_rate"]
GROUND = ["n_items", "original_claims", "unsupported_over_original_claims", "unsupported_per_item", "items_with_unsupported",
          "supported_per_item", "claims_retained", "coverage"]


def _fmt(v):
    if isinstance(v, float):
        return "nan" if v != v else f"{v:.3f}"
    return str(v)


def md_table(rows: Dict[str, Dict], cols: List[str]) -> str:
    cols = [c for c in cols if any(c in r for r in rows.values())]
    out = ["| method | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for m, r in rows.items():
        out.append(f"| {m} | " + " | ".join(_fmt(r.get(c, "")) for c in cols) + " |")
    return "\n".join(out)


def write_report(results_dir: str) -> str:
    with open(os.path.join(results_dir, "results.json"), encoding="utf-8") as f:
        R = json.load(f)
    lines = [f"# Results: {os.path.basename(results_dir.rstrip('/'))}", "",
             f"Model: `{R['config'].get('llm')}` | split: `{R['config'].get('split')}` | n = {R.get('n_samples')} | "
             f"methods: {', '.join(R['summaries'])}", "",
             "Strict = abstentions counted as errors. Claim labels check consistency with static facts, not truth.", "",
             "## Headline", "", md_table(R["summaries"], HEAD), ""]
    for m, s in R["summaries"].items():
        ci = s.get("f1_ci95", {})
        lines.append(f"- {m}: strict malicious-F1 {_fmt(ci.get('mean'))} [95% CI {_fmt(ci.get('ci_low'))}, {_fmt(ci.get('ci_high'))}]")
    if R.get("comparisons"):
        lines += ["", "## Paired comparisons (same sample ids)", "",
                  "| pair | question | n | dF1 strict | CI95 | p | dAcc | McNemar p |", "|---|---|---|---|---|---|---|---|"]
        for k, c in R["comparisons"].items():
            b = c["f1_malicious_strict"]; a = c["accuracy"]
            lines.append(f"| {k} | {c.get('question', '')} | {c['n']} | {b['delta']:+.3f} | [{b['ci_low']:+.3f}, {b['ci_high']:+.3f}] | "
                         f"{b['p_value']:.3f} | {a['delta']:+.3f} | {c['mcnemar']['p_value']:.3f} |")
    if R.get("grounding_vs_e3"):
        lines += ["", "## Grounding on the same denominator (E3 vs E4)", "",
                  "E4 reuses E3's answers: rates below divide by E3's original claims and by ALL items (abstained items "
                  "included), so filtering cannot by itself make E4 look better.", "",
                  md_table(R["grounding_vs_e3"], GROUND)]
    if R.get("cost"):
        c = R["cost"]
        lines += ["", "## Cost (measured GPU time, allocated by tokens)", "",
                  f"GPU seconds {c['gpu_seconds_total']:.0f} for {c['unique_prompts']} unique prompts; "
                  f"{c['seconds_per_1k_tokens']:.3f} s per 1k tokens; price/h {c.get('price_per_hour')}", "",
                  md_table(c["methods"], ["tokens_per_item", "gpu_seconds_per_item", "usd_per_1000_items", "reuses_e3_generation"])]
    if R.get("strata"):
        lines += ["", "## Strata", ""]
        for m, st in R["strata"].items():
            lines.append(f"**{m}**: " + "; ".join(f"{g}: n={v['n']} acc={v['accuracy_strict']:.2f} FPR={v['fpr_strict']:.2f}"
                                                  for g, v in sorted(st.items())))
    for key in ("kb_report", "data_report", "policy"):
        if R.get(key):
            lines += ["", f"## {key}", "", "```json", json.dumps(R[key], indent=1)[:4000], "```"]
    txt = "\n".join(lines) + "\n"
    with open(os.path.join(results_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(txt)
    return txt


def plot_dashboard(results_dirs: Dict[str, str], out_png: str):
    """Plan 13 figures 1-3 and 5 for one or more result dirs (label -> dir)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = []
    for lab, d in results_dirs.items():
        with open(os.path.join(d, "results.json"), encoding="utf-8") as f:
            R = json.load(f)
        for m, s in R["summaries"].items():
            rows.append((f"{lab}:{m}" if len(results_dirs) > 1 else m, s))
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    names = [r[0] for r in rows]
    x = range(len(rows))
    ax[0, 0].bar([i - 0.2 for i in x], [r[1]["strict/f1_malicious"] for r in rows], 0.4, label="malicious F1")
    ax[0, 0].bar([i + 0.2 for i in x], [r[1]["strict/macro_f1"] for r in rows], 0.4, label="macro F1")
    ax[0, 0].set_title("F1 (strict)"); ax[0, 0].legend()
    ax[0, 1].bar([i - 0.2 for i in x], [r[1]["strict/fpr"] for r in rows], 0.4, label="FPR")
    ax[0, 1].bar([i + 0.2 for i in x], [r[1]["strict/fnr"] for r in rows], 0.4, label="FNR")
    ax[0, 1].set_title("FPR / FNR (strict)"); ax[0, 1].legend()
    for n, r in rows:
        ax[1, 0].scatter(r["coverage"], r.get("evidence/unsupported_claim_rate", float("nan")))
        ax[1, 0].annotate(n, (r["coverage"], r.get("evidence/unsupported_claim_rate", 0)), fontsize=7)
        ax[1, 1].scatter(r["tokens_in_mean"], r["strict/f1_malicious"])
        ax[1, 1].annotate(n, (r["tokens_in_mean"], r["strict/f1_malicious"]), fontsize=7)
    ax[1, 0].set_xlabel("answer coverage"); ax[1, 0].set_ylabel("unsupported claim rate"); ax[1, 0].set_title("Unsupported claims vs coverage")
    ax[1, 1].set_xlabel("input tokens / package"); ax[1, 1].set_ylabel("malicious F1"); ax[1, 1].set_title("Cost vs F1")
    for a in (ax[0, 0], ax[0, 1]):
        a.set_xticks(list(x)); a.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    fig.tight_layout(); fig.savefig(out_png, dpi=130); plt.close(fig)
    return out_png
