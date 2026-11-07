"""Auto-fill PRISM paper section placeholders from experiment JSON outputs.

After running the experiment suite, this script reads `all_results.json`
(produced by runner.py) and substitutes placeholders in paper markdown/tex
files with real numbers.

Usage:

    from prism.paper_fill import fill_paper
    fill_paper(
        results_path="./results/all_results.json",
        template_dir="../paper/",
        out_dir="../paper/_filled/",
    )

Placeholder syntax in paper templates:
    {{exp1.webqsp.prism_ours.hit_at_1:.2f}}
    {{exp4_indist.PRISM (ours).ece:.4f}}
    {{exp5.pareto_frontier[0].mean_cost}}
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict


# ─────────────────────────────────────────────────────────────────────
# Placeholder resolution
# ─────────────────────────────────────────────────────────────────────

PLACEHOLDER_RE = re.compile(r"\{\{([^{}]+?)\}\}")


def _walk(data: Any, path: str) -> Any:
    """Walk dot-separated path through nested dict/list.

    Supports: a.b.c → data["a"]["b"]["c"]
              a[2].b → data["a"][2]["b"]
              a."key with space".b
    """
    cur = data
    # Tokenise the path
    tokens = re.findall(r"[^.\[\]]+|\[\d+\]", path)
    for tok in tokens:
        if tok.startswith("[") and tok.endswith("]"):
            cur = cur[int(tok[1:-1])]
        else:
            if isinstance(cur, dict) and tok in cur:
                cur = cur[tok]
            elif isinstance(cur, list) and tok.isdigit():
                cur = cur[int(tok)]
            else:
                # Try with quote stripping
                if tok.strip('"').strip("'") in cur:
                    cur = cur[tok.strip('"').strip("'")]
                else:
                    raise KeyError(f"path '{path}' missing at token '{tok}'")
    return cur


def _format(value: Any, fmt_spec: str = "") -> str:
    if fmt_spec:
        try:
            return format(value, fmt_spec)
        except Exception:
            return str(value)
    return str(value)


def resolve_placeholder(match: re.Match, results: Dict) -> str:
    spec = match.group(1).strip()
    # Split path and format spec
    if ":" in spec:
        path, fmt = spec.rsplit(":", 1)
    else:
        path, fmt = spec, ""
    try:
        val = _walk(results, path)
        return _format(val, fmt)
    except (KeyError, IndexError, TypeError) as e:
        return f"{{{{MISSING: {path}}}}}"


def fill_text(text: str, results: Dict) -> str:
    return PLACEHOLDER_RE.sub(lambda m: resolve_placeholder(m, results), text)


# ─────────────────────────────────────────────────────────────────────
# Main entry
# ─────────────────────────────────────────────────────────────────────

def fill_paper(results_path: str, template_dir: str, out_dir: str,
                pattern: str = "PRISM_*.tex"):
    """Read all_results.json, fill placeholders in every template, write
    filled versions to out_dir."""
    with open(results_path) as f:
        results = json.load(f)
    # The runner saves under "experiments" — unnest if needed
    if "experiments" in results and isinstance(results["experiments"], dict):
        results = results["experiments"]

    os.makedirs(out_dir, exist_ok=True)
    n_filled = 0
    n_missing = 0
    for tpath in Path(template_dir).glob(pattern):
        text = tpath.read_text()
        filled = fill_text(text, results)
        n_missing_here = len(re.findall(r"\{\{MISSING:", filled))
        n_filled_here = len(re.findall(PLACEHOLDER_RE, text)) - n_missing_here
        n_filled += n_filled_here
        n_missing += n_missing_here
        out_path = Path(out_dir) / tpath.name
        out_path.write_text(filled)
        print(f"  {tpath.name}: filled {n_filled_here}, missing {n_missing_here}")
    print(f"\nTotal: {n_filled} placeholders filled, {n_missing} missing")
    print(f"Output → {out_dir}")


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="./results/all_results.json")
    ap.add_argument("--templates", default="../paper")
    ap.add_argument("--out", default="../paper/_filled")
    ap.add_argument("--pattern", default="PRISM_*.tex")
    args = ap.parse_args()
    fill_paper(args.results, args.templates, args.out, args.pattern)
