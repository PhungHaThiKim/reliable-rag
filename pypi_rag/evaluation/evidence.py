"""Hallucination / evidence metrics (plan 12.3) and retrieval metrics (plan 12.2).

Claim labels come from verification.check_claims (consistency with STATIC FACTS, not truth):
  unsupported_claim_rate   UNSUPPORTED / checkable claims
  contradicted_claim_rate  CONTRADICTED / checkable claims
  citation_accuracy        cited package items that support the claim / all citations
  evidence_coverage        gold behaviors (annotated samples) that appear in the evidence the model saw
Retrieval metrics need gold relevance: a document / path is relevant to a sample when its behaviors
intersect the sample's annotated `gold_behaviors` (synthetic or manually annotated samples only).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else float("nan")


def claim_metrics(checks: Sequence[Dict]) -> Dict:
    n_chk = sum(c["n_checkable"] for c in checks)
    n_all = sum(c["n_claims"] for c in checks)
    with_claims = [c for c in checks if c["n_claims"]]
    return {"claims_per_answer": n_all / max(len(checks), 1),
            "unsupported_claim_rate": sum(c["n_unsupported"] for c in checks) / n_chk if n_chk else float("nan"),
            "contradicted_claim_rate": sum(c["n_contradicted"] for c in checks) / n_chk if n_chk else float("nan"),
            "supported_claim_rate": sum(c["n_supported"] for c in checks) / n_chk if n_chk else float("nan"),
            "uncheckable_claim_rate": sum(c["n_uncheckable"] for c in checks) / n_all if n_all else float("nan"),
            "citation_accuracy": _mean([c["citation_accuracy"] for c in checks]),
            "invalid_citation_rate": sum(c["n_invalid_citations"] for c in checks) / max(n_all, 1),
            "answers_with_unsupported": sum(1 for c in with_claims if c["n_unsupported"] + c["n_contradicted"] > 0)
                                        / max(len(with_claims), 1),
            # same-denominator views (E3 vs E4): ALL items, and the claims of the ORIGINAL answer
            "items_with_unsupported_all": sum(1 for c in checks if c["n_unsupported"] + c["n_contradicted"] > 0)
                                          / max(len(checks), 1),
            "unsupported_per_item": sum(c["n_unsupported"] + c["n_contradicted"] for c in checks) / max(len(checks), 1),
            "supported_per_item": sum(c["n_supported"] for c in checks) / max(len(checks), 1),
            "unsupported_over_original_claims": (sum(c["n_unsupported"] + c["n_contradicted"] for c in checks)
                                                 / max(1, sum((c.get("pre_policy") or c)["n_claims"] for c in checks))),
            "claims_retained": n_all / max(1, sum((c.get("pre_policy") or c)["n_claims"] for c in checks))}


def evidence_coverage(gold_behaviors: Sequence[str], visible_behaviors: Sequence[str]) -> Optional[float]:
    g = set(gold_behaviors)
    if not g:
        return None
    return len(g & set(visible_behaviors)) / len(g)


def retrieval_metrics(ranked_behaviors: List[List[str]], gold: Sequence[str], ks=(1, 3, 5)) -> Optional[Dict]:
    """ranked_behaviors: behaviors of each retrieved item in rank order."""
    g = set(gold)
    if not g:
        return None
    rel = [1 if set(b) & g else 0 for b in ranked_behaviors]
    out = {}
    for k in ks:
        top = rel[:k]
        out[f"precision@{k}"] = sum(top) / k
        covered = set().union(*[set(b) & g for b in ranked_behaviors[:k]]) if ranked_behaviors[:k] else set()
        out[f"recall@{k}"] = len(covered) / len(g)
    out["mrr"] = next((1.0 / (i + 1) for i, r in enumerate(rel) if r), 0.0)
    dcg = sum(r / np.log2(i + 2) for i, r in enumerate(rel[:max(ks)]))
    idcg = sum(1 / np.log2(i + 2) for i in range(min(sum(rel), max(ks)) or 1))
    out[f"ndcg@{max(ks)}"] = float(dcg / idcg) if idcg else 0.0
    return out


def item_lines(ev) -> set:
    """(file, line) pairs an evidence item shows: its code span, graph/fact spans and interprocedural call sites
    ('name()@file:line'). Knowledge items (K*) show nothing about the package."""
    out = set()
    if ev.evidence_id.startswith("K"):
        return out
    if ev.file_path and ev.start_line and ev.source_type == "code_span":
        for ln in range(int(ev.start_line), int(ev.end_line or ev.start_line) + 1):
            out.add((ev.file_path, ln))
    meta = ev.meta or {}
    for sp in meta.get("spans") or []:
        if len(sp) >= 2 and str(sp[1]).isdigit():
            out.add((str(sp[0]), int(sp[1])))
    for cs in meta.get("call_sites") or []:
        loc = str(cs).rsplit("@", 1)[-1]
        f, _, ln = loc.rpartition(":")
        if f and ln.isdigit():
            out.add((f, int(ln)))
    return out


def required_evidence_recall(required: Sequence[str], items) -> Optional[float]:
    """Share of the required 'path:line' locations shown by at least one of `items`."""
    if not required:
        return None
    shown = set()
    for ev in items:
        shown |= item_lines(ev)
    hit = 0
    for r in required:
        f, _, ln = r.rpartition(":")
        hit += int((f, int(ln)) in shown) if ln.isdigit() else 0
    return hit / len(required)


def answer_path_ok(gold_function_path: Sequence[str], explanation: str, claims: Sequence[Dict],
                   checked: Sequence[Dict], relational: Sequence[str]) -> Optional[float]:
    """1.0 when the answer (a) has a SUPPORTED, non-negated relational claim and (b) names every function of
    the gold function path in its explanation or claims; else 0.0. None without a gold function path."""
    if not gold_function_path:
        return None
    rel = set(relational)
    ok = any(c.get("label") == "SUPPORTED" and not c.get("negated") and set(c.get("behaviors") or []) & rel
             for c in checked)
    text = (explanation or "") + " " + " ".join(str(c.get("text", "")) for c in claims)
    named = all(fn.lower() in text.lower() for fn in gold_function_path)
    return 1.0 if (ok and named) else 0.0


def path_recall(found_api_paths: List[List[str]], gold_paths: List[List[str]]) -> Optional[float]:
    """Share of gold API paths whose first and last API both appear, in order, in one found path."""
    if not gold_paths:
        return None

    def same(a: str, g: str) -> bool:          # os.environ ~ os.environ.items, socket.create_connection.recv
        return a == g or a.startswith(g + ".") or g.startswith(a + ".")

    def hit(g):
        for f in found_api_paths:
            for i, a in enumerate(f):
                if same(a, g[0]) and any(same(b, g[-1]) for b in f[i + 1:]):
                    return True
        return False
    return sum(hit(g) for g in gold_paths) / len(gold_paths)
