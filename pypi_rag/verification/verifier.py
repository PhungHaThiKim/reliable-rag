"""Evidence verification (plan section 11). Deterministic and offline: it uses the model's parsed claims,
the evidence items that were VISIBLE in the prompt, and the package's static facts. It never sees labels.

Stages
  1 schema       verdict in {malicious, benign, uncertain}; cited ids exist in the prompt
  2 grounding    every cited code span / graph path still matches the source (hash) and was visible
  3 claims       each claim -> behavior (declared, else lexicon); knowledge items (K*) never count as
                 evidence about the package
  4 support      SUPPORTED     behavior present in package facts AND shown by a cited package item
                 UNSUPPORTED   no package fact, or cited items do not show it (citation mismatch), or
                               the claim is not checkable against facts
                 CONTRADICTED  claim says X happens, analysis had full coverage, X has no fact and no cited
                               item mentions it; or claim says X is absent while confirmed facts show X
  5 policy       keep | filter (drop unsupported claims) | cite (abstain when a malicious verdict cites
                 nothing valid) | abstain (filter + abstain when no critical claim is supported, or when a
                 benign verdict conflicts with a severe AST-confirmed graph path)
Claim checks are fallible: they test consistency with static facts, not truth (report it that way).
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Sequence

from ..analysis import taxonomy as T
from ..analysis.ast_parser import PackageFacts
from ..analysis.provenance import verify_code_evidence
from ..schema import Evidence, Sample

RELATIONAL = set(T.RELATIONAL)
_SEV = re.compile(r"severity (0\.\d+|1\.0)")


def _item_shows(ev: Evidence, b: str, facts: PackageFacts) -> bool:
    if ev.source_type in ("pattern", "rule", "benign_rule", "malicious_example", "benign_example", "yara", "advisory",
                          "pyguard_pattern"):
        return False                                           # general knowledge, not about this package
    if b in ev.behaviors:
        return True
    if ev.source_type == "code_span" and ev.file_path:
        for c in facts.calls.values():
            if c.file == ev.file_path and ev.start_line <= c.line <= (ev.end_line or ev.start_line) and b in c.behaviors:
                return True
        if b in RELATIONAL:                                    # a chain whose both ends are inside cited spans
            return False
    return False


def _cited_edges(items: Sequence[Evidence]) -> set:
    """(src cid, dst cid) pairs shown by cited graph paths (consecutive ops) and flattened edge facts."""
    out = set()
    for ev in items:
        if ev.source_type == "graph_path" and "|" in (ev.source_ref or ""):
            ops = ev.source_ref.split("|")
            out.update(zip(ops, ops[1:]))
        edge = (ev.meta or {}).get("edge")
        if edge:
            out.add(tuple(edge))
    return out


def _chain_covered(b: str, items: Sequence[Evidence], facts: PackageFacts) -> bool:
    """A relational claim (data_exfiltration, download_exec) is shown when the cited items together contain
    every edge of one chain that supports it, whether as one path (E3) or as separate facts (E3a)."""
    cited = _cited_edges(items)
    if not cited:
        return False
    for chain in facts.relational_behaviors().get(b, []):
        if all((f.src, f.dst) in cited for f in chain):
            return True
    return False


def check_claims(pred_claims: List[Dict], visible: Dict[str, Evidence], facts: PackageFacts, sample: Sample) -> Dict:
    fb = facts.behaviors()
    full_cov = facts.complete()        # absence is evidence only with full parse, no dynamic dispatch, no limits
    confirmed = {b for c in facts.calls.values() if c.kind != "regex" for b in c.behaviors}
    confirmed |= set(facts.relational_behaviors())
    rel_chains = facts.relational_behaviors()
    labels = []
    for cl in pred_claims:
        bs = [cl["behavior"]] if cl.get("behavior") else list(cl.get("inferred_behaviors", []))[:2]
        cited = [e for e in cl.get("evidence", [])]
        valid = [visible[e] for e in cited if e in visible]
        grounded = [ev for ev in valid if verify_code_evidence(sample, ev)]
        pkg_items = [ev for ev in grounded if not ev.evidence_id.startswith("K")]
        out = {"text": cl.get("text", ""), "behaviors": bs, "negated": bool(cl.get("negated")), "cited": cited,
               "invalid_ids": [e for e in cited if e not in visible], "ungrounded_ids":
               [ev.evidence_id for ev in valid if ev not in grounded], "cites_only_knowledge": bool(valid) and not pkg_items}
        if not bs:
            out.update(label="UNCHECKABLE", why="no behavior could be identified in the claim"); labels.append(out); continue
        if cl.get("negated"):
            hit = [b for b in bs if b in confirmed]
            if hit:
                out.update(label="CONTRADICTED", why=f"claims {hit[0]} is absent, AST facts show it")
            elif full_cov and not any(b in fb for b in bs):
                out.update(label="SUPPORTED", why="behavior absent from complete static facts")
            else:
                out.update(label="UNSUPPORTED", why="absence cannot be verified (incomplete coverage or regex-only facts)")
            labels.append(out); continue
        shown = [b for b in bs if any(_item_shows(ev, b, facts) for ev in pkg_items)
                 or (b in RELATIONAL and _chain_covered(b, pkg_items, facts))]
        present = [b for b in bs if b in fb]
        if shown and set(shown) & set(present):
            out.update(label="SUPPORTED", why=f"{shown[0]} in package facts and in cited item(s)")
        elif present:
            out.update(label="UNSUPPORTED", why=f"{present[0]} exists in the package but the cited items do not show it",
                       citation_mismatch=True)
        else:
            mentioned = any(set(T.behaviors_from_text(ev.content)) & set(bs) for ev in pkg_items)
            if full_cov and not mentioned and pkg_items and not (set(bs) & RELATIONAL):
                out.update(label="CONTRADICTED", why=f"{bs[0]} absent from complete static facts and from the cited items")
            else:
                out.update(label="UNSUPPORTED", why=f"no static fact for {bs[0]}"
                           + (" (relational: flow not found)" if set(bs) & RELATIONAL else ""))
        labels.append(out)
    cites = [(l, e) for l in labels for e in l["cited"]]
    acc = [1.0 if (e in visible and not e.startswith("K") and l["label"] == "SUPPORTED") else 0.0 for l, e in cites]
    crit_sup = sum(1 for l in labels if l["label"] == "SUPPORTED" and not l["negated"]
                   and set(l["behaviors"]) & set(T.CRITICAL))
    n_check = sum(1 for l in labels if l["label"] != "UNCHECKABLE")
    return {"claims": labels, "n_claims": len(labels), "n_checkable": n_check,
            "n_supported": sum(l["label"] == "SUPPORTED" for l in labels),
            "n_unsupported": sum(l["label"] == "UNSUPPORTED" for l in labels),
            "n_contradicted": sum(l["label"] == "CONTRADICTED" for l in labels),
            "n_uncheckable": sum(l["label"] == "UNCHECKABLE" for l in labels),
            "n_critical_supported": crit_sup,
            "citation_accuracy": (sum(acc) / len(acc)) if acc else None,
            "n_invalid_citations": sum(len(l["invalid_ids"]) for l in labels),
            "n_ungrounded_citations": sum(len(l["ungrounded_ids"]) for l in labels),
            "dispatch_unresolved": bool(facts.dispatch_sites), "analysis_complete": full_cov}


def restrict_check(check: Dict, kept: List[Dict]) -> Dict:
    """Counts recomputed over the claims that remain in the final answer (E4 filter / abstain)."""
    ids = {id(c) for c in kept}
    labels = [c for c in check["claims"] if id(c) in ids]
    cites = [(l, e) for l in labels for e in l["cited"]]
    acc = [1.0 if (not e.startswith("K") and e not in l["invalid_ids"] and l["label"] == "SUPPORTED") else 0.0
           for l, e in cites]
    return {**check, "claims": labels, "n_claims": len(labels),
            "n_checkable": sum(1 for l in labels if l["label"] != "UNCHECKABLE"),
            "n_supported": sum(l["label"] == "SUPPORTED" for l in labels),
            "n_unsupported": sum(l["label"] == "UNSUPPORTED" for l in labels),
            "n_contradicted": sum(l["label"] == "CONTRADICTED" for l in labels),
            "n_uncheckable": sum(l["label"] == "UNCHECKABLE" for l in labels),
            "citation_accuracy": (sum(acc) / len(acc)) if acc else None,
            "n_invalid_citations": sum(len(l["invalid_ids"]) for l in labels),
            "n_ungrounded_citations": sum(len(l["ungrounded_ids"]) for l in labels),
            "n_claims_before_policy": check["n_claims"],
            # counts of the ORIGINAL answer, so E4 can be scored on the same denominator as E3
            "pre_policy": {k: check[k] for k in ("n_claims", "n_checkable", "n_supported", "n_unsupported",
                                                  "n_contradicted", "n_uncheckable")}}


def graph_conflict(visible: Dict[str, Evidence], min_severity: float = 0.9) -> Optional[str]:
    """Id of a visible graph path with an AST-confirmed, severe pattern (used against benign verdicts)."""
    for iid, ev in visible.items():
        if ev.source_type != "graph_path" or ev.meta.get("empty") or ev.meta.get("pyguard"):
            continue
        sev = [float(x) for x in _SEV.findall(ev.content.split("\n", 1)[0])]
        if sev and max(sev) >= min_severity and all(s == "ast_confirmed" for s in ev.meta.get("statuses", [])):
            return iid
    return None


def apply_policy(verdict: str, check: Dict, visible: Dict[str, Evidence], mode: str = "abstain",
                 min_critical: int = 1, conflict_severity: Optional[float] = 0.9,
                 abstain_on_dispatch: bool = False) -> Dict:
    """-> {verdict, kept_claims, reason}. Never uses labels."""
    claims = check["claims"]
    kept = claims if mode in ("keep", "cite") else [c for c in claims if c["label"] == "SUPPORTED"]
    v, why = verdict, ""
    if mode == "cite":
        if verdict == "malicious" and not any(c["cited"] and not c["invalid_ids"] and not c["cites_only_knowledge"]
                                              for c in claims):
            v, why = "uncertain", "malicious verdict without a valid package citation"
    elif mode == "abstain":
        if verdict == "malicious" and check["n_critical_supported"] < min_critical:
            v, why = "uncertain", f"malicious verdict with {check['n_critical_supported']} supported critical claims"
        elif verdict == "benign" and conflict_severity is not None:
            g = graph_conflict(visible, conflict_severity)
            if g or any(c["label"] == "CONTRADICTED" and c["negated"] and set(c["behaviors"]) & set(T.CRITICAL)
                        for c in claims):
                v, why = "uncertain", f"benign verdict conflicts with evidence {g or 'contradicted negated claim'}"
        if v == "benign" and abstain_on_dispatch and check.get("dispatch_unresolved"):
            v, why = "uncertain", "benign verdict while dynamic dispatch targets are unresolved"
    return {"verdict": v, "kept_claims": kept, "reason": why}


def tune_policy(rows: Sequence[Dict], labels: Dict[str, str], grid_min: Iterable[int] = (1, 2),
                grid_sev: Iterable[Optional[float]] = (None, 0.85, 0.9, 0.95),
                grid_dispatch: Iterable[bool] = (False, True)) -> Dict:
    """rows: [{sample_id, verdict, check, visible}] on VALIDATION. Picks the setting with the best strict
    malicious-F1 (abstentions count as errors), tie-break on coverage."""
    from ..evaluation.classification import classification_metrics
    best = None
    for m in grid_min:
        for sv in grid_sev:
            for dz in grid_dispatch:
                preds = {r["sample_id"]: apply_policy(r["verdict"], r["check"], r["visible"], "abstain", m, sv, dz)["verdict"]
                         for r in rows}
                ids = [i for i in preds if labels.get(i) in ("benign", "malicious")]
                met = classification_metrics([labels[i] for i in ids], [preds[i] for i in ids])
                key = (met["strict"]["f1_malicious"], met["coverage"])
                if best is None or key > best[0]:
                    best = (key, {"min_critical": m, "conflict_severity": sv, "abstain_on_dispatch": dz,
                                  "val_strict_f1": key[0], "val_coverage": key[1]})
    return best[1] if best else {"min_critical": 1, "conflict_severity": 0.9}


JUDGE_PROMPT = ("You check whether evidence supports a claim about a Python package. Evidence:\n{evidence}\n\n"
                "Claim: {claim}\n\nAnswer with one word: SUPPORTED, UNSUPPORTED or CONTRADICTED.")


def llm_judge(llm, items: List[Dict]) -> List[str]:
    """Optional probabilistic check for semantic claims: items = [{claim, evidence_text}]. Results must be
    reported as LLM-judged, not as ground truth (plan 12.2)."""
    batch = [[{"role": "user", "content": JUDGE_PROMPT.format(evidence=it["evidence_text"][:4000], claim=it["claim"])}]
             for it in items]
    outs = llm.generate(batch)
    res = []
    for o in outs:
        m = re.search(r"\b(SUPPORTED|UNSUPPORTED|CONTRADICTED)\b", o.text.upper())
        res.append(m.group(1) if m else "UNCLEAR")
    return res
