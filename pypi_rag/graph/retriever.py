"""Graph retrieval for E3 and the text-linearized control E3a (plan 10.4 and 12.4 fairness note).

E3  : pattern-matched / sink-reaching paths found by traversal, scored by provenance confidence,
      behavioral specificity and execution context, linearized with file:line and the code line of
      every operation (evidence G*), plus the matched patterns' background text (evidence K*).
E3a : the SAME edge facts (one sentence per edge, no paths, no motif matching) and the SAME pattern
      descriptions, selected by the hybrid text retriever with the shared queries (evidence F*).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx

from ..analysis.code_slicer import approx_tokens
from ..knowledge.builtin import BEHAVIOR_PATTERNS
from ..knowledge.documents import KnowledgeDoc
from ..schema import Evidence, Sample
from .traversal import GraphPath, match_patterns, pyguard_matches

PATTERNS = {p.pattern_id: p for p in BEHAVIOR_PATTERNS}
EXFIL = {"P01", "P02", "P03", "P04", "P05", "P15"}
DLEXEC = {"P06", "P07"}
CTX_BONUS = {"install": 1.25, "import": 1.1, "runtime": 1.0}


def _op_line(G, sample: Sample, op: str) -> str:
    d = G.nodes[op]
    f = sample.file(d["file"])
    code = f.line(d["line"]).strip()[:160] if f else ""
    beh = ",".join(d.get("behaviors", []))
    return f"{d['file']}:{d['line']} {d['api']} {{{beh}}} [{d.get('context')}] `{code}`"


def _edge_txt(e: Dict) -> str:
    via = f" via {', '.join(e.get('via') or [])}" if e.get("via") else ""
    return f"--{e['etype']} [{e['extractor']}{via}; conf {e['confidence']:.1f}; {e['evidence_status']}]-->"


def _endpoint_pairs(G, op) -> List[Tuple[str, str]]:
    return [(k, str(G.nodes[v].get("value"))) for _u, v, k, _d in G.out_edges(op, keys=True, data=True)
            if k in ("SENDS_TO", "FETCHES_FROM", "READS", "WRITES")]


def _loc(G, op) -> str:
    return f"{G.nodes[op]['file']}:{G.nodes[op]['line']}"


def _path_edges(G, p) -> List[Dict]:
    """Structured edges of one path (meta only, never printed by E3): FLOWS_TO steps + endpoint edges."""
    out = []
    for i, e in enumerate(p.edges):
        a, b = p.ops[i], p.ops[i + 1]
        out.append({"etype": e["etype"], "src_node": a, "dst_node": b, "src": G.nodes[a]["cid"], "dst": G.nodes[b]["cid"],
                    "src_api": G.nodes[a]["api"], "dst_api": G.nodes[b]["api"], "src_loc": _loc(G, a), "dst_loc": _loc(G, b),
                    "extractor": e["extractor"], "confidence": e["confidence"], "evidence_status": e["evidence_status"],
                    "via": list(e.get("via") or []), "via_functions": list(e.get("via_functions") or []),
                    "call_sites": list(e.get("call_sites") or [])})
    for o in p.ops:
        for k, v in _endpoint_pairs(G, o):
            out.append({"etype": k, "src_node": o, "src": G.nodes[o]["cid"], "src_api": G.nodes[o]["api"],
                        "src_loc": _loc(G, o), "value": v})
    return out


def _endpoints(G, op) -> List[str]:
    out = []
    for _u, v, k, _d in G.out_edges(op, keys=True, data=True):
        if k in ("SENDS_TO", "FETCHES_FROM", "READS", "WRITES"):
            out.append(f"{k} {G.nodes[v].get('value')}")
    return out


def score_path(G, p: GraphPath) -> float:
    conf = min([e["confidence"] for e in p.edges] or [G.nodes[p.ops[0]].get("kind") == "regex" and 0.3 or 1.0])
    beh = {b for o in p.ops for b in G.nodes[o].get("behaviors", [])}
    ctx = CTX_BONUS.get(G.nodes[p.ops[0]].get("context"), 1.0)
    return (max(p.severity, 0.1) * (0.5 + 0.5 * conf) * ctx) + 0.03 * len(beh) + 0.02 * len(p.ops)


def relational_behaviors(pids: Sequence[str]) -> List[str]:
    out = []
    if set(pids) & EXFIL:
        out.append("data_exfiltration")
    if set(pids) & DLEXEC:
        out.append("download_exec")
    return out


def graph_evidence(G: nx.MultiDiGraph, sample: Sample, max_paths: int = 8, max_hops: int = 3,
                   max_tokens: int = 1500, pattern_docs: bool = True, dispatch_caveat: bool = True
                   ) -> Tuple[List[Evidence], Dict]:
    paths = match_patterns(G, max_hops=max_hops)
    merged: Dict[Tuple, GraphPath] = {}
    for p in paths:                                       # same ops -> one path with all pattern ids
        k = (tuple(p.ops), p.relation)
        if k in merged:
            m = merged[k]
            m.patterns = sorted(set(m.patterns) | set(p.patterns)); m.severity = max(m.severity, p.severity)
        else:
            merged[k] = p
    paths = list(merged.values())
    for p in paths:
        p.score = score_path(G, p)
    paths.sort(key=lambda p: -p.score)
    kept: List[GraphPath] = []
    for p in paths:                                       # drop paths contained in a kept, better path
        if any(set(p.ops) <= set(q.ops) and set(p.patterns) <= set(q.patterns) for q in kept):
            continue
        kept.append(p)
    items: List[Evidence] = []
    used = 0
    for p in kept[:max_paths]:
        lines = []
        pats = [f"{pid} {PATTERNS[pid].title} (severity {PATTERNS[pid].severity})" for pid in p.patterns if pid in PATTERNS]
        head = "; ".join(pats) if pats else "unmatched data flow into a sensitive sink (severity 0.3)"
        lines.append(f"{p.relation} path: {head}")
        for i, o in enumerate(p.ops):
            lines.append(f"  ({i + 1}) {_op_line(G, sample, o)}" + "".join(f" | {x}" for x in _endpoints(G, o)))
            if i < len(p.edges):
                lines.append(f"      {_edge_txt(p.edges[i])}")
        content = "\n".join(lines)
        t = approx_tokens(content)
        if used + t > max_tokens and items:
            break
        used += t
        beh = sorted({b for o in p.ops for b in G.nodes[o].get("behaviors", [])} | set(relational_behaviors(p.patterns)))
        spans = [(G.nodes[o]["file"], G.nodes[o]["line"]) for o in p.ops]
        edges = _path_edges(G, p)
        call_sites = sorted({c for e in edges for c in e.get("call_sites", [])})
        via_fns = [f for e in edges for f in e.get("via_functions", [])]
        items.append(Evidence(evidence_id=f"G{len(items) + 1}", sample_id=sample.sample_id, source_type="graph_path",
                              source_ref="|".join(G.nodes[o]["cid"] for o in p.ops), content=content, score=p.score,
                              file_path=spans[0][0], start_line=min(s[1] for s in spans if s[0] == spans[0][0]),
                              end_line=max(s[1] for s in spans if s[0] == spans[0][0]), behaviors=beh,
                              meta={"spans": spans, "patterns": p.patterns, "relation": p.relation,
                                    "extractors": [e["extractor"] for e in p.edges],
                                    "statuses": [e["evidence_status"] for e in p.edges],
                                    "ops_api": [G.nodes[o]["api"] for o in p.ops], "edges": edges,
                                    "call_sites": call_sites, "via_functions": list(dict.fromkeys(via_fns)),
                                    "ops": [{"node": o, "cid": G.nodes[o]["cid"], "api": G.nodes[o]["api"],
                                             "loc": _loc(G, o), "behaviors": list(G.nodes[o].get("behaviors", [])),
                                             "context": G.nodes[o].get("context")} for o in p.ops]}))
    pg = pyguard_matches(G)
    for m in pg[:3]:
        if used > max_tokens:
            break
        o = m["op"]
        content = (f"PyGuard pattern {G.nodes[m['pattern']].get('pattern_id')} [{m['kind']}, support {m['support']}] "
                   f"matched by action subsequence starting at {_op_line(G, sample, o)} (heuristic API->action mapping)")
        used += approx_tokens(content)
        items.append(Evidence(evidence_id=f"G{len(items) + 1}", sample_id=sample.sample_id, source_type="graph_path",
                              source_ref=m["pattern"], content=content, score=0.2, file_path=G.nodes[o]["file"],
                              start_line=G.nodes[o]["line"], end_line=G.nodes[o]["line"],
                              behaviors=list(G.nodes[o].get("behaviors", [])),
                              meta={"spans": [(G.nodes[o]["file"], G.nodes[o]["line"])], "pyguard": True,
                                    "statuses": ["inferred"]}))
    if not items:
        ops = [n for n, d in G.nodes(data=True) if d.get("ntype") == "Operation"]
        iso = "; ".join(_op_line(G, sample, o) for o in ops[:8]) or "none"
        cov = G.graph.get("parse_coverage", 1.0)
        note = "" if cov >= 1.0 else (f" Static coverage is incomplete: {1 - cov:.0%} of files could not be parsed "
                                      "(regex fallback, flows unknown), so absence of a path is NOT evidence of absence.")
        items.append(Evidence(evidence_id="G1", sample_id=sample.sample_id, source_type="graph_path",
                              source_ref="none", content=f"No source-to-sink data flow or known behavior pattern was found. "
                                                         f"Isolated sensitive operations: {iso}.{note}",
                              score=0.0, behaviors=[], meta={"spans": [], "empty": True}))
    cov = G.graph.get("parse_coverage", 1.0)
    if cov < 1.0 and not items[-1].meta.get("empty"):
        items.append(Evidence(evidence_id=f"G{len(items) + 1}", sample_id=sample.sample_id, source_type="graph_path",
                              source_ref="coverage", content=(
                                  f"Static coverage note: {1 - cov:.0%} of Python files could not be parsed (e.g. Python 2 "
                                  "syntax or obfuscation); their operations were found by regex only and their data flows "
                                  "are unknown, so a missing path there is NOT evidence that no flow exists."),
                              score=0.0, behaviors=[], meta={"spans": [], "coverage_note": True}))
    disp = G.graph.get("dispatch_sites") or []
    if dispatch_caveat and disp:
        where = ", ".join(f"{d['file']}:{d['line']} {d['api'].replace('builtins.', '')}" for d in disp[:4])
        items.append(Evidence(evidence_id=f"G{len(items) + 1}", sample_id=sample.sample_id, source_type="graph_path",
                              source_ref="dispatch", content=(
                                  f"Dynamic dispatch note: {len(disp)} call(s) choose their target at run time ({where}). "
                                  "The static graph cannot follow them, so the absence of a path is NOT evidence that "
                                  "no flow exists."),
                              score=0.0, behaviors=[], meta={"spans": [(d["file"], d["line"]) for d in disp[:4]],
                                                             "dispatch_note": True}))
    limits = [x for x in (G.graph.get("limits_hit") or []) if x not in ("large_file_regex_only",)]
    if limits:
        items.append(Evidence(evidence_id=f"G{len(items) + 1}", sample_id=sample.sample_id, source_type="graph_path",
                              source_ref="limits", content=(
                                  f"Analysis limit note: the static analysis was truncated ({', '.join(limits)}); some "
                                  "operations or flows may be missing."),
                              score=0.0, behaviors=[], meta={"spans": [], "limits_note": True}))
    pat_items: List[Evidence] = []
    if pattern_docs:
        seen = []
        for p in kept[:max_paths]:
            for pid in p.patterns:
                if pid in PATTERNS and pid not in seen:
                    seen.append(pid)
        for pid in seen[:3]:
            pat = PATTERNS[pid]
            pat_items.append(Evidence(evidence_id=f"K{len(pat_items) + 1}", sample_id=sample.sample_id,
                                      source_type="pattern", source_ref=f"builtin:{pid}", content=pat.text(),
                                      behaviors=sorted({b for s in pat.steps for b in s})))
    diag = {"n_candidate_paths": len(paths), "n_paths": sum(1 for i in items if i.meta.get("spans")),
            "n_pyguard_matches": len(pg), "path_tokens": used,
            "pattern_ids": sorted({pid for p in kept[:max_paths] for pid in p.patterns})}
    return items + pat_items, diag


# ---------------------------------------------------------------------------------- E3a facts
def edge_facts(G: nx.MultiDiGraph, sample: Sample) -> List[KnowledgeDoc]:
    """One text fact per behavior-relevant edge, with its provenance. Same information as the graph,
    without path structure."""
    docs = []
    for u, v, k, d in G.edges(keys=True, data=True):
        if k not in ("FLOWS_TO", "SENDS_TO", "FETCHES_FROM", "READS", "WRITES", "PRECEDES", "HAS_BEHAVIOR"):
            continue
        if k == "FLOWS_TO":
            src = G.nodes[u]["producer"]; s_op = f"op:{src}"
            txt = (f"Data produced by {_op_line(G, sample, s_op)} flows to {_op_line(G, sample, v)} "
                   f"({d['extractor']}, conf {d['confidence']:.1f}).")
            beh = sorted(set(G.nodes[s_op].get("behaviors", [])) | set(G.nodes[v].get("behaviors", [])))
            ops = [s_op, v]
        elif k == "HAS_BEHAVIOR":
            txt = f"Operation {_op_line(G, sample, u)} has behavior {G.nodes[v]['name']} ({d['extractor']})."
            beh = [G.nodes[v]["name"]]; ops = [u]
        elif k == "PRECEDES":
            txt = (f"Operation {_op_line(G, sample, u)} executes before {_op_line(G, sample, v)} in the same function "
                   f"(line order only, no confirmed data flow).")
            beh = sorted(set(G.nodes[u].get("behaviors", [])) | set(G.nodes[v].get("behaviors", []))); ops = [u, v]
        else:
            txt = f"Operation {_op_line(G, sample, u)} {k} {G.nodes[v].get('value')}."
            beh = list(G.nodes[u].get("behaviors", [])); ops = [u]
        docs.append(KnowledgeDoc(doc_id=f"fact:{len(docs)}", source_type="fact", title=k, text=txt, behaviors=beh,
                                 apis=[G.nodes[o]["api"] for o in ops],
                                 provenance={"spans": [(G.nodes[o]["file"], G.nodes[o]["line"]) for o in ops],
                                             "etype": k, "extractor": d["extractor"]}))
    return docs


def flatten_paths(G: nx.MultiDiGraph, sample: Sample, items: Sequence[Evidence], seed_key: str) -> List[Evidence]:
    """E3a (test-case document definition): EXACTLY the edges of the graph paths E3 showed, one sentence per
    edge, in a seeded random order; notes (empty / coverage / dispatch / limits) are kept verbatim. Same
    information as E3, without path structure."""
    import hashlib
    import random
    sentences: List[Tuple[str, Dict]] = []
    seen = set()
    for it in items:
        if it.source_type != "graph_path":
            continue
        edges = it.meta.get("edges")
        if edges is None:                                     # notes and PyGuard matches: keep as is
            sentences.append((it.content, {"spans": it.meta.get("spans", []), "behaviors": list(it.behaviors)}))
            continue
        if not [e for e in edges if e["etype"] == "FLOWS_TO"]:    # single-operation pattern path
            for op in it.meta.get("ops", []):
                k = ("op", op["cid"])
                if k in seen:
                    continue
                seen.add(k)
                sentences.append((f"Operation {_op_line(G, sample, op['node'])}.",
                                  {"spans": [tuple(op["loc"].rsplit(":", 1))], "behaviors": op["behaviors"]}))
        for e in edges:
            k = (e["etype"], e["src"], e.get("dst") or e.get("value"))
            if k in seen:
                continue
            seen.add(k)
            if e["etype"] == "FLOWS_TO":
                via = f" via {', '.join(e['via'])}" if e.get("via") else ""
                txt = (f"Data produced by {_op_line(G, sample, e['src_node'])} flows to {_op_line(G, sample, e['dst_node'])} "
                       f"({e['extractor']}{via}; conf {e['confidence']:.1f}; {e['evidence_status']}).")
                beh = sorted(set(G.nodes[e["src_node"]].get("behaviors", [])) | set(G.nodes[e["dst_node"]].get("behaviors", [])))
                spans = [tuple(e["src_loc"].rsplit(":", 1)), tuple(e["dst_loc"].rsplit(":", 1))] + \
                    [tuple(c.rsplit(":", 1)) for c in e.get("call_sites", [])]
                sentences.append((txt, {"spans": spans, "behaviors": beh, "edge": (e["src"], e["dst"])}))
            elif e.get("dst_node"):                           # other step between two operations (PRECEDES)
                txt = (f"Operation {_op_line(G, sample, e['src_node'])} executes before "
                       f"{_op_line(G, sample, e['dst_node'])} in the same function (line order only, no confirmed "
                       f"data flow; {e['evidence_status']}).")
                beh = sorted(set(G.nodes[e["src_node"]].get("behaviors", [])) | set(G.nodes[e["dst_node"]].get("behaviors", [])))
                sentences.append((txt, {"spans": [tuple(e["src_loc"].rsplit(":", 1)), tuple(e["dst_loc"].rsplit(":", 1))],
                                        "behaviors": beh, "edge": (e["src"], e["dst"])}))
            else:                                             # endpoint edge: SENDS_TO / READS / ... a value
                txt = f"Operation {_op_line(G, sample, e['src_node'])} {e['etype']} {e.get('value')}."
                sentences.append((txt, {"spans": [tuple(e["src_loc"].rsplit(":", 1))],
                                        "behaviors": list(G.nodes[e["src_node"]].get("behaviors", []))}))
    rng = random.Random(int(hashlib.sha256(seed_key.encode()).hexdigest()[:12], 16))
    rng.shuffle(sentences)
    out = []
    for j, (txt, m) in enumerate(sentences):
        spans = [(f, int(l)) for f, l in m["spans"] if str(l).isdigit()]
        out.append(Evidence(evidence_id=f"F{j + 1}", sample_id=sample.sample_id, source_type="fact", source_ref=f"e3a:{j}",
                            content=txt, behaviors=list(m["behaviors"]),
                            file_path=spans[0][0] if spans else None, start_line=spans[0][1] if spans else None,
                            end_line=spans[0][1] if spans else None,
                            meta={"spans": spans, "edge": m.get("edge")}))
    return out
