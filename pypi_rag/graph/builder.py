"""Build a source-grounded behavior graph (NetworkX MultiDiGraph) for ONE package from PackageFacts.
Entity linking happens here: API names are already canonical (ast_parser), behaviors come from the
taxonomy (built-in + optional PyGuard lexicon) and literal arguments become FileResource /
NetworkEndpoint nodes."""
from __future__ import annotations

import posixpath
import time
from collections import defaultdict
from typing import Dict, List, Optional

import networkx as nx

from ..analysis import taxonomy as T
from ..analysis.ast_parser import PackageFacts
from ..schema import Sample
from .schema import check_edge, edge_attrs


def _add_edge(G, u, v, etype, **kw):
    G.add_edge(u, v, key=etype, **edge_attrs(etype, **kw))


def build_behavior_graph(sample: Sample, facts: PackageFacts, pyguard_matcher=None) -> nx.MultiDiGraph:
    t0 = time.time()
    G = nx.MultiDiGraph(sample_id=sample.sample_id)
    pkg = f"pkg:{sample.package_name}"
    G.add_node(pkg, ntype="Package", name=sample.package_name, version=sample.version)
    for f in sample.files:
        if f.path.endswith((".py", ".pth", ".pyw")):
            G.add_node(f"mod:{f.path}", ntype="Module", path=f.path, parse_ok=f.path not in facts.parse_errors)
            _add_edge(G, pkg, f"mod:{f.path}", "CONTAINS", source_file=f.path, line_start=1, line_end=f.n_lines,
                      extractor="manifest", confidence=1.0)
    scopes = {c.scope for c in facts.calls.values()} | set(facts.functions)
    for sc in scopes:
        path, _, q = sc.partition("::")
        fi = facts.functions.get(sc)
        G.add_node(f"fn:{sc}", ntype="Function", qualname=q, file=path)
        if f"mod:{path}" in G:
            _add_edge(G, f"mod:{path}", f"fn:{sc}", "DEFINES", source_file=path,
                      line_start=fi.line if fi else 1, line_end=fi.end_line if fi else 1, extractor="ast", confidence=1.0)
    for caller, callee, path, line in facts.local_calls:
        if f"fn:{caller}" in G and f"fn:{callee}" in G:
            _add_edge(G, f"fn:{caller}", f"fn:{callee}", "CALLS", source_file=path, line_start=line, line_end=line,
                      extractor="ast", confidence=1.0)

    by_scope = defaultdict(list)
    for c in facts.sensitive_calls():
        op = f"op:{c.cid}"
        G.add_node(op, ntype="Operation", api=c.api, behaviors=list(c.behaviors), file=c.file, line=c.line,
                   end_line=c.end_line, context=c.context, kind=c.kind, raw=c.raw, cid=c.cid)
        ext = "regex" if c.kind == "regex" else "ast"
        _add_edge(G, f"fn:{c.scope}", op, "CALLS", source_file=c.file, line_start=c.line, line_end=c.end_line,
                  extractor=ext, confidence=c.confidence)
        G.add_node(f"api:{c.api}", ntype="API", name=c.api)
        _add_edge(G, op, f"api:{c.api}", "INSTANCE_OF", source_file=c.file, line_start=c.line, line_end=c.line,
                  extractor=ext if ext == "regex" else "taxonomy", confidence=c.confidence)
        base = set(T.behaviors_of(c.api))
        for b in c.behaviors:
            G.add_node(f"beh:{b}", ntype="Behavior", name=b)
            how = "taxonomy" if b in base else ("mode_arg" if c.api.endswith("open") else "literal_arg")
            _add_edge(G, op, f"beh:{b}", "HAS_BEHAVIOR", source_file=c.file, line_start=c.line, line_end=c.line,
                      extractor=how if ext != "regex" else "regex", confidence=c.confidence,
                      taxonomy_origin=T.rule_origin(c.api) if b in base else "")
        for kind, val in c.indicators:
            if kind in ("url", "ip", "exfil_host"):
                n = f"net:{val}"; G.add_node(n, ntype="NetworkEndpoint", value=val)
                et = "SENDS_TO" if "network_send" in c.behaviors else "FETCHES_FROM"
                _add_edge(G, op, n, et, source_file=c.file, line_start=c.line, line_end=c.end_line,
                          extractor="literal_arg" if ext != "regex" else "regex", confidence=c.confidence)
            elif kind in ("sensitive_path", "persistence_path"):
                n = f"res:{val}"; G.add_node(n, ntype="FileResource", value=val, kind=kind)
                et = "WRITES" if "file_write" in c.behaviors else "READS"
                _add_edge(G, op, n, et, source_file=c.file, line_start=c.line, line_end=c.end_line,
                          extractor="literal_arg" if ext != "regex" else "regex", confidence=c.confidence)
        if c.kind != "hook":
            by_scope[c.scope].append(c)

    for fl in facts.flows:
        s, d = facts.calls.get(fl.src), facts.calls.get(fl.dst)
        if not s or not d or f"op:{s.cid}" not in G or f"op:{d.cid}" not in G:
            continue
        val = f"val:{s.cid}"
        if val not in G:
            G.add_node(val, ntype="DataValue", producer=s.cid)
            _add_edge(G, f"op:{s.cid}", val, "PRODUCES", source_file=s.file, line_start=s.line, line_end=s.end_line,
                      extractor="ast", confidence=1.0)
        _add_edge(G, val, f"op:{d.cid}", "FLOWS_TO", source_file=fl.file, line_start=fl.line_start,
                  line_end=fl.line_end, extractor=fl.extractor, confidence=fl.confidence, via=list(fl.via),
                  via_functions=list(fl.via_functions), call_sites=list(fl.call_sites))

    for sc, ops in by_scope.items():                        # PRECEDES: sensitive ops in line order (inferred)
        ops = sorted(ops, key=lambda c: (c.line, c.col))
        for a, b in zip(ops, ops[1:]):
            if a.cid != b.cid:
                _add_edge(G, f"op:{a.cid}", f"op:{b.cid}", "PRECEDES", source_file=a.file, line_start=a.line,
                          line_end=b.end_line, extractor="line_order", confidence=0.5)

    if pyguard_matcher is not None:
        for m in pyguard_matcher.match(facts):
            pn = f"pat:pyguard:{m['pattern_id']}"
            G.add_node(pn, ntype="Pattern", pattern_id=m["pattern_id"], origin="pyguard", kind=m["kind"],
                       support=m.get("support"), steps=m["steps"])
            first = facts.calls[m["ops"][0]]; last = facts.calls[m["ops"][-1]]
            if f"op:{first.cid}" in G:
                _add_edge(G, f"op:{first.cid}", pn, "MATCHES_PATTERN", source_file=first.file, line_start=first.line,
                          line_end=last.end_line if last.file == first.file else first.end_line,
                          extractor="pyguard_subsequence", confidence=0.5, ops=list(m["ops"]))
    G.graph["build_ms"] = (time.time() - t0) * 1000
    G.graph["parse_coverage"] = facts.parse_coverage()
    G.graph["dispatch_sites"] = list(facts.dispatch_sites)
    G.graph["limits_hit"] = list(facts.limits_hit)
    return G


def graph_stats(G: nx.MultiDiGraph) -> Dict:
    et = defaultdict(int); st = defaultdict(int); bad = 0
    for _u, _v, d in G.edges(data=True):
        et[d["etype"]] += 1; st[d["evidence_status"]] += 1
        bad += 0 if check_edge(d) else 1
    nt = defaultdict(int)
    for _n, d in G.nodes(data=True):
        nt[d.get("ntype", "?")] += 1
    return {"n_nodes": G.number_of_nodes(), "n_edges": G.number_of_edges(), "node_types": dict(nt),
            "edge_types": dict(et), "evidence_status": dict(st), "edges_missing_provenance": bad,
            "build_ms": round(G.graph.get("build_ms", 0.0), 2), "parse_coverage": G.graph.get("parse_coverage", 1.0)}
