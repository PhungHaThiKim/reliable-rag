"""Bounded traversal (1-3 hops) and motif matching over the behavior graph (plan 10.4)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx

from ..knowledge.builtin import BEHAVIOR_PATTERNS, BehaviorPattern


@dataclass
class GraphPath:
    ops: List[str]                                  # operation node ids, in order
    edges: List[Dict]                               # edge attribute dicts between consecutive ops
    relation: str                                   # FLOWS_TO | PRECEDES | SINGLE
    patterns: List[str] = field(default_factory=list)
    severity: float = 0.0
    score: float = 0.0


def _flow_succ(G, op) -> List[Tuple[str, Dict]]:
    out = []
    for _u, val, k, d in G.out_edges(op, keys=True, data=True):
        if k == "PRODUCES":
            for _v, nxt, k2, d2 in G.out_edges(val, keys=True, data=True):
                if k2 == "FLOWS_TO":
                    out.append((nxt, d2))
    return out


def _prec_succ(G, op) -> List[Tuple[str, Dict]]:
    return [(v, d) for _u, v, k, d in G.out_edges(op, keys=True, data=True) if k == "PRECEDES"]


def chains(G: nx.MultiDiGraph, relation: str = "FLOWS_TO", max_hops: int = 3, max_chains: int = 2000) -> List[GraphPath]:
    succ = _flow_succ if relation == "FLOWS_TO" else _prec_succ
    ops = [n for n, d in G.nodes(data=True) if d.get("ntype") == "Operation"]
    has_in = set()
    for o in ops:
        for nxt, _ in succ(G, o):
            has_in.add(nxt)
    starts = [o for o in ops if succ(G, o)]
    out: List[GraphPath] = []
    for s in starts:
        stack = [(s, [s], [])]
        while stack and len(out) < max_chains:
            node, path, edges = stack.pop()
            for nxt, d in succ(G, node):
                if nxt in path:
                    continue
                p2, e2 = path + [nxt], edges + [d]
                out.append(GraphPath(p2, e2, relation))
                if len(e2) < max_hops:
                    stack.append((nxt, p2, e2))
    return out


def _subseq_match(G, ops: Sequence[str], steps: Sequence[Sequence[str]]) -> Optional[List[int]]:
    """Indices i1<i2<.. such that set(behaviors(op[i_k])) & set(steps[k]) is not empty (greedy, first op of each step)."""
    idx, j = [], 0
    for i, o in enumerate(ops):
        if j < len(steps) and set(G.nodes[o].get("behaviors", [])) & set(steps[j]):
            idx.append(i); j += 1
    return idx if j == len(steps) else None


def match_patterns(G: nx.MultiDiGraph, patterns: Sequence[BehaviorPattern] = BEHAVIOR_PATTERNS, max_hops: int = 3
                   ) -> List[GraphPath]:
    flows = chains(G, "FLOWS_TO", max_hops)
    precs = chains(G, "PRECEDES", max_hops)
    flow_pairs = {(p.ops[0], p.ops[-1]) for p in flows}
    ops = [n for n, d in G.nodes(data=True) if d.get("ntype") == "Operation"]
    found: Dict[Tuple, GraphPath] = {}
    for pat in patterns:
        if pat.relation == "SINGLE":
            for o in ops:
                d = G.nodes[o]
                if set(d.get("behaviors", [])) & set(pat.steps[0]) and (pat.context is None or d.get("context") == pat.context):
                    key = (pat.pattern_id, (o,))
                    found.setdefault(key, GraphPath([o], [], "SINGLE", [pat.pattern_id], pat.severity))
            continue
        pool = flows if pat.relation == "FLOWS_TO" else precs
        for p in pool:
            idx = _subseq_match(G, p.ops, pat.steps)
            if idx is None or idx[0] != 0:
                continue
            last = idx[-1]
            sub_ops, sub_edges = p.ops[: last + 1], p.edges[: last]
            if pat.context and G.nodes[sub_ops[0]].get("context") != pat.context:
                continue
            if pat.relation == "PRECEDES" and (sub_ops[0], sub_ops[-1]) in flow_pairs:
                continue                                    # a confirmed flow exists: the FLOWS_TO pattern covers it
            key = (pat.pattern_id, tuple(sub_ops))
            if key not in found:
                found[key] = GraphPath(list(sub_ops), list(sub_edges), pat.relation, [pat.pattern_id], pat.severity)
    # unmatched flows into critical sinks are still evidence (lower severity)
    matched_ops = {tuple(p.ops) for p in found.values()}
    for p in flows:
        if tuple(p.ops) in matched_ops:
            continue
        last = set(G.nodes[p.ops[-1]].get("behaviors", []))
        if last & {"network_send", "code_exec", "process_exec"}:
            found[("flow", tuple(p.ops))] = GraphPath(p.ops, p.edges, "FLOWS_TO", [], 0.3)
    return list(found.values())


def pyguard_matches(G: nx.MultiDiGraph) -> List[Dict]:
    out = []
    for u, v, k, d in G.edges(keys=True, data=True):
        if k == "MATCHES_PATTERN":
            out.append({"pattern": v, "op": u, **{x: d.get(x) for x in ("ops", "confidence", "extractor")},
                        "kind": G.nodes[v].get("kind"), "steps": G.nodes[v].get("steps"), "support": G.nodes[v].get("support")})
    return out
