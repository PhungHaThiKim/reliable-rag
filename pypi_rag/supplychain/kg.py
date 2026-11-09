"""Task B knowledge graph, built by PARSING the corpus documents (never from the in-memory org), so the KG
and the text index hold exactly the same information. Every edge keeps `doc` and `line` provenance.

Nodes  svc:<name> pkg:<name> pv:<name>@<ver> team:<name> adv:<id> pol:<id>[-v<n>] adr:<id> inc:<id>
Edges  DEPENDS_ON{scope, extras, unpinned} (svc -> pv, or svc -> pkg when unpinned), VERSION_OF (pv -> pkg),
       REQUIRES{extra} (pv -> pv), CALLS, OWNED_BY, AFFECTS{spec} (adv -> pkg), MATCHES (adv -> pv, derived
       from the advisory range and the package versions present in the corpus), SUPERSEDES, BANS{spec},
       EXEMPTS{spec, until} (adr -> svc and adr -> pkg), INVOLVES (inc -> svc), MENTIONS (inc -> pkg).

Retrieval (`kg_evidence`) links entity names in the question, then:
  advisory / package -> reverse traversal over DEPENDS_ON and REQUIRES up to `max_hops`, one path per
                        (service, route), with scope / extras / version on every step
  service            -> node facts, direct dependencies, policy clauses that mention its class, exceptions
It returns paths and facts; it does NOT drop dev-only or extra-only routes: the reader decides.
"""
from __future__ import annotations

import hashlib
import random
import re
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import networkx as nx

from ..knowledge.documents import KnowledgeDoc
from ..schema import Evidence
from .org import in_spec, spec_text

_DEP = re.compile(r"^\s+- ([A-Za-z0-9_.\-]+)(?:==([0-9][\w.]*))?(.*)$")


def _prov(doc: str, line: int) -> Dict:
    return {"doc": doc, "line": line}


def build_kg(docs: Sequence[KnowledgeDoc]) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    by_type = defaultdict(list)
    for d in docs:
        by_type[d.doc_id.split(":", 1)[0]].append(d)

    def node(n, **attrs):
        if n not in G:
            G.add_node(n, **attrs)
        else:
            G.nodes[n].update(attrs)

    for d in by_type["pkg"]:
        L = d.text.splitlines()
        name, ver = L[0].split(": ", 1)[1], L[1].split(": ", 1)[1]
        pv = f"pv:{name}@{ver}"
        node(f"pkg:{name}", kind="pkg", name=name, doc=d.doc_id)
        node(pv, kind="pv", name=name, version=ver, doc=d.doc_id, maintainer=L[2].split(": ", 1)[1])
        G.add_edge(pv, f"pkg:{name}", key="VERSION_OF", **_prov(d.doc_id, 2))
        for i, line in enumerate(L[4:], 5):
            m = _DEP.match(line)
            if not m:
                continue
            extra = re.search(r"only with extra: (\w+)", m.group(3) or "")
            G.add_edge(pv, f"pv:{m.group(1)}@{m.group(2)}", key="REQUIRES", extra=extra.group(1) if extra else "",
                       **_prov(d.doc_id, i))
    for d in by_type["svc"]:
        L = d.text.splitlines()
        kv = {}
        for i, line in enumerate(L, 1):
            if ": " in line and not line.startswith(" "):
                k, v = line.split(": ", 1)
                kv[k] = (v, i)
        s = f"svc:{kv['service'][0]}"
        node(s, kind="svc", name=kv["service"][0], tier=int(kv["tier"][0]), internet_facing=kv["internet_facing"][0] == "yes",
             handles_pii=kv["handles_pii"][0] == "yes", status=kv["status"][0], team=kv["team"][0], doc=d.doc_id,
             plugins_from_config=any(x.startswith("plugins:") for x in L), runtime=kv["runtime"][0])
        node(f"team:{kv['team'][0]}", kind="team", name=kv["team"][0])
        G.add_edge(s, f"team:{kv['team'][0]}", key="OWNED_BY", **_prov(d.doc_id, kv["team"][1]))
        if kv["calls"][0] != "none":
            for c in kv["calls"][0].split(", "):
                G.add_edge(s, f"svc:{c}", key="CALLS", **_prov(d.doc_id, kv["calls"][1]))
        for i, line in enumerate(L, 1):
            m = _DEP.match(line)
            if not m:
                continue
            rest = m.group(3) or ""
            sc = re.search(r"\((runtime|dev)(?:; extras: ([^)]*))?\)\s*$", rest)
            scope = sc.group(1) if sc else "runtime"
            extras = [x.strip() for x in sc.group(2).split(",")] if sc and sc.group(2) else []
            if m.group(2):
                tgt = f"pv:{m.group(1)}@{m.group(2)}"
            else:
                tgt = f"pkg:{m.group(1)}"
                node(tgt, kind="pkg", name=m.group(1))
            G.add_edge(s, tgt, key="DEPENDS_ON", scope=scope, extras=extras, unpinned=not m.group(2),
                       **_prov(d.doc_id, i))
    versions = defaultdict(list)
    for n, a in G.nodes(data=True):
        if a.get("kind") == "pv":
            versions[a["name"]].append(a["version"])
    for d in by_type["adv"]:
        L = d.text.splitlines()
        aid, pkg = L[0].split(": ", 1)[1], L[1].split(": ", 1)[1]
        spec = L[2].split(": ", 1)[1]
        spec = "*" if spec == "all versions" else spec
        node(f"adv:{aid}", kind="adv", name=aid, spec=spec, published=L[3].split(": ", 1)[1], summary=L[5][9:], doc=d.doc_id)
        node(f"pkg:{pkg}", kind="pkg", name=pkg)
        G.add_edge(f"adv:{aid}", f"pkg:{pkg}", key="AFFECTS", spec=spec, **_prov(d.doc_id, 3))
        for v in versions.get(pkg, []):
            if in_spec(v, spec):
                G.add_edge(f"adv:{aid}", f"pv:{pkg}@{v}", key="MATCHES", spec=spec, **_prov(d.doc_id, 3))
    for d in by_type["pol"]:
        L = d.text.splitlines()
        head = L[0].split(": ", 1)[1]
        m = re.match(r"(\S+) version (\d+)", head)
        pid = f"pol:{m.group(1)}-v{m.group(2)}" if m else f"pol:{head}"
        status = L[1].split(": ", 1)[1]
        windows, rules, dev_exempt, adv_ban = [], [], None, False
        for i, line in enumerate(L, 1):
            w = re.match(r"\s+3\.\d+ (.+): (\d+) hours", line)
            if w:
                windows.append((w.group(1), int(w.group(2)), i))
            b = re.match(r"\s+(4\.\d+) (\S+) (.+?) is banned \((.*)\)", line)
            if b:
                spec = "*" if b.group(3) == "all versions" else b.group(3)
                node(f"pkg:{b.group(2)}", kind="pkg", name=b.group(2))
                G.add_edge(pid, f"pkg:{b.group(2)}", key="BANS", spec=spec, clause=b.group(1), reason=b.group(4),
                           **_prov(d.doc_id, i))
            if "listed in an open ACME security advisory is banned" in line:
                adv_ban = True
            if line.strip().startswith("2.3"):
                dev_exempt = "do not apply" in line
            r = re.match(r"\s+rule (\d+): tokens of (\S+) services expire after at most (\d+) days", line)
            if r:
                rules.append((r.group(2), int(r.group(3)), i))
            s = re.match(r"supersedes: (\S+) v(\d+)", line)
            if s:
                G.add_edge(pid, f"pol:{s.group(1)}-v{s.group(2)}", key="SUPERSEDES", **_prov(d.doc_id, i))
        node(pid, kind="pol", name=pid[4:], status=status, effective=L[2].split(": ", 1)[1], windows=windows,
             token_rules=rules, dev_exempt=dev_exempt, advisory_ban=adv_ban, doc=d.doc_id)
    for d in by_type["adr"]:
        L = d.text.splitlines()
        kv = {line.split(": ", 1)[0]: (line.split(": ", 1)[1], i) for i, line in enumerate(L, 1) if ": " in line}
        rid = f"adr:{kv['record'][0]}"
        pk, _, spec = kv["package"][0].partition(" ")
        spec = "*" if spec == "all versions" else spec
        node(rid, kind="adr", name=kv["record"][0], until=kv["valid until"][0], decided=kv["decided"][0],
             approver=kv["approved by"][0], doc=d.doc_id)
        G.add_edge(rid, f"svc:{kv['service'][0]}", key="EXEMPTS", package=pk, spec=spec, until=kv["valid until"][0],
                   **_prov(d.doc_id, kv["service"][1]))
        G.add_edge(rid, f"pkg:{pk}", key="EXEMPTS", service=kv["service"][0], spec=spec, until=kv["valid until"][0],
                   **_prov(d.doc_id, kv["package"][1]))
    names = sorted((a["name"] for _, a in G.nodes(data=True) if a.get("kind") == "pkg"), key=len, reverse=True)
    for d in by_type["inc"]:
        L = d.text.splitlines()
        iid = f"inc:{L[0].split(': ', 1)[1]}"
        node(iid, kind="inc", name=iid[4:], date=L[1].split(": ", 1)[1], summary=L[3][9:], doc=d.doc_id)
        for s in L[2].split(": ", 1)[1].split(", "):
            G.add_edge(iid, f"svc:{s}", key="INVOLVES", **_prov(d.doc_id, 3))
        for n in names:
            if re.search(rf"(?<![\w-]){re.escape(n)}(?![\w-])", L[3]):
                G.add_edge(iid, f"pkg:{n}", key="MENTIONS", **_prov(d.doc_id, 4))
    return G


# ---------------------------------------------------------------------------------- rendering
def _svc_desc(G, s: str) -> str:
    a = G.nodes[s]
    if a.get("kind") != "svc" or "tier" not in a:
        return s[4:]
    bits = [f"tier {a['tier']}", "internet-facing" if a["internet_facing"] else "internal",
            "handles PII" if a["handles_pii"] else "no PII", a["status"], f"team {a['team']}"]
    return f"{a['name']} ({', '.join(bits)})"


def _label(G, n: str) -> str:
    a = G.nodes[n]
    if a.get("kind") == "pv":
        return f"{a['name']}=={a['version']}"
    if a.get("kind") == "svc":
        return _svc_desc(G, n)
    return a.get("name", n.split(":", 1)[1])


def _edge_text(G, u: str, v: str, k: str, d: Dict) -> str:
    if k == "DEPENDS_ON":
        bits = [d["scope"]] + ([f"extras enabled: {', '.join(d['extras'])}"] if d.get("extras") else []) + \
               (["unpinned, version not recorded"] if d.get("unpinned") else [])
        return f"-DEPENDS_ON ({'; '.join(bits)})->"
    if k == "REQUIRES":
        return f"-REQUIRES (only with extra {d['extra']})->" if d.get("extra") else "-REQUIRES->"
    return f"-{k}->"


def edge_sentence(G, u: str, v: str, k: str, d: Dict) -> str:
    """One self-contained sentence per edge (E3a / E3b)."""
    src = G.nodes[u].get("name", u) if G.nodes[u].get("kind") != "pv" else _label(G, u)
    dst = _label(G, v) if G.nodes[v].get("kind") == "pv" else G.nodes[v].get("name", v.split(":", 1)[1])
    if k == "DEPENDS_ON":
        bits = [d["scope"]] + ([f"extras enabled: {', '.join(d['extras'])}"] if d.get("extras") else []) + \
               (["unpinned, version not recorded"] if d.get("unpinned") else [])
        txt = f"Service {src} depends on {dst} ({'; '.join(bits)})."
    elif k == "REQUIRES":
        txt = f"Package {src} requires {dst}" + (f" only with extra {d['extra']}." if d.get("extra") else ".")
    elif k == "MATCHES":
        txt = f"Advisory {src} applies to {dst} (affected versions {spec_text(d['spec'])})."
    elif k == "AFFECTS":
        txt = f"Advisory {src} affects package {dst}, versions {spec_text(d['spec'])}."
    elif k == "BANS":
        pa = G.nodes[u]
        txt = (f"Policy {src} ({pa.get('status')}) clause {d['clause']} bans {dst} {spec_text(d['spec'])} ({d['reason']}).")
    elif k == "EXEMPTS":
        what = f"service {dst}" if G.nodes[v].get("kind") == "svc" else f"package {dst} {spec_text(d['spec'])}"
        txt = f"Exception {src} covers {what}, valid until {d['until']}."
    elif k == "SUPERSEDES":
        txt = f"Policy {src} supersedes {dst}."
    elif k == "OWNED_BY":
        txt = f"Service {src} is owned by team {dst}."
    elif k == "CALLS":
        txt = f"Service {src} calls service {dst}."
    elif k == "INVOLVES":
        txt = f"Incident {src} involves service {dst}."
    elif k == "MENTIONS":
        txt = f"Incident {src} mentions package {dst}."
    else:
        txt = f"{src} {k} {dst}."
    return f"{txt} [{d['doc']}:{d['line']}]"


def node_facts(G, n: str, topics: Sequence[str] = ()) -> List[Tuple[str, List[str]]]:
    """(sentence, docs) facts stored on a node."""
    a = G.nodes[n]
    out = []
    if a.get("kind") == "svc" and "tier" in a:
        out.append((f"Service {_svc_desc(G, n)}; runtime {a['runtime']}"
                    + ("; loads additional plugin packages from runtime configuration that is not in the repository"
                       if a["plugins_from_config"] else "") + f". [{a['doc']}]", [a["doc"]]))
    elif a.get("kind") == "adv":
        out.append((f"Advisory {a['name']} (open, published {a['published']}): affected versions {spec_text(a['spec'])}; "
                    f"{a['summary']} [{a['doc']}:2]", [a["doc"]]))
    elif a.get("kind") == "pol":
        if "windows" in topics and a["windows"]:
            out.append((f"Policy {a['name']} ({a['status']}, effective {a['effective']}) patch windows, first matching class "
                        "wins: " + "; ".join(f"{c}: {h} hours" for c, h, _ in a["windows"]) + f". [{a['doc']}]", [a["doc"]]))
        if "dev" in topics and a.get("dev_exempt") is not None:
            out.append((f"Policy {a['name']} ({a['status']}): "
                        + ("dev-scope dependencies are exempt from bans and patch windows" if a["dev_exempt"]
                           else "rules apply to runtime and dev dependencies alike") + f". [{a['doc']}]", [a["doc"]]))
        if "bans" in topics and a.get("windows"):         # the complete ban list of this policy version
            bans = [f"{d['clause']} {G.nodes[v].get('name')} {spec_text(d['spec'])}"
                    for _u, v, k, d in G.out_edges(n, keys=True, data=True) if k == "BANS"]
            if a.get("advisory_ban"):
                bans.append("every package version listed in an open advisory")
            out.append((f"Policy {a['name']} ({a['status']}, effective {a['effective']}) bans: "
                        + ("; ".join(bans) if bans else "nothing") + f". [{a['doc']}]", [a["doc"]]))
        if "tokens" in topics and a["token_rules"]:
            out.append((f"Policy {a['name']} ({a['status']}, effective {a['effective']}): "
                        + "; ".join(f"tokens of {c} services expire after at most {dd} days" for c, dd, _ in a["token_rules"])
                        + f". [{a['doc']}]", [a["doc"]]))
    elif a.get("kind") == "adr":
        out.append((f"Exception record {a['name']}: decided {a['decided']}, valid until {a['until']}, approved by "
                    f"{a['approver']}. [{a['doc']}]", [a["doc"]]))
    return out


# ---------------------------------------------------------------------------------- retrieval
class KGIndex:
    def __init__(self, G: nx.MultiDiGraph):
        self.G = G
        names = []
        for n, a in G.nodes(data=True):
            if a.get("kind") in ("svc", "pkg", "adv", "adr", "team", "inc"):
                names.append((a.get("name", n.split(":", 1)[1]), n))
            elif a.get("kind") == "pol":
                names.append((a["name"].rsplit("-v", 1)[0] if re.search(r"-v\d+$", a["name"]) else a["name"], n))
        by: Dict[str, List[str]] = defaultdict(list)
        for name, n in names:
            by[name].append(n)
        self.names = sorted(by.items(), key=lambda x: -len(x[0]))
        self._facts_docs: Optional[List[KnowledgeDoc]] = None

    def link(self, text: str) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = defaultdict(list)
        taken = []
        for name, nodes in self.names:
            for m in re.finditer(rf"(?<![\w-]){re.escape(name)}(?![\w-])", text):
                if any(s <= m.start() < e for s, e in taken):
                    continue
                taken.append((m.start(), m.end()))
                for n in nodes:
                    kind = self.G.nodes[n].get("kind")
                    if n not in out[kind]:
                        out[kind].append(n)
                    if kind == "pkg":
                        v = re.match(r"==([0-9][\w.]*)", text[m.end():])
                        if v and f"pv:{name}@{v.group(1)}" in self.G and f"pv:{name}@{v.group(1)}" not in out["pv"]:
                            out["pv"].append(f"pv:{name}@{v.group(1)}")
        return dict(out)

    def _reverse_paths(self, targets: Sequence[str], max_hops: int, only_svc: Optional[set] = None,
                       cap: int = 400) -> List[List[Tuple[str, str, str, Dict]]]:
        """Paths [svc -> ... -> target] as edge lists, found by walking DEPENDS_ON / REQUIRES backwards."""
        G, out = self.G, []
        stack = [(t, []) for t in targets]
        while stack and len(out) < cap:
            n, suffix = stack.pop()
            if len(suffix) >= max_hops:
                continue
            for u, _v, k, d in G.in_edges(n, keys=True, data=True):
                if k not in ("DEPENDS_ON", "REQUIRES"):
                    continue
                path = [(u, n, k, d)] + suffix
                if k == "DEPENDS_ON":
                    if only_svc is None or u in only_svc:
                        out.append(path)
                else:
                    stack.append((u, path))
        return out

    def evidence(self, qid: str, question: str, max_hops: int = 4, max_items: int = 20,
                 budget: int = 2000, count_tokens: Callable[[str], int] = lambda s: len(s) // 3 + 1
                 ) -> Tuple[List[Evidence], Dict]:
        G = self.G
        ents = self.link(question)
        ql = question.lower()
        topics = [t for t, kws in (("windows", ("patch", "window", "hours")), ("tokens", ("token",)),
                                   ("dev", ("dev", "development")), ("bans", ("deploy", "ban", "allowed", "may ", "exception",
                                                                            "policy", "permitted")))
                  if any(k in ql for k in kws)]
        items: List[Tuple[str, List[Tuple], List[str], str]] = []    # (text, edges, docs, kind)
        seen_facts = set()

        def fact(text, docs):
            if text not in seen_facts:
                seen_facts.add(text); items.append((text, [], docs, "fact"))

        svcs = set(ents.get("svc", []))
        targets: List[Tuple[str, List[str]]] = []
        for a in ents.get("adv", []):
            for t, dd in node_facts(G, a):
                fact(t, dd)
            pvs = [v for _u, v, k in G.out_edges(a, keys=True) if k == "MATCHES"]
            pkg = [v for _u, v, k in G.out_edges(a, keys=True) if k == "AFFECTS"]
            targets.append((a, pvs + pkg))                      # pkg: unpinned dependencies on the package
        for p in ents.get("pkg", []):
            if any(G.has_edge(a, p, "AFFECTS") for a in ents.get("adv", [])):
                continue
            pvs = [x for x in ents.get("pv", []) if x.startswith(f"pv:{G.nodes[p]['name']}@")] or \
                  [u for u, _v, k in G.in_edges(p, keys=True) if k == "VERSION_OF"]
            targets.append((p, pvs + [p]))
            for u, _v, k, d in G.in_edges(p, keys=True, data=True):          # advisories, bans, exceptions
                if k in ("AFFECTS", "BANS", "EXEMPTS"):
                    fact(edge_sentence(G, u, p, k, d), [d["doc"]])
                    for t, dd in node_facts(G, u, topics):
                        fact(t, dd)
            for u, _v, k, d in G.in_edges(p, keys=True, data=True):
                if k == "MENTIONS":
                    fact(edge_sentence(G, u, p, k, d), [d["doc"]])
        for s in sorted(svcs):
            for t, dd in node_facts(G, s):
                fact(t, dd)
            if not targets:
                for _u, v, k, d in G.out_edges(s, keys=True, data=True):
                    if k in ("DEPENDS_ON", "OWNED_BY"):
                        fact(edge_sentence(G, s, v, k, d), [d["doc"]])
            for u, _v, k, d in G.in_edges(s, keys=True, data=True):
                if k == "EXEMPTS":
                    fact(edge_sentence(G, u, s, k, d), [d["doc"]])
                    for t, dd in node_facts(G, u):
                        fact(t, dd)
        if svcs or ents.get("pol") or topics:
            for n, a in G.nodes(data=True):
                if a.get("kind") == "pol" and (topics or n in ents.get("pol", [])):
                    for t, dd in node_facts(G, n, topics or ("windows", "tokens", "dev", "bans")):
                        fact(t, dd)
                    for _u, v, k, d in G.out_edges(n, keys=True, data=True):
                        if k == "SUPERSEDES":
                            fact(edge_sentence(G, n, v, k, d), [d["doc"]])
        paths = []
        for src, pvs in targets:
            for p in self._reverse_paths(pvs, max_hops, svcs or None):
                paths.append((src, p))
        paths.sort(key=lambda x: (len(x[1]), x[1][0][0], x[1][-1][1]))
        for src, p in paths:
            s = p[0][0]
            txt = _label(G, s)
            docs = []
            for u, v, k, d in p:
                txt += f" {_edge_text(G, u, v, k, d)} {_label(G, v)}"
                docs.append(f"{d['doc']}:{d['line']}")
            if G.nodes[src].get("kind") == "adv":
                last = p[-1][1]
                md = G.get_edge_data(src, last, "MATCHES")
                if md:
                    txt += f" | {G.nodes[src]['name']} applies to this version"
                    docs.append(f"{md['doc']}:{md['line']}")
            want_team = "team" in ql
            if want_team:
                for _u, v, k, d in G.out_edges(s, keys=True, data=True):
                    if k == "OWNED_BY":
                        docs.append(f"{d['doc']}:{d['line']}")
            items.append((txt + " | sources: " + ", ".join(docs), list(p), [x.rsplit(":", 1)[0] for x in docs], "path"))
        if not items:
            items.append(("No entity in the question matches the knowledge graph (no service, package, advisory or "
                          "policy with that name).", [], [], "empty"))
        out, used, cut = [], 0, 0
        n_paths = sum(1 for it in items if it[3] == "path")
        for txt, edges, docs, kind in items:
            t = count_tokens(txt) + 3
            if len(out) >= max_items or used + t > budget:
                cut += 1
                continue
            used += t
            out.append(Evidence(evidence_id=f"G{len(out) + 1}", sample_id=qid, source_type="graph_path", source_ref=kind,
                                content=txt, meta={"docs": sorted(set(docs)), "kind": kind,
                                                   "edges": [(u, v, k) for u, v, k, _d in edges]}))
        if cut:
            out.append(Evidence(evidence_id=f"G{len(out) + 1}", sample_id=qid, source_type="graph_path", source_ref="note",
                                content=f"Note: {cut} more graph item(s) were found but not shown (budget).",
                                meta={"docs": [], "kind": "note", "edges": []}))
        diag = {"linked": {k: len(v) for k, v in ents.items()}, "n_paths": n_paths, "n_items": len(items), "cut": cut,
                "topics": topics}
        return out, diag

    def flatten(self, qid: str, items: Sequence[Evidence], seed_key: str) -> List[Evidence]:
        """E3a: exactly the edges and facts of E3's items, one sentence each, in a seeded random order."""
        G = self.G
        sents, seen = [], set()
        for it in items:
            if it.meta.get("kind") != "path":
                if it.content not in seen:
                    seen.add(it.content); sents.append((it.content, it.meta.get("docs", [])))
                continue
            for u, v, k in it.meta["edges"]:
                d = G.get_edge_data(u, v, k)
                s = edge_sentence(G, u, v, k, d)
                if s not in seen:
                    seen.add(s); sents.append((s, [d["doc"]]))
            if "applies to this version" in it.content:
                last = it.meta["edges"][-1][1]
                for a, _v, k, d in G.in_edges(last, keys=True, data=True):
                    if k == "MATCHES" and f"{G.nodes[a]['name']} applies" in it.content:
                        s = edge_sentence(G, a, last, k, d)
                        if s not in seen:
                            seen.add(s); sents.append((s, [d["doc"]]))
            s0 = it.meta["edges"][0][0] if it.meta["edges"] else None
            if s0 and "(tier" in it.content.split(" -")[0]:
                for t, dd in node_facts(G, s0):
                    if t not in seen:
                        seen.add(t); sents.append((t, dd))
        rng = random.Random(int(hashlib.sha256(seed_key.encode()).hexdigest()[:12], 16))
        rng.shuffle(sents)
        return [Evidence(evidence_id=f"F{j + 1}", sample_id=qid, source_type="fact", source_ref="e3a",
                         content=t, meta={"docs": sorted(set(dd)), "kind": "fact"}) for j, (t, dd) in enumerate(sents)]

    def fact_docs(self) -> List[KnowledgeDoc]:
        """E3b: every edge and node fact of the KG as a retrievable sentence."""
        if self._facts_docs is None:
            G, docs = self.G, []
            for u, v, k, d in G.edges(keys=True, data=True):
                if k == "VERSION_OF":
                    continue
                docs.append(KnowledgeDoc(doc_id=f"kgf:{len(docs)}", source_type="fact", title=k,
                                         text=edge_sentence(G, u, v, k, d), provenance={"docs": [d["doc"]]}))
            for n in G.nodes:
                for t, dd in node_facts(G, n, ("windows", "tokens", "dev", "bans")):
                    docs.append(KnowledgeDoc(doc_id=f"kgf:{len(docs)}", source_type="fact", title="node", text=t,
                                             provenance={"docs": dd}))
            self._facts_docs = docs
        return self._facts_docs


def kg_stats(G: nx.MultiDiGraph) -> Dict:
    from collections import Counter
    return {"nodes": G.number_of_nodes(), "edges": G.number_of_edges(),
            "node_kinds": dict(Counter(a.get("kind", "?") for _, a in G.nodes(data=True))),
            "edge_types": dict(Counter(k for _, _, k in G.edges(keys=True))),
            "edges_missing_provenance": sum(1 for *_x, d in G.edges(data=True) if not d.get("doc"))}
