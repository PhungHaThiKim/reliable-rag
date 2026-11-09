"""Task B methods: the SAME question set, prompt, reader and evidence budget for every method.

  e0          LLM only (the question, plus the code for code questions)
  e1          + dense top-k ACME documents (K*)
  e2          + hybrid (dense + BM25, RRF) documents (K*)        e2_bm25   ablation: BM25 only
  e3          + KG paths and facts (G*) from the graph parsed out of the same documents
  e3a_exact   exactly E3's edges and facts as separate shuffled sentences (F*): path structure removed
  e3a         (= E3b) KG edge sentences chosen by hybrid retrieval, no traversal (F*)
  e4          E3 + answer verification + abstention (policy tuned on the dev org; reuses E3's generation)
Verification also runs on every method in measurement mode (claim support, fabricated entities).
"""
from __future__ import annotations

import json
import re
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from ..generation.output_schema import _first_json_object
from ..generation.prompts import fit_items, render_item
from ..retrieval import Retriever
from ..schema import Evidence, sha256_text
from .kg import KGIndex
from .questions import INSUFFICIENT

QA_METHODS: Dict[str, Dict] = {
    "e0": {"kind": "none"},
    "e1": {"kind": "text", "mode": "vector"},
    "e2": {"kind": "text", "mode": "rrf"},
    "e2_bm25": {"kind": "text", "mode": "bm25"},
    "e3": {"kind": "kg"},
    "e3a_exact": {"kind": "kg_flat"},
    "e3a": {"kind": "kg_facts"},
    "e4": {"kind": "kg", "policy": True},
}

SYSTEM = (
    "You answer questions about the software supply chain of ACME, a company that runs Python services. "
    "Today is {today}.\n"
    "Rules:\n"
    "- The evidence items given to you (C* code, K* ACME documents, G* knowledge-graph paths, F* knowledge-graph facts) "
    "are the only source of truth about ACME. Do not use outside knowledge about ACME.\n"
    "- If the evidence does not let you answer reliably, set \"insufficient_evidence\" to true and give null as the answer "
    "instead of guessing.\n"
    "- Every claim must cite the ids of the items that show it.\n"
    "- Do not name services, packages, advisories, policies or records that appear neither in the question nor in the evidence.\n"
    "Return ONLY one JSON object, no markdown:\n"
    '{{"answer": <in the requested format, or null>, "insufficient_evidence": true or false, "confidence": <number 0-1>, '
    '"claims": [{{"text": "<short claim>", "evidence": ["K1"]}}]}}\n'
    "Use at most 5 claims."
)

FORMATS = {"yes_no": '"yes" or "no"', "service_set": "a JSON list of service names ([] if there are none)",
           "team_set": "a JSON list of team names ([] if there are none)",
           "package_path": "a JSON list of package names in order",
           "hours": "an integer number of hours", "days": "an integer number of days",
           "version": 'a version string such as "1.2.3"', "env_set": "a JSON list of environment variable names",
           "host": 'a host name such as "api.example.com"', "date": 'a date such as "2026-01-31"'}
SECTION = {"knowledge": "ACME documents (retrieved)", "graph_path": "Knowledge graph paths and facts",
           "fact": "Knowledge graph facts"}
ENTITY_RX = re.compile(r"\b(svc-[a-z0-9-]+|acme-[a-z0-9-]+|ADV-\d{4}-\d{3}|POL-[A-Z]{3}-\d{3}|ADR-\d{3}|INC-\d{4}-\d{3})\b")


# ---------------------------------------------------------------------------------- prompt
def build_qa_messages(q: Dict, extra: Dict[str, Sequence[Evidence]], count_tokens: Callable[[str], int],
                      today: str, budget: int = 2000, code_budget: int = 1500) -> Tuple[List[Dict], List[Evidence], Dict]:
    parts = [f"## Question\n{q['question']}\nAnswer format: {FORMATS[q['answer_type']]}"]
    visible: List[Evidence] = []
    if q.get("code"):
        code = q["code"]
        while count_tokens(code) > code_budget:
            code = code[: int(len(code) * 0.9)]
        c1 = Evidence("C1", q["qid"], "code_span", "question_code", f"```python\n{code}```", meta={"docs": ["code"]})
        parts.append("## Code\n" + render_item(c1, 100000)); visible.append(c1)
    remaining, cut_any = budget, False
    for key in ("graph_path", "fact", "knowledge"):
        its = list(extra.get(key, []))
        if not its:
            continue
        kept, cut = fit_items(its, remaining, count_tokens)
        cut_any |= cut
        remaining -= sum(count_tokens(render_item(i)) for i in kept)
        if kept:
            parts.append(f"## {SECTION[key]}\n" + "\n\n".join(render_item(i) for i in kept)); visible += kept
    parts.append("Answer with the JSON object only.")
    sysmsg = SYSTEM.format(today=today)
    user = "\n\n".join(parts)
    meta = {"evidence_tokens": budget - remaining, "evidence_cut": cut_any, "prompt_sha256": sha256_text(sysmsg + user)}
    return [{"role": "system", "content": sysmsg}, {"role": "user", "content": user}], visible, meta


# ---------------------------------------------------------------------------------- parsing
def _norm_list(x) -> List[str]:
    if x is None:
        return []
    if isinstance(x, str):
        x = [s for s in re.split(r"\s*(?:,|->|>|\n|;)\s*", x.strip("[] ")) if s]
    out = []
    for v in x if isinstance(x, list) else [x]:
        s = str(v).strip().strip("\"'` ").strip()
        if s:
            out.append(s)
    return out


def normalize_answer(ans, answer_type: str):
    if ans is None:
        return None
    if answer_type == "yes_no":
        if isinstance(ans, bool):
            return "yes" if ans else "no"
        s = str(ans).strip().lower()
        return "yes" if s.startswith(("yes", "true")) else ("no" if s.startswith(("no", "false")) else None)
    if answer_type in ("service_set", "team_set", "env_set"):
        return sorted({v.lower() if answer_type != "env_set" else v for v in _norm_list(ans)})
    if answer_type == "package_path":
        return [v.lower().split("==")[0] for v in _norm_list(ans)]
    if answer_type in ("hours", "days"):
        s = str(ans).lower()
        m = re.search(r"(\d+(?:\.\d+)?)", s)
        if not m:
            return None
        v = float(m.group(1))
        if answer_type == "hours" and "day" in s:
            v *= 24
        if answer_type == "days" and "hour" in s:
            v /= 24
        return int(round(v))
    s = str(ans).strip().strip("\"'` ")
    if answer_type == "version":
        m = re.search(r"\d+(?:\.\d+)+", s)
        return m.group(0) if m else s.lower()
    return s.lower()


def parse_qa(text: str, answer_type: str, valid_ids: Sequence[str] = ()) -> Dict:
    js = _first_json_object(text or "")
    out = {"answer": None, "insufficient": False, "confidence": None, "claims": [], "parse_ok": False, "parse_error": ""}
    if js is None:
        out["parse_error"] = "no JSON object"
        low = (text or "").lower()
        out["insufficient"] = "insufficient" in low or "cannot determine" in low
        return out
    try:
        d = json.loads(js)
    except json.JSONDecodeError as e:
        out["parse_error"] = f"json: {e}"
        return out
    ins = d.get("insufficient_evidence")
    ins = ins if isinstance(ins, bool) else str(ins).strip().lower() in ("true", "yes", "1")
    raw = d.get("answer")
    if isinstance(raw, str) and re.search(r"insufficient|cannot (be )?determin|unknown|not enough", raw.lower()):
        ins = True
    out["insufficient"] = bool(ins)
    out["answer"] = None if out["insufficient"] else normalize_answer(raw, answer_type)
    if not out["insufficient"] and out["answer"] is None and raw is None:
        out["insufficient"] = True                       # null answer without the flag: treat as abstention
    try:
        out["confidence"] = max(0.0, min(1.0, float(d.get("confidence")) / (100 if float(d.get("confidence")) > 1 else 1)))
    except (TypeError, ValueError):
        out["confidence"] = None
    vs = set(valid_ids)
    for c in (d.get("claims") or [])[:8]:
        if not isinstance(c, dict):
            continue
        ev = [str(e).strip() for e in (c.get("evidence") or []) if str(e).strip()]
        out["claims"].append({"text": str(c.get("text", ""))[:500], "evidence": ev,
                              "invalid_ids": [e for e in ev if e not in vs]})
    out["parse_ok"] = True
    return out


# ---------------------------------------------------------------------------------- verification
def _mentions(text: str, known_lower: set) -> Tuple[List[str], List[str]]:
    """(known entities, entity-looking tokens that are not in the corpus) mentioned in `text`."""
    found, fake = [], []
    for m in ENTITY_RX.finditer(text or ""):
        t = m.group(1)
        (found if t.lower() in known_lower else fake).append(t)
    return found, fake


def verify(q: Dict, parsed: Dict, visible: Dict[str, Evidence], known_lower: set, kinds: Dict[str, set]) -> Dict:
    """Deterministic checks (no labels): claim support, answer support, fabricated entities."""
    qtext = q["question"].lower() + " " + (q.get("code") or "").lower()
    vis_text = " ".join(e.content.lower() for e in visible.values())
    labels = []
    for c in parsed["claims"]:
        cited = [visible[e] for e in c["evidence"] if e in visible]
        found, fake = _mentions(c["text"], known_lower)
        ctext = " ".join(e.content.lower() for e in cited)
        missing = [f for f in found if f.lower() not in ctext and f.lower() not in qtext]
        ok = bool(cited) and not fake and not missing
        labels.append({**c, "label": "SUPPORTED" if ok else "UNSUPPORTED",
                       "why": "" if ok else ("no valid citation" if not cited else
                                             f"not in cited items: {(fake + missing)[:3]}")})
    fabricated = []
    ans = parsed["answer"]
    if not parsed["insufficient"] and ans is not None:
        elems = ans if isinstance(ans, list) else [str(ans)]
        kind = {"service_set": "svc", "team_set": "team", "package_path": "pkg"}.get(q["answer_type"])
        if kind:
            fabricated += [e for e in elems if e.lower() not in kinds[kind] and e.lower() not in qtext]
    for c in parsed["claims"]:
        fabricated += [t for t in _mentions(c["text"], known_lower)[1] if t.lower() not in qtext]
    unsupported_elems = []
    if not parsed["insufficient"] and isinstance(ans, list) and q["answer_type"] in ("service_set", "team_set", "package_path"):
        unsupported_elems = [e for e in ans if e.lower() not in vis_text]
    n_sup = sum(l["label"] == "SUPPORTED" for l in labels)
    return {"claims": labels, "n_claims": len(labels), "n_supported": n_sup, "n_unsupported": len(labels) - n_sup,
            "fabricated": sorted(set(fabricated)), "unsupported_answer_elements": unsupported_elems,
            "support_ratio": n_sup / len(labels) if labels else 0.0}


def apply_qa_policy(parsed: Dict, check: Dict, tau: float = 0.5, filter_elements: bool = True) -> Tuple[Dict, str]:
    """E4: drop answer elements that no visible item mentions; abstain when the answer is not supported."""
    p = dict(parsed)
    if p["insufficient"]:
        return p, ""
    why = ""
    if filter_elements and isinstance(p["answer"], list) and check["unsupported_answer_elements"]:
        drop = {e.lower() for e in check["unsupported_answer_elements"]}
        p["answer"] = [e for e in p["answer"] if e.lower() not in drop]
        why = f"dropped {len(drop)} unsupported element(s)"
    if check["fabricated"] or (tau > 0 and check["support_ratio"] < tau):
        p["insufficient"], p["answer"] = True, None
        why = (why + "; " if why else "") + (f"fabricated {check['fabricated'][:2]}" if check["fabricated"]
                                             else f"support ratio {check['support_ratio']:.2f} < {tau}")
    return p, why


# ---------------------------------------------------------------------------------- context
class QAContext:
    def __init__(self, llm, docs, kg: KGIndex, embedder, today: str, known: Sequence[str], kinds: Dict[str, set],
                 k_text: int = 8, k_facts: int = 16, max_hops: int = 4, max_items: int = 20, evidence_tokens: int = 2000,
                 cache_dir: Optional[str] = None, seed: int = 42):
        self.llm, self.docs, self.kg, self.embedder, self.today = llm, list(docs), kg, embedder, today
        self.known_lower = {k.lower() for k in known}
        self.kinds = kinds
        self.k_text, self.k_facts, self.max_hops, self.max_items = k_text, k_facts, max_hops, max_items
        self.evidence_tokens, self.cache_dir, self.seed = evidence_tokens, cache_dir, seed
        self._retr: Dict[str, Retriever] = {}

    def retriever(self, mode: str) -> Retriever:
        if mode not in self._retr:
            self._retr[mode] = Retriever(self.docs, mode=mode, embedder=self.embedder, final_k=self.k_text,
                                         vector_k=max(10, self.k_text * 2), bm25_k=max(10, self.k_text * 2),
                                         max_per_source=None,
                                         cache_path=f"{self.cache_dir}/sc_emb" if self.cache_dir and mode != "bm25" else None)
        return self._retr[mode]

    def facts_retriever(self) -> Retriever:
        if "facts" not in self._retr:
            self._retr["facts"] = Retriever(self.kg.fact_docs(), mode="rrf", embedder=self.embedder, final_k=self.k_facts,
                                            vector_k=self.k_facts * 2, bm25_k=self.k_facts * 2, max_per_source=None,
                                            cache_path=f"{self.cache_dir}/sc_fact_emb" if self.cache_dir else None)
        return self._retr["facts"]


def build_request(method: str, q: Dict, ctx: QAContext) -> Dict:
    spec = QA_METHODS[method]
    extra: Dict[str, List[Evidence]] = {}
    trace: Dict = {}
    if spec["kind"] == "text":
        hits = ctx.retriever(spec["mode"]).retrieve([q["question"]])
        extra["knowledge"] = [Evidence(f"K{j + 1}", q["qid"], d.source_type, d.doc_id, d.render(1500), score=sc,
                                       meta={"docs": [d.doc_id]}) for j, (d, sc, _t) in enumerate(hits)]
        trace["retrieved"] = [d.doc_id for d, _, _ in hits]
    elif spec["kind"] in ("kg", "kg_flat"):
        items, diag = ctx.kg.evidence(q["qid"], q["question"], ctx.max_hops, ctx.max_items,
                                      int(ctx.evidence_tokens * 0.95), ctx.llm.count_tokens)
        if spec["kind"] == "kg":
            extra["graph_path"] = items
        else:
            extra["fact"] = ctx.kg.flatten(q["qid"], items, seed_key=f"{ctx.seed}:{q['qid']}")
        trace["kg"] = diag
    elif spec["kind"] == "kg_facts":
        hits = ctx.facts_retriever().retrieve([q["question"]])
        extra["fact"] = [Evidence(f"F{j + 1}", q["qid"], "fact", d.doc_id, d.text, score=sc,
                                  meta={"docs": d.provenance.get("docs", [])}) for j, (d, sc, _t) in enumerate(hits)]
        trace["retrieved"] = [d.doc_id for d, _, _ in hits]
    msgs, visible, meta = build_qa_messages(q, extra, ctx.llm.count_tokens, ctx.today, ctx.evidence_tokens)
    return {"messages": msgs, "visible": visible, "meta": meta, "trace": trace}


def predict_qa(methods: Sequence[str], questions: Sequence[Dict], ctx: QAContext, policy: Optional[Dict] = None,
               progress: Optional[Callable[[str], None]] = None) -> Dict[str, List[Dict]]:
    """All methods in one generation pass, prompts ordered question by question (shared prefix)."""
    policy = policy or {"tau": 0.5, "filter_elements": True}
    t0 = time.time()
    shared = "e4" in methods and "e3" in methods          # E4 = E3's prompt and generation + the policy
    gen = [m for m in methods if not (shared and m == "e4")]
    reqs = {m: [build_request(m, q, ctx) for q in questions] for m in gen}
    if shared:
        reqs["e4"] = reqs["e3"]
    order = [(i, m) for i in range(len(questions)) for m in gen]
    if progress:
        progress(f"[sc] built {len(order)} prompts ({len(methods)} methods x {len(questions)} questions) in {time.time() - t0:.0f}s")
    outs = ctx.llm.generate([reqs[m][i]["messages"] for i, m in order])
    by = {(i, m): o for (i, m), o in zip(order, outs)}
    res: Dict[str, List[Dict]] = {}
    for m in methods:
        src = "e3" if (shared and m == "e4") else m
        rows = []
        for i, q in enumerate(questions):
            r, o = reqs[m][i], by[(i, src)]
            vis = {e.evidence_id: e for e in r["visible"]}
            parsed = parse_qa(o.text, q["answer_type"], list(vis))
            check = verify(q, parsed, vis, ctx.known_lower, ctx.kinds)
            final, why = parsed, ""
            if QA_METHODS[m].get("policy"):
                final, why = apply_qa_policy(parsed, check, policy.get("tau", 0.5), policy.get("filter_elements", True))
            rows.append({"qid": q["qid"], "method": m, "answer": final["answer"], "insufficient": final["insufficient"],
                         "raw_answer": parsed["answer"], "raw_insufficient": parsed["insufficient"],
                         "confidence": parsed["confidence"], "claims": check["claims"], "check": {k: v for k, v in check.items() if k != "claims"},
                         "policy_reason": why, "parse_ok": parsed["parse_ok"], "parse_error": parsed["parse_error"],
                         "visible": [{"id": e.evidence_id, "docs": e.meta.get("docs", []), "content": e.content[:600]}
                                     for e in r["visible"]],
                         "raw_response": o.text, "input_tokens": o.input_tokens, "output_tokens": o.output_tokens,
                         "latency_ms": o.latency_ms, "evidence_tokens": r["meta"]["evidence_tokens"],
                         "evidence_cut": r["meta"]["evidence_cut"], "trace": r["trace"]})
        res[m] = rows
    return res


def reapply_policy(rows_e3: Sequence[Dict], questions: Dict[str, Dict], tau: float, filter_elements: bool) -> List[Dict]:
    """E4 rows from saved E3 rows (offline; used by tuning)."""
    out = []
    for r in rows_e3:
        parsed = {"answer": r["raw_answer"], "insufficient": r["raw_insufficient"]}
        check = dict(r["check"]); check["claims"] = r["claims"]
        final, why = apply_qa_policy(parsed, check, tau, filter_elements)
        out.append({**r, "method": "e4", "answer": final["answer"], "insufficient": final["insufficient"], "policy_reason": why})
    return out
