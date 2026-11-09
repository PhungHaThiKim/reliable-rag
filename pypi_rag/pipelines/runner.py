"""E0..E4 pipelines behind one interface: predict_many(samples) -> [PredictionResult].

  e0              LLM-only: shared package block (metadata + code slices C*)
  e1              + Simple RAG: dense top-k knowledge (K*)
  e2              + Hybrid RAG: dense + BM25, reciprocal-rank fusion (K*)
  e2_bm25         ablation: BM25 only            e2_rerank   ablation: RRF + cross-encoder rerank
  e3              + Behavioral KG-RAG: traversal paths (G*) + matched pattern text (K*)
  e3a_exact       E3a of the test-case document: EXACTLY the edges of E3's paths as separate sentences (F*),
                  shuffled, same pattern docs (K*), same budget; isolates path structure
  e3a             (= E3b in the slides) graph facts as sentences chosen by hybrid retrieval, no traversal
  e4              E3 + verification + abstention     e4_filter / e4_cite   verification ablations
  e0_malllm, e1_malllm_yara, e1_malllm_code      mal-LLM reproduction (first 300 chars of setup.py)
Verification also runs on every method in `keep` mode, as a MEASUREMENT (claim labels, citation
accuracy) without changing the verdict. No method ever receives the label (Sample.public_view()).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from ..analysis import analyze_package, malllm_representation, package_representation
from ..analysis.ast_parser import PackageFacts
from ..generation.output_schema import parse_json_response, parse_malllm_response
from ..generation.prompts import build_messages, malllm_messages
from ..graph import build_behavior_graph, edge_facts, flatten_paths, graph_evidence, graph_stats
from ..knowledge.build_index import builtin_docs
from ..retrieval import Retriever, build_queries
from ..schema import Evidence, PredictionResult, Sample, sha256_text
from ..verification import apply_policy, check_claims, restrict_check

METHODS: Dict[str, Dict] = {
    "e0": {"kind": "llm_only"},
    "e1": {"kind": "text_rag", "mode": "vector"},
    "e2": {"kind": "text_rag", "mode": "rrf"},
    "e2_bm25": {"kind": "text_rag", "mode": "bm25"},
    "e2_rerank": {"kind": "text_rag", "mode": "rrf", "rerank": True},
    "e3": {"kind": "kg_rag"},
    "e3a": {"kind": "facts_rag"},
    "e3a_exact": {"kind": "facts_exact"},
    "e4": {"kind": "kg_rag", "policy": "abstain"},
    "e4_filter": {"kind": "kg_rag", "policy": "filter"},
    "e4_cite": {"kind": "kg_rag", "policy": "cite"},
    "e0_malllm": {"kind": "malllm", "ctx": None},
    "e1_malllm_yara": {"kind": "malllm", "ctx": "yara", "k": 2},
    "e1_malllm_code": {"kind": "malllm", "ctx": "malicious_example", "k": 1},
}


@dataclass
class RunConfig:
    code_tokens: int = 3000             # shared package block budget (every method)
    evidence_tokens: int = 1500         # added evidence budget (E1..E4)
    vector_k: int = 10
    bm25_k: int = 10
    final_k: int = 5
    rrf_k: int = 60
    max_per_source: int = 3
    n_queries: int = 4
    kb_source_types: Optional[List[str]] = None
    max_hops: int = 3
    max_paths: int = 6
    facts_k: int = 8
    policy_min_critical: int = 1
    policy_conflict_severity: Optional[float] = 0.9
    policy_abstain_on_dispatch: bool = False
    dispatch_caveat: bool = True        # tell the reader when dynamic dispatch hides call targets
    use_sample_question: bool = False   # add sample.meta["question"] (test cases) to the prompt
    seed: int = 42


class Context:
    """Shared resources of one run (LLM, KB, retrievers, per-sample facts and graphs)."""

    def __init__(self, llm, cfg: RunConfig, kb_docs: Sequence = (), embedder=None, reranker=None,
                 pyguard_matcher=None, cache_dir: Optional[str] = None):
        self.llm, self.cfg, self.kb_docs = llm, cfg, list(kb_docs)
        self.embedder, self.reranker, self.pyguard_matcher = embedder, reranker, pyguard_matcher
        self.cache_dir = cache_dir
        self._facts: Dict[str, PackageFacts] = {}
        self._graph: Dict[str, object] = {}
        self._retr: Dict[Tuple, Retriever] = {}
        self._pattern_docs = [d for d in builtin_docs() if d.source_type == "pattern"]

    def facts(self, s: Sample) -> PackageFacts:
        if s.sample_id not in self._facts:
            self._facts[s.sample_id] = analyze_package(s)
        return self._facts[s.sample_id]

    def graph(self, s: Sample):
        if s.sample_id not in self._graph:
            self._graph[s.sample_id] = build_behavior_graph(s, self.facts(s), self.pyguard_matcher)
        return self._graph[s.sample_id]

    def retriever(self, mode: str, rerank: bool = False, source_types=None, final_k=None) -> Retriever:
        key = (mode, rerank, tuple(source_types or ()), final_k)
        if key not in self._retr:
            c = self.cfg
            self._retr[key] = Retriever(self.kb_docs, mode=mode, embedder=self.embedder,
                                        reranker=self.reranker if rerank else None, vector_k=c.vector_k,
                                        bm25_k=c.bm25_k, final_k=final_k or c.final_k, rrf_k=c.rrf_k,
                                        max_per_source=c.max_per_source, source_types=source_types or c.kb_source_types,
                                        cache_path=f"{self.cache_dir}/kb_emb" if self.cache_dir else None)
        return self._retr[key]


def _kdoc_evidence(sample_id: str, hits, start: int = 1, prefix: str = "K") -> List[Evidence]:
    out = []
    for j, (d, score, trace) in enumerate(hits):
        out.append(Evidence(evidence_id=f"{prefix}{start + j}", sample_id=sample_id, source_type=d.source_type,
                            source_ref=d.doc_id, content=d.render(), score=score, behaviors=list(d.behaviors),
                            meta={"trace": trace, "provenance": {k: v for k, v in d.provenance.items()
                                                                 if k in ("origin", "label_hint", "split", "spans")}}))
    return out


def build_request(method: str, s: Sample, ctx: Context) -> Dict:
    """Everything up to the LLM call. `s` is the public view (no label)."""
    spec = METHODS[method]
    cfg = ctx.cfg
    facts = ctx.facts(s)
    trace: Dict = {}
    if spec["kind"] == "malllm":
        rep = malllm_representation(s)
        context, hits = "", []
        if spec.get("ctx"):
            r = ctx.retriever("vector", source_types=[spec["ctx"]], final_k=spec["k"])
            hits = r.retrieve([rep["snippet"]], k=spec["k"])
            context = "\n\n".join(d.text for d, _, _ in hits)
        msgs = malllm_messages(rep, context, "YARA rule" if spec.get("ctx") == "yara" else "malicious code")
        trace["retrieved"] = [{"doc_id": d.doc_id, "score": sc, **tr} for d, sc, tr in hits]
        visible = _kdoc_evidence(s.sample_id, hits)
        return {"messages": msgs, "visible": visible, "meta": {"prompt_sha256": sha256_text(str(msgs))},
                "trace": trace, "parser": "malllm"}

    header, code_items, rep_meta = package_representation(s, facts, max_tokens=cfg.code_tokens,
                                                          count_tokens=ctx.llm.count_tokens)
    extra: Dict[str, List[Evidence]] = {}
    queries = build_queries(code_items, facts, n=cfg.n_queries)
    if spec["kind"] == "text_rag":
        r = ctx.retriever(spec["mode"], rerank=spec.get("rerank", False))
        hits = r.retrieve(queries)
        extra["knowledge"] = _kdoc_evidence(s.sample_id, hits)
        trace["retrieved"] = [{"doc_id": d.doc_id, "source_type": d.source_type, "score": sc, "behaviors": d.behaviors,
                               **tr} for d, sc, tr in hits]
    elif spec["kind"] in ("kg_rag", "facts_exact"):
        G = ctx.graph(s)
        items, diag = graph_evidence(G, s, max_paths=cfg.max_paths, max_hops=cfg.max_hops,
                                     max_tokens=int(cfg.evidence_tokens * 0.8), dispatch_caveat=cfg.dispatch_caveat)
        gitems = [i for i in items if i.source_type == "graph_path"]
        extra["knowledge"] = [i for i in items if i.source_type != "graph_path"]
        if spec["kind"] == "kg_rag":
            extra["graph_path"] = gitems
        else:                                             # E3a: the same edges, flattened and shuffled
            extra["fact"] = flatten_paths(G, s, gitems, seed_key=f"{cfg.seed}:{s.sample_id}")
        trace["graph"] = {**diag, **graph_stats(G),
                          "paths": [{"id": i.evidence_id, "apis": i.meta.get("ops_api"), "patterns": i.meta.get("patterns"),
                                     "spans": i.meta.get("spans"), "statuses": i.meta.get("statuses"), "score": i.score,
                                     "via_functions": i.meta.get("via_functions"), "call_sites": i.meta.get("call_sites")}
                                    for i in gitems]}
    elif spec["kind"] == "facts_rag":
        G = ctx.graph(s)
        fdocs = edge_facts(G, s)
        r = Retriever(fdocs + ctx._pattern_docs, mode="rrf", embedder=ctx.embedder, vector_k=cfg.vector_k,
                      bm25_k=cfg.bm25_k, final_k=cfg.facts_k, rrf_k=cfg.rrf_k, max_per_source=None)
        hits = r.retrieve(queries)
        facts_hits = [h for h in hits if h[0].source_type == "fact"]
        pat_hits = [h for h in hits if h[0].source_type != "fact"][:3]
        extra["fact"] = [Evidence(evidence_id=f"F{j + 1}", sample_id=s.sample_id, source_type="fact", source_ref=d.doc_id,
                                  content=d.text, score=sc, behaviors=list(d.behaviors),
                                  file_path=(d.provenance["spans"][0][0] if d.provenance.get("spans") else None),
                                  start_line=(d.provenance["spans"][0][1] if d.provenance.get("spans") else None),
                                  end_line=(d.provenance["spans"][0][1] if d.provenance.get("spans") else None),
                                  meta={"spans": d.provenance.get("spans", []), "etype": d.provenance.get("etype")})
                         for j, (d, sc, _t) in enumerate(facts_hits)]
        extra["knowledge"] = _kdoc_evidence(s.sample_id, pat_hits)
        trace["facts"] = {"n_facts": len(fdocs), "retrieved": [d.doc_id for d, _, _ in hits]}
    question = (s.meta or {}).get("question") if cfg.use_sample_question else None
    msgs, visible, meta = build_messages(header, code_items, extra, ctx.llm.count_tokens, cfg.evidence_tokens,
                                         question=question)
    meta.update({"code_tokens_budget": cfg.code_tokens, "code_truncated": rep_meta["truncated"],
                 "n_code_items": len(code_items), "parse_coverage": rep_meta["parse_coverage"]})
    trace["queries"] = [q[:300] for q in queries]
    return {"messages": msgs, "visible": visible, "meta": meta, "trace": trace, "parser": "json"}


def _finish(method: str, pub: Sequence[Sample], reqs: List[Dict], outs, ctx: Context, run_id: str):
    spec = METHODS[method]
    preds, traces = [], []
    for s, r, o in zip(pub, reqs, outs):
        vis = {e.evidence_id: e for e in r["visible"]}
        parsed = parse_malllm_response(o.text) if r["parser"] == "malllm" else parse_json_response(o.text, vis)
        check = check_claims(parsed["claims"], vis, ctx.facts(s), s)
        verdict, kept, reason = parsed["verdict"], parsed["claims"], ""
        if spec.get("policy"):
            pol = apply_policy(parsed["verdict"], check, vis, spec["policy"], ctx.cfg.policy_min_critical,
                               ctx.cfg.policy_conflict_severity, ctx.cfg.policy_abstain_on_dispatch)
            verdict, reason = pol["verdict"], pol["reason"]
            keep_ids = {id(l) for l in pol["kept_claims"]}
            kept = [c for c, l in zip(parsed["claims"], check["claims"]) if id(l) in keep_ids]
            check = restrict_check(check, pol["kept_claims"])
        preds.append(PredictionResult(
            sample_id=s.sample_id, method=method, verdict=verdict, explanation=parsed["explanation"],
            evidence=[Evidence(**{**e.__dict__, "content": e.content[:800]}) for e in r["visible"]],
            claimed_evidence_ids=parsed["cited_ids"], latency_ms=o.latency_ms, input_tokens=o.input_tokens,
            output_tokens=o.output_tokens, run_id=run_id, confidence=parsed["confidence"], claims=kept,
            raw_response=o.text, prompt_sha256=r["meta"].get("prompt_sha256", ""), parse_ok=parsed["parse_ok"],
            parse_error=parsed["parse_error"], truncated=bool(o.truncated or r["meta"].get("evidence_cut")),
            model=ctx.llm.name, raw_verdict=parsed["verdict"],
            verification={k: v for k, v in check.items()} | {"policy": spec.get("policy", "keep"), "policy_reason": reason},
            extra={"prompt_meta": r["meta"], "cached": o.cached}))
        traces.append({"sample_id": s.sample_id, "method": method, **r["trace"]})
    return preds, traces


def predict_all(methods: Sequence[str], samples: Sequence[Sample], ctx: Context, run_id: str = "",
                progress: Optional[Callable[[str], None]] = None) -> Dict[str, Tuple[List[PredictionResult], List[Dict]]]:
    """All methods in ONE generation pass. Prompts are ordered package by package, so the prompts of one
    package (same system prompt + same package block, different evidence) are adjacent: vLLM's prefix cache
    then computes the shared prefix once. E4 variants reuse E3's prompts (deduplicated by the cache)."""
    pub = [s.public_view() for s in samples]
    t0 = time.time()
    reqs = {m: [build_request(m, s, ctx) for s in pub] for m in methods}
    order = [(i, m) for i in range(len(pub)) for m in methods]
    if progress:
        progress(f"[run] built {len(order)} prompts ({len(methods)} methods x {len(pub)} packages) in "
                 f"{time.time() - t0:.0f}s; generating ...")
    t1 = time.time()
    outs = ctx.llm.generate([reqs[m][i]["messages"] for i, m in order])
    if progress:
        progress(f"[run] generation done in {(time.time() - t1) / 60:.1f} min")
    by = {(i, m): o for (i, m), o in zip(order, outs)}
    return {m: _finish(m, pub, reqs[m], [by[(i, m)] for i in range(len(pub))], ctx, run_id) for m in methods}


def predict_many(method: str, samples: Sequence[Sample], ctx: Context, run_id: str = "",
                 progress: Optional[Callable[[str], None]] = None) -> Tuple[List[PredictionResult], List[Dict]]:
    return predict_all([method], samples, ctx, run_id, progress)[method]
