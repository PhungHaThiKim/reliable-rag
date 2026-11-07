"""Hallucination / groundedness evaluation for KGQA reader outputs.

Separates two things the old `llm_unsupported_rate` conflated:

    faithfulness  — is each asserted answer ENTAILED by the retrieved graph
                    evidence (not merely mentioned somewhere in the context)?
    correctness   — is the answer right w.r.t. gold?

Runs OFFLINE on the per-question records the CLI already writes
(`records_<dataset>_<method>.jsonl` with `llm_pred`, `selected_paths`,
`confidence`, ...) plus the BenchDataset (full subgraph, topic entities, gold).

Verification is two-tier, per asserted entity ("claim"):

  Tier 1  deterministic graph check
          evidence_type ∈ {path_endpoint, in_context_no_proof, in_subgraph_only, absent}
          path_endpoint      the entity terminates a selected path that starts at a
                             topic entity (a graph proof exists in the shown evidence)
          in_context_no_proof entity appears in the context only as an
                             intermediate/other node  (the "Alice works at X, X is in
                             London → born in London" case)
          in_subgraph_only   entity exists in the KG subgraph but was not retrieved
          absent             not in the subgraph at all (parametric / fabricated)
  Tier 2  semantic check of path_endpoint claims
          the path's directed relation sequence is compared with the REFERENCE
          PROOFS of the question (relation sequences topic→gold in the full
          subgraph; gold is used for evaluation only, never shown to the reader).
          same sequence (or the entity is gold) → SUPPORTED, else UNCERTAIN
          (open world: a proof for *some* relation ≠ proof for the asked one).
          An optional LLM judge can turn UNCERTAIN into SUPPORTED / CONTRADICTED.

Claim labels: SUPPORTED | UNSUPPORTED | CONTRADICTED | UNCERTAIN.
Question-level hallucination  h_i = any UNSUPPORTED or CONTRADICTED claim
(`h_strict_i` additionally counts UNCERTAIN).

Metric groups
  A faithfulness : entity_unsupported_rate (legacy proxy), claim_hallucination_rate,
                   question_hallucination_rate, graph_proof_support_rate,
                   contradiction_rate, uncertain_rate
  B correctness  : llm_hit / llm_f1 (unchanged), grounded_correct_rate,
                   grounded_answer_f1, abstention_rate, answer_coverage
  C risk         : hallucination risk–coverage (AURC_h, risk@cov) and
                   correctness risk–coverage, both ranked by the retriever's confidence
plus an error taxonomy per question and paired tests between methods.

CLI:
  python -m kgqa_bench.hallucination --results results_gnn/b08_reader_Qwen2.5-7B-Instruct \
      --eval gnnrag:ckpt/gnn_features_webqsp_test.pkl:webqsp:test \
      --methods gnnrag_all gnnrag_top5 sbert_top5 cgar --ref gnnrag_all --review-sample 200
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import re
from collections import Counter, defaultdict
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from prism.qa_text_eval import extract_answer_entities, match, normalize
from significance import mcnemar, paired_bootstrap

from .data import BenchDataset, BenchQuestion, build_adjacency, shortest_relation_paths

SUPPORTED, UNSUPPORTED, CONTRADICTED, UNCERTAIN = "SUPPORTED", "UNSUPPORTED", "CONTRADICTED", "UNCERTAIN"
LABELS = (SUPPORTED, UNSUPPORTED, CONTRADICTED, UNCERTAIN)

_ABSTAIN_RE = re.compile(
    r"\b(unknown|not enough information|insufficient (information|evidence)|cannot (be )?(determine|answer)"
    r"|can ?not (determine|answer)|no (sufficient |relevant )?(information|evidence)|unable to (determine|answer)"
    r"|does not (contain|provide)|not (provided|mentioned|available))\b", re.I)


# ─────────────────────────────────────────────────────────────────────
# path / entity helpers
# ─────────────────────────────────────────────────────────────────────

def parse_path_text(text: str) -> List[Tuple[str, str, str]]:
    """'h -> r -> t -> r2 -> t2' (grl_rag.path_to_string) → [(h,r,t),(t,r2,t2)]."""
    parts = [p.strip() for p in str(text).split(" -> ")]
    if len(parts) < 3:
        return []
    return [(parts[i], parts[i + 1], parts[i + 2]) for i in range(0, len(parts) - 2, 2)]


def _norm(s: str) -> str:
    return normalize(s)


class EntityIndex:
    """Normalised-name lookup over a set of entity strings (alias-tolerant)."""

    def __init__(self, entities: Sequence[str]):
        self.by_norm: Dict[str, str] = {}
        for e in entities:
            n = _norm(e)
            if n:
                self.by_norm.setdefault(n, e)

    def resolve(self, text: str) -> List[str]:
        """Entities whose normalised name equals, or is contained in / contains,
        the normalised claim text. Exact match wins; containment needs ≥ 3 chars
        on both sides to avoid 'a' ⊂ everything."""
        n = _norm(text)
        if not n:
            return []
        if n in self.by_norm:
            return [self.by_norm[n]]
        out = []
        if len(n) >= 3:
            for k, e in self.by_norm.items():
                if len(k) >= 3 and (k in n or n in k):
                    out.append(e)
        return out


def _directed_relation_sequence(path: List[Tuple[str, str, str]], triple_set: set) -> Tuple[str, ...]:
    """Relation sequence with direction recovered from the ORIGINAL directed
    triples ('inv:r' when the path traverses an edge backwards, 'missing:r'
    when the edge does not exist in the subgraph at all)."""
    seq = []
    for h, r, t in path:
        if (h, r, t) in triple_set:
            seq.append(r)
        elif (t, r, h) in triple_set:
            seq.append("inv:" + r)
        else:
            seq.append("missing:" + r)
    return tuple(seq)


def reference_proofs(q: BenchQuestion, max_hops: int = 3, max_paths_per_target: int = 5
                     ) -> Dict[str, List[Tuple[str, ...]]]:
    """gold answer → list of directed relation sequences (topic → gold) found in
    the FULL subgraph. Evaluation-only: never shown to the reader."""
    adj = build_adjacency(q.triples, undirected=True)
    by_target = shortest_relation_paths(adj, q.q_entities, max_hops, max_paths_per_target=max_paths_per_target)
    out: Dict[str, List[Tuple[str, ...]]] = {}
    for a in q.answers:
        seqs = [tuple(r for (_h, r, _t) in p) for p in by_target.get(a, [])]
        if seqs:
            out[a] = seqs
    return out


def is_abstention(pred: str) -> bool:
    return bool(_ABSTAIN_RE.search(str(pred)))


# ─────────────────────────────────────────────────────────────────────
# optional LLM judge (tier-2 refinement of UNCERTAIN claims)
# ─────────────────────────────────────────────────────────────────────

JUDGE_PROMPT = (
    "You are verifying whether a knowledge-graph evidence path entails an answer.\n"
    "Question: {question}\n"
    "Evidence paths (head -> relation -> tail):\n{paths}\n"
    "Proposed answer entity: {claim}\n\n"
    "Does the evidence, by itself and following the relation meanings and directions, "
    "establish that the proposed entity answers the question? "
    "Reply with exactly one word: SUPPORTED, CONTRADICTED, or UNCERTAIN."
)


class HFJudge:
    """Small wrapper around an HF chat model used ONLY as entailment judge.
    judge(question, paths, claim) → SUPPORTED | CONTRADICTED | UNCERTAIN."""

    def __init__(self, model_name: str, max_new_tokens: int = 8):
        from .llm_reader import HFReader
        self.reader = HFReader(model_name, prompt_type="chat", max_new_tokens=max_new_tokens)

    def __call__(self, question: str, paths: Sequence[str], claim: str) -> str:
        import torch
        rd = self.reader
        prompt = JUDGE_PROMPT.format(question=question, paths="\n".join(paths), claim=claim)
        prompt = rd.tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                            add_generation_prompt=True)
        enc = rd.tok(prompt, return_tensors="pt", truncation=True, max_length=rd.max_context).to(rd.model.device)
        with torch.inference_mode():
            out = rd.model.generate(**enc, max_new_tokens=rd.max_new_tokens, do_sample=False,
                                    pad_token_id=rd.tok.pad_token_id)
        txt = rd.tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).upper()
        for lab in (CONTRADICTED, SUPPORTED, UNCERTAIN):
            if lab in txt:
                return lab
        return UNCERTAIN


def judge_from_spec(spec: Optional[str]) -> Optional[Callable]:
    if spec in (None, "", "none"):
        return None
    if spec.startswith("hf:"):
        return HFJudge(spec.split(":", 1)[1])
    raise ValueError(f"unknown judge spec: {spec}")


# ─────────────────────────────────────────────────────────────────────
# per-question verification
# ─────────────────────────────────────────────────────────────────────

def verify_question(q: BenchQuestion, record: Dict, judge: Optional[Callable] = None,
                    max_hops: int = 3) -> Dict:
    """Verify every asserted entity of `record['llm_pred']` against the selected
    paths of the record and the question's full subgraph. Returns a verdict dict
    (one row per question) with the per-claim detail inside `claims`."""
    pred = record.get("llm_pred", "") or ""
    paths_text: List[str] = list(record.get("selected_paths", []) or [])
    paths = [parse_path_text(t) for t in paths_text]
    triple_set = {(str(h), str(r), str(t)) for h, r, t in q.triples}
    sub_entities = {e for h, _r, t in q.triples for e in (h, t)}
    ctx_entities = {e for p in paths for h, _r, t in p for e in (h, t)}
    topic_norm = {_norm(e) for e in q.q_entities}
    gold = list(q.answers)
    proofs = reference_proofs(q, max_hops=max_hops)
    gold_seqs = {s for seqs in proofs.values() for s in seqs}

    # endpoint → list of (path index, directed relation sequence)
    endpoints: Dict[str, List[Tuple[int, Tuple[str, ...]]]] = defaultdict(list)
    for i, p in enumerate(paths):
        if not p:
            continue
        if _norm(p[0][0]) in topic_norm:
            endpoints[p[-1][2]].append((i, _directed_relation_sequence(p, triple_set)))

    ctx_index, sub_index = EntityIndex(sorted(ctx_entities)), EntityIndex(sorted(sub_entities))
    raw_claims = [c for c in dict.fromkeys(x.strip() for x in extract_answer_entities(pred)) if c]
    abstained = is_abstention(pred) and not any(ctx_index.resolve(c) for c in raw_claims)
    claims: List[Dict] = []
    if not abstained:
        for c in raw_claims:
            if is_abstention(c):
                continue
            is_gold = any(match(c, a) for a in gold)
            in_ctx, in_sub = ctx_index.resolve(c), sub_index.resolve(c)
            proof_paths = [(i, seq) for e in in_ctx for (i, seq) in endpoints.get(e, [])]
            if proof_paths:
                ev = "path_endpoint"
                if is_gold or any(seq in gold_seqs for _i, seq in proof_paths):
                    label, why = SUPPORTED, "endpoint of a topic→entity path whose relation sequence matches a reference proof" \
                        if not is_gold else "endpoint of a topic→entity path; entity is gold"
                else:
                    label, why = UNCERTAIN, "endpoint of a path, but its relation sequence differs from every reference proof"
                    if judge is not None:
                        label = judge(q.question, [paths_text[i] for i, _s in proof_paths], c)
                        why += f"; judge={label}"
            elif in_ctx:
                ev, label = "in_context_no_proof", UNSUPPORTED
                why = "entity appears in the context only as an intermediate/other node (no topic→entity path)"
                if judge is not None:
                    jl = judge(q.question, paths_text, c)
                    if jl == SUPPORTED:
                        label, why = SUPPORTED, why + "; judge=SUPPORTED"
                    elif jl == CONTRADICTED:
                        label, why = CONTRADICTED, why + "; judge=CONTRADICTED"
            elif in_sub:
                ev, label, why = "in_subgraph_only", UNSUPPORTED, "entity is in the KG subgraph but was not retrieved"
            else:
                ev, label, why = "absent", UNSUPPORTED, "entity is not in the subgraph (parametric / fabricated)"
            claims.append({"claim": c, "label": label, "evidence_type": ev, "is_gold": bool(is_gold),
                           "resolved": in_ctx[:3], "proof_path_ids": [i for i, _s in proof_paths][:5],
                           "relation_seqs": [list(s) for _i, s in proof_paths][:3],
                           "legacy_in_context": c.lower() in "\n".join(paths_text).lower(), "why": why})

    n = len(claims)
    cnt = Counter(cl["label"] for cl in claims)
    n_unsup, n_contra, n_unc, n_sup = cnt[UNSUPPORTED], cnt[CONTRADICTED], cnt[UNCERTAIN], cnt[SUPPORTED]
    answered = (n > 0) and not abstained
    llm_hit = int(record.get("llm_hit", 0) or 0)
    gold_hit_supported = [any(cl["label"] == SUPPORTED and match(cl["claim"], a) for cl in claims) for a in gold]
    g_prec = (sum(1 for cl in claims if cl["label"] == SUPPORTED and cl["is_gold"]) / n) if n else 0.0
    g_rec = (sum(gold_hit_supported) / len(gold)) if gold else 0.0
    g_f1 = (2 * g_prec * g_rec / (g_prec + g_rec)) if (g_prec + g_rec) > 0 else 0.0
    hallucinated = int(answered and (n_unsup + n_contra) > 0)
    hallucinated_strict = int(answered and (n_unsup + n_contra + n_unc) > 0)
    v = {
        "qid": q.qid, "method": record.get("method"), "dataset": record.get("dataset"),
        "answered": int(answered), "abstained": int(abstained), "n_claims": n,
        "n_supported": n_sup, "n_unsupported": n_unsup, "n_contradicted": n_contra, "n_uncertain": n_unc,
        "hallucinated": hallucinated, "hallucinated_strict": hallucinated_strict,
        "proof_support": (sum(1 for cl in claims if cl["evidence_type"] == "path_endpoint") / n) if n else None,
        "legacy_unsupported_rate": (sum(1 for cl in claims if not cl["legacy_in_context"]) / n) if n else None,
        "llm_hit": llm_hit, "llm_f1": float(record.get("llm_f1", 0.0) or 0.0),
        "grounded_correct": int(answered and llm_hit == 1 and (n_unsup + n_contra) == 0),
        "grounded_f1": g_f1, "grounded_prec": g_prec, "grounded_rec": g_rec,
        "correct_ungrounded": int(answered and llm_hit == 1 and (n_unsup + n_contra) > 0),
        "hit_ctx": int(record.get("hit_ctx", 0) or 0), "confidence": record.get("confidence"),
        "cost": record.get("cost"), "n_gold": len(gold), "hops": q.meta.get("hops"),
        "answer_in_subgraph": int(any(a in sub_entities for a in gold)),
        "answer_in_pool": int(any(a in set(q.top_candidates) for a in gold)),
        "has_reference_proof": int(bool(gold_seqs)),
        "claims": claims, "llm_pred": pred,
    }
    v["error_category"] = error_category(v)
    return v


def error_category(v: Dict) -> str:
    """First matching category (priority order) — see §6 of the design note."""
    any_wrong_rel = any(cl["evidence_type"] == "in_context_no_proof" for cl in v["claims"])
    any_unsup = (v["n_unsupported"] + v["n_contradicted"]) > 0
    if not v["answered"]:
        return "abstained"
    if v["llm_hit"] == 1 and not any_unsup:
        return "ok_grounded"
    if v["llm_hit"] == 1 and any_unsup:
        return "correct_ungrounded"
    if not v["answer_in_subgraph"]:
        return "missing_kg_evidence"
    if not v["answer_in_pool"]:
        return "candidate_omission"
    if v["hit_ctx"] == 0:
        return "path_selection_failure"
    if any_wrong_rel:
        return "wrong_relation_evidence"
    if any_unsup:
        return "reader_unsupported_generation"
    return "reader_reasoning_failure"


# ─────────────────────────────────────────────────────────────────────
# aggregation
# ─────────────────────────────────────────────────────────────────────

def risk_coverage(confidences: Sequence[float], risks: Sequence[int],
                  grid: Sequence[float] = (0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 1.0)) -> Dict:
    """Rank by confidence (desc) and report risk among the answered top-c fraction.
    AURC = mean over prefixes of cumulative risk (lower is better)."""
    pairs = sorted(zip([float(c) for c in confidences], [int(r) for r in risks]), key=lambda x: -x[0])
    n = len(pairs)
    if n == 0:
        return {"aurc": None, "curve": [], "risk_at": {}}
    cum, auc, curve = 0, 0.0, []
    for i, (_c, r) in enumerate(pairs, 1):
        cum += r
        auc += (cum / i) / n
        curve.append((i / n, cum / i))
    risk_at = {}
    for c in grid:
        k = max(1, int(round(c * n)))
        risk_at[f"{c:.1f}"] = float(sum(r for _c, r in pairs[:k]) / k)
    return {"aurc": float(auc), "curve": curve, "risk_at": risk_at}


def summarize_verdicts(verdicts: List[Dict], strict: bool = False) -> Dict:
    if not verdicts:
        return {"n": 0}
    ans = [v for v in verdicts if v["answered"]]
    n_claims = sum(v["n_claims"] for v in ans)
    hk = "hallucinated_strict" if strict else "hallucinated"
    out = {
        "n": len(verdicts), "n_answered": len(ans),
        "answer_coverage": len(ans) / len(verdicts),
        "abstention_rate": float(np.mean([v["abstained"] for v in verdicts])),
        # A — faithfulness
        "entity_unsupported_rate": float(np.mean([v["legacy_unsupported_rate"] for v in ans
                                                  if v["legacy_unsupported_rate"] is not None])) if ans else None,
        "claim_hallucination_rate": (sum(v["n_unsupported"] + v["n_contradicted"] for v in ans) / n_claims) if n_claims else None,
        "question_hallucination_rate": float(np.mean([v[hk] for v in ans])) if ans else None,
        "question_hallucination_rate_strict": float(np.mean([v["hallucinated_strict"] for v in ans])) if ans else None,
        "graph_proof_support_rate": (sum(1 for v in ans for cl in v["claims"] if cl["evidence_type"] == "path_endpoint") / n_claims) if n_claims else None,
        "contradiction_rate": (sum(v["n_contradicted"] for v in ans) / n_claims) if n_claims else None,
        "uncertain_rate": (sum(v["n_uncertain"] for v in ans) / n_claims) if n_claims else None,
        "claims_per_answer": (n_claims / len(ans)) if ans else 0.0,
        # B — correctness
        "llm_hit": float(np.mean([v["llm_hit"] for v in verdicts])),
        "llm_f1": float(np.mean([v["llm_f1"] for v in verdicts])),
        "grounded_correct_rate": float(np.mean([v["grounded_correct"] for v in verdicts])),
        "grounded_answer_f1": float(np.mean([v["grounded_f1"] for v in verdicts])),
        "correct_ungrounded_rate": float(np.mean([v["correct_ungrounded"] for v in verdicts])),
        "error_taxonomy": dict(Counter(v["error_category"] for v in verdicts)),
        "evidence_types": dict(Counter(cl["evidence_type"] for v in ans for cl in v["claims"])),
        "labels": dict(Counter(cl["label"] for v in ans for cl in v["claims"])),
    }
    # C — risk / coverage ranked by retriever confidence
    conf_ok = [v for v in ans if v.get("confidence") is not None]
    if conf_ok:
        rc_h = risk_coverage([v["confidence"] for v in conf_ok], [v[hk] for v in conf_ok])
        rc_c = risk_coverage([v["confidence"] for v in conf_ok], [1 - v["llm_hit"] for v in conf_ok])
        rc_g = risk_coverage([v["confidence"] for v in conf_ok], [1 - v["grounded_correct"] for v in conf_ok])
        out.update({"aurc_hallucination": rc_h["aurc"], "halluc_risk_at": rc_h["risk_at"],
                    "aurc_incorrect": rc_c["aurc"], "incorrect_risk_at": rc_c["risk_at"],
                    "aurc_not_grounded_correct": rc_g["aurc"],
                    "curves": {"hallucination": rc_h["curve"], "incorrect": rc_c["curve"],
                               "not_grounded_correct": rc_g["curve"]}})
    return out


def compare_methods(va: List[Dict], vb: List[Dict], keys: Sequence[str] = ("hallucinated", "grounded_correct", "llm_hit"),
                    n_bootstrap: int = 10000, answered_only: bool = False) -> Dict[str, Dict]:
    """Paired A − B on common qids: bootstrap CI + McNemar per binary key."""
    A, B = {v["qid"]: v for v in va}, {v["qid"]: v for v in vb}
    qids = sorted(set(A) & set(B))
    if answered_only:
        qids = [q for q in qids if A[q]["answered"] and B[q]["answered"]]
    out = {}
    for k in keys:
        a = [int(A[q][k]) for q in qids]; b = [int(B[q][k]) for q in qids]
        if not a:
            out[k] = {"n": 0}; continue
        out[k] = {"n": len(a), "bootstrap": paired_bootstrap(a, b, n_bootstrap=n_bootstrap),
                  "mcnemar": mcnemar(a, b)}
        out[k]["significant_0.05"] = out[k]["bootstrap"]["p_value"] < 0.05
    return out


def disagreement_table(va: List[Dict], vb: List[Dict], key: str = "hallucinated") -> Dict[str, int]:
    A, B = {v["qid"]: v for v in va}, {v["qid"]: v for v in vb}
    c = Counter()
    for q in set(A) & set(B):
        c[f"a{A[q][key]}_b{B[q][key]}"] += 1
    return dict(c)


# ─────────────────────────────────────────────────────────────────────
# human-review sample
# ─────────────────────────────────────────────────────────────────────

def review_sample(verdicts_by_method: Dict[str, List[Dict]], dataset: BenchDataset, n: int = 200,
                  seed: int = 0, records_by_method: Optional[Dict[str, List[Dict]]] = None) -> List[Dict]:
    """Pick up to `n` (qid, method) rows for human labelling. Priority:
    1) qids where methods disagree on `hallucinated`, 2) UNCERTAIN claims,
    3) stratified random fill over (n_gold>1, hops>1, hit_ctx, llm_hit)."""
    rng = random.Random(seed)
    by_q = {q.qid: q for q in dataset}
    recs = {m: {r["qid"]: r for r in rs} for m, rs in (records_by_method or {}).items()}
    methods = list(verdicts_by_method)
    V = {m: {v["qid"]: v for v in vs} for m, vs in verdicts_by_method.items()}
    common = sorted(set.intersection(*[set(V[m]) for m in methods])) if methods else []
    chosen: List[Tuple[str, str, str]] = []
    seen = set()

    def add(q, m, reason):
        if (q, m) not in seen and len(chosen) < n:
            seen.add((q, m)); chosen.append((q, m, reason))

    dis = [q for q in common if len({V[m][q]["hallucinated"] for m in methods}) > 1]
    rng.shuffle(dis)
    for q in dis:
        for m in methods:
            add(q, m, "method_disagreement")
    unc = [(q, m) for m in methods for q in V[m] if V[m][q]["n_uncertain"] > 0]
    rng.shuffle(unc)
    for q, m in unc:
        add(q, m, "uncertain_claim")
    strata = defaultdict(list)
    for m in methods:
        for q, v in V[m].items():
            strata[(int(v["n_gold"] > 1), int((v.get("hops") or 1) > 1), v["hit_ctx"], v["llm_hit"])].append((q, m))
    keys = sorted(strata)
    while len(chosen) < n and any(strata.values()):
        for k in keys:
            if strata[k]:
                q, m = strata[k].pop(rng.randrange(len(strata[k])))
                add(q, m, "stratified_" + "_".join(map(str, k)))
            if len(chosen) >= n:
                break
    rows = []
    for q, m, reason in chosen:
        v = V[m][q]; bq = by_q.get(q)
        rows.append({
            "qid": q, "method": m, "reason": reason, "question": bq.question if bq else "",
            "gold": " | ".join(bq.answers) if bq else "", "llm_pred": v["llm_pred"],
            "selected_paths": "\n".join((recs.get(m, {}).get(q, {}) or {}).get("selected_paths", [])),
            "auto_claims": json.dumps([{k: cl[k] for k in ("claim", "label", "evidence_type", "is_gold")} for cl in v["claims"]],
                                      ensure_ascii=False),
            "auto_hallucinated": v["hallucinated"], "auto_error_category": v["error_category"],
            "human_hallucinated": "", "human_label_per_claim": "", "human_notes": "",
        })
    return rows


# ─────────────────────────────────────────────────────────────────────
# driver
# ─────────────────────────────────────────────────────────────────────

def evaluate_records(dataset: BenchDataset, records: List[Dict], judge: Optional[Callable] = None,
                     verbose: bool = False) -> List[Dict]:
    by_q = {q.qid: q for q in dataset}
    out = []
    it = records
    if verbose:
        try:
            from tqdm import tqdm; it = tqdm(records, desc="verify", leave=False)
        except ImportError:
            pass
    for r in it:
        q = by_q.get(r["qid"])
        if q is None or "llm_pred" not in r:
            continue
        out.append(verify_question(q, r, judge=judge))
    return out


def _load_records(path: str) -> List[Dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def discover_methods(results_dir: str, ds_key: str) -> List[str]:
    pre = f"records_{ds_key.replace(':', '_')}_"
    ms = []
    for p in sorted(glob.glob(os.path.join(results_dir, pre + "*.jsonl"))):
        m = os.path.basename(p)[len(pre):-len(".jsonl")]
        if "@seed" not in m:
            ms.append(m)
    return ms


def markdown_report(summaries: Dict[str, Dict], comparisons: Dict[str, Dict], ref: str, ds_key: str,
                    strict: bool) -> str:
    L = [f"# Hallucination / groundedness report — `{ds_key}`", "",
         "Claim = one asserted answer entity. Labels: SUPPORTED (graph proof topic→entity whose relation "
         "sequence matches a reference proof, or entity is gold), UNSUPPORTED (not in context / only an "
         "intermediate node / not retrieved / absent), CONTRADICTED (judge), UNCERTAIN (proof of a different "
         "relation; open world). Question hallucinated = any UNSUPPORTED/CONTRADICTED claim"
         + (" **or UNCERTAIN (strict)**" if strict else "") + ".", "",
         "## A. Faithfulness  (lower is better except proof-support)", "",
         "| Method | answered | claims/ans | entity-unsupported (legacy) | claim-HR | question-HR | question-HR strict | proof-support | uncertain | contradicted |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda x, nd=3: "—" if x is None else f"{x:.{nd}f}"
    for m, s in summaries.items():
        L.append(f"| {m} | {s['n_answered']}/{s['n']} | {f(s['claims_per_answer'], 2)} | {f(s['entity_unsupported_rate'])} | "
                 f"{f(s['claim_hallucination_rate'])} | **{f(s['question_hallucination_rate'])}** | {f(s['question_hallucination_rate_strict'])} | "
                 f"{f(s['graph_proof_support_rate'])} | {f(s['uncertain_rate'])} | {f(s['contradiction_rate'])} |")
    L += ["", "## B. Correctness", "",
          "| Method | llm_hit | llm_f1 | grounded-correct | grounded-F1 | correct-but-ungrounded | abstention |",
          "|---|---|---|---|---|---|---|"]
    for m, s in summaries.items():
        L.append(f"| {m} | {f(s['llm_hit'])} | {f(s['llm_f1'])} | **{f(s['grounded_correct_rate'])}** | {f(s['grounded_answer_f1'])} | "
                 f"{f(s['correct_ungrounded_rate'])} | {f(s['abstention_rate'])} |")
    L += ["", "## C. Risk–coverage ranked by retriever confidence (lower is better)", "",
          "| Method | AURC-halluc | halluc@50% | halluc@80% | halluc@100% | AURC-incorrect | incorrect@50% | AURC-not-grounded-correct |",
          "|---|---|---|---|---|---|---|---|"]
    for m, s in summaries.items():
        if "aurc_hallucination" not in s:
            L.append(f"| {m} | — (no confidence) | | | | | | |"); continue
        L.append(f"| {m} | {f(s['aurc_hallucination'])} | {f(s['halluc_risk_at']['0.5'])} | {f(s['halluc_risk_at']['0.8'])} | "
                 f"{f(s['halluc_risk_at']['1.0'])} | {f(s['aurc_incorrect'])} | {f(s['incorrect_risk_at']['0.5'])} | {f(s['aurc_not_grounded_correct'])} |")
    L += ["", "## Error taxonomy (count per question)", ""]
    cats = sorted({c for s in summaries.values() for c in s["error_taxonomy"]})
    L.append("| Method | " + " | ".join(cats) + " |"); L.append("|---|" + "---|" * len(cats))
    for m, s in summaries.items():
        L.append(f"| {m} | " + " | ".join(str(s["error_taxonomy"].get(c, 0)) for c in cats) + " |")
    L += ["", "## Evidence types of claims", ""]
    evs = sorted({c for s in summaries.values() for c in s["evidence_types"]})
    L.append("| Method | " + " | ".join(evs) + " |"); L.append("|---|" + "---|" * len(evs))
    for m, s in summaries.items():
        L.append(f"| {m} | " + " | ".join(str(s["evidence_types"].get(c, 0)) for c in evs) + " |")
    if comparisons:
        L += ["", f"## Paired tests vs `{ref}` (A − ref; negative hallucination delta = fewer hallucinations)", "",
              "| Method | key | n | Δ (pp) | 95% CI | p (bootstrap) | p (McNemar) | sig |", "|---|---|---|---|---|---|---|---|"]
        for m, byk in comparisons.items():
            for k, c in byk.items():
                if not c.get("n"):
                    continue
                b = c["bootstrap"]
                L.append(f"| {m} | {k} | {c['n']} | {b['delta']*100:+.2f} | [{b['ci_low']*100:+.2f}, {b['ci_high']*100:+.2f}] | "
                         f"{b['p_value']:.4f} | {c['mcnemar']['p_value']:.4f} | {'yes' if c['significant_0.05'] else 'no'} |")
    L += ["", "Notes: reference proofs are built from gold answers at EVALUATION time only. UNCERTAIN claims are "
          "not hallucinations under the open-world assumption; report both HR and HR-strict. Validate the automatic "
          "labels against the human-review sample before drawing conclusions."]
    return "\n".join(L)


def risk_coverage_plot(summaries: Dict[str, Dict], out_path: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, title in ((axes[0], "hallucination", "Hallucination risk vs coverage"),
                           (axes[1], "incorrect", "Incorrect-answer risk vs coverage")):
        for m, s in summaries.items():
            cur = s.get("curves", {}).get(key)
            if cur:
                ax.plot([c for c, _r in cur], [r for _c, r in cur], label=m)
        ax.set_xlabel("answer coverage (ranked by confidence)"); ax.set_ylabel("risk"); ax.set_title(title)
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)


def run(results_dir: str, eval_specs: Sequence[str], methods: Optional[Sequence[str]] = None,
        ref: str = "gnnrag_all", judge_spec: Optional[str] = None, strict: bool = False,
        review_n: int = 0, out_dir: Optional[str] = None, n_bootstrap: int = 10000,
        cache_dir: Optional[str] = None, verbose: bool = True) -> Dict:
    from .data import load_spec
    out_dir = out_dir or os.path.join(results_dir, "hallucination")
    os.makedirs(out_dir, exist_ok=True)
    judge = judge_from_spec(judge_spec)
    result = {"datasets": {}, "config": {"results": results_dir, "eval": list(eval_specs), "ref": ref,
                                        "judge": judge_spec, "strict": strict}}
    for spec in eval_specs:
        ds = load_spec(spec, cache_dir=cache_dir or os.path.join(results_dir, "data_cache"))
        ms = list(methods) if methods else discover_methods(results_dir, ds.key)
        summaries, verdicts, recs_by_m = {}, {}, {}
        for m in ms:
            p = os.path.join(results_dir, f"records_{ds.key.replace(':', '_')}_{m}.jsonl")
            if not os.path.exists(p):
                if verbose: print(f"[skip] {p} not found")
                continue
            recs = _load_records(p)
            if not any("llm_pred" in r for r in recs):
                if verbose: print(f"[skip] {m}: no llm_pred in records (run with --reader first)")
                continue
            recs_by_m[m] = recs
            vs = evaluate_records(ds, recs, judge=judge, verbose=verbose)
            verdicts[m] = vs
            summaries[m] = summarize_verdicts(vs, strict=strict)
            with open(os.path.join(out_dir, f"claims_{ds.key.replace(':', '_')}_{m}.jsonl"), "w") as f:
                for v in vs:
                    f.write(json.dumps(v, ensure_ascii=False) + "\n")
            if verbose:
                s = summaries[m]
                print(f"{ds.key} {m:>14s}: answered {s['n_answered']}/{s['n']}  claim-HR {s['claim_hallucination_rate']}  "
                      f"question-HR {s['question_hallucination_rate']}  grounded-correct {s['grounded_correct_rate']:.3f}  "
                      f"AURC-h {s.get('aurc_hallucination')}")
        comparisons = {}
        if ref in verdicts:
            for m in verdicts:
                if m != ref:
                    comparisons[m] = compare_methods(verdicts[m], verdicts[ref], n_bootstrap=n_bootstrap)
                    comparisons[m]["disagreement_hallucinated"] = disagreement_table(verdicts[m], verdicts[ref])
        rep = markdown_report(summaries, {m: {k: c for k, c in v.items() if isinstance(c, dict) and "bootstrap" in c}
                                          for m, v in comparisons.items()}, ref, ds.key, strict)
        with open(os.path.join(out_dir, f"report_{ds.key.replace(':', '_')}.md"), "w") as f:
            f.write(rep)
        risk_coverage_plot(summaries, os.path.join(out_dir, f"risk_coverage_{ds.key.replace(':', '_')}.png"))
        if review_n > 0 and verdicts:
            rows = review_sample(verdicts, ds, n=review_n, records_by_method=recs_by_m)
            with open(os.path.join(out_dir, f"review_sample_{ds.key.replace(':', '_')}.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["qid"])
                w.writeheader(); w.writerows(rows)
        slim = {m: {k: v for k, v in s.items() if k != "curves"} for m, s in summaries.items()}
        result["datasets"][ds.key] = {"summaries": slim, "comparisons": comparisons, "methods": list(verdicts)}
    with open(os.path.join(out_dir, "hallucination.json"), "w") as f:
        json.dump(result, f, indent=1, ensure_ascii=False)
    if verbose:
        print("→", out_dir)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description="Groundedness / hallucination evaluation on reader records")
    ap.add_argument("--results", required=True, help="dir with records_<dataset>_<method>.jsonl (a --reader run)")
    ap.add_argument("--eval", nargs="+", required=True, help="dataset spec(s) exactly as used for the run")
    ap.add_argument("--methods", nargs="*", default=None, help="default: every records_* file found")
    ap.add_argument("--ref", default="gnnrag_all")
    ap.add_argument("--judge", default=None, help="none | hf:<model>  (tier-2 LLM entailment judge for UNCERTAIN claims)")
    ap.add_argument("--strict", action="store_true", help="count UNCERTAIN claims as hallucinations")
    ap.add_argument("--review-sample", type=int, default=0, help="write a CSV of N (qid, method) rows for human labelling")
    ap.add_argument("--out", default=None, help="default: <results>/hallucination")
    ap.add_argument("--n-bootstrap", type=int, default=10000)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    run(a.results, a.eval, a.methods, ref=a.ref, judge_spec=a.judge, strict=a.strict, review_n=a.review_sample,
        out_dir=a.out, n_bootstrap=a.n_bootstrap, cache_dir=a.cache_dir, verbose=not a.quiet)


if __name__ == "__main__":
    main()
