"""One metric suite for every method + the research-question comparisons (plan 12)."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..schema import PredictionResult, Sample
from .classification import bootstrap_metric_ci, classification_metrics, risk_coverage
from ..analysis.taxonomy import RELATIONAL
from .evidence import (answer_path_ok, claim_metrics, evidence_coverage, path_recall, required_evidence_recall,
                       retrieval_metrics)
from .statistical_tests import compare_methods

RQ_PAIRS = [("e1", "e0", "RQ1 retrieval vs LLM-only"), ("e2", "e1", "RQ2 hybrid vs dense"),
            ("e3", "e2", "RQ3 KG-RAG vs hybrid"),
            ("e3", "e3a_exact", "RQ3 path structure vs exactly the same edges as separate sentences (E3a)"),
            ("e3", "e3a", "RQ3 graph traversal vs graph facts chosen by retrieval (E3b)"),
            ("e4", "e3", "RQ4 verification vs none"), ("e2_bm25", "e2", "ablation BM25 only vs RRF"),
            ("e2_rerank", "e2", "ablation RRF+rerank vs RRF"), ("e4_filter", "e3", "ablation claim filter"),
            ("e4_cite", "e3", "ablation citation-only"), ("e1_malllm_yara", "e0_malllm", "mal-LLM repro RAG-YARA vs zero-shot"),
            ("e1_malllm_code", "e0_malllm", "mal-LLM repro RAG-code vs zero-shot")]


def _pct(xs, q):
    xs = [x for x in xs if x is not None]
    return float(np.percentile(xs, q)) if xs else float("nan")


def _mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.mean(xs)) if xs else float("nan")


def summarize_method(preds: Sequence[PredictionResult], samples: Dict[str, Sample], traces: Sequence[Dict] = (),
                     n_bootstrap: int = 1000) -> Dict:
    lab = [(p, samples[p.sample_id]) for p in preds if p.sample_id in samples]
    scored = [(p, s) for p, s in lab if s.label in ("benign", "malicious")]
    y = [s.label for _, s in scored]; yp = [p.verdict for p, _ in scored]
    conf = [p.confidence for p, _ in scored]
    out = {"n": len(preds), "n_scored": len(scored)}
    cm = classification_metrics(y, yp)
    out.update({"coverage": cm["coverage"], "abstention_rate": cm["abstention_rate"]})
    for pol in ("strict", "conservative", "answered"):
        for k in ("f1_malicious", "macro_f1", "precision_malicious", "recall_malicious", "fpr", "fnr", "balanced_accuracy",
                  "accuracy"):
            out[f"{pol}/{k}"] = cm[pol][k]
    out["confusion_strict"] = {k: cm["strict"][k] for k in ("tp", "fp", "fn", "tn")}
    out["f1_ci95"] = bootstrap_metric_ci(y, yp, n_boot=n_bootstrap)
    rc = risk_coverage(y, yp, conf)
    out["aurc"] = rc["aurc"]; out["risk_at"] = rc["risk_at"]
    out["rc_curve"] = rc["curve"]; out["rc_max_coverage"] = rc["max_coverage"]
    out["parse_failure_rate"] = _mean([0.0 if p.parse_ok else 1.0 for p in preds])
    out["truncated_rate"] = _mean([1.0 if p.truncated else 0.0 for p in preds])
    out["changed_by_policy"] = _mean([1.0 if p.raw_verdict and p.raw_verdict != p.verdict else 0.0 for p in preds])
    out.update({f"evidence/{k}": v for k, v in claim_metrics([p.verification for p in preds if p.verification]).items()})
    # cost
    out["tokens_in_mean"] = _mean([p.input_tokens for p in preds]); out["tokens_out_mean"] = _mean([p.output_tokens for p in preds])
    out["latency_p50_ms"] = _pct([p.latency_ms for p in preds], 50)      # cache keeps the original latency
    out["latency_p95_ms"] = _pct([p.latency_ms for p in preds], 95)
    out["tokens_per_100_pkgs"] = 100 * (_mean([p.input_tokens for p in preds]) + _mean([p.output_tokens for p in preds]))
    out["evidence_tokens_mean"] = _mean([p.extra.get("prompt_meta", {}).get("evidence_tokens") for p in preds])
    # gold-annotated samples only (synthetic / manually annotated)
    cov, rmet, prec = [], defaultdict(list), []
    tr_by = {t["sample_id"]: t for t in traces}
    for p, s in lab:
        gb = s.gold.get("gold_behaviors") or []
        if not gb:
            continue
        pkg_beh = [b for e in p.evidence if not e.evidence_id.startswith("K") for b in e.behaviors]
        cov.append(evidence_coverage(gb, pkg_beh))
        kb = [e.behaviors for e in p.evidence if e.evidence_id.startswith("K")]
        if kb:
            for k, v in (retrieval_metrics(kb, gb) or {}).items():
                rmet[k].append(v)
        t = tr_by.get(p.sample_id, {})
        if t.get("graph") and s.gold.get("gold_paths"):
            prec.append(path_recall([pp["apis"] or [] for pp in t["graph"].get("paths", [])], s.gold["gold_paths"]))
    out["gold/evidence_coverage"] = _mean(cov)
    out["gold/n_annotated"] = len(cov)
    for k, v in rmet.items():
        out[f"gold/kb_{k}"] = _mean(v)
    out["gold/path_recall"] = _mean(prec)
    # controlled cases (gold expected_verdict / required_evidence / gold_function_path)
    exp = [(p, s) for p, s in lab if s.gold.get("expected_verdict")]
    if exp:
        out["case/n"] = len(exp)
        out["case/expected_verdict_accuracy"] = _mean([float(p.verdict == s.gold["expected_verdict"]) for p, s in exp])
        unc = [float(p.verdict == "uncertain") for p, s in exp if s.gold["expected_verdict"] == "uncertain"]
        dec = [float(p.verdict == "uncertain") for p, s in exp if s.gold["expected_verdict"] != "uncertain"]
        out["case/abstention_accuracy"] = _mean(unc)
        out["case/false_abstention_rate"] = _mean(dec)
        out["case/required_evidence_recall_visible"] = _mean(
            [required_evidence_recall(s.gold.get("required_evidence") or [], p.evidence) for p, s in exp])
        out["case/required_evidence_recall_cited"] = _mean(
            [required_evidence_recall(s.gold.get("required_evidence") or [],
                                      [e for e in p.evidence if e.evidence_id in set(p.claimed_evidence_ids)])
             for p, s in exp])
        out["case/answer_path_accuracy"] = _mean(
            [answer_path_ok(s.gold.get("gold_function_path") or [], p.explanation, p.claims,
                            (p.verification or {}).get("claims", []), RELATIONAL) for p, s in exp])
    gtr = [t["graph"] for t in traces if t.get("graph")]
    if gtr:
        out["graph/paths_per_sample"] = _mean([g["n_paths"] for g in gtr])
        out["graph/edges_mean"] = _mean([g["n_edges"] for g in gtr])
        out["graph/build_ms_mean"] = _mean([g["build_ms"] for g in gtr])
        out["graph/edge_grounding_rate"] = _mean([1 - g["edges_missing_provenance"] / max(g["n_edges"], 1) for g in gtr])
        st = defaultdict(int)
        for g in gtr:
            for k, v in g["evidence_status"].items():
                st[k] += v
        out["graph/inferred_edge_share"] = st.get("inferred", 0) / max(sum(st.values()), 1)
    return out


def stratify(preds: Sequence[PredictionResult], samples: Dict[str, Sample], key: str = "difficulty") -> Dict[str, Dict]:
    groups: Dict[str, List] = defaultdict(list)
    for p in preds:
        s = samples.get(p.sample_id)
        if s is None or s.label not in ("benign", "malicious"):
            continue
        g = s.meta.get(key) or ("parse_incomplete" if key == "coverage" and p.extra.get("prompt_meta", {}).get("parse_coverage", 1) < 1 else "all")
        groups[str(g)].append(p)
    out = {}
    for g, ps in groups.items():
        cm = classification_metrics([samples[p.sample_id].label for p in ps], [p.verdict for p in ps])
        out[g] = {"n": len(ps), "accuracy_strict": cm["strict"]["accuracy"], "f1_malicious_strict": cm["strict"]["f1_malicious"],
                  "fpr_strict": cm["strict"]["fpr"], "coverage": cm["coverage"]}
    return out


def case_table(preds_by_method: Dict[str, Sequence[PredictionResult]], samples: Dict[str, Sample],
               traces_by_method: Optional[Dict[str, Sequence[Dict]]] = None) -> List[Dict]:
    """One row per (controlled case, method): verdict vs expected, evidence recall, path answer, patterns."""
    rows = []
    for m, preds in preds_by_method.items():
        tr = {t["sample_id"]: t for t in (traces_by_method or {}).get(m, [])}
        for p in preds:
            s = samples.get(p.sample_id)
            if s is None or not s.meta.get("case"):
                continue
            req = s.gold.get("required_evidence") or []
            cited = [e for e in p.evidence if e.evidence_id in set(p.claimed_evidence_ids)]
            g = tr.get(p.sample_id, {}).get("graph") or {}
            rows.append({"case": s.meta["case"], "method": m, "expected": s.gold.get("expected_verdict"),
                         "verdict": p.verdict, "raw_verdict": p.raw_verdict,
                         "correct": p.verdict == s.gold.get("expected_verdict"),
                         "req_evidence_visible": required_evidence_recall(req, p.evidence),
                         "req_evidence_cited": required_evidence_recall(req, cited),
                         "answer_path_ok": answer_path_ok(s.gold.get("gold_function_path") or [], p.explanation, p.claims,
                                                          (p.verification or {}).get("claims", []), RELATIONAL),
                         "n_supported": (p.verification or {}).get("n_supported"),
                         "n_unsupported": (p.verification or {}).get("n_unsupported"),
                         "policy_reason": (p.verification or {}).get("policy_reason", ""),
                         "patterns": sorted({x for pp in g.get("paths", []) for x in (pp.get("patterns") or [])})})
    return sorted(rows, key=lambda r: (r["case"], r["method"]))


def comparisons(preds_by_method: Dict[str, Sequence[PredictionResult]], samples: Dict[str, Sample],
                ref: Optional[str] = None, n_bootstrap: int = 2000) -> Dict[str, Dict]:
    """Paired comparisons; resampling unit = meta.group_id (grouped split) when present, else the sample."""
    y = {i: s.label for i, s in samples.items() if s.label in ("benign", "malicious")}
    groups = {i: s.meta["group_id"] for i, s in samples.items() if s.meta.get("group_id") is not None} or None
    pv = {m: {p.sample_id: p.verdict for p in ps} for m, ps in preds_by_method.items()}
    out = {}
    pairs = [(a, b, n) for a, b, n in RQ_PAIRS if a in pv and b in pv]
    if ref and ref in pv:
        pairs += [(m, ref, f"vs {ref}") for m in pv if m != ref and (m, ref) not in {(a, b) for a, b, _ in pairs}]
    for a, b, name in pairs:
        c = compare_methods(y, pv[a], pv[b], n_bootstrap, groups)
        c["question"] = name
        out[f"{a}_vs_{b}"] = c
    return out


def grounding_vs_base(preds_by_method: Dict[str, Sequence[PredictionResult]], base: str = "e3",
                      policy_methods: Sequence[str] = ("e4", "e4_filter", "e4_cite")) -> Dict[str, Dict]:
    """E3 vs E4 on the SAME items and the SAME claims. E4 reuses E3's generation, so E3's claims are the original
    claims of both: unsupported claims left in the final answer are divided by E3's claim count and by ALL items
    (an abstained item counts as an item, with no claim). Dropping hard cases therefore cannot, by itself, make
    the rate look better; `supported_per_item`, `claims_retained` and `coverage` show what the filter cost."""
    if base not in preds_by_method:
        return {}
    b = {p.sample_id: p for p in preds_by_method[base]}

    def bad(v):
        v = v or {}
        return v.get("n_unsupported", 0) + v.get("n_contradicted", 0)
    out = {}
    for m in [base] + [x for x in policy_methods if x in preds_by_method]:
        ps = [p for p in preds_by_method[m] if p.sample_id in b]
        n = max(len(ps), 1)
        orig = sum((b[p.sample_id].verification or {}).get("n_claims", 0) for p in ps)
        out[m] = {"n_items": len(ps), "original_claims": orig,
                  "unsupported_over_original_claims": sum(bad(p.verification) for p in ps) / max(orig, 1),
                  "unsupported_per_item": sum(bad(p.verification) for p in ps) / n,
                  "items_with_unsupported": sum(1 for p in ps if bad(p.verification) > 0) / n,
                  "supported_per_item": sum((p.verification or {}).get("n_supported", 0) for p in ps) / n,
                  "claims_retained": sum((p.verification or {}).get("n_claims", 0) for p in ps) / max(orig, 1),
                  "coverage": sum(1 for p in ps if p.verdict != "uncertain") / n}
    return out


def cost_estimate(preds_by_method: Dict[str, Sequence[PredictionResult]], price_per_hour: Optional[float] = None) -> Dict:
    """Measured GPU time of the run (sum of the per-prompt share of each generation batch over UNIQUE prompts),
    allocated to methods by their input + output tokens. E4 variants reuse E3's generation (no extra GPU time)."""
    uniq: Dict[str, tuple] = {}
    for m, ps in preds_by_method.items():
        for p in ps:
            key = p.prompt_sha256 or f"{m}:{p.sample_id}"
            if key not in uniq:
                uniq[key] = ((p.latency_ms or 0.0) / 1000.0, (p.input_tokens or 0) + (p.output_tokens or 0))
    T = sum(x[0] for x in uniq.values()); W = sum(x[1] for x in uniq.values())
    spt = T / W if W else 0.0
    out = {"gpu_seconds_total": T, "unique_prompts": len(uniq), "seconds_per_1k_tokens": spt * 1000,
           "price_per_hour": price_per_hour, "methods": {}}
    for m, ps in preds_by_method.items():
        tok = _mean([(p.input_tokens or 0) + (p.output_tokens or 0) for p in ps])
        reuse = m.startswith("e4")
        sec = 0.0 if reuse else tok * spt
        out["methods"][m] = {"tokens_per_item": tok, "gpu_seconds_per_item": sec, "reuses_e3_generation": reuse,
                             "usd_per_1000_items": (sec * 1000 * price_per_hour / 3600) if price_per_hour else None}
    return out
