"""Task B scoring: one rule per answer type, the same for every method. NO pooled accuracy over all questions:
answerable groups (G1-G3) and the unanswerable group (G4) measure different things and are reported apart.

  per group G1-G3   exact_accuracy (scalars exact; sets and paths exactly equal), set_f1 (F1 of sets / paths),
                    answer_coverage, answered_accuracy, risk-coverage / AURC from the model's confidence
  G4                correct_abstention_rate (the method said insufficient_evidence)
  answerable        the same quality metrics over G1-G3 together (secondary view)
  grounding_all     over ALL questions, same denominators for every method: unsupported claims in the FINAL
                    output / claims of the ORIGINAL output (an E4 answer turned into an abstention states nothing),
                    questions with any unsupported claim, fabricated names, cited gold evidence (recall/precision)
  cost              tokens, latency, measured GPU time allocated by tokens (E4 reuses E3's generation)
Paired comparisons run per group on identical question ids: bootstrap over question CLUSTERS (`entity`) and
McNemar on exact correctness (G4: on correct abstention).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..evaluation.classification import selective_curve
from ..evaluation.statistical_tests import mcnemar, paired_bootstrap

GROUPS = ("G1", "G2", "G3", "G4")
ANSWERABLE = ("G1", "G2", "G3")
PAIRS = [("e1", "e0", "documents vs LLM only"), ("e2", "e1", "hybrid vs dense"), ("e2_bm25", "e2", "BM25 only vs hybrid"),
         ("e3", "e2", "KG vs hybrid documents"), ("e3", "e3a_exact", "path structure vs same edges as sentences (E3a)"),
         ("e3", "e3a", "KG traversal vs KG facts chosen by retrieval (E3b)"), ("e4", "e3", "verification vs none"),
         ("e3", "e0", "KG vs LLM only")]


def _f1(pred: Sequence[str], gold: Sequence[str]) -> float:
    p, g = {x.lower() for x in pred}, {x.lower() for x in gold}
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    tp = len(p & g)
    return 0.0 if tp == 0 else 2 * tp / (len(p) + len(g))


def score_one(q: Dict, r: Dict) -> Dict:
    ins = bool(r["insufficient"])
    if not q["answerable"]:
        c = float(ins)
        return {"correct": c, "partial": c, "abstained": ins}
    if ins or r["answer"] is None:
        return {"correct": 0.0, "partial": 0.0, "abstained": True}
    a, g, t = r["answer"], q["gold"], q["answer_type"]
    if t in ("service_set", "team_set", "env_set"):
        part = _f1(a, g)
        c = float({x.lower() for x in a} == {x.lower() for x in g})
    elif t == "package_path":
        c = float([x.lower() for x in a] == [x.lower() for x in g])
        part = _f1(a, g) if not c else 1.0
    elif t in ("hours", "days"):
        c = part = float(a == g)
    else:
        c = part = float(str(a).lower().strip() == str(g).lower().strip())
    return {"correct": c, "partial": part, "abstained": False}


def evidence_scores(q: Dict, r: Dict) -> Dict:
    gold = set(q.get("gold_evidence") or [])
    cited_ids = {e for c in r["claims"] for e in c.get("evidence", [])}
    docs = set()
    for v in r["visible"]:
        if v["id"] in cited_ids:
            docs |= set(v["docs"])
    out = {"cited_docs": len(docs)}
    if gold:
        out["ev_recall"] = len(docs & gold) / len(gold)
        vis_docs = {d for v in r["visible"] for d in v["docs"]}
        out["ev_recall_visible"] = len(vis_docs & gold) / len(gold)
        if docs:
            out["ev_precision"] = len(docs & gold) / len(docs)
    return out


def final_claims(r: Dict):
    """(claims, unsupported claims) in the FINAL output. An answer the E4 policy turned into an abstention
    states nothing; every other output keeps the claims of its original answer."""
    if r.get("method", "").startswith("e4") and r["insufficient"] and not r.get("raw_insufficient"):
        return 0, 0
    return r["check"]["n_claims"], r["check"]["n_unsupported"]


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else float("nan")


def _scored(rows, qs):
    return [(qs[r["qid"]], r, score_one(qs[r["qid"]], r), evidence_scores(qs[r["qid"]], r)) for r in rows]


def _quality(sc) -> Dict:
    if not sc:
        return {"n": 0}
    answered = [x for x in sc if not x[2]["abstained"]]
    rc = selective_curve([x[1].get("confidence") for x in answered], [x[2]["correct"] for x in answered], len(sc))
    acc = _mean([x[2]["correct"] for x in sc])
    return {"n": len(sc), "accuracy": acc, "exact_accuracy": acc, "set_f1": _mean([x[2]["partial"] for x in sc]),
            "answer_coverage": len(answered) / len(sc), "false_abstention_rate": 1 - len(answered) / len(sc),
            "answered_accuracy": _mean([x[2]["correct"] for x in answered]) if answered else float("nan"),
            "aurc": rc["aurc"], "risk_at": rc["risk_at"], "rc_curve": rc["curve"]}


def _grounding(sc) -> Dict:
    if not sc:
        return {}
    orig = sum(x[1]["check"]["n_claims"] for x in sc)
    fin = [final_claims(x[1]) for x in sc]
    return {"unsupported_claim_rate": sum(x[1]["check"]["n_unsupported"] for x in sc) / max(1, orig),
            "unsupported_over_original_claims": sum(u for _, u in fin) / max(1, orig),
            "questions_with_unsupported": _mean([float(u > 0) for _, u in fin]),
            "claims_retained": sum(c for c, _ in fin) / max(1, orig),
            "fabricated_answer_rate": _mean([float(bool(x[1]["check"]["fabricated"])) for x in sc]),
            "correct_but_unsupported": _mean([float(x[2]["correct"] == 1 and x[1]["check"]["n_supported"] == 0
                                                    and not x[2]["abstained"]) for x in sc]),
            "ev_recall": _mean([x[3].get("ev_recall") for x in sc]),
            "ev_recall_visible": _mean([x[3].get("ev_recall_visible") for x in sc]),
            "ev_precision": _mean([x[3].get("ev_precision") for x in sc])}


def _cost(sc) -> Dict:
    if not sc:
        return {}
    return {"tokens_in_mean": _mean([x[1]["input_tokens"] for x in sc]),
            "tokens_out_mean": _mean([x[1]["output_tokens"] for x in sc]),
            "evidence_tokens_mean": _mean([x[1]["evidence_tokens"] for x in sc]),
            "latency_p50_ms": float(np.percentile([x[1]["latency_ms"] or 0 for x in sc], 50)),
            "parse_failure_rate": _mean([0.0 if x[1]["parse_ok"] else 1.0 for x in sc])}


def summarize(rows: Sequence[Dict], qs: Dict[str, Dict]) -> Dict:
    sc = _scored(rows, qs)
    res: Dict = {}
    for g in ANSWERABLE:
        sel = [x for x in sc if x[0]["group"] == g]
        res[g] = {**_quality(sel), **_grounding(sel), **_cost(sel)}
    g4 = [x for x in sc if x[0]["group"] == "G4"]
    car = _mean([float(x[2]["abstained"]) for x in g4]) if g4 else float("nan")
    res["G4"] = {"n": len(g4), "accuracy": car, "correct_abstention_rate": car, **_grounding(g4), **_cost(g4)}
    ans = [x for x in sc if x[0]["answerable"]]
    res["answerable"] = {**_quality(ans), **_grounding(ans)}
    res["grounding_all"] = {"n": len(sc), **_grounding(sc), **_cost(sc)}
    traps, types = defaultdict(list), defaultdict(list)
    for x in sc:
        traps[x[0]["trap"]].append(x)
        types[f"{x[0]['group']}/{x[0]['qtype']}"].append(x)
    res["by_trap"] = {t: {"n": len(v), "accuracy": _mean([x[2]["correct"] for x in v])} for t, v in sorted(traps.items())}
    res["by_type"] = {t: {"n": len(v), "accuracy": _mean([x[2]["correct"] for x in v])} for t, v in sorted(types.items())}
    return res


def compare(rows_by: Dict[str, Sequence[Dict]], qs: Dict[str, Dict], n_bootstrap: int = 2000,
            pairs: Optional[List] = None) -> Dict:
    """Per group (and the answerable groups together); never a pool of answerable and unanswerable questions."""
    out = {}
    corr = {m: {r["qid"]: score_one(qs[r["qid"]], r)["correct"] for r in rows} for m, rows in rows_by.items()}
    for a, b, name in pairs or PAIRS:
        if a not in corr or b not in corr:
            continue
        for g in GROUPS + ("answerable",):
            ids = sorted(i for i in set(corr[a]) & set(corr[b])
                         if (qs[i]["group"] in ANSWERABLE if g == "answerable" else qs[i]["group"] == g))
            if not ids:
                continue
            ca, cb = [corr[a][i] for i in ids], [corr[b][i] for i in ids]
            bs = paired_bootstrap(ca, cb, n_bootstrap, groups=[qs[i]["entity"] for i in ids])
            out[f"{a}_vs_{b}/{g}"] = {"question": name, "group": g, "n": len(ids), **bs,
                                      "metric": "correct_abstention" if g == "G4" else "exact_accuracy",
                                      "mcnemar_p": mcnemar(ca, cb)["p_value"],
                                      "significant_0.05": bool(bs["ci_low"] > 0 or bs["ci_high"] < 0)}
    return out


def _objective(s: Dict) -> float:
    """Balanced: answer quality on answerable questions and correct abstention on G4 weigh the same."""
    return 0.5 * s["answerable"].get("exact_accuracy", 0.0) + 0.5 * (s["G4"].get("correct_abstention_rate") or 0.0)


def policy_sweep(rows_e3: Sequence[Dict], qs: Dict[str, Dict],
                 taus=(0.0, 0.25, 0.34, 0.5, 0.67, 0.75, 1.0), filter_elements: bool = True) -> List[Dict]:
    """E4 operating points (descriptive trade-off curve, not a tuning step)."""
    from .qa import reapply_policy
    out = []
    for tau in taus:
        s = summarize(reapply_policy(rows_e3, qs, tau, filter_elements), qs)
        out.append({"tau": tau, "answer_coverage": s["answerable"]["answer_coverage"],
                    "answered_accuracy": s["answerable"]["answered_accuracy"],
                    "exact_accuracy": s["answerable"]["exact_accuracy"],
                    "correct_abstention_rate": s["G4"]["correct_abstention_rate"],
                    "unsupported_over_original_claims": s["grounding_all"]["unsupported_over_original_claims"]})
    return out


def tune_policy(rows_e3: Sequence[Dict], qs: Dict[str, Dict], grid_tau=(0.0, 0.34, 0.5, 0.67, 1.0),
                grid_filter=(False, True)) -> Dict:
    """Pick (tau, filter) on the DEV org by the balanced objective; ties -> higher answer coverage."""
    from .qa import reapply_policy
    best = None
    for tau in grid_tau:
        for fe in grid_filter:
            s = summarize(reapply_policy(rows_e3, qs, tau, fe), qs)
            key = (round(_objective(s), 6), s["answerable"]["answer_coverage"])
            if best is None or key > best[0]:
                best = (key, {"tau": tau, "filter_elements": fe, "objective": "0.5*answerable_exact + 0.5*correct_abstention",
                              "dev_objective": key[0], "dev_answerable_exact": s["answerable"]["exact_accuracy"],
                              "dev_correct_abstention": s["G4"]["correct_abstention_rate"],
                              "dev_answer_coverage": s["answerable"]["answer_coverage"]})
    return best[1]


def cost_estimate(rows_by: Dict[str, Sequence[Dict]], price_per_hour: Optional[float] = None) -> Dict:
    """Measured GPU time (sum of each prompt's share of its generation batch, over unique generations), allocated
    to methods by input + output tokens. E4 reuses E3's generation."""
    uniq = {}
    for m, rows in rows_by.items():
        if m.startswith("e4") and "e3" in rows_by:
            continue
        for r in rows:
            uniq[(m, r["qid"])] = ((r["latency_ms"] or 0.0) / 1000.0, (r["input_tokens"] or 0) + (r["output_tokens"] or 0))
    T = sum(x[0] for x in uniq.values()); W = sum(x[1] for x in uniq.values())
    spt = T / W if W else 0.0
    out = {"gpu_seconds_total": T, "unique_prompts": len(uniq), "seconds_per_1k_tokens": spt * 1000,
           "price_per_hour": price_per_hour, "methods": {}}
    for m, rows in rows_by.items():
        tok = _mean([(r["input_tokens"] or 0) + (r["output_tokens"] or 0) for r in rows])
        reuse = m.startswith("e4") and "e3" in rows_by
        sec = 0.0 if reuse else tok * spt
        out["methods"][m] = {"tokens_per_item": tok, "gpu_seconds_per_item": sec, "reuses_e3_generation": reuse,
                             "usd_per_1000_items": (sec * 1000 * price_per_hour / 3600) if price_per_hour else None}
    return out
