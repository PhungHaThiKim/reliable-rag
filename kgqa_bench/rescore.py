"""Re-score saved reader outputs offline (no GPU, no model reload).

Why: records_*.jsonl keep the raw ``llm_pred`` text, so whenever the answer
parser (qa_text_eval.extract_answer_entities, D2 fix Nov 2026) or the
abstention rule changes, every llm_* metric can be recomputed from disk in
seconds instead of re-running the reader (25-40 min per model on a T4).

What changes versus the pre-Nov-2026 attach_llm_answers() (now shares split_answer):
  * abstentions ("unknown", "the paths do not provide ...") are detected with
    qa_text_eval.split_answer (no answer entity left + abstention phrase)
    and scored as llm_abstained=1, hit/hit1/f1=0,
    unsupported_rate=0  (the legacy proxy counted the string "unknown" as an
    unsupported entity, which inflated unsupported_rate for the grounded prompt);
  * llm_unsupported_rate is therefore "fabricated entities among ANSWERED
    questions"; the old value is kept as llm_unsupported_rate_legacy.

Usage:
  python -m kgqa_bench.rescore --results results_gnn/b08_reader_X [more dirs]
        [--eval SPEC ...]   # defaults to the eval specs stored in results.json config
        [--ref gnnrag_all] [--n-bootstrap 10000] [--no-backup]
The original records are copied once to records_*.jsonl.orig; results.json and
report.md are rewritten in place.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
from typing import Dict, List, Optional

from qa_text_eval import eval_f1, eval_hit, eval_hit1, match, split_answer

from . import metrics as M
from .data import load_spec

LLM_KEYS = ("llm_hit", "llm_hit1", "llm_f1", "llm_acc", "llm_unsupported_rate", "llm_abstained")
EXTRA_KEYS = ("llm_n_dropped", "llm_fallback", "llm_grounded")


def rescore_record(r: Dict, q, prefix: str = "llm", verify: bool = False) -> Dict:
    """Recompute {prefix}_hit/hit1/f1/acc/unsupported_rate/abstained in place from the reader's
    raw text ({prefix}_pred_raw, else {prefix}_pred). verify=True applies the CGAR-G grounding
    step (llm_reader.ground_answer) first; verify=False restores the raw text as {prefix}_pred."""
    from .llm_reader import candidates_of, ground_answer
    raw = r.get(f"{prefix}_pred_raw")
    if raw is None:
        raw = r.get(f"{prefix}_pred")
        if raw is None:
            return r
        r[f"{prefix}_pred_raw"] = raw
    if f"{prefix}_unsupported_rate_legacy" not in r and f"{prefix}_unsupported_rate" in r:
        r[f"{prefix}_unsupported_rate_legacy"] = r[f"{prefix}_unsupported_rate"]
    pred = raw
    if verify:
        g = ground_answer(raw, candidates_of(r))
        pred = g["pred"]
        r[f"{prefix}_n_dropped"], r[f"{prefix}_fallback"], r[f"{prefix}_grounded"] = len(g["dropped"]), g["fallback"], g["grounded"]
    else:
        for k in ("n_dropped", "fallback", "grounded"):
            r.pop(f"{prefix}_{k}", None)
    r[f"{prefix}_pred"] = pred
    pred_list, abstained = split_answer(pred)
    pred_str = " ".join(pred_list)
    f1, _p, _r = eval_f1(pred_list, q.answers)
    r[f"{prefix}_abstained"] = int(abstained)
    r[f"{prefix}_hit"] = eval_hit(pred_str, q.answers) if pred_list else 0
    r[f"{prefix}_hit1"] = eval_hit1(pred_list, q.answers)
    r[f"{prefix}_f1"] = f1
    r[f"{prefix}_acc"] = (sum(1 for a in q.answers if match(pred_str, a)) / len(q.answers)) if (q.answers and pred_list) else 0.0
    ctx = "\n".join(r.get("selected_paths", [])).lower()
    r[f"{prefix}_unsupported_rate"] = (sum(1 for p in pred_list if p.lower() not in ctx) / len(pred_list)) if pred_list else 0.0
    return r


def _load_records(path: str) -> List[Dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def rescore_dir(results_dir: str, eval_specs: Optional[List[str]] = None, ref: Optional[str] = None,
                n_bootstrap: int = 10000, backup: bool = True, verbose: bool = True,
                verify: bool = False) -> Dict:
    """Rescore every records_<ds>_<method>.jsonl in results_dir; update results.json + report.md.
    verify=True scores the CGAR-G grounded answer (see rescore_record); the raw text is always kept.
    Returns {ds_key: {method: {metric: (before, after)}}}."""
    rj = os.path.join(results_dir, "results.json")
    with open(rj) as f:
        R = json.load(f)
    cfg = R.get("config", {})
    eval_specs = eval_specs or cfg.get("eval") or []
    if not eval_specs:
        raise ValueError("no eval specs: pass --eval or make sure results.json config has 'eval'")
    load_kw = {}
    if cfg.get("max_items"):
        load_kw["max_items"] = cfg["max_items"]
    if cfg.get("subset_seed") is not None:
        load_kw["subset_seed"] = cfg["subset_seed"]
    datasets = [load_spec(s, **load_kw) for s in eval_specs]
    ref = ref or cfg.get("compare_ref")
    changes: Dict[str, Dict] = {}
    for ds in datasets:
        key = ds.key.replace(":", "_")
        by_qid = {q.qid: q for q in ds}
        recs_by_method: Dict[str, List[Dict]] = {}
        for path in sorted(glob.glob(os.path.join(results_dir, f"records_{key}_*.jsonl"))):
            method = os.path.basename(path)[len(f"records_{key}_"):-len(".jsonl")]
            recs = _load_records(path)
            if not any(r.get("llm_pred") is not None for r in recs):
                continue
            if backup and not os.path.exists(path + ".orig"):
                shutil.copy(path, path + ".orig")
            before = M.summarize_records(recs, n_bootstrap=50)
            for r in recs:
                q = by_qid.get(r["qid"])
                if q is not None:
                    rescore_record(r, q, verify=verify)
            with open(path, "w") as f:
                for r in recs:
                    f.write(json.dumps(r, default=float) + "\n")
            after = M.summarize_records(recs, n_bootstrap=n_bootstrap)
            recs_by_method[method] = recs
            summ = R.setdefault("summaries", {}).setdefault(ds.key, {}).setdefault(method, {})
            for k, v in after.items():
                if k in LLM_KEYS or k.endswith("_llm") or k.startswith("llm_hit_ci") or k in EXTRA_KEYS:
                    summ[k] = v
            for k in EXTRA_KEYS:
                if k not in after:
                    summ.pop(k, None)
            changes.setdefault(ds.key, {})[method] = {k: (before.get(k), after.get(k)) for k in LLM_KEYS}
            if verbose:
                d = changes[ds.key][method]
                print(f"  {ds.key:14s} {method:16s} " + "  ".join(
                    f"{k[4:]}: {(d[k][0] if d[k][0] is not None else float('nan')):.3f}->{d[k][1]:.3f}"
                    for k in LLM_KEYS if d[k][1] is not None))
        # paired comparisons on the recomputed llm metrics
        if ref in recs_by_method:
            cmp = R.setdefault("comparisons", {}).setdefault(ds.key, {})
            for m, recs in recs_by_method.items():
                if m == ref:
                    continue
                for metric in ("llm_hit", "llm_hit1", "llm_f1"):
                    cmp[f"{m}_vs_{ref}:{metric}"] = M.compare_records(recs, recs_by_method[ref], metric,
                                                                      n_bootstrap=n_bootstrap)
    R.setdefault("config", {})["rescored"] = True
    R["config"]["reader_verify"] = bool(verify)
    with open(rj, "w") as f:
        json.dump(R, f, indent=2, default=float)
    try:
        from .report import write_report
        write_report(results_dir)
    except Exception as e:  # report is cosmetic; never lose the rescored json over it
        if verbose:
            print(f"  [warn] report.md not rewritten: {e}")
    return changes


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", nargs="+", required=True, help="one or more results dirs (each with results.json)")
    ap.add_argument("--eval", nargs="*", default=None, help="dataset specs; default = results.json config['eval']")
    ap.add_argument("--ref", default=None, help="reference method for paired tests; default = config['compare_ref']")
    ap.add_argument("--n-bootstrap", type=int, default=10000)
    ap.add_argument("--no-backup", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--verify", action="store_true", help="CGAR-G: score the grounded answer (candidate-resolved entities, first candidate as fallback)")
    a = ap.parse_args(argv)
    out = {}
    for d in a.results:
        if not os.path.exists(os.path.join(d, "results.json")):
            print(f"[skip] {d}: no results.json"); continue
        if not a.quiet:
            print(f"== rescore {d}")
        out[d] = rescore_dir(d, a.eval, a.ref, a.n_bootstrap, backup=not a.no_backup, verbose=not a.quiet, verify=a.verify)
    return out


if __name__ == "__main__":
    main()
