"""Statistical significance utilities for paired LLM-QA predictions.

Two tests provided:
  - paired_bootstrap_hit: paired bootstrap CI + p-value for Hit/F1 difference
  - mcnemar_hit: McNemar's exact test on per-question Hit (binary 0/1)

Usage on saved predictions_*.jsonl files:

    from significance import compare_methods_on_jsonl
    compare_methods_on_jsonl(
        '/path/predictions_llama2.jsonl',
        method_a='gnn_rag', method_b='cgar_v2',
        metric='hit', n_bootstrap=10000)
"""
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


# ── Re-use the GNN-RAG normalisation (kept identical to P6 cell 4) ──────
import re, string


def _normalize(s):
    s = str(s).lower()
    exclude = set(string.punctuation)
    s = ''.join(c for c in s if c not in exclude)
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = re.sub(r'\b(<pad>)\b', ' ', s)
    return ' '.join(s.split())


def _match(prediction, answer):
    return _normalize(answer) in _normalize(prediction)


def hit_for(pred):
    """Per-question Hit (1 if any gold in prediction)."""
    answers = pred.get('ground_truth', [])
    text = pred.get('prediction', '')
    if not answers:
        return None
    return int(any(_match(text, a) for a in answers))


def f1_for(pred):
    """Per-question F1 (entity-level, matching GNN-RAG)."""
    answers = pred.get('ground_truth', [])
    text = pred.get('prediction', '')
    if not answers:
        return None
    pred_list = [l.strip() for l in text.split('\n') if l.strip()]
    if not pred_list:
        return 0.0
    pred_str = ' '.join(pred_list)
    matched = sum(1 for a in answers if _match(pred_str, a))
    if matched == 0:
        return 0.0
    p = matched / len(pred_list)
    r = matched / len(answers)
    return 0.0 if (p + r) == 0 else 2 * p * r / (p + r)


# ── Tests ──────────────────────────────────────────────────────────

def paired_bootstrap(scores_a, scores_b, n_bootstrap=10000, seed=42):
    """Two-sided paired bootstrap for mean(a) vs mean(b).

    Returns dict with:
        delta:       mean(a) - mean(b)
        ci_low/high: 95% CI of delta
        p_value:     two-sided p-value (proportion of bootstrap deltas with
                     opposite sign to observed)
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    assert a.shape == b.shape, 'paired arrays must align'
    n = len(a)
    rng = np.random.default_rng(seed)
    delta_obs = a.mean() - b.mean()

    deltas = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        deltas[i] = a[idx].mean() - b[idx].mean()
    ci_low, ci_high = np.percentile(deltas, [2.5, 97.5])

    if delta_obs >= 0:
        p = (deltas <= 0).mean()
    else:
        p = (deltas >= 0).mean()
    return {
        'mean_a': float(a.mean()),
        'mean_b': float(b.mean()),
        'delta': float(delta_obs),
        'ci_low': float(ci_low),
        'ci_high': float(ci_high),
        'p_value': float(min(2 * p, 1.0)),  # two-sided
        'n': int(n),
    }


def mcnemar(hits_a, hits_b, exact=True):
    """McNemar's test on per-question binary Hit (0/1).

    Builds 2x2 table:        a=1   a=0
                       b=1    n11   n01
                       b=0    n10   n00
    Tests H0: P(a=1, b=0) = P(a=0, b=1).
    """
    a = np.asarray(hits_a, dtype=int)
    b = np.asarray(hits_b, dtype=int)
    assert a.shape == b.shape
    n10 = int(((a == 1) & (b == 0)).sum())  # a wins
    n01 = int(((a == 0) & (b == 1)).sum())  # b wins
    n11 = int(((a == 1) & (b == 1)).sum())
    n00 = int(((a == 0) & (b == 0)).sum())
    discordant = n10 + n01
    if discordant == 0:
        return {'n10': 0, 'n01': 0, 'n11': n11, 'n00': n00,
                'p_value': 1.0, 'method': 'no_discordant'}

    if exact:
        # Exact binomial: P(X >= max(n10,n01)) under Binomial(discordant, 0.5)
        from math import comb
        k = max(n10, n01)
        p_one = sum(comb(discordant, i) * (0.5 ** discordant)
                    for i in range(k, discordant + 1))
        p_value = min(2 * p_one, 1.0)
        method = 'exact_binomial'
    else:
        # Chi-square with continuity correction (Edwards)
        chi2 = ((abs(n10 - n01) - 1) ** 2) / discordant
        # Survival function of chi2 with df=1
        p_value = math.erfc(math.sqrt(chi2 / 2))
        method = 'chi2_with_continuity'

    return {'n10_a_wins': n10, 'n01_b_wins': n01, 'n11_both': n11, 'n00_neither': n00,
            'p_value': float(p_value), 'method': method}


# ── High-level helper for jsonl files (P6 format) ──────────────────

def load_predictions_grouped(jsonl_path):
    """Load predictions_*.jsonl and group by method, indexed by qid.

    Returns: {method: {qid: pred_dict}}
    """
    by_method = defaultdict(dict)
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            p = json.loads(line)
            by_method[p.get('method', 'unknown')][p['id']] = p
    return dict(by_method)


def compare_methods_on_jsonl(jsonl_path, method_a, method_b, metric='hit',
                              n_bootstrap=10000, seed=42, verbose=True):
    """Run paired tests between two retrieval methods on the same LLM file.

    Args:
        jsonl_path: P6-format predictions file
        method_a, method_b: names that appear in 'method' field
        metric: 'hit' or 'f1'

    Returns: dict with 'bootstrap', 'mcnemar' (if hit), and per-question scores.
    """
    grouped = load_predictions_grouped(jsonl_path)
    if method_a not in grouped or method_b not in grouped:
        raise ValueError(f'methods present: {list(grouped)}')

    metric_fn = {'hit': hit_for, 'f1': f1_for}[metric]
    common_qids = set(grouped[method_a]) & set(grouped[method_b])
    if not common_qids:
        raise ValueError('no overlapping qids between methods')

    qids = sorted(common_qids)
    scores_a, scores_b = [], []
    for qid in qids:
        sa = metric_fn(grouped[method_a][qid])
        sb = metric_fn(grouped[method_b][qid])
        if sa is None or sb is None:
            continue
        scores_a.append(sa); scores_b.append(sb)

    res = {
        'jsonl': str(jsonl_path),
        'method_a': method_a, 'method_b': method_b, 'metric': metric,
        'n_compared': len(scores_a),
        'bootstrap': paired_bootstrap(scores_a, scores_b, n_bootstrap, seed),
    }
    if metric == 'hit':
        res['mcnemar'] = mcnemar([int(s) for s in scores_a], [int(s) for s in scores_b])

    if verbose:
        b = res['bootstrap']
        print(f'  Paired bootstrap ({metric}, n={res["n_compared"]}, n_boot={n_bootstrap}):')
        print(f'    {method_a}: {b["mean_a"]*100:.2f}%')
        print(f'    {method_b}: {b["mean_b"]*100:.2f}%')
        print(f'    Δ = {b["delta"]*100:+.2f}pp  '
              f'95% CI = [{b["ci_low"]*100:+.2f}, {b["ci_high"]*100:+.2f}]  '
              f'p = {b["p_value"]:.4f}')
        if 'mcnemar' in res:
            m = res['mcnemar']
            print(f'  McNemar ({m["method"]}): a-only={m["n10_a_wins"]} '
                  f'b-only={m["n01_b_wins"]} both={m["n11_both"]} '
                  f'neither={m["n00_neither"]} p={m["p_value"]:.4f}')
        sig = b['p_value'] < 0.05
        print(f'  → {"SIGNIFICANT" if sig else "NOT significant"} at α=0.05')
        print(f'  → Paper sentence: "{method_a} {"significantly differs from" if sig else "is statistically indistinguishable from"} '
              f'{method_b} (Δ = {b["delta"]*100:+.2f}pp, 95% CI [{b["ci_low"]*100:+.2f}, {b["ci_high"]*100:+.2f}], p = {b["p_value"]:.3f})."')
    return res


# ── CLI for quick CI use ───────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('jsonl')
    ap.add_argument('--a', required=True)
    ap.add_argument('--b', required=True)
    ap.add_argument('--metric', default='hit', choices=['hit', 'f1'])
    ap.add_argument('--bootstrap', type=int, default=10000)
    args = ap.parse_args()
    compare_methods_on_jsonl(args.jsonl, args.a, args.b,
                              metric=args.metric, n_bootstrap=args.bootstrap)
