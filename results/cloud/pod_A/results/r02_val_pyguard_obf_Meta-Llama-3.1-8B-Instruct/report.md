# Results: r02_val_pyguard_obf_Meta-Llama-3.1-8B-Instruct

Model: `vllm:NousResearch/Meta-Llama-3.1-8B-Instruct` | split: `val` | n = 200 | methods: e1, e2, e2_bm25, e2_rerank, e3

Strict = abstentions counted as errors. Claim labels check consistency with static facts, not truth.

## Headline

| method | strict/f1_malicious | strict/macro_f1 | strict/fpr | strict/fnr | answered/f1_malicious | coverage | aurc | rc_max_coverage | evidence/unsupported_claim_rate | evidence/unsupported_over_original_claims | evidence/items_with_unsupported_all | evidence/citation_accuracy | gold/evidence_coverage | gold/path_recall | tokens_in_mean | tokens_out_mean | latency_p50_ms | parse_failure_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e1 | 0.733 | 0.590 | 0.707 | 0.020 | 0.733 | 1.000 | 0.092 | 1.000 | 0.359 | 0.519 | 0.825 | 0.423 | nan | nan | 2586.940 | 291.685 | 210.312 | 0.005 |
| e2 | 0.731 | 0.582 | 0.717 | 0.020 | 0.731 | 1.000 | 0.096 | 1.000 | 0.294 | 0.432 | 0.805 | 0.523 | nan | nan | 2617.750 | 295.445 | 210.312 | 0.000 |
| e2_bm25 | 0.742 | 0.611 | 0.677 | 0.020 | 0.742 | 1.000 | 0.084 | 1.000 | 0.344 | 0.491 | 0.760 | 0.431 | nan | nan | 2612.895 | 290.375 | 210.312 | 0.000 |
| e2_rerank | 0.744 | 0.618 | 0.667 | 0.020 | 0.744 | 1.000 | 0.075 | 1.000 | 0.362 | 0.490 | 0.790 | 0.424 | nan | nan | 2587.290 | 289.560 | 210.312 | 0.000 |
| e3 | 0.792 | 0.723 | 0.505 | 0.020 | 0.792 | 1.000 | 0.053 | 1.000 | 0.262 | 0.351 | 0.550 | 0.681 | nan | nan | 1650.535 | 247.905 | 210.312 | 0.000 |

- e1: strict malicious-F1 0.733 [95% CI 0.672, 0.791]
- e2: strict malicious-F1 0.731 [95% CI 0.672, 0.786]
- e2_bm25: strict malicious-F1 0.742 [95% CI 0.683, 0.796]
- e2_rerank: strict malicious-F1 0.744 [95% CI 0.685, 0.800]
- e3: strict malicious-F1 0.792 [95% CI 0.733, 0.842]

## Paired comparisons (same sample ids)

| pair | question | n | dF1 strict | CI95 | p | dAcc | McNemar p |
|---|---|---|---|---|---|---|---|
| e2_vs_e1 | RQ2 hybrid vs dense | 200 | -0.003 | [-0.017, +0.011] | 0.858 | -0.005 | 1.000 |
| e3_vs_e2 | RQ3 KG-RAG vs hybrid | 200 | +0.061 | [+0.038, +0.089] | 0.000 | +0.105 | 0.000 |
| e2_bm25_vs_e2 | ablation BM25 only vs RRF | 200 | +0.011 | [-0.003, +0.027] | 0.210 | +0.020 | 0.289 |
| e2_rerank_vs_e2 | ablation RRF+rerank vs RRF | 200 | +0.014 | [+0.003, +0.027] | 0.012 | +0.025 | 0.062 |

## Grounding on the same denominator (E3 vs E4)

E4 reuses E3's answers: rates below divide by E3's original claims and by ALL items (abstained items included), so filtering cannot by itself make E4 look better.

| method | n_items | original_claims | unsupported_over_original_claims | unsupported_per_item | items_with_unsupported | supported_per_item | claims_retained | coverage |
|---|---|---|---|---|---|---|---|---|
| e3 | 200 | 817 | 0.351 | 1.435 | 0.550 | 2.585 | 1.000 | 1.000 |

## Cost (measured GPU time, allocated by tokens)

GPU seconds 209 for 995 unique prompts; 0.078 s per 1k tokens; price/h 1.79

| method | tokens_per_item | gpu_seconds_per_item | usd_per_1000_items | reuses_e3_generation |
|---|---|---|---|---|
| e1 | 2878.625 | 0.225 | 0.112 | False |
| e2 | 2913.195 | 0.227 | 0.113 | False |
| e2_bm25 | 2903.270 | 0.227 | 0.113 | False |
| e2_rerank | 2876.850 | 0.224 | 0.112 | False |
| e3 | 1898.440 | 0.148 | 0.074 | False |

## Strata

**e1**: all: n=200 acc=0.64 FPR=0.71
**e2**: all: n=200 acc=0.64 FPR=0.72
**e2_bm25**: all: n=200 acc=0.66 FPR=0.68
**e2_rerank**: all: n=200 acc=0.66 FPR=0.67
**e3**: all: n=200 acc=0.74 FPR=0.51

## kb_report

```json
{
 "sources": [
  "builtin",
  "examples",
  "pyguard",
  "yara"
 ],
 "counts_before_audit": {
  "builtin": 30,
  "examples": 1567,
  "pyguard": 304,
  "yara": 8
 },
 "n_docs": 1908,
 "n_dropped": 1,
 "leakage_before": {
  "n_docs": 1909,
  "docs_from_eval_samples": [],
  "n_docs_from_eval_samples": 0,
  "docs_mentioning_eval_package": [],
  "n_docs_mentioning_eval_package": 0,
  "docs_with_eval_code": [
   "ex:pyguard-obf-m-2675d2f2ca4dab13:C1"
  ],
  "n_docs_with_eval_code": 1,
  "clean": false
 },
 "leakage_after": {
  "n_docs": 1908,
  "docs_from_eval_samples": [],
  "n_docs_from_eval_samples": 0,
  "docs_mentioning_eval_package": [],
  "n_docs_mentioning_eval_package": 0,
  "docs_with_eval_code": [],
  "n_docs_with_eval_code": 0,
  "clean": true
 },
 "by_source_type": {
  "rule": 7,
  "benign_rule": 7,
  "pattern": 16,
  "malicious_example": 828,
  "benign_example": 738,
  "pyguard_pattern": 304,
  "yara": 8
 }
}
```
