# Results: r02_val_pyguard_obf_Qwen2.5-Coder-7B-Instruct

Model: `vllm:Qwen/Qwen2.5-Coder-7B-Instruct` | split: `val` | n = 200 | methods: e1, e2, e2_bm25, e2_rerank, e3

Strict = abstentions counted as errors. Claim labels check consistency with static facts, not truth.

## Headline

| method | strict/f1_malicious | strict/macro_f1 | strict/fpr | strict/fnr | answered/f1_malicious | coverage | aurc | rc_max_coverage | evidence/unsupported_claim_rate | evidence/unsupported_over_original_claims | evidence/items_with_unsupported_all | evidence/citation_accuracy | gold/evidence_coverage | gold/path_recall | tokens_in_mean | tokens_out_mean | latency_p50_ms | parse_failure_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e1 | 0.933 | 0.930 | 0.111 | 0.030 | 0.933 | 1.000 | 0.009 | 1.000 | 0.261 | 0.378 | 0.615 | 0.449 | nan | nan | 2636.520 | 272.120 | 163.654 | 0.000 |
| e2 | 0.915 | 0.910 | 0.141 | 0.040 | 0.915 | 1.000 | 0.012 | 1.000 | 0.282 | 0.399 | 0.705 | 0.462 | nan | nan | 2671.915 | 288.795 | 163.654 | 0.000 |
| e2_bm25 | 0.938 | 0.935 | 0.101 | 0.030 | 0.938 | 1.000 | 0.005 | 1.000 | 0.231 | 0.377 | 0.685 | 0.499 | nan | nan | 2662.130 | 282.165 | 163.654 | 0.000 |
| e2_rerank | 0.929 | 0.925 | 0.121 | 0.030 | 0.929 | 1.000 | 0.009 | 1.000 | 0.244 | 0.375 | 0.650 | 0.480 | nan | nan | 2633.105 | 273.525 | 163.654 | 0.000 |
| e3 | 0.888 | 0.885 | 0.131 | 0.099 | 0.888 | 1.000 | 0.033 | 1.000 | 0.238 | 0.270 | 0.435 | 0.781 | nan | nan | 1695.000 | 234.770 | 163.654 | 0.000 |

- e1: strict malicious-F1 0.933 [95% CI 0.897, 0.965]
- e2: strict malicious-F1 0.915 [95% CI 0.874, 0.950]
- e2_bm25: strict malicious-F1 0.938 [95% CI 0.903, 0.966]
- e2_rerank: strict malicious-F1 0.929 [95% CI 0.892, 0.961]
- e3: strict malicious-F1 0.888 [95% CI 0.841, 0.932]

## Paired comparisons (same sample ids)

| pair | question | n | dF1 strict | CI95 | p | dAcc | McNemar p |
|---|---|---|---|---|---|---|---|
| e2_vs_e1 | RQ2 hybrid vs dense | 200 | -0.018 | [-0.049, +0.009] | 0.212 | -0.020 | 0.344 |
| e3_vs_e2 | RQ3 KG-RAG vs hybrid | 200 | -0.027 | [-0.109, +0.036] | 0.618 | -0.025 | 0.383 |
| e2_bm25_vs_e2 | ablation BM25 only vs RRF | 200 | +0.023 | [-0.004, +0.051] | 0.108 | +0.025 | 0.180 |
| e2_rerank_vs_e2 | ablation RRF+rerank vs RRF | 200 | +0.014 | [-0.017, +0.046] | 0.412 | +0.015 | 0.581 |

## Grounding on the same denominator (E3 vs E4)

E4 reuses E3's answers: rates below divide by E3's original claims and by ALL items (abstained items included), so filtering cannot by itself make E4 look better.

| method | n_items | original_claims | unsupported_over_original_claims | unsupported_per_item | items_with_unsupported | supported_per_item | claims_retained | coverage |
|---|---|---|---|---|---|---|---|---|
| e3 | 200 | 784 | 0.270 | 1.060 | 0.435 | 2.675 | 1.000 | 1.000 |

## Cost (measured GPU time, allocated by tokens)

GPU seconds 163 for 995 unique prompts; 0.060 s per 1k tokens; price/h 1.79

| method | tokens_per_item | gpu_seconds_per_item | usd_per_1000_items | reuses_e3_generation |
|---|---|---|---|---|
| e1 | 2908.640 | 0.174 | 0.087 | False |
| e2 | 2960.710 | 0.177 | 0.088 | False |
| e2_bm25 | 2944.295 | 0.176 | 0.088 | False |
| e2_rerank | 2906.630 | 0.174 | 0.087 | False |
| e3 | 1929.770 | 0.116 | 0.058 | False |

## Strata

**e1**: all: n=200 acc=0.93 FPR=0.11
**e2**: all: n=200 acc=0.91 FPR=0.14
**e2_bm25**: all: n=200 acc=0.94 FPR=0.10
**e2_rerank**: all: n=200 acc=0.93 FPR=0.12
**e3**: all: n=200 acc=0.89 FPR=0.13

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
