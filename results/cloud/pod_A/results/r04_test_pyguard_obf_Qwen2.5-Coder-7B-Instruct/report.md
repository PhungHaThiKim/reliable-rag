# Results: r04_test_pyguard_obf_Qwen2.5-Coder-7B-Instruct

Model: `vllm:Qwen/Qwen2.5-Coder-7B-Instruct` | split: `test` | n = 310 | methods: e0, e1, e2, e2_bm25, e2_rerank, e3, e3a_exact, e3a, e4, e4_filter, e4_cite

Strict = abstentions counted as errors. Claim labels check consistency with static facts, not truth.

## Headline

| method | strict/f1_malicious | strict/macro_f1 | strict/fpr | strict/fnr | answered/f1_malicious | coverage | aurc | rc_max_coverage | evidence/unsupported_claim_rate | evidence/unsupported_over_original_claims | evidence/items_with_unsupported_all | evidence/citation_accuracy | gold/evidence_coverage | gold/path_recall | tokens_in_mean | tokens_out_mean | latency_p50_ms | parse_failure_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e0 | 0.953 | 0.952 | 0.078 | 0.019 | 0.953 | 1.000 | 0.007 | 1.000 | 0.291 | 0.479 | 0.758 | 0.460 | nan | nan | 1402.697 | 274.248 | 135.058 | 0.000 |
| e1 | 0.925 | 0.919 | 0.149 | 0.013 | 0.925 | 1.000 | 0.006 | 1.000 | 0.242 | 0.381 | 0.642 | 0.485 | nan | nan | 2625.565 | 277.158 | 135.058 | 0.000 |
| e2 | 0.914 | 0.906 | 0.175 | 0.013 | 0.914 | 1.000 | 0.007 | 1.000 | 0.277 | 0.387 | 0.652 | 0.461 | nan | nan | 2665.116 | 284.842 | 135.058 | 0.000 |
| e2_bm25 | 0.919 | 0.912 | 0.162 | 0.013 | 0.919 | 1.000 | 0.007 | 1.000 | 0.251 | 0.390 | 0.716 | 0.483 | nan | nan | 2655.403 | 282.065 | 135.058 | 0.000 |
| e2_rerank | 0.928 | 0.922 | 0.143 | 0.013 | 0.928 | 1.000 | 0.006 | 1.000 | 0.268 | 0.390 | 0.645 | 0.481 | nan | nan | 2640.532 | 265.739 | 135.058 | 0.000 |
| e3 | 0.939 | 0.935 | 0.110 | 0.019 | 0.939 | 1.000 | 0.011 | 1.000 | 0.270 | 0.326 | 0.548 | 0.711 | nan | nan | 1735.948 | 247.561 | 135.058 | 0.000 |
| e3a_exact | 0.942 | 0.939 | 0.104 | 0.019 | 0.942 | 1.000 | 0.006 | 1.000 | 0.259 | 0.404 | 0.703 | 0.600 | nan | nan | 1716.826 | 275.100 | 135.058 | 0.000 |
| e3a | 0.928 | 0.922 | 0.143 | 0.013 | 0.928 | 1.000 | 0.003 | 1.000 | 0.253 | 0.334 | 0.623 | 0.677 | nan | nan | 1979.213 | 280.374 | 135.058 | 0.000 |
| e4 | 0.887 | 0.887 | 0.110 | 0.115 | 0.936 | 0.948 | 0.012 | 0.948 | 0.000 | 0.000 | 0.000 | 0.998 | nan | nan | 1735.948 | 247.561 | 135.058 | 0.000 |
| e4_filter | 0.939 | 0.935 | 0.110 | 0.019 | 0.939 | 1.000 | 0.011 | 1.000 | 0.000 | 0.000 | 0.000 | 0.998 | nan | nan | 1735.948 | 247.561 | 135.058 | 0.000 |
| e4_cite | 0.939 | 0.935 | 0.110 | 0.019 | 0.939 | 1.000 | 0.011 | 1.000 | 0.270 | 0.326 | 0.548 | 0.711 | nan | nan | 1735.948 | 247.561 | 135.058 | 0.000 |

- e0: strict malicious-F1 0.953 [95% CI 0.929, 0.975]
- e1: strict malicious-F1 0.925 [95% CI 0.894, 0.952]
- e2: strict malicious-F1 0.914 [95% CI 0.880, 0.943]
- e2_bm25: strict malicious-F1 0.919 [95% CI 0.886, 0.948]
- e2_rerank: strict malicious-F1 0.928 [95% CI 0.897, 0.955]
- e3: strict malicious-F1 0.939 [95% CI 0.911, 0.964]
- e3a_exact: strict malicious-F1 0.942 [95% CI 0.914, 0.965]
- e3a: strict malicious-F1 0.928 [95% CI 0.896, 0.954]
- e4: strict malicious-F1 0.887 [95% CI 0.847, 0.921]
- e4_filter: strict malicious-F1 0.939 [95% CI 0.911, 0.964]
- e4_cite: strict malicious-F1 0.939 [95% CI 0.911, 0.964]

## Paired comparisons (same sample ids)

| pair | question | n | dF1 strict | CI95 | p | dAcc | McNemar p |
|---|---|---|---|---|---|---|---|
| e1_vs_e0 | RQ1 retrieval vs LLM-only | 310 | -0.028 | [-0.051, -0.011] | 0.002 | -0.032 | 0.006 |
| e2_vs_e1 | RQ2 hybrid vs dense | 310 | -0.011 | [-0.036, +0.013] | 0.472 | -0.013 | 0.481 |
| e3_vs_e2 | RQ3 KG-RAG vs hybrid | 310 | +0.025 | [-0.003, +0.054] | 0.074 | +0.029 | 0.122 |
| e3_vs_e3a_exact | RQ3 path structure vs exactly the same edges as separate sentences (E3a) | 310 | -0.003 | [-0.013, +0.006] | 0.758 | -0.003 | 1.000 |
| e3_vs_e3a | RQ3 graph traversal vs graph facts chosen by retrieval (E3b) | 310 | +0.011 | [-0.013, +0.036] | 0.442 | +0.013 | 0.481 |
| e4_vs_e3 | RQ4 verification vs none | 310 | -0.051 | [-0.084, -0.024] | 0.000 | -0.048 | 0.000 |
| e2_bm25_vs_e2 | ablation BM25 only vs RRF | 310 | +0.005 | [-0.017, +0.027] | 0.712 | +0.006 | 0.815 |
| e2_rerank_vs_e2 | ablation RRF+rerank vs RRF | 310 | +0.014 | [-0.010, +0.038] | 0.296 | +0.016 | 0.359 |
| e4_filter_vs_e3 | ablation claim filter | 310 | +0.000 | [+0.000, +0.000] | 1.000 | +0.000 | 1.000 |
| e4_cite_vs_e3 | ablation citation-only | 310 | +0.000 | [+0.000, +0.000] | 1.000 | +0.000 | 1.000 |
| e2_vs_e0 | vs e0 | 310 | -0.039 | [-0.063, -0.018] | 0.000 | -0.045 | 0.001 |
| e2_bm25_vs_e0 | vs e0 | 310 | -0.034 | [-0.059, -0.011] | 0.002 | -0.039 | 0.008 |
| e2_rerank_vs_e0 | vs e0 | 310 | -0.026 | [-0.048, -0.008] | 0.014 | -0.029 | 0.022 |
| e3_vs_e0 | vs e0 | 310 | -0.015 | [-0.037, +0.006] | 0.188 | -0.016 | 0.302 |
| e3a_exact_vs_e0 | vs e0 | 310 | -0.012 | [-0.033, +0.009] | 0.310 | -0.013 | 0.424 |
| e3a_vs_e0 | vs e0 | 310 | -0.026 | [-0.051, -0.005] | 0.016 | -0.029 | 0.035 |
| e4_vs_e0 | vs e0 | 310 | -0.066 | [-0.104, -0.030] | 0.000 | -0.065 | 0.000 |
| e4_filter_vs_e0 | vs e0 | 310 | -0.015 | [-0.037, +0.006] | 0.188 | -0.016 | 0.302 |
| e4_cite_vs_e0 | vs e0 | 310 | -0.015 | [-0.037, +0.006] | 0.188 | -0.016 | 0.302 |

## Grounding on the same denominator (E3 vs E4)

E4 reuses E3's answers: rates below divide by E3's original claims and by ALL items (abstained items included), so filtering cannot by itself make E4 look better.

| method | n_items | original_claims | unsupported_over_original_claims | unsupported_per_item | items_with_unsupported | supported_per_item | claims_retained | coverage |
|---|---|---|---|---|---|---|---|---|
| e3 | 310 | 1275 | 0.326 | 1.342 | 0.548 | 2.535 | 1.000 | 1.000 |
| e4 | 310 | 1275 | 0.000 | 0.000 | 0.000 | 2.535 | 0.616 | 0.948 |
| e4_filter | 310 | 1275 | 0.000 | 0.000 | 0.000 | 2.535 | 0.616 | 1.000 |
| e4_cite | 310 | 1275 | 0.326 | 1.342 | 0.548 | 2.535 | 1.000 | 1.000 |

## Cost (measured GPU time, allocated by tokens)

GPU seconds 328 for 2467 unique prompts; 0.054 s per 1k tokens; price/h 1.79

| method | tokens_per_item | gpu_seconds_per_item | usd_per_1000_items | reuses_e3_generation |
|---|---|---|---|---|
| e0 | 1676.945 | 0.091 | 0.045 | False |
| e1 | 2902.723 | 0.157 | 0.078 | False |
| e2 | 2949.958 | 0.160 | 0.080 | False |
| e2_bm25 | 2937.468 | 0.159 | 0.079 | False |
| e2_rerank | 2906.271 | 0.158 | 0.078 | False |
| e3 | 1983.510 | 0.108 | 0.053 | False |
| e3a_exact | 1991.926 | 0.108 | 0.054 | False |
| e3a | 2259.587 | 0.123 | 0.061 | False |
| e4 | 1983.510 | 0.000 | 0.000 | True |
| e4_filter | 1983.510 | 0.000 | 0.000 | True |
| e4_cite | 1983.510 | 0.000 | 0.000 | True |

## Strata

**e0**: all: n=310 acc=0.95 FPR=0.08
**e1**: all: n=310 acc=0.92 FPR=0.15
**e2**: all: n=310 acc=0.91 FPR=0.18
**e2_bm25**: all: n=310 acc=0.91 FPR=0.16
**e2_rerank**: all: n=310 acc=0.92 FPR=0.14
**e3**: all: n=310 acc=0.94 FPR=0.11
**e3a_exact**: all: n=310 acc=0.94 FPR=0.10
**e3a**: all: n=310 acc=0.92 FPR=0.14
**e4**: all: n=310 acc=0.89 FPR=0.11
**e4_filter**: all: n=310 acc=0.94 FPR=0.11
**e4_cite**: all: n=310 acc=0.94 FPR=0.11

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
