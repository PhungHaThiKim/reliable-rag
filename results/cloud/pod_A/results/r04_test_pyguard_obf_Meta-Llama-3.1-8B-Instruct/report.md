# Results: r04_test_pyguard_obf_Meta-Llama-3.1-8B-Instruct

Model: `vllm:NousResearch/Meta-Llama-3.1-8B-Instruct` | split: `test` | n = 310 | methods: e0, e1, e2, e2_bm25, e2_rerank, e3, e3a_exact, e3a, e4, e4_filter, e4_cite

Strict = abstentions counted as errors. Claim labels check consistency with static facts, not truth.

## Headline

| method | strict/f1_malicious | strict/macro_f1 | strict/fpr | strict/fnr | answered/f1_malicious | coverage | aurc | rc_max_coverage | evidence/unsupported_claim_rate | evidence/unsupported_over_original_claims | evidence/items_with_unsupported_all | evidence/citation_accuracy | gold/evidence_coverage | gold/path_recall | tokens_in_mean | tokens_out_mean | latency_p50_ms | parse_failure_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e0 | 0.754 | 0.629 | 0.662 | 0.000 | 0.755 | 0.997 | 0.061 | 0.997 | 0.244 | 0.480 | 0.829 | 0.409 | nan | nan | 1371.223 | 285.506 | 163.870 | 0.003 |
| e1 | 0.727 | 0.557 | 0.760 | 0.000 | 0.729 | 0.997 | 0.084 | 0.997 | 0.376 | 0.525 | 0.842 | 0.395 | nan | nan | 2578.300 | 294.574 | 163.870 | 0.006 |
| e2 | 0.731 | 0.567 | 0.747 | 0.000 | 0.731 | 1.000 | 0.080 | 1.000 | 0.328 | 0.451 | 0.839 | 0.497 | nan | nan | 2618.061 | 293.971 | 163.870 | 0.000 |
| e2_bm25 | 0.729 | 0.562 | 0.753 | 0.000 | 0.731 | 0.997 | 0.081 | 0.997 | 0.388 | 0.507 | 0.855 | 0.407 | nan | nan | 2613.435 | 296.071 | 163.870 | 0.003 |
| e2_rerank | 0.727 | 0.557 | 0.760 | 0.000 | 0.729 | 0.997 | 0.084 | 0.997 | 0.397 | 0.535 | 0.842 | 0.406 | nan | nan | 2599.129 | 294.355 | 163.870 | 0.006 |
| e3 | 0.805 | 0.743 | 0.481 | 0.006 | 0.805 | 1.000 | 0.047 | 1.000 | 0.277 | 0.366 | 0.658 | 0.630 | nan | nan | 1695.342 | 258.626 | 163.870 | 0.003 |
| e3a_exact | 0.814 | 0.758 | 0.455 | 0.006 | 0.814 | 1.000 | 0.036 | 1.000 | 0.308 | 0.400 | 0.687 | 0.601 | nan | nan | 1677.061 | 263.284 | 163.870 | 0.000 |
| e3a | 0.722 | 0.542 | 0.779 | 0.000 | 0.722 | 1.000 | 0.090 | 1.000 | 0.245 | 0.339 | 0.681 | 0.668 | nan | nan | 1934.313 | 294.561 | 163.870 | 0.000 |
| e4 | 0.757 | 0.698 | 0.481 | 0.103 | 0.918 | 0.790 | 0.021 | 0.790 | 0.000 | 0.000 | 0.000 | 0.983 | nan | nan | 1695.342 | 258.626 | 163.870 | 0.003 |
| e4_filter | 0.805 | 0.743 | 0.481 | 0.006 | 0.805 | 1.000 | 0.047 | 1.000 | 0.000 | 0.000 | 0.000 | 0.983 | nan | nan | 1695.342 | 258.626 | 163.870 | 0.003 |
| e4_cite | 0.802 | 0.740 | 0.481 | 0.013 | 0.804 | 0.997 | 0.046 | 0.997 | 0.277 | 0.366 | 0.658 | 0.630 | nan | nan | 1695.342 | 258.626 | 163.870 | 0.003 |

- e0: strict malicious-F1 0.754 [95% CI 0.708, 0.795]
- e1: strict malicious-F1 0.727 [95% CI 0.679, 0.775]
- e2: strict malicious-F1 0.731 [95% CI 0.683, 0.776]
- e2_bm25: strict malicious-F1 0.729 [95% CI 0.681, 0.773]
- e2_rerank: strict malicious-F1 0.727 [95% CI 0.680, 0.772]
- e3: strict malicious-F1 0.805 [95% CI 0.762, 0.844]
- e3a_exact: strict malicious-F1 0.814 [95% CI 0.768, 0.852]
- e3a: strict malicious-F1 0.722 [95% CI 0.675, 0.769]
- e4: strict malicious-F1 0.757 [95% CI 0.707, 0.801]
- e4_filter: strict malicious-F1 0.805 [95% CI 0.762, 0.844]
- e4_cite: strict malicious-F1 0.802 [95% CI 0.758, 0.842]

## Paired comparisons (same sample ids)

| pair | question | n | dF1 strict | CI95 | p | dAcc | McNemar p |
|---|---|---|---|---|---|---|---|
| e1_vs_e0 | RQ1 retrieval vs LLM-only | 310 | -0.026 | [-0.045, -0.009] | 0.002 | -0.048 | 0.003 |
| e2_vs_e1 | RQ2 hybrid vs dense | 310 | +0.003 | [-0.005, +0.013] | 0.584 | +0.006 | 0.727 |
| e3_vs_e2 | RQ3 KG-RAG vs hybrid | 310 | +0.075 | [+0.051, +0.100] | 0.000 | +0.129 | 0.000 |
| e3_vs_e3a_exact | RQ3 path structure vs exactly the same edges as separate sentences (E3a) | 310 | -0.008 | [-0.018, -0.002] | 0.032 | -0.013 | 0.125 |
| e3_vs_e3a | RQ3 graph traversal vs graph facts chosen by retrieval (E3b) | 310 | +0.083 | [+0.057, +0.112] | 0.000 | +0.145 | 0.000 |
| e4_vs_e3 | RQ4 verification vs none | 310 | -0.048 | [-0.073, -0.024] | 0.000 | -0.048 | 0.000 |
| e2_bm25_vs_e2 | ablation BM25 only vs RRF | 310 | -0.002 | [-0.016, +0.012] | 0.948 | -0.003 | 1.000 |
| e2_rerank_vs_e2 | ablation RRF+rerank vs RRF | 310 | -0.003 | [-0.020, +0.012] | 0.764 | -0.006 | 0.824 |
| e4_filter_vs_e3 | ablation claim filter | 310 | +0.000 | [+0.000, +0.000] | 1.000 | +0.000 | 1.000 |
| e4_cite_vs_e3 | ablation citation-only | 310 | -0.003 | [-0.012, +0.000] | 0.724 | -0.003 | 1.000 |
| e2_vs_e0 | vs e0 | 310 | -0.023 | [-0.040, -0.007] | 0.008 | -0.042 | 0.011 |
| e2_bm25_vs_e0 | vs e0 | 310 | -0.025 | [-0.041, -0.010] | 0.000 | -0.045 | 0.003 |
| e2_rerank_vs_e0 | vs e0 | 310 | -0.026 | [-0.044, -0.011] | 0.000 | -0.048 | 0.001 |
| e3_vs_e0 | vs e0 | 310 | +0.052 | [+0.030, +0.074] | 0.000 | +0.087 | 0.000 |
| e3a_exact_vs_e0 | vs e0 | 310 | +0.060 | [+0.039, +0.085] | 0.000 | +0.100 | 0.000 |
| e3a_vs_e0 | vs e0 | 310 | -0.031 | [-0.052, -0.014] | 0.000 | -0.058 | 0.001 |
| e4_vs_e0 | vs e0 | 310 | +0.003 | [-0.029, +0.037] | 0.850 | +0.039 | 0.104 |
| e4_filter_vs_e0 | vs e0 | 310 | +0.052 | [+0.030, +0.074] | 0.000 | +0.087 | 0.000 |
| e4_cite_vs_e0 | vs e0 | 310 | +0.048 | [+0.026, +0.073] | 0.000 | +0.084 | 0.000 |

## Grounding on the same denominator (E3 vs E4)

E4 reuses E3's answers: rates below divide by E3's original claims and by ALL items (abstained items included), so filtering cannot by itself make E4 look better.

| method | n_items | original_claims | unsupported_over_original_claims | unsupported_per_item | items_with_unsupported | supported_per_item | claims_retained | coverage |
|---|---|---|---|---|---|---|---|---|
| e3 | 310 | 1340 | 0.366 | 1.584 | 0.658 | 2.671 | 1.000 | 1.000 |
| e4 | 310 | 1340 | 0.000 | 0.000 | 0.000 | 2.671 | 0.618 | 0.790 |
| e4_filter | 310 | 1340 | 0.000 | 0.000 | 0.000 | 2.671 | 0.618 | 1.000 |
| e4_cite | 310 | 1340 | 0.366 | 1.584 | 0.658 | 2.671 | 1.000 | 0.997 |

## Cost (measured GPU time, allocated by tokens)

GPU seconds 413 for 2466 unique prompts; 0.069 s per 1k tokens; price/h 1.79

| method | tokens_per_item | gpu_seconds_per_item | usd_per_1000_items | reuses_e3_generation |
|---|---|---|---|---|
| e0 | 1656.729 | 0.115 | 0.057 | False |
| e1 | 2872.874 | 0.199 | 0.099 | False |
| e2 | 2912.032 | 0.201 | 0.100 | False |
| e2_bm25 | 2909.506 | 0.201 | 0.100 | False |
| e2_rerank | 2893.484 | 0.200 | 0.099 | False |
| e3 | 1953.968 | 0.135 | 0.067 | False |
| e3a_exact | 1940.345 | 0.134 | 0.067 | False |
| e3a | 2228.874 | 0.154 | 0.077 | False |
| e4 | 1953.968 | 0.000 | 0.000 | True |
| e4_filter | 1953.968 | 0.000 | 0.000 | True |
| e4_cite | 1953.968 | 0.000 | 0.000 | True |

## Strata

**e0**: all: n=310 acc=0.67 FPR=0.66
**e1**: all: n=310 acc=0.62 FPR=0.76
**e2**: all: n=310 acc=0.63 FPR=0.75
**e2_bm25**: all: n=310 acc=0.63 FPR=0.75
**e2_rerank**: all: n=310 acc=0.62 FPR=0.76
**e3**: all: n=310 acc=0.76 FPR=0.48
**e3a_exact**: all: n=310 acc=0.77 FPR=0.45
**e3a**: all: n=310 acc=0.61 FPR=0.78
**e4**: all: n=310 acc=0.71 FPR=0.48
**e4_filter**: all: n=310 acc=0.76 FPR=0.48
**e4_cite**: all: n=310 acc=0.75 FPR=0.48

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
