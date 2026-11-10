# Results: r03_test_malllm_Qwen2.5-Coder-7B-Instruct_nodup

Model: `vllm:Qwen/Qwen2.5-Coder-7B-Instruct` | split: `test` | n = 405 | methods: e0, e1, e2, e2_bm25, e2_rerank, e3, e3a_exact, e3a, e4, e4_filter, e4_cite

Strict = abstentions counted as errors. Claim labels check consistency with static facts, not truth.

## Headline

| method | strict/f1_malicious | strict/macro_f1 | strict/fpr | strict/fnr | answered/f1_malicious | coverage | aurc | rc_max_coverage | evidence/unsupported_claim_rate | evidence/unsupported_over_original_claims | evidence/items_with_unsupported_all | evidence/citation_accuracy | gold/evidence_coverage | gold/path_recall | tokens_in_mean | tokens_out_mean | latency_p50_ms | parse_failure_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e0 | 0.829 | 0.905 | 0.000 | 0.292 | 0.829 | 1.000 | 0.004 | 1.000 | 0.583 | 0.645 | 0.701 | 0.705 | nan | nan | 628.602 | 267.514 | 105.908 | 0.000 |
| e1 | 0.867 | 0.925 | 0.008 | 0.188 | 0.876 | 0.998 | 0.002 | 0.998 | 0.458 | 0.475 | 0.489 | 0.332 | nan | nan | 1369.242 | 231.235 | 105.908 | 0.002 |
| e2 | 0.911 | 0.950 | 0.003 | 0.146 | 0.911 | 1.000 | 0.001 | 1.000 | 0.452 | 0.468 | 0.509 | 0.319 | nan | nan | 1458.180 | 239.385 | 105.908 | 0.000 |
| e2_bm25 | 0.911 | 0.950 | 0.003 | 0.146 | 0.911 | 1.000 | 0.001 | 1.000 | 0.503 | 0.510 | 0.533 | 0.282 | nan | nan | 1537.435 | 235.906 | 105.908 | 0.000 |
| e2_rerank | 0.864 | 0.924 | 0.006 | 0.208 | 0.864 | 1.000 | 0.002 | 1.000 | 0.450 | 0.492 | 0.538 | 0.340 | nan | nan | 1503.336 | 238.985 | 105.908 | 0.000 |
| e3 | 0.569 | 0.743 | 0.126 | 0.229 | 0.569 | 1.000 | 0.045 | 1.000 | 0.459 | 0.455 | 0.570 | 0.607 | nan | nan | 784.198 | 233.010 | 105.908 | 0.000 |
| e3a_exact | 0.607 | 0.768 | 0.104 | 0.229 | 0.607 | 1.000 | 0.049 | 1.000 | 0.452 | 0.456 | 0.602 | 0.569 | nan | nan | 769.378 | 243.659 | 105.908 | 0.000 |
| e3a | 0.796 | 0.885 | 0.022 | 0.229 | 0.796 | 1.000 | 0.007 | 1.000 | 0.487 | 0.493 | 0.640 | 0.862 | nan | nan | 1015.240 | 257.469 | 105.908 | 0.000 |
| e4 | 0.547 | 0.731 | 0.126 | 0.271 | 0.560 | 0.993 | 0.045 | 0.993 | 0.000 | 0.000 | 0.000 | 0.999 | nan | nan | 784.198 | 233.010 | 105.908 | 0.000 |
| e4_filter | 0.569 | 0.743 | 0.126 | 0.229 | 0.569 | 1.000 | 0.045 | 1.000 | 0.000 | 0.000 | 0.000 | 0.999 | nan | nan | 784.198 | 233.010 | 105.908 | 0.000 |
| e4_cite | 0.569 | 0.743 | 0.126 | 0.229 | 0.569 | 1.000 | 0.045 | 1.000 | 0.459 | 0.455 | 0.570 | 0.607 | nan | nan | 784.198 | 233.010 | 105.908 | 0.000 |

- e0: strict malicious-F1 0.829 [95% CI 0.730, 0.914]
- e1: strict malicious-F1 0.867 [95% CI 0.785, 0.939]
- e2: strict malicious-F1 0.911 [95% CI 0.841, 0.965]
- e2_bm25: strict malicious-F1 0.911 [95% CI 0.844, 0.969]
- e2_rerank: strict malicious-F1 0.864 [95% CI 0.780, 0.933]
- e3: strict malicious-F1 0.569 [95% CI 0.458, 0.662]
- e3a_exact: strict malicious-F1 0.607 [95% CI 0.495, 0.703]
- e3a: strict malicious-F1 0.796 [95% CI 0.694, 0.881]
- e4: strict malicious-F1 0.547 [95% CI 0.432, 0.641]
- e4_filter: strict malicious-F1 0.569 [95% CI 0.458, 0.662]
- e4_cite: strict malicious-F1 0.569 [95% CI 0.458, 0.662]

## Paired comparisons (same sample ids)

| pair | question | n | dF1 strict | CI95 | p | dAcc | McNemar p |
|---|---|---|---|---|---|---|---|
| e1_vs_e0 | RQ1 retrieval vs LLM-only | 405 | +0.037 | [-0.027, +0.111] | 0.282 | +0.005 | 0.727 |
| e2_vs_e1 | RQ2 hybrid vs dense | 405 | +0.044 | [+0.000, +0.101] | 0.078 | +0.010 | 0.219 |
| e3_vs_e2 | RQ3 KG-RAG vs hybrid | 405 | -0.342 | [-0.453, -0.246] | 0.000 | -0.119 | 0.000 |
| e3_vs_e3a_exact | RQ3 path structure vs exactly the same edges as separate sentences (E3a) | 405 | -0.037 | [-0.074, +0.000] | 0.076 | -0.020 | 0.096 |
| e3_vs_e3a | RQ3 graph traversal vs graph facts chosen by retrieval (E3b) | 405 | -0.226 | [-0.308, -0.159] | 0.000 | -0.091 | 0.000 |
| e4_vs_e3 | RQ4 verification vs none | 405 | -0.022 | [-0.063, +0.000] | 0.260 | -0.005 | 0.500 |
| e2_bm25_vs_e2 | ablation BM25 only vs RRF | 405 | +0.000 | [-0.043, +0.044] | 1.000 | +0.000 | 1.000 |
| e2_rerank_vs_e2 | ablation RRF+rerank vs RRF | 405 | -0.047 | [-0.116, +0.012] | 0.148 | -0.010 | 0.289 |
| e4_filter_vs_e3 | ablation claim filter | 405 | +0.000 | [+0.000, +0.000] | 1.000 | +0.000 | 1.000 |
| e4_cite_vs_e3 | ablation citation-only | 405 | +0.000 | [+0.000, +0.000] | 1.000 | +0.000 | 1.000 |
| e2_vs_e0 | vs e0 | 405 | +0.082 | [+0.017, +0.164] | 0.012 | +0.015 | 0.070 |
| e2_bm25_vs_e0 | vs e0 | 405 | +0.082 | [+0.017, +0.160] | 0.012 | +0.015 | 0.070 |
| e2_rerank_vs_e0 | vs e0 | 405 | +0.034 | [-0.023, +0.106] | 0.274 | +0.005 | 0.688 |
| e3_vs_e0 | vs e0 | 405 | -0.260 | [-0.364, -0.166] | 0.000 | -0.104 | 0.000 |
| e3a_exact_vs_e0 | vs e0 | 405 | -0.223 | [-0.322, -0.134] | 0.000 | -0.084 | 0.000 |
| e3a_vs_e0 | vs e0 | 405 | -0.034 | [-0.105, +0.032] | 0.342 | -0.012 | 0.227 |
| e4_vs_e0 | vs e0 | 405 | -0.282 | [-0.391, -0.183] | 0.000 | -0.109 | 0.000 |
| e4_filter_vs_e0 | vs e0 | 405 | -0.260 | [-0.364, -0.166] | 0.000 | -0.104 | 0.000 |
| e4_cite_vs_e0 | vs e0 | 405 | -0.260 | [-0.364, -0.166] | 0.000 | -0.104 | 0.000 |

## Grounding on the same denominator (E3 vs E4)

E4 reuses E3's answers: rates below divide by E3's original claims and by ALL items (abstained items included), so filtering cannot by itself make E4 look better.

| method | n_items | original_claims | unsupported_over_original_claims | unsupported_per_item | items_with_unsupported | supported_per_item | claims_retained | coverage |
|---|---|---|---|---|---|---|---|---|
| e3 | 405 | 1665 | 0.455 | 1.872 | 0.570 | 2.099 | 1.000 | 1.000 |
| e4 | 405 | 1665 | 0.000 | 0.000 | 0.000 | 2.099 | 0.511 | 0.993 |
| e4_filter | 405 | 1665 | 0.000 | 0.000 | 0.000 | 2.099 | 0.511 | 1.000 |
| e4_cite | 405 | 1665 | 0.455 | 1.872 | 0.570 | 2.099 | 1.000 | 1.000 |

## Cost (measured GPU time, allocated by tokens)

GPU seconds 361 for 3207 unique prompts; 0.082 s per 1k tokens; price/h 1.09

| method | tokens_per_item | gpu_seconds_per_item | usd_per_1000_items | reuses_e3_generation |
|---|---|---|---|---|
| e0 | 896.116 | 0.073 | 0.022 | False |
| e1 | 1600.477 | 0.131 | 0.040 | False |
| e2 | 1697.565 | 0.139 | 0.042 | False |
| e2_bm25 | 1773.341 | 0.145 | 0.044 | False |
| e2_rerank | 1742.321 | 0.143 | 0.043 | False |
| e3 | 1017.207 | 0.083 | 0.025 | False |
| e3a_exact | 1013.037 | 0.083 | 0.025 | False |
| e3a | 1272.709 | 0.104 | 0.032 | False |
| e4 | 1017.207 | 0.000 | 0.000 | True |
| e4_filter | 1017.207 | 0.000 | 0.000 | True |
| e4_cite | 1017.207 | 0.000 | 0.000 | True |

## Strata

**e0**: all: n=405 acc=0.97 FPR=0.00
**e1**: all: n=405 acc=0.97 FPR=0.01
**e2**: all: n=405 acc=0.98 FPR=0.00
**e2_bm25**: all: n=405 acc=0.98 FPR=0.00
**e2_rerank**: all: n=405 acc=0.97 FPR=0.01
**e3**: all: n=405 acc=0.86 FPR=0.13
**e3a_exact**: all: n=405 acc=0.88 FPR=0.10
**e3a**: all: n=405 acc=0.95 FPR=0.02
**e4**: all: n=405 acc=0.86 FPR=0.13
**e4_filter**: all: n=405 acc=0.86 FPR=0.13
**e4_cite**: all: n=405 acc=0.86 FPR=0.13

## kb_report

```json
{
 "sources": [
  "builtin",
  "examples",
  "malllm_code",
  "pyguard",
  "yara"
 ],
 "counts_before_audit": {
  "builtin": 30,
  "examples": 1842,
  "malllm_code": 744,
  "pyguard": 304,
  "yara": 8
 },
 "n_docs": 2859,
 "n_dropped": 69,
 "leakage_before": {
  "n_docs": 2928,
  "docs_from_eval_samples": [],
  "n_docs_from_eval_samples": 0,
  "docs_mentioning_eval_package": [],
  "n_docs_mentioning_eval_package": 0,
  "docs_with_eval_code": [
   "ex:malllm-train-m-00105:C1",
   "ex:malllm-train-m-00116:C1",
   "ex:malllm-train-m-00815:C1",
   "ex:malllm-train-m-00718:C1",
   "ex:malllm-train-m-00992:C1",
   "ex:malllm-train-m-00755:C1",
   "ex:malllm-train-m-00579:C1",
   "ex:malllm-train-m-00103:C1",
   "ex:malllm-train-m-00995:C1",
   "ex:malllm-train-m-01157:C1"
  ],
  "n_docs_with_eval_code": 69,
  "clean": false
 },
 "leakage_after": {
  "n_docs": 2859,
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
  "malicious_example": 1019,
  "benign_example": 1498,
  "pyguard_pattern": 304,
  "yara": 8
 }
}
```
