# Results: r03_test_malllm_Qwen2.5-Coder-7B-Instruct

Model: `vllm:Qwen/Qwen2.5-Coder-7B-Instruct` | split: `test` | n = 500 | methods: e0, e1, e2, e2_bm25, e2_rerank, e3, e3a_exact, e3a, e4, e4_filter, e4_cite

Strict = abstentions counted as errors. Claim labels check consistency with static facts, not truth.

## Headline

| method | strict/f1_malicious | strict/macro_f1 | strict/fpr | strict/fnr | answered/f1_malicious | coverage | aurc | rc_max_coverage | evidence/unsupported_claim_rate | evidence/unsupported_over_original_claims | evidence/items_with_unsupported_all | evidence/citation_accuracy | gold/evidence_coverage | gold/path_recall | tokens_in_mean | tokens_out_mean | latency_p50_ms | parse_failure_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e0 | 0.866 | 0.912 | 0.003 | 0.231 | 0.866 | 1.000 | 0.014 | 1.000 | 0.503 | 0.624 | 0.716 | 0.607 | nan | nan | 653.358 | 270.524 | 105.908 | 0.000 |
| e1 | 0.934 | 0.956 | 0.011 | 0.097 | 0.938 | 0.998 | 0.003 | 0.998 | 0.424 | 0.452 | 0.488 | 0.327 | nan | nan | 1432.676 | 229.296 | 105.908 | 0.002 |
| e2 | 0.945 | 0.963 | 0.003 | 0.097 | 0.949 | 0.998 | 0.003 | 0.998 | 0.415 | 0.447 | 0.504 | 0.323 | nan | nan | 1509.850 | 238.552 | 105.908 | 0.002 |
| e2_bm25 | 0.946 | 0.963 | 0.005 | 0.090 | 0.946 | 1.000 | 0.003 | 1.000 | 0.460 | 0.485 | 0.526 | 0.296 | nan | nan | 1575.600 | 232.844 | 105.908 | 0.000 |
| e2_rerank | 0.938 | 0.958 | 0.005 | 0.104 | 0.938 | 1.000 | 0.004 | 1.000 | 0.401 | 0.488 | 0.586 | 0.333 | nan | nan | 1596.640 | 251.764 | 105.908 | 0.000 |
| e3 | 0.760 | 0.831 | 0.128 | 0.172 | 0.760 | 1.000 | 0.057 | 1.000 | 0.414 | 0.416 | 0.534 | 0.655 | nan | nan | 839.300 | 230.484 | 105.908 | 0.000 |
| e3a_exact | 0.766 | 0.837 | 0.109 | 0.194 | 0.774 | 0.994 | 0.066 | 0.994 | 0.394 | 0.441 | 0.630 | 0.564 | nan | nan | 819.094 | 253.118 | 105.908 | 0.006 |
| e3a | 0.865 | 0.910 | 0.025 | 0.187 | 0.865 | 1.000 | 0.013 | 1.000 | 0.448 | 0.462 | 0.648 | 0.833 | nan | nan | 1058.902 | 260.660 | 105.908 | 0.000 |
| e4 | 0.743 | 0.820 | 0.128 | 0.201 | 0.756 | 0.990 | 0.057 | 0.990 | 0.000 | 0.000 | 0.000 | 0.998 | nan | nan | 839.300 | 230.484 | 105.908 | 0.000 |
| e4_filter | 0.760 | 0.831 | 0.128 | 0.172 | 0.760 | 1.000 | 0.057 | 1.000 | 0.000 | 0.000 | 0.000 | 0.998 | nan | nan | 839.300 | 230.484 | 105.908 | 0.000 |
| e4_cite | 0.760 | 0.831 | 0.128 | 0.172 | 0.760 | 1.000 | 0.057 | 1.000 | 0.414 | 0.416 | 0.534 | 0.655 | nan | nan | 839.300 | 230.484 | 105.908 | 0.000 |

- e0: strict malicious-F1 0.866 [95% CI 0.816, 0.910]
- e1: strict malicious-F1 0.934 [95% CI 0.901, 0.962]
- e2: strict malicious-F1 0.945 [95% CI 0.916, 0.971]
- e2_bm25: strict malicious-F1 0.946 [95% CI 0.917, 0.972]
- e2_rerank: strict malicious-F1 0.938 [95% CI 0.905, 0.964]
- e3: strict malicious-F1 0.760 [95% CI 0.701, 0.810]
- e3a_exact: strict malicious-F1 0.766 [95% CI 0.706, 0.820]
- e3a: strict malicious-F1 0.865 [95% CI 0.815, 0.904]
- e4: strict malicious-F1 0.743 [95% CI 0.681, 0.794]
- e4_filter: strict malicious-F1 0.760 [95% CI 0.701, 0.810]
- e4_cite: strict malicious-F1 0.760 [95% CI 0.701, 0.810]

## Paired comparisons (same sample ids)

| pair | question | n | dF1 strict | CI95 | p | dAcc | McNemar p |
|---|---|---|---|---|---|---|---|
| e1_vs_e0 | RQ1 retrieval vs LLM-only | 500 | +0.069 | [+0.025, +0.149] | 0.000 | +0.030 | 0.001 |
| e2_vs_e1 | RQ2 hybrid vs dense | 500 | +0.011 | [-0.019, +0.048] | 0.456 | +0.006 | 0.549 |
| e3_vs_e2 | RQ3 KG-RAG vs hybrid | 500 | -0.185 | [-0.303, -0.106] | 0.000 | -0.112 | 0.000 |
| e3_vs_e3a_exact | RQ3 path structure vs exactly the same edges as separate sentences (E3a) | 500 | -0.006 | [-0.040, +0.034] | 0.724 | -0.008 | 0.523 |
| e3_vs_e3a | RQ3 graph traversal vs graph facts chosen by retrieval (E3b) | 500 | -0.105 | [-0.169, -0.060] | 0.000 | -0.072 | 0.000 |
| e4_vs_e3 | RQ4 verification vs none | 500 | -0.017 | [-0.046, -0.003] | 0.028 | -0.008 | 0.125 |
| e2_bm25_vs_e2 | ablation BM25 only vs RRF | 500 | +0.000 | [-0.021, +0.023] | 0.942 | +0.000 | 1.000 |
| e2_rerank_vs_e2 | ablation RRF+rerank vs RRF | 500 | -0.008 | [-0.044, +0.024] | 0.680 | -0.004 | 0.774 |
| e4_filter_vs_e3 | ablation claim filter | 500 | +0.000 | [+0.000, +0.000] | 1.000 | +0.000 | 1.000 |
| e4_cite_vs_e3 | ablation citation-only | 500 | +0.000 | [+0.000, +0.000] | 1.000 | +0.000 | 1.000 |
| e2_vs_e0 | vs e0 | 500 | +0.080 | [+0.031, +0.168] | 0.000 | +0.036 | 0.000 |
| e2_bm25_vs_e0 | vs e0 | 500 | +0.080 | [+0.031, +0.170] | 0.000 | +0.036 | 0.000 |
| e2_rerank_vs_e0 | vs e0 | 500 | +0.072 | [+0.026, +0.154] | 0.000 | +0.032 | 0.000 |
| e3_vs_e0 | vs e0 | 500 | -0.105 | [-0.178, -0.055] | 0.000 | -0.076 | 0.000 |
| e3a_exact_vs_e0 | vs e0 | 500 | -0.100 | [-0.179, -0.045] | 0.000 | -0.068 | 0.000 |
| e3a_vs_e0 | vs e0 | 500 | -0.000 | [-0.035, +0.041] | 0.962 | -0.004 | 0.791 |
| e4_vs_e0 | vs e0 | 500 | -0.122 | [-0.210, -0.066] | 0.000 | -0.084 | 0.000 |
| e4_filter_vs_e0 | vs e0 | 500 | -0.105 | [-0.178, -0.055] | 0.000 | -0.076 | 0.000 |
| e4_cite_vs_e0 | vs e0 | 500 | -0.105 | [-0.178, -0.055] | 0.000 | -0.076 | 0.000 |

## Grounding on the same denominator (E3 vs E4)

E4 reuses E3's answers: rates below divide by E3's original claims and by ALL items (abstained items included), so filtering cannot by itself make E4 look better.

| method | n_items | original_claims | unsupported_over_original_claims | unsupported_per_item | items_with_unsupported | supported_per_item | claims_retained | coverage |
|---|---|---|---|---|---|---|---|---|
| e3 | 500 | 1988 | 0.416 | 1.656 | 0.534 | 2.182 | 1.000 | 1.000 |
| e4 | 500 | 1988 | 0.000 | 0.000 | 0.000 | 2.182 | 0.549 | 0.990 |
| e4_filter | 500 | 1988 | 0.000 | 0.000 | 0.000 | 2.182 | 0.549 | 1.000 |
| e4_cite | 500 | 1988 | 0.416 | 1.656 | 0.534 | 2.182 | 1.000 | 1.000 |

## Cost (measured GPU time, allocated by tokens)

GPU seconds 455 for 3934 unique prompts; 0.081 s per 1k tokens; price/h 1.09

| method | tokens_per_item | gpu_seconds_per_item | usd_per_1000_items | reuses_e3_generation |
|---|---|---|---|---|
| e0 | 923.882 | 0.075 | 0.023 | False |
| e1 | 1661.972 | 0.135 | 0.041 | False |
| e2 | 1748.402 | 0.142 | 0.043 | False |
| e2_bm25 | 1808.444 | 0.146 | 0.044 | False |
| e2_rerank | 1848.404 | 0.150 | 0.045 | False |
| e3 | 1069.784 | 0.087 | 0.026 | False |
| e3a_exact | 1072.212 | 0.087 | 0.026 | False |
| e3a | 1319.562 | 0.107 | 0.032 | False |
| e4 | 1069.784 | 0.000 | 0.000 | True |
| e4_filter | 1069.784 | 0.000 | 0.000 | True |
| e4_cite | 1069.784 | 0.000 | 0.000 | True |

## Strata

**e0**: all: n=500 acc=0.94 FPR=0.00
**e1**: all: n=500 acc=0.97 FPR=0.01
**e2**: all: n=500 acc=0.97 FPR=0.00
**e2_bm25**: all: n=500 acc=0.97 FPR=0.01
**e2_rerank**: all: n=500 acc=0.97 FPR=0.01
**e3**: all: n=500 acc=0.86 FPR=0.13
**e3a_exact**: all: n=500 acc=0.87 FPR=0.11
**e3a**: all: n=500 acc=0.93 FPR=0.02
**e4**: all: n=500 acc=0.85 FPR=0.13
**e4_filter**: all: n=500 acc=0.86 FPR=0.13
**e4_cite**: all: n=500 acc=0.86 FPR=0.13

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
