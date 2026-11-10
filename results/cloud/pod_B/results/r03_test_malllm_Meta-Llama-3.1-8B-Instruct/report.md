# Results: r03_test_malllm_Meta-Llama-3.1-8B-Instruct

Model: `vllm:NousResearch/Meta-Llama-3.1-8B-Instruct` | split: `test` | n = 500 | methods: e0, e1, e2, e2_bm25, e2_rerank, e3, e3a_exact, e3a, e4, e4_filter, e4_cite

Strict = abstentions counted as errors. Claim labels check consistency with static facts, not truth.

## Headline

| method | strict/f1_malicious | strict/macro_f1 | strict/fpr | strict/fnr | answered/f1_malicious | coverage | aurc | rc_max_coverage | evidence/unsupported_claim_rate | evidence/unsupported_over_original_claims | evidence/items_with_unsupported_all | evidence/citation_accuracy | gold/evidence_coverage | gold/path_recall | tokens_in_mean | tokens_out_mean | latency_p50_ms | parse_failure_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e0 | 0.822 | 0.877 | 0.077 | 0.157 | 0.825 | 0.998 | 0.011 | 0.998 | 0.567 | 0.660 | 0.726 | 0.308 | nan | nan | 633.816 | 274.556 | 132.436 | 0.002 |
| e1 | 0.674 | 0.748 | 0.273 | 0.112 | 0.674 | 1.000 | 0.035 | 1.000 | 0.536 | 0.563 | 0.698 | 0.335 | nan | nan | 1387.596 | 251.082 | 132.436 | 0.000 |
| e2 | 0.709 | 0.782 | 0.221 | 0.119 | 0.709 | 1.000 | 0.031 | 1.000 | 0.570 | 0.589 | 0.756 | 0.293 | nan | nan | 1452.980 | 258.432 | 132.436 | 0.000 |
| e2_bm25 | 0.706 | 0.783 | 0.205 | 0.149 | 0.706 | 1.000 | 0.028 | 1.000 | 0.536 | 0.555 | 0.690 | 0.336 | nan | nan | 1517.618 | 262.556 | 132.436 | 0.000 |
| e2_rerank | 0.700 | 0.774 | 0.232 | 0.119 | 0.702 | 0.998 | 0.029 | 0.998 | 0.515 | 0.558 | 0.694 | 0.333 | nan | nan | 1544.126 | 258.052 | 132.436 | 0.002 |
| e3 | 0.704 | 0.782 | 0.202 | 0.157 | 0.704 | 1.000 | 0.043 | 1.000 | 0.488 | 0.508 | 0.568 | 0.557 | nan | nan | 813.802 | 239.758 | 132.436 | 0.000 |
| e3a_exact | 0.718 | 0.795 | 0.180 | 0.164 | 0.720 | 0.998 | 0.035 | 0.998 | 0.495 | 0.532 | 0.678 | 0.477 | nan | nan | 794.482 | 243.772 | 132.436 | 0.002 |
| e3a | 0.699 | 0.762 | 0.290 | 0.037 | 0.703 | 0.996 | 0.027 | 0.996 | 0.472 | 0.510 | 0.750 | 0.510 | nan | nan | 1031.724 | 271.476 | 132.436 | 0.004 |
| e4 | 0.679 | 0.766 | 0.202 | 0.201 | 0.709 | 0.974 | 0.041 | 0.974 | 0.000 | 0.000 | 0.000 | 1.000 | nan | nan | 813.802 | 239.758 | 132.436 | 0.000 |
| e4_filter | 0.704 | 0.782 | 0.202 | 0.157 | 0.704 | 1.000 | 0.043 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | nan | nan | 813.802 | 239.758 | 132.436 | 0.000 |
| e4_cite | 0.700 | 0.779 | 0.202 | 0.164 | 0.702 | 0.998 | 0.043 | 0.998 | 0.488 | 0.508 | 0.568 | 0.557 | nan | nan | 813.802 | 239.758 | 132.436 | 0.000 |

- e0: strict malicious-F1 0.822 [95% CI 0.771, 0.868]
- e1: strict malicious-F1 0.674 [95% CI 0.612, 0.726]
- e2: strict malicious-F1 0.709 [95% CI 0.649, 0.763]
- e2_bm25: strict malicious-F1 0.706 [95% CI 0.644, 0.759]
- e2_rerank: strict malicious-F1 0.700 [95% CI 0.638, 0.752]
- e3: strict malicious-F1 0.704 [95% CI 0.643, 0.758]
- e3a_exact: strict malicious-F1 0.718 [95% CI 0.656, 0.771]
- e3a: strict malicious-F1 0.699 [95% CI 0.644, 0.750]
- e4: strict malicious-F1 0.679 [95% CI 0.615, 0.735]
- e4_filter: strict malicious-F1 0.704 [95% CI 0.643, 0.758]
- e4_cite: strict malicious-F1 0.700 [95% CI 0.637, 0.754]

## Paired comparisons (same sample ids)

| pair | question | n | dF1 strict | CI95 | p | dAcc | McNemar p |
|---|---|---|---|---|---|---|---|
| e1_vs_e0 | RQ1 retrieval vs LLM-only | 500 | -0.148 | [-0.221, -0.092] | 0.000 | -0.132 | 0.000 |
| e2_vs_e1 | RQ2 hybrid vs dense | 500 | +0.034 | [-0.003, +0.073] | 0.074 | +0.036 | 0.036 |
| e3_vs_e2 | RQ3 KG-RAG vs hybrid | 500 | -0.005 | [-0.053, +0.038] | 0.850 | +0.004 | 0.915 |
| e3_vs_e3a_exact | RQ3 path structure vs exactly the same edges as separate sentences (E3a) | 500 | -0.014 | [-0.030, +0.002] | 0.074 | -0.014 | 0.039 |
| e3_vs_e3a | RQ3 graph traversal vs graph facts chosen by retrieval (E3b) | 500 | +0.005 | [-0.051, +0.049] | 0.908 | +0.032 | 0.101 |
| e4_vs_e3 | RQ4 verification vs none | 500 | -0.025 | [-0.058, -0.006] | 0.002 | -0.012 | 0.031 |
| e2_bm25_vs_e2 | ablation BM25 only vs RRF | 500 | -0.003 | [-0.039, +0.032] | 0.874 | +0.004 | 0.890 |
| e2_rerank_vs_e2 | ablation RRF+rerank vs RRF | 500 | -0.008 | [-0.040, +0.020] | 0.620 | -0.008 | 0.665 |
| e4_filter_vs_e3 | ablation claim filter | 500 | +0.000 | [+0.000, +0.000] | 1.000 | +0.000 | 1.000 |
| e4_cite_vs_e3 | ablation citation-only | 500 | -0.004 | [-0.016, +0.000] | 0.756 | -0.002 | 1.000 |
| e2_vs_e0 | vs e0 | 500 | -0.113 | [-0.177, -0.063] | 0.000 | -0.096 | 0.000 |
| e2_bm25_vs_e0 | vs e0 | 500 | -0.116 | [-0.175, -0.071] | 0.000 | -0.092 | 0.000 |
| e2_rerank_vs_e0 | vs e0 | 500 | -0.122 | [-0.184, -0.071] | 0.000 | -0.104 | 0.000 |
| e3_vs_e0 | vs e0 | 500 | -0.118 | [-0.185, -0.068] | 0.000 | -0.092 | 0.000 |
| e3a_exact_vs_e0 | vs e0 | 500 | -0.104 | [-0.174, -0.057] | 0.000 | -0.078 | 0.000 |
| e3a_vs_e0 | vs e0 | 500 | -0.123 | [-0.194, -0.061] | 0.000 | -0.124 | 0.000 |
| e4_vs_e0 | vs e0 | 500 | -0.142 | [-0.223, -0.085] | 0.000 | -0.104 | 0.000 |
| e4_filter_vs_e0 | vs e0 | 500 | -0.118 | [-0.185, -0.068] | 0.000 | -0.092 | 0.000 |
| e4_cite_vs_e0 | vs e0 | 500 | -0.122 | [-0.192, -0.070] | 0.000 | -0.094 | 0.000 |

## Grounding on the same denominator (E3 vs E4)

E4 reuses E3's answers: rates below divide by E3's original claims and by ALL items (abstained items included), so filtering cannot by itself make E4 look better.

| method | n_items | original_claims | unsupported_over_original_claims | unsupported_per_item | items_with_unsupported | supported_per_item | claims_retained | coverage |
|---|---|---|---|---|---|---|---|---|
| e3 | 500 | 2025 | 0.508 | 2.056 | 0.568 | 1.950 | 1.000 | 1.000 |
| e4 | 500 | 2025 | 0.000 | 0.000 | 0.000 | 1.950 | 0.481 | 0.974 |
| e4_filter | 500 | 2025 | 0.000 | 0.000 | 0.000 | 1.950 | 0.481 | 1.000 |
| e4_cite | 500 | 2025 | 0.508 | 2.056 | 0.568 | 1.950 | 1.000 | 0.998 |

## Cost (measured GPU time, allocated by tokens)

GPU seconds 549 for 3934 unique prompts; 0.100 s per 1k tokens; price/h 1.09

| method | tokens_per_item | gpu_seconds_per_item | usd_per_1000_items | reuses_e3_generation |
|---|---|---|---|---|
| e0 | 908.372 | 0.090 | 0.027 | False |
| e1 | 1638.678 | 0.163 | 0.049 | False |
| e2 | 1711.412 | 0.170 | 0.052 | False |
| e2_bm25 | 1780.174 | 0.177 | 0.054 | False |
| e2_rerank | 1802.178 | 0.179 | 0.054 | False |
| e3 | 1053.560 | 0.105 | 0.032 | False |
| e3a_exact | 1038.254 | 0.103 | 0.031 | False |
| e3a | 1303.200 | 0.130 | 0.039 | False |
| e4 | 1053.560 | 0.000 | 0.000 | True |
| e4_filter | 1053.560 | 0.000 | 0.000 | True |
| e4_cite | 1053.560 | 0.000 | 0.000 | True |

## Strata

**e0**: all: n=500 acc=0.90 FPR=0.08
**e1**: all: n=500 acc=0.77 FPR=0.27
**e2**: all: n=500 acc=0.81 FPR=0.22
**e2_bm25**: all: n=500 acc=0.81 FPR=0.20
**e2_rerank**: all: n=500 acc=0.80 FPR=0.23
**e3**: all: n=500 acc=0.81 FPR=0.20
**e3a_exact**: all: n=500 acc=0.82 FPR=0.18
**e3a**: all: n=500 acc=0.78 FPR=0.29
**e4**: all: n=500 acc=0.80 FPR=0.20
**e4_filter**: all: n=500 acc=0.81 FPR=0.20
**e4_cite**: all: n=500 acc=0.81 FPR=0.20

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
