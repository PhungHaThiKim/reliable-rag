# Results: r03_test_malllm_Meta-Llama-3.1-8B-Instruct_nodup

Model: `vllm:NousResearch/Meta-Llama-3.1-8B-Instruct` | split: `test` | n = 405 | methods: e0, e1, e2, e2_bm25, e2_rerank, e3, e3a_exact, e3a, e4, e4_filter, e4_cite

Strict = abstentions counted as errors. Claim labels check consistency with static facts, not truth.

## Headline

| method | strict/f1_malicious | strict/macro_f1 | strict/fpr | strict/fnr | answered/f1_malicious | coverage | aurc | rc_max_coverage | evidence/unsupported_claim_rate | evidence/unsupported_over_original_claims | evidence/items_with_unsupported_all | evidence/citation_accuracy | gold/evidence_coverage | gold/path_recall | tokens_in_mean | tokens_out_mean | latency_p50_ms | parse_failure_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e0 | 0.642 | 0.793 | 0.073 | 0.271 | 0.648 | 0.998 | 0.007 | 0.998 | 0.647 | 0.668 | 0.701 | 0.300 | nan | nan | 609.264 | 280.605 | 132.436 | 0.002 |
| e1 | 0.407 | 0.617 | 0.272 | 0.229 | 0.407 | 1.000 | 0.044 | 1.000 | 0.577 | 0.579 | 0.664 | 0.332 | nan | nan | 1321.758 | 243.672 | 132.436 | 0.000 |
| e2 | 0.463 | 0.664 | 0.218 | 0.208 | 0.463 | 1.000 | 0.036 | 1.000 | 0.618 | 0.631 | 0.723 | 0.289 | nan | nan | 1400.916 | 255.163 | 132.436 | 0.000 |
| e2_bm25 | 0.462 | 0.667 | 0.202 | 0.250 | 0.462 | 1.000 | 0.031 | 1.000 | 0.589 | 0.597 | 0.662 | 0.330 | nan | nan | 1479.301 | 260.775 | 132.436 | 0.000 |
| e2_rerank | 0.443 | 0.649 | 0.230 | 0.229 | 0.443 | 1.000 | 0.035 | 1.000 | 0.577 | 0.590 | 0.664 | 0.312 | nan | nan | 1445.985 | 251.027 | 132.436 | 0.000 |
| e3 | 0.474 | 0.674 | 0.199 | 0.229 | 0.474 | 1.000 | 0.037 | 1.000 | 0.553 | 0.558 | 0.588 | 0.505 | nan | nan | 759.644 | 244.010 | 132.436 | 0.000 |
| e3a_exact | 0.500 | 0.694 | 0.176 | 0.229 | 0.503 | 0.998 | 0.031 | 0.998 | 0.565 | 0.569 | 0.657 | 0.446 | nan | nan | 745.548 | 247.790 | 132.436 | 0.002 |
| e3a | 0.451 | 0.639 | 0.289 | 0.083 | 0.456 | 0.995 | 0.038 | 0.995 | 0.536 | 0.564 | 0.746 | 0.482 | nan | nan | 988.654 | 265.588 | 132.436 | 0.005 |
| e4 | 0.455 | 0.663 | 0.199 | 0.271 | 0.483 | 0.978 | 0.033 | 0.978 | 0.000 | 0.000 | 0.000 | 1.000 | nan | nan | 759.644 | 244.010 | 132.436 | 0.000 |
| e4_filter | 0.474 | 0.674 | 0.199 | 0.229 | 0.474 | 1.000 | 0.037 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | nan | nan | 759.644 | 244.010 | 132.436 | 0.000 |
| e4_cite | 0.474 | 0.674 | 0.199 | 0.229 | 0.474 | 1.000 | 0.037 | 1.000 | 0.553 | 0.558 | 0.588 | 0.505 | nan | nan | 759.644 | 244.010 | 132.436 | 0.000 |

- e0: strict malicious-F1 0.642 [95% CI 0.523, 0.742]
- e1: strict malicious-F1 0.407 [95% CI 0.309, 0.495]
- e2: strict malicious-F1 0.463 [95% CI 0.348, 0.552]
- e2_bm25: strict malicious-F1 0.462 [95% CI 0.351, 0.556]
- e2_rerank: strict malicious-F1 0.443 [95% CI 0.331, 0.526]
- e3: strict malicious-F1 0.474 [95% CI 0.370, 0.568]
- e3a_exact: strict malicious-F1 0.500 [95% CI 0.392, 0.594]
- e3a: strict malicious-F1 0.451 [95% CI 0.353, 0.531]
- e4: strict malicious-F1 0.455 [95% CI 0.346, 0.548]
- e4_filter: strict malicious-F1 0.474 [95% CI 0.370, 0.568]
- e4_cite: strict malicious-F1 0.474 [95% CI 0.370, 0.568]

## Paired comparisons (same sample ids)

| pair | question | n | dF1 strict | CI95 | p | dAcc | McNemar p |
|---|---|---|---|---|---|---|---|
| e1_vs_e0 | RQ1 retrieval vs LLM-only | 405 | -0.236 | [-0.298, -0.165] | 0.000 | -0.170 | 0.000 |
| e2_vs_e1 | RQ2 hybrid vs dense | 405 | +0.057 | [+0.016, +0.101] | 0.006 | +0.049 | 0.017 |
| e3_vs_e2 | RQ3 KG-RAG vs hybrid | 405 | +0.011 | [-0.052, +0.067] | 0.776 | +0.015 | 0.586 |
| e3_vs_e3a_exact | RQ3 path structure vs exactly the same edges as separate sentences (E3a) | 405 | -0.026 | [-0.044, -0.010] | 0.002 | -0.020 | 0.008 |
| e3_vs_e3a | RQ3 graph traversal vs graph facts chosen by retrieval (E3b) | 405 | +0.023 | [-0.046, +0.084] | 0.512 | +0.062 | 0.005 |
| e4_vs_e3 | RQ4 verification vs none | 405 | -0.020 | [-0.055, +0.000] | 0.260 | -0.005 | 0.500 |
| e2_bm25_vs_e2 | ablation BM25 only vs RRF | 405 | -0.002 | [-0.051, +0.047] | 0.900 | +0.010 | 0.672 |
| e2_rerank_vs_e2 | ablation RRF+rerank vs RRF | 405 | -0.020 | [-0.062, +0.021] | 0.312 | -0.012 | 0.560 |
| e4_filter_vs_e3 | ablation claim filter | 405 | +0.000 | [+0.000, +0.000] | 1.000 | +0.000 | 1.000 |
| e4_cite_vs_e3 | ablation citation-only | 405 | +0.000 | [+0.000, +0.000] | 1.000 | +0.000 | 1.000 |
| e2_vs_e0 | vs e0 | 405 | -0.179 | [-0.247, -0.103] | 0.000 | -0.121 | 0.000 |
| e2_bm25_vs_e0 | vs e0 | 405 | -0.181 | [-0.244, -0.111] | 0.000 | -0.111 | 0.000 |
| e2_rerank_vs_e0 | vs e0 | 405 | -0.199 | [-0.265, -0.131] | 0.000 | -0.133 | 0.000 |
| e3_vs_e0 | vs e0 | 405 | -0.168 | [-0.244, -0.097] | 0.000 | -0.106 | 0.000 |
| e3a_exact_vs_e0 | vs e0 | 405 | -0.142 | [-0.220, -0.071] | 0.000 | -0.086 | 0.000 |
| e3a_vs_e0 | vs e0 | 405 | -0.191 | [-0.278, -0.097] | 0.000 | -0.168 | 0.000 |
| e4_vs_e0 | vs e0 | 405 | -0.188 | [-0.269, -0.111] | 0.000 | -0.111 | 0.000 |
| e4_filter_vs_e0 | vs e0 | 405 | -0.168 | [-0.244, -0.097] | 0.000 | -0.106 | 0.000 |
| e4_cite_vs_e0 | vs e0 | 405 | -0.168 | [-0.244, -0.097] | 0.000 | -0.106 | 0.000 |

## Grounding on the same denominator (E3 vs E4)

E4 reuses E3's answers: rates below divide by E3's original claims and by ALL items (abstained items included), so filtering cannot by itself make E4 look better.

| method | n_items | original_claims | unsupported_over_original_claims | unsupported_per_item | items_with_unsupported | supported_per_item | claims_retained | coverage |
|---|---|---|---|---|---|---|---|---|
| e3 | 405 | 1705 | 0.558 | 2.348 | 0.588 | 1.830 | 1.000 | 1.000 |
| e4 | 405 | 1705 | 0.000 | 0.000 | 0.000 | 1.830 | 0.435 | 0.978 |
| e4_filter | 405 | 1705 | 0.000 | 0.000 | 0.000 | 1.830 | 0.435 | 1.000 |
| e4_cite | 405 | 1705 | 0.558 | 2.348 | 0.588 | 1.830 | 1.000 | 1.000 |

## Cost (measured GPU time, allocated by tokens)

GPU seconds 439 for 3207 unique prompts; 0.102 s per 1k tokens; price/h 1.09

| method | tokens_per_item | gpu_seconds_per_item | usd_per_1000_items | reuses_e3_generation |
|---|---|---|---|---|
| e0 | 889.869 | 0.090 | 0.027 | False |
| e1 | 1565.430 | 0.159 | 0.048 | False |
| e2 | 1656.079 | 0.168 | 0.051 | False |
| e2_bm25 | 1740.077 | 0.177 | 0.054 | False |
| e2_rerank | 1697.012 | 0.173 | 0.052 | False |
| e3 | 1003.654 | 0.102 | 0.031 | False |
| e3a_exact | 993.338 | 0.101 | 0.031 | False |
| e3a | 1254.242 | 0.128 | 0.039 | False |
| e4 | 1003.654 | 0.000 | 0.000 | True |
| e4_filter | 1003.654 | 0.000 | 0.000 | True |
| e4_cite | 1003.654 | 0.000 | 0.000 | True |

## Strata

**e0**: all: n=405 acc=0.90 FPR=0.07
**e1**: all: n=405 acc=0.73 FPR=0.27
**e2**: all: n=405 acc=0.78 FPR=0.22
**e2_bm25**: all: n=405 acc=0.79 FPR=0.20
**e2_rerank**: all: n=405 acc=0.77 FPR=0.23
**e3**: all: n=405 acc=0.80 FPR=0.20
**e3a_exact**: all: n=405 acc=0.82 FPR=0.18
**e3a**: all: n=405 acc=0.74 FPR=0.29
**e4**: all: n=405 acc=0.79 FPR=0.20
**e4_filter**: all: n=405 acc=0.80 FPR=0.20
**e4_cite**: all: n=405 acc=0.80 FPR=0.20

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
