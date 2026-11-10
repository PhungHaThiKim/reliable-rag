# Results: r02_val_malllm_Qwen2.5-Coder-7B-Instruct

Model: `vllm:Qwen/Qwen2.5-Coder-7B-Instruct` | split: `val` | n = 200 | methods: e1, e2, e2_bm25, e2_rerank, e3

Strict = abstentions counted as errors. Claim labels check consistency with static facts, not truth.

## Headline

| method | strict/f1_malicious | strict/macro_f1 | strict/fpr | strict/fnr | answered/f1_malicious | coverage | aurc | rc_max_coverage | evidence/unsupported_claim_rate | evidence/unsupported_over_original_claims | evidence/items_with_unsupported_all | evidence/citation_accuracy | gold/evidence_coverage | gold/path_recall | tokens_in_mean | tokens_out_mean | latency_p50_ms | parse_failure_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e1 | 0.756 | 0.842 | 0.000 | 0.393 | 0.756 | 1.000 | 0.024 | 1.000 | 0.407 | 0.437 | 0.505 | 0.332 | nan | nan | 1471.990 | 239.725 | 146.888 | 0.000 |
| e2 | 0.747 | 0.836 | 0.007 | 0.393 | 0.756 | 0.995 | 0.026 | 0.995 | 0.411 | 0.449 | 0.520 | 0.260 | nan | nan | 1583.725 | 246.285 | 146.888 | 0.005 |
| e2_bm25 | 0.753 | 0.839 | 0.014 | 0.375 | 0.753 | 1.000 | 0.027 | 1.000 | 0.401 | 0.429 | 0.500 | 0.304 | nan | nan | 1591.530 | 245.330 | 146.888 | 0.000 |
| e2_rerank | 0.737 | 0.827 | 0.028 | 0.375 | 0.737 | 1.000 | 0.030 | 1.000 | 0.403 | 0.466 | 0.560 | 0.305 | nan | nan | 1627.045 | 253.920 | 146.888 | 0.000 |
| e3 | 0.587 | 0.716 | 0.146 | 0.429 | 0.587 | 1.000 | 0.195 | 1.000 | 0.421 | 0.416 | 0.600 | 0.630 | nan | nan | 929.920 | 245.695 | 146.888 | 0.000 |

- e1: strict malicious-F1 0.756 [95% CI 0.640, 0.846]
- e2: strict malicious-F1 0.747 [95% CI 0.634, 0.837]
- e2_bm25: strict malicious-F1 0.753 [95% CI 0.643, 0.844]
- e2_rerank: strict malicious-F1 0.737 [95% CI 0.625, 0.830]
- e3: strict malicious-F1 0.587 [95% CI 0.478, 0.690]

## Paired comparisons (same sample ids)

| pair | question | n | dF1 strict | CI95 | p | dAcc | McNemar p |
|---|---|---|---|---|---|---|---|
| e2_vs_e1 | RQ2 hybrid vs dense | 200 | -0.008 | [-0.049, +0.050] | 0.810 | -0.005 | 1.000 |
| e3_vs_e2 | RQ3 KG-RAG vs hybrid | 200 | -0.160 | [-0.414, -0.060] | 0.000 | -0.110 | 0.000 |
| e2_bm25_vs_e2 | ablation BM25 only vs RRF | 200 | +0.005 | [-0.040, +0.039] | 0.864 | +0.000 | 1.000 |
| e2_rerank_vs_e2 | ablation RRF+rerank vs RRF | 200 | -0.010 | [-0.088, +0.027] | 0.676 | -0.010 | 0.688 |

## Grounding on the same denominator (E3 vs E4)

E4 reuses E3's answers: rates below divide by E3's original claims and by ALL items (abstained items included), so filtering cannot by itself make E4 look better.

| method | n_items | original_claims | unsupported_over_original_claims | unsupported_per_item | items_with_unsupported | supported_per_item | claims_retained | coverage |
|---|---|---|---|---|---|---|---|---|
| e3 | 200 | 847 | 0.416 | 1.760 | 0.600 | 2.300 | 1.000 | 1.000 |

## Cost (measured GPU time, allocated by tokens)

GPU seconds 144 for 980 unique prompts; 0.087 s per 1k tokens; price/h 1.09

| method | tokens_per_item | gpu_seconds_per_item | usd_per_1000_items | reuses_e3_generation |
|---|---|---|---|---|
| e1 | 1711.715 | 0.148 | 0.045 | False |
| e2 | 1830.010 | 0.159 | 0.048 | False |
| e2_bm25 | 1836.860 | 0.159 | 0.048 | False |
| e2_rerank | 1880.965 | 0.163 | 0.049 | False |
| e3 | 1175.615 | 0.102 | 0.031 | False |

## Strata

**e1**: all: n=200 acc=0.89 FPR=0.00
**e2**: all: n=200 acc=0.89 FPR=0.01
**e2_bm25**: all: n=200 acc=0.89 FPR=0.01
**e2_rerank**: all: n=200 acc=0.88 FPR=0.03
**e3**: all: n=200 acc=0.78 FPR=0.15

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
