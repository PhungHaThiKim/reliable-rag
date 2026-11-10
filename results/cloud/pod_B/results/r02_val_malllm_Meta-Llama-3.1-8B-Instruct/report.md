# Results: r02_val_malllm_Meta-Llama-3.1-8B-Instruct

Model: `vllm:NousResearch/Meta-Llama-3.1-8B-Instruct` | split: `val` | n = 200 | methods: e1, e2, e2_bm25, e2_rerank, e3

Strict = abstentions counted as errors. Claim labels check consistency with static facts, not truth.

## Headline

| method | strict/f1_malicious | strict/macro_f1 | strict/fpr | strict/fnr | answered/f1_malicious | coverage | aurc | rc_max_coverage | evidence/unsupported_claim_rate | evidence/unsupported_over_original_claims | evidence/items_with_unsupported_all | evidence/citation_accuracy | gold/evidence_coverage | gold/path_recall | tokens_in_mean | tokens_out_mean | latency_p50_ms | parse_failure_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e1 | 0.512 | 0.640 | 0.278 | 0.411 | 0.512 | 1.000 | 0.095 | 1.000 | 0.515 | 0.512 | 0.625 | 0.345 | nan | nan | 1425.250 | 236.485 | 175.821 | 0.000 |
| e2 | 0.541 | 0.670 | 0.229 | 0.411 | 0.541 | 1.000 | 0.118 | 1.000 | 0.546 | 0.541 | 0.650 | 0.375 | nan | nan | 1523.610 | 244.890 | 175.821 | 0.000 |
| e2_bm25 | 0.555 | 0.683 | 0.208 | 0.411 | 0.555 | 1.000 | 0.101 | 1.000 | 0.471 | 0.487 | 0.575 | 0.343 | nan | nan | 1547.210 | 260.620 | 175.821 | 0.000 |
| e2_rerank | 0.559 | 0.687 | 0.201 | 0.411 | 0.559 | 1.000 | 0.107 | 1.000 | 0.536 | 0.558 | 0.635 | 0.299 | nan | nan | 1587.660 | 247.760 | 175.821 | 0.000 |
| e3 | 0.529 | 0.662 | 0.229 | 0.429 | 0.529 | 1.000 | 0.111 | 1.000 | 0.442 | 0.447 | 0.500 | 0.586 | nan | nan | 907.480 | 242.740 | 175.821 | 0.010 |

- e1: strict malicious-F1 0.512 [95% CI 0.406, 0.607]
- e2: strict malicious-F1 0.541 [95% CI 0.430, 0.641]
- e2_bm25: strict malicious-F1 0.555 [95% CI 0.439, 0.651]
- e2_rerank: strict malicious-F1 0.559 [95% CI 0.453, 0.657]
- e3: strict malicious-F1 0.529 [95% CI 0.422, 0.631]

## Paired comparisons (same sample ids)

| pair | question | n | dF1 strict | CI95 | p | dAcc | McNemar p |
|---|---|---|---|---|---|---|---|
| e2_vs_e1 | RQ2 hybrid vs dense | 200 | +0.029 | [-0.007, +0.078] | 0.158 | +0.035 | 0.189 |
| e3_vs_e2 | RQ3 KG-RAG vs hybrid | 200 | -0.012 | [-0.079, +0.048] | 0.692 | -0.005 | 1.000 |
| e2_bm25_vs_e2 | ablation BM25 only vs RRF | 200 | +0.014 | [-0.025, +0.057] | 0.556 | +0.015 | 0.629 |
| e2_rerank_vs_e2 | ablation RRF+rerank vs RRF | 200 | +0.018 | [-0.036, +0.080] | 0.542 | +0.020 | 0.503 |

## Grounding on the same denominator (E3 vs E4)

E4 reuses E3's answers: rates below divide by E3's original claims and by ALL items (abstained items included), so filtering cannot by itself make E4 look better.

| method | n_items | original_claims | unsupported_over_original_claims | unsupported_per_item | items_with_unsupported | supported_per_item | claims_retained | coverage |
|---|---|---|---|---|---|---|---|---|
| e3 | 200 | 836 | 0.447 | 1.870 | 0.500 | 2.305 | 1.000 | 1.000 |

## Cost (measured GPU time, allocated by tokens)

GPU seconds 172 for 980 unique prompts; 0.106 s per 1k tokens; price/h 1.09

| method | tokens_per_item | gpu_seconds_per_item | usd_per_1000_items | reuses_e3_generation |
|---|---|---|---|---|
| e1 | 1661.735 | 0.176 | 0.053 | False |
| e2 | 1768.500 | 0.188 | 0.057 | False |
| e2_bm25 | 1807.830 | 0.192 | 0.058 | False |
| e2_rerank | 1835.420 | 0.195 | 0.059 | False |
| e3 | 1150.220 | 0.122 | 0.037 | False |

## Strata

**e1**: all: n=200 acc=0.69 FPR=0.28
**e2**: all: n=200 acc=0.72 FPR=0.23
**e2_bm25**: all: n=200 acc=0.73 FPR=0.21
**e2_rerank**: all: n=200 acc=0.74 FPR=0.20
**e3**: all: n=200 acc=0.71 FPR=0.23

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
