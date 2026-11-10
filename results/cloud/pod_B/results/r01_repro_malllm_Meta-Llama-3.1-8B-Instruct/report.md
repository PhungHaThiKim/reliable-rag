# Results: r01_repro_malllm_Meta-Llama-3.1-8B-Instruct

Model: `vllm:NousResearch/Meta-Llama-3.1-8B-Instruct` | split: `test` | n = 1029 | methods: e0_malllm, e1_malllm_yara, e1_malllm_code

Strict = abstentions counted as errors. Claim labels check consistency with static facts, not truth.

## Headline

| method | strict/f1_malicious | strict/macro_f1 | strict/fpr | strict/fnr | answered/f1_malicious | coverage | aurc | rc_max_coverage | evidence/unsupported_claim_rate | evidence/unsupported_over_original_claims | evidence/items_with_unsupported_all | evidence/citation_accuracy | gold/evidence_coverage | gold/path_recall | tokens_in_mean | tokens_out_mean | latency_p50_ms | parse_failure_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e0_malllm | 0.559 | 0.727 | 0.013 | 0.598 | 0.774 | 0.893 | 0.003 | 0.893 | nan | 0.000 | 0.000 | nan | nan | nan | 436.822 | 69.034 | 90.751 | 0.107 |
| e1_malllm_yara | 0.285 | 0.576 | 0.001 | 0.833 | 0.430 | 0.894 | 0.009 | 0.894 | nan | 0.000 | 0.000 | nan | nan | nan | 584.645 | 71.903 | 90.751 | 0.106 |
| e1_malllm_code | 0.471 | 0.427 | 0.758 | 0.054 | 0.476 | 0.989 | 0.203 | 0.989 | nan | 0.000 | 0.000 | nan | nan | nan | 1021.307 | 76.871 | 90.751 | 0.011 |

- e0_malllm: strict malicious-F1 0.559 [95% CI 0.499, 0.615]
- e1_malllm_yara: strict malicious-F1 0.285 [95% CI 0.226, 0.346]
- e1_malllm_code: strict malicious-F1 0.471 [95% CI 0.433, 0.504]

## Paired comparisons (same sample ids)

| pair | question | n | dF1 strict | CI95 | p | dAcc | McNemar p |
|---|---|---|---|---|---|---|---|
| e1_malllm_yara_vs_e0_malllm | mal-LLM repro RAG-YARA vs zero-shot | 1029 | -0.274 | [-0.407, -0.169] | 0.000 | -0.054 | 0.000 |
| e1_malllm_code_vs_e0_malllm | mal-LLM repro RAG-code vs zero-shot | 1029 | -0.088 | [-0.379, +0.212] | 0.526 | -0.399 | 0.000 |

## Cost (measured GPU time, allocated by tokens)

GPU seconds 252 for 3036 unique prompts; 0.110 s per 1k tokens; price/h 1.09

| method | tokens_per_item | gpu_seconds_per_item | usd_per_1000_items | reuses_e3_generation |
|---|---|---|---|---|
| e0_malllm | 505.856 | 0.055 | 0.017 | False |
| e1_malllm_yara | 656.548 | 0.072 | 0.022 | False |
| e1_malllm_code | 1098.178 | 0.120 | 0.036 | False |

## Strata

**e0_malllm**: all: n=1029 acc=0.83 FPR=0.01
**e1_malllm_yara**: all: n=1029 acc=0.78 FPR=0.00
**e1_malllm_code**: all: n=1029 acc=0.43 FPR=0.76

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
