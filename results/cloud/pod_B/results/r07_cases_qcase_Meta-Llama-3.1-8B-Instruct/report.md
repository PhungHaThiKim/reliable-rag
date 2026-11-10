# Results: r07_cases_qcase_Meta-Llama-3.1-8B-Instruct

Model: `vllm:NousResearch/Meta-Llama-3.1-8B-Instruct` | split: `test` | n = 6 | methods: e0, e1, e2, e3, e3a_exact, e3a, e4

Strict = abstentions counted as errors. Claim labels check consistency with static facts, not truth.

## Headline

| method | strict/f1_malicious | strict/macro_f1 | strict/fpr | strict/fnr | answered/f1_malicious | coverage | aurc | rc_max_coverage | evidence/unsupported_claim_rate | evidence/unsupported_over_original_claims | evidence/items_with_unsupported_all | evidence/citation_accuracy | gold/evidence_coverage | gold/path_recall | case/required_evidence_recall_cited | case/abstention_accuracy | tokens_in_mean | tokens_out_mean | latency_p50_ms | parse_failure_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e0 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 1.000 | 0.042 | 0.250 | 0.500 | 0.600 | 0.833 | nan | 0.889 | 0.000 | 577.833 | 242.667 | 310.978 | 0.000 |
| e1 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 1.000 | 0.174 | 0.308 | 0.667 | 0.483 | 0.833 | nan | 0.722 | 0.000 | 904.167 | 297.500 | 310.978 | 0.000 |
| e2 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 1.000 | 0.276 | 0.400 | 0.833 | 0.533 | 0.833 | nan | 0.889 | 0.000 | 904.500 | 313.500 | 310.978 | 0.000 |
| e3 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 1.000 | 0.038 | 0.231 | 0.667 | 0.678 | 0.933 | 1.000 | 0.806 | 0.000 | 805.167 | 257.000 | 310.978 | 0.000 |
| e3a_exact | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 1.000 | 0.080 | 0.154 | 0.500 | 0.726 | 0.833 | 1.000 | 0.639 | 0.000 | 807.167 | 268.167 | 310.978 | 0.000 |
| e3a | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 1.000 | 0.167 | 0.192 | 0.667 | 0.625 | 0.900 | nan | 0.833 | 0.000 | 1006.000 | 294.333 | 310.978 | 0.000 |
| e4 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.933 | 1.000 | 0.806 | 0.000 | 805.167 | 257.000 | 310.978 | 0.000 |

- e0: strict malicious-F1 1.000 [95% CI 1.000, 1.000]
- e1: strict malicious-F1 1.000 [95% CI 1.000, 1.000]
- e2: strict malicious-F1 1.000 [95% CI 1.000, 1.000]
- e3: strict malicious-F1 1.000 [95% CI 1.000, 1.000]
- e3a_exact: strict malicious-F1 1.000 [95% CI 1.000, 1.000]
- e3a: strict malicious-F1 1.000 [95% CI 1.000, 1.000]
- e4: strict malicious-F1 1.000 [95% CI 1.000, 1.000]

## Paired comparisons (same sample ids)

| pair | question | n | dF1 strict | CI95 | p | dAcc | McNemar p |
|---|---|---|---|---|---|---|---|
| e1_vs_e0 | RQ1 retrieval vs LLM-only | 5 | +0.000 | [+0.000, +0.000] | 1.000 | +0.000 | 1.000 |
| e2_vs_e1 | RQ2 hybrid vs dense | 5 | +0.000 | [+0.000, +0.000] | 1.000 | +0.000 | 1.000 |
| e3_vs_e2 | RQ3 KG-RAG vs hybrid | 5 | +0.000 | [+0.000, +0.000] | 1.000 | +0.000 | 1.000 |
| e3_vs_e3a_exact | RQ3 path structure vs exactly the same edges as separate sentences (E3a) | 5 | +0.000 | [+0.000, +0.000] | 1.000 | +0.000 | 1.000 |
| e3_vs_e3a | RQ3 graph traversal vs graph facts chosen by retrieval (E3b) | 5 | +0.000 | [+0.000, +0.000] | 1.000 | +0.000 | 1.000 |
| e4_vs_e3 | RQ4 verification vs none | 5 | +0.000 | [+0.000, +0.000] | 1.000 | +0.000 | 1.000 |
| e2_vs_e0 | vs e0 | 5 | +0.000 | [+0.000, +0.000] | 1.000 | +0.000 | 1.000 |
| e3_vs_e0 | vs e0 | 5 | +0.000 | [+0.000, +0.000] | 1.000 | +0.000 | 1.000 |
| e3a_exact_vs_e0 | vs e0 | 5 | +0.000 | [+0.000, +0.000] | 1.000 | +0.000 | 1.000 |
| e3a_vs_e0 | vs e0 | 5 | +0.000 | [+0.000, +0.000] | 1.000 | +0.000 | 1.000 |
| e4_vs_e0 | vs e0 | 5 | +0.000 | [+0.000, +0.000] | 1.000 | +0.000 | 1.000 |

## Grounding on the same denominator (E3 vs E4)

E4 reuses E3's answers: rates below divide by E3's original claims and by ALL items (abstained items included), so filtering cannot by itself make E4 look better.

| method | n_items | original_claims | unsupported_over_original_claims | unsupported_per_item | items_with_unsupported | supported_per_item | claims_retained | coverage |
|---|---|---|---|---|---|---|---|---|
| e3 | 6 | 26 | 0.231 | 1.000 | 0.667 | 3.333 | 1.000 | 1.000 |
| e4 | 6 | 26 | 0.000 | 0.000 | 0.000 | 3.333 | 0.769 | 1.000 |

## Cost (measured GPU time, allocated by tokens)

GPU seconds 11 for 36 unique prompts; 0.279 s per 1k tokens; price/h 1.09

| method | tokens_per_item | gpu_seconds_per_item | usd_per_1000_items | reuses_e3_generation |
|---|---|---|---|---|
| e0 | 820.500 | 0.229 | 0.069 | False |
| e1 | 1201.667 | 0.336 | 0.102 | False |
| e2 | 1218.000 | 0.340 | 0.103 | False |
| e3 | 1062.167 | 0.297 | 0.090 | False |
| e3a_exact | 1075.333 | 0.300 | 0.091 | False |
| e3a | 1300.333 | 0.363 | 0.110 | False |
| e4 | 1062.167 | 0.000 | 0.000 | True |

## Strata

**e0**: C01: n=1 acc=1.00 FPR=0.00; C02: n=1 acc=1.00 FPR=0.00; C03: n=1 acc=1.00 FPR=0.00; C04: n=1 acc=1.00 FPR=0.00; C05: n=1 acc=1.00 FPR=0.00
**e1**: C01: n=1 acc=1.00 FPR=0.00; C02: n=1 acc=1.00 FPR=0.00; C03: n=1 acc=1.00 FPR=0.00; C04: n=1 acc=1.00 FPR=0.00; C05: n=1 acc=1.00 FPR=0.00
**e2**: C01: n=1 acc=1.00 FPR=0.00; C02: n=1 acc=1.00 FPR=0.00; C03: n=1 acc=1.00 FPR=0.00; C04: n=1 acc=1.00 FPR=0.00; C05: n=1 acc=1.00 FPR=0.00
**e3**: C01: n=1 acc=1.00 FPR=0.00; C02: n=1 acc=1.00 FPR=0.00; C03: n=1 acc=1.00 FPR=0.00; C04: n=1 acc=1.00 FPR=0.00; C05: n=1 acc=1.00 FPR=0.00
**e3a_exact**: C01: n=1 acc=1.00 FPR=0.00; C02: n=1 acc=1.00 FPR=0.00; C03: n=1 acc=1.00 FPR=0.00; C04: n=1 acc=1.00 FPR=0.00; C05: n=1 acc=1.00 FPR=0.00
**e3a**: C01: n=1 acc=1.00 FPR=0.00; C02: n=1 acc=1.00 FPR=0.00; C03: n=1 acc=1.00 FPR=0.00; C04: n=1 acc=1.00 FPR=0.00; C05: n=1 acc=1.00 FPR=0.00
**e4**: C01: n=1 acc=1.00 FPR=0.00; C02: n=1 acc=1.00 FPR=0.00; C03: n=1 acc=1.00 FPR=0.00; C04: n=1 acc=1.00 FPR=0.00; C05: n=1 acc=1.00 FPR=0.00

## kb_report

```json
{
 "sources": [
  "builtin",
  "case_rules"
 ],
 "counts_before_audit": {
  "builtin": 30,
  "case_rules": 5
 },
 "n_docs": 35,
 "n_dropped": 0,
 "leakage_before": {
  "n_docs": 35,
  "docs_from_eval_samples": [],
  "n_docs_from_eval_samples": 0,
  "docs_mentioning_eval_package": [],
  "n_docs_mentioning_eval_package": 0,
  "docs_with_eval_code": [],
  "n_docs_with_eval_code": 0,
  "clean": true
 },
 "leakage_after": {
  "n_docs": 35,
  "docs_from_eval_samples": [],
  "n_docs_from_eval_samples": 0,
  "docs_mentioning_eval_package": [],
  "n_docs_mentioning_eval_package": 0,
  "docs_with_eval_code": [],
  "n_docs_with_eval_code": 0,
  "clean": true
 },
 "by_source_type": {
  "rule": 10,
  "benign_rule": 9,
  "pattern": 16
 }
}
```
