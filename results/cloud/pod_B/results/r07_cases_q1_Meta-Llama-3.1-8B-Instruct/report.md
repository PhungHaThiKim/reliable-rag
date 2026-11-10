# Results: r07_cases_q1_Meta-Llama-3.1-8B-Instruct

Model: `vllm:NousResearch/Meta-Llama-3.1-8B-Instruct` | split: `test` | n = 6 | methods: e0, e1, e2, e3, e3a_exact, e3a, e4

Strict = abstentions counted as errors. Claim labels check consistency with static facts, not truth.

## Headline

| method | strict/f1_malicious | strict/macro_f1 | strict/fpr | strict/fnr | answered/f1_malicious | coverage | aurc | rc_max_coverage | evidence/unsupported_claim_rate | evidence/unsupported_over_original_claims | evidence/items_with_unsupported_all | evidence/citation_accuracy | gold/evidence_coverage | gold/path_recall | case/required_evidence_recall_cited | case/abstention_accuracy | tokens_in_mean | tokens_out_mean | latency_p50_ms | parse_failure_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e0 | 0.857 | 0.762 | 0.500 | 0.000 | 0.857 | 1.000 | 0.040 | 1.000 | 0.240 | 0.259 | 0.500 | 0.650 | 0.833 | nan | 0.722 | 0.000 | 544.333 | 277.167 | 279.206 | 0.000 |
| e1 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 1.000 | 0.296 | 0.367 | 0.833 | 0.511 | 0.833 | nan | 0.889 | 0.000 | 870.667 | 315.500 | 279.206 | 0.000 |
| e2 | 0.857 | 0.762 | 0.500 | 0.000 | 0.857 | 1.000 | 0.040 | 1.000 | 0.310 | 0.467 | 0.833 | 0.489 | 0.833 | nan | 0.889 | 0.000 | 871.000 | 318.167 | 279.206 | 0.000 |
| e3 | 0.857 | 0.762 | 0.500 | 0.000 | 0.857 | 1.000 | 0.040 | 1.000 | 0.185 | 0.321 | 0.667 | 0.725 | 0.933 | 1.000 | 0.583 | 0.000 | 771.667 | 269.333 | 279.206 | 0.000 |
| e3a_exact | 0.857 | 0.762 | 0.500 | 0.000 | 0.857 | 1.000 | 0.040 | 1.000 | 0.269 | 0.333 | 0.500 | 0.750 | 0.833 | 1.000 | 0.750 | 0.000 | 773.667 | 262.000 | 279.206 | 0.000 |
| e3a | 0.857 | 0.762 | 0.500 | 0.000 | 0.857 | 1.000 | 0.040 | 1.000 | 0.185 | 0.167 | 0.500 | 0.802 | 0.900 | nan | 0.750 | 0.000 | 972.500 | 304.333 | 279.206 | 0.000 |
| e4 | 0.857 | 0.762 | 0.500 | 0.000 | 0.857 | 1.000 | 0.040 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.933 | 1.000 | 0.583 | 0.000 | 771.667 | 269.333 | 279.206 | 0.000 |

- e0: strict malicious-F1 0.857 [95% CI 0.498, 1.000]
- e1: strict malicious-F1 1.000 [95% CI 1.000, 1.000]
- e2: strict malicious-F1 0.857 [95% CI 0.498, 1.000]
- e3: strict malicious-F1 0.857 [95% CI 0.498, 1.000]
- e3a_exact: strict malicious-F1 0.857 [95% CI 0.498, 1.000]
- e3a: strict malicious-F1 0.857 [95% CI 0.498, 1.000]
- e4: strict malicious-F1 0.857 [95% CI 0.498, 1.000]

## Paired comparisons (same sample ids)

| pair | question | n | dF1 strict | CI95 | p | dAcc | McNemar p |
|---|---|---|---|---|---|---|---|
| e1_vs_e0 | RQ1 retrieval vs LLM-only | 5 | +0.143 | [+0.000, +0.500] | 0.718 | +0.200 | 1.000 |
| e2_vs_e1 | RQ2 hybrid vs dense | 5 | -0.143 | [-0.500, +0.000] | 0.718 | -0.200 | 1.000 |
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
| e3 | 6 | 28 | 0.321 | 1.500 | 0.667 | 3.000 | 1.000 | 1.000 |
| e4 | 6 | 28 | 0.000 | 0.000 | 0.000 | 3.000 | 0.643 | 1.000 |

## Cost (measured GPU time, allocated by tokens)

GPU seconds 10 for 36 unique prompts; 0.256 s per 1k tokens; price/h 1.09

| method | tokens_per_item | gpu_seconds_per_item | usd_per_1000_items | reuses_e3_generation |
|---|---|---|---|---|
| e0 | 821.500 | 0.210 | 0.064 | False |
| e1 | 1186.167 | 0.303 | 0.092 | False |
| e2 | 1189.167 | 0.304 | 0.092 | False |
| e3 | 1041.000 | 0.266 | 0.081 | False |
| e3a_exact | 1035.667 | 0.265 | 0.080 | False |
| e3a | 1276.833 | 0.327 | 0.099 | False |
| e4 | 1041.000 | 0.000 | 0.000 | True |

## Strata

**e0**: C01: n=1 acc=1.00 FPR=0.00; C02: n=1 acc=1.00 FPR=0.00; C03: n=1 acc=1.00 FPR=0.00; C04: n=1 acc=0.00 FPR=1.00; C05: n=1 acc=1.00 FPR=0.00
**e1**: C01: n=1 acc=1.00 FPR=0.00; C02: n=1 acc=1.00 FPR=0.00; C03: n=1 acc=1.00 FPR=0.00; C04: n=1 acc=1.00 FPR=0.00; C05: n=1 acc=1.00 FPR=0.00
**e2**: C01: n=1 acc=1.00 FPR=0.00; C02: n=1 acc=1.00 FPR=0.00; C03: n=1 acc=1.00 FPR=0.00; C04: n=1 acc=0.00 FPR=1.00; C05: n=1 acc=1.00 FPR=0.00
**e3**: C01: n=1 acc=1.00 FPR=0.00; C02: n=1 acc=1.00 FPR=0.00; C03: n=1 acc=1.00 FPR=0.00; C04: n=1 acc=0.00 FPR=1.00; C05: n=1 acc=1.00 FPR=0.00
**e3a_exact**: C01: n=1 acc=1.00 FPR=0.00; C02: n=1 acc=1.00 FPR=0.00; C03: n=1 acc=1.00 FPR=0.00; C04: n=1 acc=0.00 FPR=1.00; C05: n=1 acc=1.00 FPR=0.00
**e3a**: C01: n=1 acc=1.00 FPR=0.00; C02: n=1 acc=1.00 FPR=0.00; C03: n=1 acc=1.00 FPR=0.00; C04: n=1 acc=0.00 FPR=1.00; C05: n=1 acc=1.00 FPR=0.00
**e4**: C01: n=1 acc=1.00 FPR=0.00; C02: n=1 acc=1.00 FPR=0.00; C03: n=1 acc=1.00 FPR=0.00; C04: n=1 acc=0.00 FPR=1.00; C05: n=1 acc=1.00 FPR=0.00

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
