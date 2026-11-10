# Results: r05_test_synthetic_Meta-Llama-3.1-8B-Instruct

Model: `vllm:NousResearch/Meta-Llama-3.1-8B-Instruct` | split: `test` | n = 48 | methods: e0, e1, e2, e2_bm25, e3, e3a_exact, e3a, e4, e4_filter, e4_cite

Strict = abstentions counted as errors. Claim labels check consistency with static facts, not truth.

## Headline

| method | strict/f1_malicious | strict/macro_f1 | strict/fpr | strict/fnr | answered/f1_malicious | coverage | aurc | rc_max_coverage | evidence/unsupported_claim_rate | evidence/unsupported_over_original_claims | evidence/items_with_unsupported_all | evidence/citation_accuracy | gold/evidence_coverage | gold/path_recall | tokens_in_mean | tokens_out_mean | latency_p50_ms | parse_failure_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e0 | 0.702 | 0.633 | 0.607 | 0.000 | 0.702 | 1.000 | 0.075 | 1.000 | 0.000 | 0.176 | 0.312 | 0.675 | 0.660 | nan | 586.896 | 233.396 | 127.490 | 0.000 |
| e1 | 0.667 | 0.556 | 0.714 | 0.000 | 0.667 | 1.000 | 0.107 | 1.000 | 0.140 | 0.228 | 0.646 | 0.503 | 0.660 | nan | 1192.292 | 299.083 | 127.490 | 0.000 |
| e2 | 0.690 | 0.608 | 0.643 | 0.000 | 0.690 | 1.000 | 0.085 | 1.000 | 0.161 | 0.242 | 0.729 | 0.431 | 0.660 | nan | 1215.042 | 310.083 | 127.490 | 0.000 |
| e2_bm25 | 0.678 | 0.582 | 0.679 | 0.000 | 0.678 | 1.000 | 0.095 | 1.000 | 0.160 | 0.220 | 0.625 | 0.587 | 0.660 | nan | 1229.917 | 294.438 | 127.490 | 0.000 |
| e3 | 0.702 | 0.633 | 0.607 | 0.000 | 0.702 | 1.000 | 0.075 | 1.000 | 0.014 | 0.103 | 0.375 | 0.741 | 0.930 | 1.000 | 887.021 | 280.792 | 127.490 | 0.000 |
| e3a_exact | 0.702 | 0.633 | 0.607 | 0.000 | 0.702 | 1.000 | 0.163 | 1.000 | 0.035 | 0.113 | 0.354 | 0.891 | 0.660 | 1.000 | 877.292 | 257.500 | 127.490 | 0.000 |
| e3a | 0.635 | 0.469 | 0.821 | 0.000 | 0.635 | 1.000 | 0.144 | 1.000 | 0.119 | 0.237 | 0.583 | 0.750 | 0.730 | nan | 1084.479 | 295.083 | 127.490 | 0.000 |
| e4 | 0.702 | 0.633 | 0.607 | 0.000 | 0.702 | 1.000 | 0.075 | 1.000 | 0.000 | 0.000 | 0.000 | 0.931 | 0.930 | 1.000 | 887.021 | 280.792 | 127.490 | 0.000 |
| e4_filter | 0.702 | 0.633 | 0.607 | 0.000 | 0.702 | 1.000 | 0.075 | 1.000 | 0.000 | 0.000 | 0.000 | 0.931 | 0.930 | 1.000 | 887.021 | 280.792 | 127.490 | 0.000 |
| e4_cite | 0.702 | 0.633 | 0.607 | 0.000 | 0.702 | 1.000 | 0.075 | 1.000 | 0.014 | 0.103 | 0.375 | 0.741 | 0.930 | 1.000 | 887.021 | 280.792 | 127.490 | 0.000 |

- e0: strict malicious-F1 0.702 [95% CI 0.549, 0.818]
- e1: strict malicious-F1 0.667 [95% CI 0.517, 0.788]
- e2: strict malicious-F1 0.690 [95% CI 0.537, 0.807]
- e2_bm25: strict malicious-F1 0.678 [95% CI 0.526, 0.800]
- e3: strict malicious-F1 0.702 [95% CI 0.549, 0.818]
- e3a_exact: strict malicious-F1 0.702 [95% CI 0.549, 0.818]
- e3a: strict malicious-F1 0.635 [95% CI 0.483, 0.761]
- e4: strict malicious-F1 0.702 [95% CI 0.549, 0.818]
- e4_filter: strict malicious-F1 0.702 [95% CI 0.549, 0.818]
- e4_cite: strict malicious-F1 0.702 [95% CI 0.549, 0.818]

## Paired comparisons (same sample ids)

| pair | question | n | dF1 strict | CI95 | p | dAcc | McNemar p |
|---|---|---|---|---|---|---|---|
| e1_vs_e0 | RQ1 retrieval vs LLM-only | 48 | -0.035 | [-0.151, +0.000] | 0.670 | -0.062 | 0.250 |
| e2_vs_e1 | RQ2 hybrid vs dense | 48 | +0.023 | [+0.000, +0.062] | 0.224 | +0.042 | 0.500 |
| e3_vs_e2 | RQ3 KG-RAG vs hybrid | 48 | +0.012 | [-0.028, +0.101] | 0.884 | +0.021 | 1.000 |
| e3_vs_e3a_exact | RQ3 path structure vs exactly the same edges as separate sentences (E3a) | 48 | +0.000 | [+0.000, +0.000] | 1.000 | +0.000 | 1.000 |
| e3_vs_e3a | RQ3 graph traversal vs graph facts chosen by retrieval (E3b) | 48 | +0.067 | [+0.000, +0.243] | 0.186 | +0.125 | 0.031 |
| e4_vs_e3 | RQ4 verification vs none | 48 | +0.000 | [+0.000, +0.000] | 1.000 | +0.000 | 1.000 |
| e2_bm25_vs_e2 | ablation BM25 only vs RRF | 48 | -0.012 | [-0.036, +0.000] | 0.726 | -0.021 | 1.000 |
| e4_filter_vs_e3 | ablation claim filter | 48 | +0.000 | [+0.000, +0.000] | 1.000 | +0.000 | 1.000 |
| e4_cite_vs_e3 | ablation citation-only | 48 | +0.000 | [+0.000, +0.000] | 1.000 | +0.000 | 1.000 |
| e2_vs_e0 | vs e0 | 48 | -0.012 | [-0.101, +0.028] | 0.884 | -0.021 | 1.000 |
| e2_bm25_vs_e0 | vs e0 | 48 | -0.024 | [-0.107, +0.000] | 0.670 | -0.042 | 0.500 |
| e3_vs_e0 | vs e0 | 48 | +0.000 | [+0.000, +0.000] | 1.000 | +0.000 | 1.000 |
| e3a_exact_vs_e0 | vs e0 | 48 | +0.000 | [+0.000, +0.000] | 1.000 | +0.000 | 1.000 |
| e3a_vs_e0 | vs e0 | 48 | -0.067 | [-0.243, +0.000] | 0.186 | -0.125 | 0.031 |
| e4_vs_e0 | vs e0 | 48 | +0.000 | [+0.000, +0.000] | 1.000 | +0.000 | 1.000 |
| e4_filter_vs_e0 | vs e0 | 48 | +0.000 | [+0.000, +0.000] | 1.000 | +0.000 | 1.000 |
| e4_cite_vs_e0 | vs e0 | 48 | +0.000 | [+0.000, +0.000] | 1.000 | +0.000 | 1.000 |

## Grounding on the same denominator (E3 vs E4)

E4 reuses E3's answers: rates below divide by E3's original claims and by ALL items (abstained items included), so filtering cannot by itself make E4 look better.

| method | n_items | original_claims | unsupported_over_original_claims | unsupported_per_item | items_with_unsupported | supported_per_item | claims_retained | coverage |
|---|---|---|---|---|---|---|---|---|
| e3 | 48 | 232 | 0.103 | 0.500 | 0.375 | 4.000 | 1.000 | 1.000 |
| e4 | 48 | 232 | 0.000 | 0.000 | 0.000 | 4.000 | 0.828 | 1.000 |
| e4_filter | 48 | 232 | 0.000 | 0.000 | 0.000 | 4.000 | 0.828 | 1.000 |
| e4_cite | 48 | 232 | 0.103 | 0.500 | 0.375 | 4.000 | 1.000 | 1.000 |

## Cost (measured GPU time, allocated by tokens)

GPU seconds 42 for 331 unique prompts; 0.099 s per 1k tokens; price/h 1.09

| method | tokens_per_item | gpu_seconds_per_item | usd_per_1000_items | reuses_e3_generation |
|---|---|---|---|---|
| e0 | 820.292 | 0.081 | 0.025 | False |
| e1 | 1491.375 | 0.147 | 0.045 | False |
| e2 | 1525.125 | 0.151 | 0.046 | False |
| e2_bm25 | 1524.354 | 0.151 | 0.046 | False |
| e3 | 1167.812 | 0.115 | 0.035 | False |
| e3a_exact | 1134.792 | 0.112 | 0.034 | False |
| e3a | 1379.562 | 0.136 | 0.041 | False |
| e4 | 1167.812 | 0.000 | 0.000 | True |
| e4_filter | 1167.812 | 0.000 | 0.000 | True |
| e4_cite | 1167.812 | 0.000 | 0.000 | True |

## Strata

**e0**: benign_sensitive: n=25 acc=0.32 FPR=0.68; direct: n=20 acc=1.00 FPR=0.00; plain: n=3 acc=1.00 FPR=0.00
**e1**: benign_sensitive: n=25 acc=0.20 FPR=0.80; direct: n=20 acc=1.00 FPR=0.00; plain: n=3 acc=1.00 FPR=0.00
**e2**: benign_sensitive: n=25 acc=0.28 FPR=0.72; direct: n=20 acc=1.00 FPR=0.00; plain: n=3 acc=1.00 FPR=0.00
**e2_bm25**: benign_sensitive: n=25 acc=0.24 FPR=0.76; direct: n=20 acc=1.00 FPR=0.00; plain: n=3 acc=1.00 FPR=0.00
**e3**: benign_sensitive: n=25 acc=0.32 FPR=0.68; direct: n=20 acc=1.00 FPR=0.00; plain: n=3 acc=1.00 FPR=0.00
**e3a_exact**: benign_sensitive: n=25 acc=0.32 FPR=0.68; direct: n=20 acc=1.00 FPR=0.00; plain: n=3 acc=1.00 FPR=0.00
**e3a**: benign_sensitive: n=25 acc=0.12 FPR=0.88; direct: n=20 acc=1.00 FPR=0.00; plain: n=3 acc=0.67 FPR=0.33
**e4**: benign_sensitive: n=25 acc=0.32 FPR=0.68; direct: n=20 acc=1.00 FPR=0.00; plain: n=3 acc=1.00 FPR=0.00
**e4_filter**: benign_sensitive: n=25 acc=0.32 FPR=0.68; direct: n=20 acc=1.00 FPR=0.00; plain: n=3 acc=1.00 FPR=0.00
**e4_cite**: benign_sensitive: n=25 acc=0.32 FPR=0.68; direct: n=20 acc=1.00 FPR=0.00; plain: n=3 acc=1.00 FPR=0.00

## kb_report

```json
{
 "sources": [
  "builtin",
  "examples",
  "pyguard"
 ],
 "counts_before_audit": {
  "builtin": 30,
  "examples": 119,
  "pyguard": 304
 },
 "n_docs": 451,
 "n_dropped": 2,
 "leakage_before": {
  "n_docs": 453,
  "docs_from_eval_samples": [],
  "n_docs_from_eval_samples": 0,
  "docs_mentioning_eval_package": [],
  "n_docs_mentioning_eval_package": 0,
  "docs_with_eval_code": [
   "ex:syn-0-0185-simpleutils27:C1",
   "ex:syn-0-0130-smartcolors69:C1"
  ],
  "n_docs_with_eval_code": 2,
  "clean": false
 },
 "leakage_after": {
  "n_docs": 451,
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
  "malicious_example": 75,
  "benign_example": 42,
  "pyguard_pattern": 304
 }
}
```
