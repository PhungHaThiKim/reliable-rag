# Results: r05_test_synthetic_Qwen2.5-Coder-7B-Instruct

Model: `vllm:Qwen/Qwen2.5-Coder-7B-Instruct` | split: `test` | n = 48 | methods: e0, e1, e2, e2_bm25, e3, e3a_exact, e3a, e4, e4_filter, e4_cite

Strict = abstentions counted as errors. Claim labels check consistency with static facts, not truth.

## Headline

| method | strict/f1_malicious | strict/macro_f1 | strict/fpr | strict/fnr | answered/f1_malicious | coverage | aurc | rc_max_coverage | evidence/unsupported_claim_rate | evidence/unsupported_over_original_claims | evidence/items_with_unsupported_all | evidence/citation_accuracy | gold/evidence_coverage | gold/path_recall | tokens_in_mean | tokens_out_mean | latency_p50_ms | parse_failure_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e0 | 0.889 | 0.895 | 0.179 | 0.000 | 0.889 | 1.000 | 0.007 | 1.000 | 0.103 | 0.230 | 0.562 | 0.426 | 0.660 | nan | 595.083 | 224.312 | 94.506 | 0.000 |
| e1 | 0.851 | 0.854 | 0.250 | 0.000 | 0.851 | 1.000 | 0.013 | 1.000 | 0.123 | 0.232 | 0.625 | 0.367 | 0.660 | nan | 1211.917 | 242.042 | 94.506 | 0.000 |
| e2 | 0.909 | 0.916 | 0.143 | 0.000 | 0.909 | 1.000 | 0.004 | 1.000 | 0.085 | 0.113 | 0.396 | 0.477 | 0.660 | nan | 1236.708 | 262.333 | 94.506 | 0.000 |
| e2_bm25 | 0.909 | 0.916 | 0.143 | 0.000 | 0.909 | 1.000 | 0.004 | 1.000 | 0.101 | 0.176 | 0.500 | 0.493 | 0.660 | nan | 1253.104 | 253.646 | 94.506 | 0.000 |
| e3 | 0.870 | 0.875 | 0.214 | 0.000 | 0.870 | 1.000 | 0.099 | 1.000 | 0.030 | 0.062 | 0.229 | 0.927 | 0.930 | 1.000 | 903.167 | 226.354 | 94.506 | 0.000 |
| e3a_exact | 0.769 | 0.748 | 0.429 | 0.000 | 0.769 | 1.000 | 0.088 | 1.000 | 0.047 | 0.128 | 0.354 | 0.871 | 0.660 | 1.000 | 893.146 | 234.729 | 94.506 | 0.000 |
| e3a | 0.727 | 0.681 | 0.536 | 0.000 | 0.727 | 1.000 | 0.058 | 1.000 | 0.088 | 0.197 | 0.583 | 0.566 | 0.730 | nan | 1104.854 | 261.083 | 94.506 | 0.000 |
| e4 | 0.870 | 0.875 | 0.214 | 0.000 | 0.870 | 1.000 | 0.099 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.930 | 1.000 | 903.167 | 226.354 | 94.506 | 0.000 |
| e4_filter | 0.870 | 0.875 | 0.214 | 0.000 | 0.870 | 1.000 | 0.099 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.930 | 1.000 | 903.167 | 226.354 | 94.506 | 0.000 |
| e4_cite | 0.870 | 0.875 | 0.214 | 0.000 | 0.870 | 1.000 | 0.099 | 1.000 | 0.030 | 0.062 | 0.229 | 0.927 | 0.930 | 1.000 | 903.167 | 226.354 | 94.506 | 0.000 |

- e0: strict malicious-F1 0.889 [95% CI 0.769, 0.979]
- e1: strict malicious-F1 0.851 [95% CI 0.732, 0.945]
- e2: strict malicious-F1 0.909 [95% CI 0.800, 0.982]
- e2_bm25: strict malicious-F1 0.909 [95% CI 0.809, 0.981]
- e3: strict malicious-F1 0.870 [95% CI 0.756, 0.955]
- e3a_exact: strict malicious-F1 0.769 [95% CI 0.625, 0.877]
- e3a: strict malicious-F1 0.727 [95% CI 0.583, 0.839]
- e4: strict malicious-F1 0.870 [95% CI 0.756, 0.955]
- e4_filter: strict malicious-F1 0.870 [95% CI 0.756, 0.955]
- e4_cite: strict malicious-F1 0.870 [95% CI 0.756, 0.955]

## Paired comparisons (same sample ids)

| pair | question | n | dF1 strict | CI95 | p | dAcc | McNemar p |
|---|---|---|---|---|---|---|---|
| e1_vs_e0 | RQ1 retrieval vs LLM-only | 48 | -0.038 | [-0.334, +0.197] | 0.788 | -0.042 | 0.754 |
| e2_vs_e1 | RQ2 hybrid vs dense | 48 | +0.058 | [-0.063, +0.226] | 0.740 | +0.062 | 0.453 |
| e3_vs_e2 | RQ3 KG-RAG vs hybrid | 48 | -0.040 | [-0.249, +0.185] | 0.870 | -0.042 | 0.727 |
| e3_vs_e3a_exact | RQ3 path structure vs exactly the same edges as separate sentences (E3a) | 48 | +0.100 | [+0.000, +0.346] | 0.686 | +0.125 | 0.031 |
| e3_vs_e3a | RQ3 graph traversal vs graph facts chosen by retrieval (E3b) | 48 | +0.142 | [+0.000, +0.443] | 0.686 | +0.188 | 0.004 |
| e4_vs_e3 | RQ4 verification vs none | 48 | +0.000 | [+0.000, +0.000] | 1.000 | +0.000 | 1.000 |
| e2_bm25_vs_e2 | ablation BM25 only vs RRF | 48 | +0.000 | [-0.117, +0.123] | 1.000 | +0.000 | 1.000 |
| e4_filter_vs_e3 | ablation claim filter | 48 | +0.000 | [+0.000, +0.000] | 1.000 | +0.000 | 1.000 |
| e4_cite_vs_e3 | ablation citation-only | 48 | +0.000 | [+0.000, +0.000] | 1.000 | +0.000 | 1.000 |
| e2_vs_e0 | vs e0 | 48 | +0.020 | [-0.143, +0.147] | 0.916 | +0.021 | 1.000 |
| e2_bm25_vs_e0 | vs e0 | 48 | +0.020 | [-0.220, +0.225] | 0.898 | +0.021 | 1.000 |
| e3_vs_e0 | vs e0 | 48 | -0.019 | [-0.360, +0.306] | 0.928 | -0.021 | 1.000 |
| e3a_exact_vs_e0 | vs e0 | 48 | -0.120 | [-0.375, +0.000] | 0.188 | -0.146 | 0.065 |
| e3a_vs_e0 | vs e0 | 48 | -0.162 | [-0.375, +0.000] | 0.188 | -0.208 | 0.006 |
| e4_vs_e0 | vs e0 | 48 | -0.019 | [-0.360, +0.306] | 0.928 | -0.021 | 1.000 |
| e4_filter_vs_e0 | vs e0 | 48 | -0.019 | [-0.360, +0.306] | 0.928 | -0.021 | 1.000 |
| e4_cite_vs_e0 | vs e0 | 48 | -0.019 | [-0.360, +0.306] | 0.928 | -0.021 | 1.000 |

## Grounding on the same denominator (E3 vs E4)

E4 reuses E3's answers: rates below divide by E3's original claims and by ALL items (abstained items included), so filtering cannot by itself make E4 look better.

| method | n_items | original_claims | unsupported_over_original_claims | unsupported_per_item | items_with_unsupported | supported_per_item | claims_retained | coverage |
|---|---|---|---|---|---|---|---|---|
| e3 | 48 | 176 | 0.062 | 0.229 | 0.229 | 3.188 | 1.000 | 1.000 |
| e4 | 48 | 176 | 0.000 | 0.000 | 0.000 | 3.188 | 0.869 | 1.000 |
| e4_filter | 48 | 176 | 0.000 | 0.000 | 0.000 | 3.188 | 0.869 | 1.000 |
| e4_cite | 48 | 176 | 0.062 | 0.229 | 0.229 | 3.188 | 1.000 | 1.000 |

## Cost (measured GPU time, allocated by tokens)

GPU seconds 31 for 331 unique prompts; 0.074 s per 1k tokens; price/h 1.09

| method | tokens_per_item | gpu_seconds_per_item | usd_per_1000_items | reuses_e3_generation |
|---|---|---|---|---|
| e0 | 819.396 | 0.061 | 0.018 | False |
| e1 | 1453.958 | 0.108 | 0.033 | False |
| e2 | 1499.042 | 0.112 | 0.034 | False |
| e2_bm25 | 1506.750 | 0.112 | 0.034 | False |
| e3 | 1129.521 | 0.084 | 0.025 | False |
| e3a_exact | 1127.875 | 0.084 | 0.025 | False |
| e3a | 1365.938 | 0.102 | 0.031 | False |
| e4 | 1129.521 | 0.000 | 0.000 | True |
| e4_filter | 1129.521 | 0.000 | 0.000 | True |
| e4_cite | 1129.521 | 0.000 | 0.000 | True |

## Strata

**e0**: benign_sensitive: n=25 acc=0.80 FPR=0.20; direct: n=20 acc=1.00 FPR=0.00; plain: n=3 acc=1.00 FPR=0.00
**e1**: benign_sensitive: n=25 acc=0.76 FPR=0.24; direct: n=20 acc=1.00 FPR=0.00; plain: n=3 acc=0.67 FPR=0.33
**e2**: benign_sensitive: n=25 acc=0.88 FPR=0.12; direct: n=20 acc=1.00 FPR=0.00; plain: n=3 acc=0.67 FPR=0.33
**e2_bm25**: benign_sensitive: n=25 acc=0.84 FPR=0.16; direct: n=20 acc=1.00 FPR=0.00; plain: n=3 acc=1.00 FPR=0.00
**e3**: benign_sensitive: n=25 acc=0.76 FPR=0.24; direct: n=20 acc=1.00 FPR=0.00; plain: n=3 acc=1.00 FPR=0.00
**e3a_exact**: benign_sensitive: n=25 acc=0.52 FPR=0.48; direct: n=20 acc=1.00 FPR=0.00; plain: n=3 acc=1.00 FPR=0.00
**e3a**: benign_sensitive: n=25 acc=0.40 FPR=0.60; direct: n=20 acc=1.00 FPR=0.00; plain: n=3 acc=1.00 FPR=0.00
**e4**: benign_sensitive: n=25 acc=0.76 FPR=0.24; direct: n=20 acc=1.00 FPR=0.00; plain: n=3 acc=1.00 FPR=0.00
**e4_filter**: benign_sensitive: n=25 acc=0.76 FPR=0.24; direct: n=20 acc=1.00 FPR=0.00; plain: n=3 acc=1.00 FPR=0.00
**e4_cite**: benign_sensitive: n=25 acc=0.76 FPR=0.24; direct: n=20 acc=1.00 FPR=0.00; plain: n=3 acc=1.00 FPR=0.00

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
