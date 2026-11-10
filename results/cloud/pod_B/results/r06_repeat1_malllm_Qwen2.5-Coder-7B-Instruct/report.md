# Results: r06_repeat1_malllm_Qwen2.5-Coder-7B-Instruct

Model: `vllm:Qwen/Qwen2.5-Coder-7B-Instruct` | split: `test` | n = 100 | methods: e0

Strict = abstentions counted as errors. Claim labels check consistency with static facts, not truth.

## Headline

| method | strict/f1_malicious | strict/macro_f1 | strict/fpr | strict/fnr | answered/f1_malicious | coverage | aurc | rc_max_coverage | evidence/unsupported_claim_rate | evidence/unsupported_over_original_claims | evidence/items_with_unsupported_all | evidence/citation_accuracy | gold/evidence_coverage | gold/path_recall | tokens_in_mean | tokens_out_mean | latency_p50_ms | parse_failure_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e0 | 0.920 | 0.947 | 0.000 | 0.148 | 0.920 | 1.000 | 0.011 | 1.000 | 0.508 | 0.675 | 0.770 | 0.506 | nan | nan | 637.220 | 274.600 | 131.637 | 0.000 |

- e0: strict malicious-F1 0.920 [95% CI 0.839, 0.984]

## Cost (measured GPU time, allocated by tokens)

GPU seconds 13 for 100 unique prompts; 0.144 s per 1k tokens; price/h 1.09

| method | tokens_per_item | gpu_seconds_per_item | usd_per_1000_items | reuses_e3_generation |
|---|---|---|---|---|
| e0 | 911.820 | 0.132 | 0.040 | False |

## Strata

**e0**: all: n=100 acc=0.96 FPR=0.00
