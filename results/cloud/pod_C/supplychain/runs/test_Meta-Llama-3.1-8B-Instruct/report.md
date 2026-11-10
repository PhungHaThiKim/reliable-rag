# Task B supply-chain QA (400 questions)

LLM: NousResearch/Meta-Llama-3.1-8B-Instruct
E4 policy: {'tau': 0.0, 'filter_elements': False, 'objective': '0.5*answerable_exact + 0.5*correct_abstention', 'dev_objective': 0.591667, 'dev_answerable_exact': 0.23333333333333334, 'dev_correct_abstention': 0.95, 'dev_answer_coverage': 0.38333333333333336, 'tuned_on': 'results/cloud/pod_C/supplychain/runs/dev_Meta-Llama-3.1-8B-Instruct:e3', 'n': 160}

## G1 (answer quality)

| method | n | exact_accuracy | set_f1 | answer_coverage | answered_accuracy | aurc | unsupported_over_original_claims | fabricated_answer_rate | ev_recall | tokens_in_mean |
|---|---|---|---|---|---|---|---|---|---|---|
| e0 | 100 | 0.620 | 0.620 | 0.780 | 0.795 | 0.024 | 0.013 | 0.000 | 0.800 | 343.400 |
| e1 | 100 | 0.630 | 0.630 | 0.790 | 0.797 | 0.023 | 0.000 | 0.000 | 0.800 | 1141.740 |
| e2 | 100 | 0.510 | 0.510 | 0.630 | 0.810 | 0.021 | 0.003 | 0.000 | 0.640 | 1028.550 |
| e2_bm25 | 100 | 0.570 | 0.570 | 0.690 | 0.826 | 0.017 | 0.038 | 0.000 | 0.670 | 827.730 |
| e3 | 100 | 0.260 | 0.260 | 0.370 | 0.703 | 0.053 | 0.141 | 0.000 | 0.450 | 375.400 |
| e3a_exact | 100 | 0.380 | 0.380 | 0.430 | 0.884 | 0.008 | 0.107 | 0.000 | 0.500 | 373.400 |
| e3a | 100 | 0.580 | 0.580 | 0.730 | 0.795 | 0.024 | 0.027 | 0.000 | 0.740 | 900.600 |
| e4 | 100 | 0.260 | 0.260 | 0.370 | 0.703 | 0.053 | 0.141 | 0.000 | 0.450 | 375.400 |

## G2 (answer quality)

| method | n | exact_accuracy | set_f1 | answer_coverage | answered_accuracy | aurc | unsupported_over_original_claims | fabricated_answer_rate | ev_recall | tokens_in_mean |
|---|---|---|---|---|---|---|---|---|---|---|
| e0 | 100 | 0.000 | 0.000 | 0.000 | nan | nan | 1.000 | 0.000 | 0.000 | 276.380 |
| e1 | 100 | 0.030 | 0.030 | 0.060 | 0.500 | 0.192 | 0.075 | 0.000 | 0.090 | 1732.440 |
| e2 | 100 | 0.260 | 0.260 | 0.460 | 0.565 | 0.117 | 0.039 | 0.000 | 0.405 | 1702.710 |
| e2_bm25 | 100 | 0.260 | 0.260 | 0.690 | 0.377 | 0.260 | 0.013 | 0.000 | 0.625 | 1530.020 |
| e3 | 100 | 0.330 | 0.330 | 0.490 | 0.673 | 0.064 | 0.251 | 0.000 | 0.525 | 750.980 |
| e3a_exact | 100 | 0.330 | 0.330 | 0.470 | 0.702 | 0.053 | 0.310 | 0.000 | 0.510 | 748.480 |
| e3a | 100 | 0.060 | 0.060 | 0.100 | 0.600 | 0.113 | 0.571 | 0.000 | 0.215 | 822.880 |
| e4 | 100 | 0.330 | 0.330 | 0.490 | 0.673 | 0.064 | 0.251 | 0.000 | 0.525 | 750.980 |

## G3 (answer quality)

| method | n | exact_accuracy | set_f1 | answer_coverage | answered_accuracy | aurc | unsupported_over_original_claims | fabricated_answer_rate | ev_recall | tokens_in_mean |
|---|---|---|---|---|---|---|---|---|---|---|
| e0 | 100 | 0.050 | 0.050 | 0.110 | 0.455 | 0.211 | 1.000 | 0.020 | 0.000 | 276.960 |
| e1 | 100 | 0.020 | 0.033 | 0.180 | 0.111 | 0.667 | 0.085 | 0.010 | 0.234 | 1425.190 |
| e2 | 100 | 0.070 | 0.082 | 0.260 | 0.269 | 0.391 | 0.127 | 0.010 | 0.282 | 1363.410 |
| e2_bm25 | 100 | 0.040 | 0.040 | 0.110 | 0.364 | 0.296 | 0.180 | 0.040 | 0.221 | 1206.970 |
| e3 | 100 | 0.200 | 0.251 | 0.390 | 0.513 | 0.151 | 0.125 | 0.040 | 0.534 | 639.560 |
| e3a_exact | 100 | 0.150 | 0.199 | 0.340 | 0.441 | 0.206 | 0.130 | 0.020 | 0.431 | 612.630 |
| e3a | 100 | 0.020 | 0.030 | 0.200 | 0.100 | 0.690 | 0.177 | 0.020 | 0.302 | 900.850 |
| e4 | 100 | 0.200 | 0.251 | 0.350 | 0.571 | 0.115 | 0.112 | 0.040 | 0.534 | 639.560 |

## answerable (answer quality)

| method | n | exact_accuracy | set_f1 | answer_coverage | answered_accuracy | aurc | unsupported_over_original_claims | fabricated_answer_rate | ev_recall | tokens_in_mean |
|---|---|---|---|---|---|---|---|---|---|---|
| e0 | 300 | 0.223 | 0.223 | 0.297 | 0.753 | 0.035 | 0.457 | 0.007 | 0.267 | - |
| e1 | 300 | 0.227 | 0.231 | 0.343 | 0.660 | 0.067 | 0.044 | 0.003 | 0.375 | - |
| e2 | 300 | 0.280 | 0.284 | 0.450 | 0.622 | 0.084 | 0.055 | 0.003 | 0.442 | - |
| e2_bm25 | 300 | 0.290 | 0.290 | 0.497 | 0.584 | 0.103 | 0.071 | 0.013 | 0.505 | - |
| e3 | 300 | 0.263 | 0.280 | 0.417 | 0.632 | 0.079 | 0.177 | 0.013 | 0.503 | - |
| e3a_exact | 300 | 0.287 | 0.303 | 0.413 | 0.694 | 0.054 | 0.196 | 0.007 | 0.480 | - |
| e3a | 300 | 0.220 | 0.223 | 0.343 | 0.641 | 0.076 | 0.260 | 0.007 | 0.419 | - |
| e4 | 300 | 0.263 | 0.280 | 0.403 | 0.653 | 0.070 | 0.173 | 0.013 | 0.503 | - |

## G4 (insufficient evidence)

| method | n | correct_abstention_rate | unsupported_over_original_claims | fabricated_answer_rate | tokens_in_mean |
|---|---|---|---|---|---|
| e0 | 100 | 0.900 | 0.783 | 0.000 | 276.290 |
| e1 | 100 | 0.860 | 0.308 | 0.000 | 1282.290 |
| e2 | 100 | 0.760 | 0.282 | 0.020 | 1260.810 |
| e2_bm25 | 100 | 0.760 | 0.346 | 0.000 | 1091.440 |
| e3 | 100 | 0.930 | 0.437 | 0.000 | 635.370 |
| e3a_exact | 100 | 0.820 | 0.454 | 0.000 | 616.100 |
| e3a | 100 | 0.870 | 0.450 | 0.010 | 836.920 |
| e4 | 100 | 0.930 | 0.437 | 0.000 | 635.370 |

## Grounding and cost over all questions (same denominators for every method)

| method | unsupported_claim_rate | unsupported_over_original_claims | questions_with_unsupported | claims_retained | fabricated_answer_rate | ev_recall | ev_recall_visible | tokens_in_mean | tokens_out_mean | latency_p50_ms |
|---|---|---|---|---|---|---|---|---|---|---|
| e0 | 0.536 | 0.536 | 0.320 | 1.000 | 0.005 | 0.244 | 0.314 | 293.257 | 215.040 | 166.001 |
| e1 | 0.114 | 0.114 | 0.110 | 1.000 | 0.003 | 0.390 | 0.657 | 1395.415 | 208.067 | 166.001 |
| e2 | 0.115 | 0.115 | 0.128 | 1.000 | 0.007 | 0.475 | 0.771 | 1338.870 | 219.420 | 166.001 |
| e2_bm25 | 0.140 | 0.140 | 0.158 | 1.000 | 0.010 | 0.491 | 0.815 | 1164.040 | 212.985 | 166.001 |
| e3 | 0.241 | 0.241 | 0.212 | 1.000 | 0.010 | 0.482 | 0.991 | 600.327 | 176.583 | 166.001 |
| e3a_exact | 0.262 | 0.262 | 0.233 | 1.000 | 0.005 | 0.465 | 0.991 | 587.653 | 186.558 | 166.001 |
| e3a | 0.304 | 0.304 | 0.312 | 1.000 | 0.007 | 0.418 | 0.793 | 865.312 | 188.350 | 166.001 |
| e4 | 0.241 | 0.238 | 0.210 | 0.983 | 0.010 | 0.482 | 0.991 | 600.327 | 176.583 | 166.001 |

Measured GPU seconds 487.281, 2800 unique prompts, 0.159 s per 1k tokens; price/h 1.09

| method | tokens/item | GPU s/item | USD/1000 | reuses E3 |
|---|---|---|---|---|
| e0 | 508.298 | 0.081 | 0.025 | False |
| e1 | 1603.483 | 0.255 | 0.077 | False |
| e2 | 1558.290 | 0.248 | 0.075 | False |
| e2_bm25 | 1377.025 | 0.219 | 0.066 | False |
| e3 | 776.910 | 0.124 | 0.037 | False |
| e3a_exact | 774.210 | 0.123 | 0.037 | False |
| e3a | 1053.662 | 0.168 | 0.051 | False |
| e4 | 776.910 | 0.000 | 0.000 | True |

## E4 operating points (tau sweep on E3 answers, descriptive)

| tau | answer coverage | answered accuracy | exact accuracy | correct abstention | unsupported/original |
|---|---|---|---|---|---|
| 0.0 | 0.403 | 0.653 | 0.263 | 0.930 | 0.238 |
| 0.25 | 0.360 | 0.667 | 0.240 | 0.950 | 0.175 |
| 0.34 | 0.360 | 0.667 | 0.240 | 0.950 | 0.175 |
| 0.5 | 0.357 | 0.673 | 0.240 | 0.950 | 0.173 |
| 0.67 | 0.333 | 0.680 | 0.227 | 0.960 | 0.159 |
| 0.75 | 0.333 | 0.680 | 0.227 | 0.960 | 0.159 |
| 1.0 | 0.323 | 0.691 | 0.223 | 0.960 | 0.157 |

## Paired comparisons per group (cluster bootstrap over entities, McNemar)

| pair | group | metric | n | delta | 95% CI | McNemar p |
|---|---|---|---|---|---|---|
| e1_vs_e0 | G1 | exact_accuracy | 100 | +0.010 | [-0.087, +0.112] | 1.0000 |
| e1_vs_e0 | G2 | exact_accuracy | 100 | +0.030 | [+0.000, +0.072] | 0.2500 |
| e1_vs_e0 | G3 | exact_accuracy | 100 | -0.030 | [-0.080, +0.020] | 0.4531 |
| e1_vs_e0 | G4 | correct_abstention | 100 | -0.040 | [-0.133, +0.052] | 0.4545 |
| e1_vs_e0 | answerable | exact_accuracy | 300 | +0.003 | [-0.039, +0.044] | 1.0000 |
| e2_vs_e1 | G1 | exact_accuracy | 100 | -0.120 | [-0.210, -0.030] | 0.0227 |
| e2_vs_e1 | G2 | exact_accuracy | 100 | +0.230 | [+0.129, +0.423] | 0.0000 |
| e2_vs_e1 | G3 | exact_accuracy | 100 | +0.050 | [-0.008, +0.128] | 0.1797 |
| e2_vs_e1 | G4 | correct_abstention | 100 | -0.100 | [-0.198, +0.000] | 0.0414 |
| e2_vs_e1 | answerable | exact_accuracy | 300 | +0.053 | [-0.004, +0.106] | 0.0519 |
| e2_bm25_vs_e2 | G1 | exact_accuracy | 100 | +0.060 | [-0.063, +0.178] | 0.3771 |
| e2_bm25_vs_e2 | G2 | exact_accuracy | 100 | +0.000 | [-0.222, +0.110] | 1.0000 |
| e2_bm25_vs_e2 | G3 | exact_accuracy | 100 | -0.030 | [-0.069, +0.000] | 0.4531 |
| e2_bm25_vs_e2 | G4 | correct_abstention | 100 | +0.000 | [-0.158, +0.151] | 1.0000 |
| e2_bm25_vs_e2 | answerable | exact_accuracy | 300 | +0.010 | [-0.056, +0.073] | 0.8099 |
| e3_vs_e2 | G1 | exact_accuracy | 100 | -0.250 | [-0.379, -0.126] | 0.0003 |
| e3_vs_e2 | G2 | exact_accuracy | 100 | +0.070 | [-0.129, +0.180] | 0.2100 |
| e3_vs_e2 | G3 | exact_accuracy | 100 | +0.130 | [+0.020, +0.247] | 0.0146 |
| e3_vs_e2 | G4 | correct_abstention | 100 | +0.170 | [+0.054, +0.283] | 0.0002 |
| e3_vs_e2 | answerable | exact_accuracy | 300 | -0.017 | [-0.109, +0.064] | 0.6817 |
| e3_vs_e3a_exact | G1 | exact_accuracy | 100 | -0.120 | [-0.219, -0.021] | 0.0428 |
| e3_vs_e3a_exact | G2 | exact_accuracy | 100 | +0.000 | [-0.075, +0.119] | 1.0000 |
| e3_vs_e3a_exact | G3 | exact_accuracy | 100 | +0.050 | [-0.023, +0.159] | 0.3593 |
| e3_vs_e3a_exact | G4 | correct_abstention | 100 | +0.110 | [+0.036, +0.180] | 0.0127 |
| e3_vs_e3a_exact | answerable | exact_accuracy | 300 | -0.023 | [-0.075, +0.030] | 0.4828 |
| e3_vs_e3a | G1 | exact_accuracy | 100 | -0.320 | [-0.431, -0.215] | 0.0000 |
| e3_vs_e3a | G2 | exact_accuracy | 100 | +0.270 | [+0.132, +0.388] | 0.0000 |
| e3_vs_e3a | G3 | exact_accuracy | 100 | +0.180 | [+0.113, +0.286] | 0.0000 |
| e3_vs_e3a | G4 | correct_abstention | 100 | +0.060 | [-0.018, +0.154] | 0.2101 |
| e3_vs_e3a | answerable | exact_accuracy | 300 | +0.043 | [-0.045, +0.129] | 0.2229 |
| e4_vs_e3 | G1 | exact_accuracy | 100 | +0.000 | [+0.000, +0.000] | 1.0000 |
| e4_vs_e3 | G2 | exact_accuracy | 100 | +0.000 | [+0.000, +0.000] | 1.0000 |
| e4_vs_e3 | G3 | exact_accuracy | 100 | +0.000 | [+0.000, +0.000] | 1.0000 |
| e4_vs_e3 | G4 | correct_abstention | 100 | +0.000 | [+0.000, +0.000] | 1.0000 |
| e4_vs_e3 | answerable | exact_accuracy | 300 | +0.000 | [+0.000, +0.000] | 1.0000 |
| e3_vs_e0 | G1 | exact_accuracy | 100 | -0.360 | [-0.469, -0.257] | 0.0000 |
| e3_vs_e0 | G2 | exact_accuracy | 100 | +0.330 | [+0.179, +0.446] | 0.0000 |
| e3_vs_e0 | G3 | exact_accuracy | 100 | +0.150 | [+0.075, +0.258] | 0.0041 |
| e3_vs_e0 | G4 | correct_abstention | 100 | +0.030 | [-0.038, +0.102] | 0.5488 |
| e3_vs_e0 | answerable | exact_accuracy | 300 | +0.040 | [-0.062, +0.136] | 0.2713 |
