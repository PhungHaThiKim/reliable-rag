# Task B supply-chain QA (400 questions)

LLM: Qwen/Qwen2.5-Coder-7B-Instruct
E4 policy: {'tau': 1.0, 'filter_elements': False, 'objective': '0.5*answerable_exact + 0.5*correct_abstention', 'dev_objective': 0.666667, 'dev_answerable_exact': 0.5583333333333333, 'dev_correct_abstention': 0.775, 'dev_answer_coverage': 0.85, 'tuned_on': 'results/cloud/pod_C/supplychain/runs/dev_Qwen2.5-Coder-7B-Instruct:e3', 'n': 160}

## G1 (answer quality)

| method | n | exact_accuracy | set_f1 | answer_coverage | answered_accuracy | aurc | unsupported_over_original_claims | fabricated_answer_rate | ev_recall | tokens_in_mean |
|---|---|---|---|---|---|---|---|---|---|---|
| e0 | 100 | 0.750 | 0.750 | 0.980 | 0.765 | 0.031 | 0.005 | 0.000 | 0.990 | 346.680 |
| e1 | 100 | 0.800 | 0.800 | 1.000 | 0.800 | 0.022 | 0.000 | 0.000 | 1.000 | 1168.340 |
| e2 | 100 | 0.750 | 0.750 | 0.990 | 0.758 | 0.033 | 0.000 | 0.000 | 0.990 | 1073.630 |
| e2_bm25 | 100 | 0.660 | 0.660 | 0.990 | 0.667 | 0.065 | 0.000 | 0.000 | 1.000 | 871.050 |
| e3 | 100 | 0.730 | 0.730 | 0.950 | 0.768 | 0.030 | 0.000 | 0.000 | 1.000 | 378.680 |
| e3a_exact | 100 | 0.720 | 0.720 | 0.940 | 0.766 | 0.031 | 0.000 | 0.000 | 1.000 | 376.680 |
| e3a | 100 | 0.790 | 0.790 | 0.990 | 0.798 | 0.023 | 0.013 | 0.000 | 1.000 | 955.440 |
| e4 | 100 | 0.730 | 0.730 | 0.950 | 0.768 | 0.030 | 0.000 | 0.000 | 1.000 | 378.680 |

## G2 (answer quality)

| method | n | exact_accuracy | set_f1 | answer_coverage | answered_accuracy | aurc | unsupported_over_original_claims | fabricated_answer_rate | ev_recall | tokens_in_mean |
|---|---|---|---|---|---|---|---|---|---|---|
| e0 | 100 | 0.000 | 0.000 | 0.010 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 278.960 |
| e1 | 100 | 0.360 | 0.360 | 0.710 | 0.507 | 0.152 | 0.051 | 0.020 | 0.235 | 1781.720 |
| e2 | 100 | 0.520 | 0.520 | 0.890 | 0.584 | 0.104 | 0.007 | 0.000 | 0.615 | 1761.180 |
| e2_bm25 | 100 | 0.450 | 0.450 | 0.840 | 0.536 | 0.133 | 0.006 | 0.000 | 0.885 | 1596.880 |
| e3 | 100 | 0.590 | 0.590 | 1.000 | 0.590 | 0.101 | 0.035 | 0.000 | 0.975 | 819.630 |
| e3a_exact | 100 | 0.580 | 0.580 | 0.960 | 0.604 | 0.093 | 0.204 | 0.000 | 0.795 | 817.200 |
| e3a | 100 | 0.360 | 0.360 | 0.690 | 0.522 | 0.142 | 0.051 | 0.010 | 0.650 | 867.010 |
| e4 | 100 | 0.560 | 0.560 | 0.950 | 0.589 | 0.101 | 0.000 | 0.000 | 0.975 | 819.630 |

## G3 (answer quality)

| method | n | exact_accuracy | set_f1 | answer_coverage | answered_accuracy | aurc | unsupported_over_original_claims | fabricated_answer_rate | ev_recall | tokens_in_mean |
|---|---|---|---|---|---|---|---|---|---|---|
| e0 | 100 | 0.000 | 0.000 | 0.030 | 0.000 | 1.000 | 1.000 | 0.030 | 0.000 | 282.210 |
| e1 | 100 | 0.220 | 0.247 | 0.490 | 0.449 | 0.197 | 0.049 | 0.040 | 0.475 | 1495.420 |
| e2 | 100 | 0.230 | 0.252 | 0.500 | 0.460 | 0.188 | 0.088 | 0.020 | 0.461 | 1442.620 |
| e2_bm25 | 100 | 0.210 | 0.210 | 0.370 | 0.568 | 0.117 | 0.003 | 0.000 | 0.396 | 1298.210 |
| e3 | 100 | 0.540 | 0.603 | 0.790 | 0.684 | 0.058 | 0.019 | 0.050 | 0.948 | 676.230 |
| e3a_exact | 100 | 0.540 | 0.583 | 0.820 | 0.659 | 0.068 | 0.076 | 0.050 | 0.838 | 646.830 |
| e3a | 100 | 0.150 | 0.160 | 0.360 | 0.417 | 0.227 | 0.042 | 0.010 | 0.559 | 996.930 |
| e4 | 100 | 0.520 | 0.583 | 0.720 | 0.722 | 0.045 | 0.003 | 0.050 | 0.948 | 676.230 |

## answerable (answer quality)

| method | n | exact_accuracy | set_f1 | answer_coverage | answered_accuracy | aurc | unsupported_over_original_claims | fabricated_answer_rate | ev_recall | tokens_in_mean |
|---|---|---|---|---|---|---|---|---|---|---|
| e0 | 300 | 0.250 | 0.250 | 0.340 | 0.735 | 0.040 | 0.787 | 0.010 | 0.330 | - |
| e1 | 300 | 0.460 | 0.469 | 0.733 | 0.627 | 0.081 | 0.037 | 0.020 | 0.570 | - |
| e2 | 300 | 0.500 | 0.507 | 0.793 | 0.630 | 0.080 | 0.034 | 0.007 | 0.689 | - |
| e2_bm25 | 300 | 0.440 | 0.440 | 0.733 | 0.600 | 0.094 | 0.003 | 0.000 | 0.760 | - |
| e3 | 300 | 0.620 | 0.641 | 0.913 | 0.679 | 0.059 | 0.021 | 0.017 | 0.974 | - |
| e3a_exact | 300 | 0.613 | 0.628 | 0.907 | 0.676 | 0.060 | 0.109 | 0.017 | 0.878 | - |
| e3a | 300 | 0.433 | 0.437 | 0.680 | 0.637 | 0.076 | 0.037 | 0.007 | 0.736 | - |
| e4 | 300 | 0.603 | 0.624 | 0.873 | 0.691 | 0.054 | 0.001 | 0.017 | 0.974 | - |

## G4 (insufficient evidence)

| method | n | correct_abstention_rate | unsupported_over_original_claims | fabricated_answer_rate | tokens_in_mean |
|---|---|---|---|---|---|
| e0 | 100 | 0.850 | 0.792 | 0.000 | 279.690 |
| e1 | 100 | 0.590 | 0.096 | 0.000 | 1340.060 |
| e2 | 100 | 0.470 | 0.115 | 0.010 | 1322.110 |
| e2_bm25 | 100 | 0.500 | 0.120 | 0.020 | 1148.510 |
| e3 | 100 | 0.560 | 0.052 | 0.000 | 671.420 |
| e3a_exact | 100 | 0.490 | 0.175 | 0.000 | 649.530 |
| e3a | 100 | 0.560 | 0.081 | 0.000 | 892.790 |
| e4 | 100 | 0.650 | 0.015 | 0.000 | 671.420 |

## Grounding and cost over all questions (same denominators for every method)

| method | unsupported_claim_rate | unsupported_over_original_claims | questions_with_unsupported | claims_retained | fabricated_answer_rate | ev_recall | ev_recall_visible | tokens_in_mean | tokens_out_mean | latency_p50_ms |
|---|---|---|---|---|---|---|---|---|---|---|
| e0 | 0.788 | 0.788 | 0.713 | 1.000 | 0.007 | 0.312 | 0.314 | 296.885 | 147.885 | 137.944 |
| e1 | 0.050 | 0.050 | 0.102 | 1.000 | 0.015 | 0.623 | 0.657 | 1446.385 | 173.330 | 137.944 |
| e2 | 0.053 | 0.053 | 0.113 | 1.000 | 0.007 | 0.712 | 0.771 | 1399.885 | 173.100 | 137.944 |
| e2_bm25 | 0.032 | 0.032 | 0.055 | 1.000 | 0.005 | 0.767 | 0.815 | 1228.662 | 182.532 | 137.944 |
| e3 | 0.030 | 0.030 | 0.055 | 1.000 | 0.013 | 0.977 | 0.991 | 636.490 | 179.768 | 137.944 |
| e3a_exact | 0.125 | 0.125 | 0.170 | 1.000 | 0.013 | 0.874 | 0.991 | 622.560 | 193.375 | 137.944 |
| e3a | 0.048 | 0.048 | 0.095 | 1.000 | 0.005 | 0.763 | 0.793 | 928.043 | 190.143 | 137.944 |
| e4 | 0.030 | 0.005 | 0.013 | 0.931 | 0.013 | 0.977 | 0.991 | 636.490 | 179.768 | 137.944 |

Measured GPU seconds 411.149, 2800 unique prompts, 0.132 s per 1k tokens; price/h 1.09

| method | tokens/item | GPU s/item | USD/1000 | reuses E3 |
|---|---|---|---|---|
| e0 | 444.770 | 0.059 | 0.018 | False |
| e1 | 1619.715 | 0.213 | 0.065 | False |
| e2 | 1572.985 | 0.207 | 0.063 | False |
| e2_bm25 | 1411.195 | 0.186 | 0.056 | False |
| e3 | 816.258 | 0.108 | 0.033 | False |
| e3a_exact | 815.935 | 0.108 | 0.033 | False |
| e3a | 1118.185 | 0.147 | 0.045 | False |
| e4 | 816.258 | 0.000 | 0.000 | True |

## E4 operating points (tau sweep on E3 answers, descriptive)

| tau | answer coverage | answered accuracy | exact accuracy | correct abstention | unsupported/original |
|---|---|---|---|---|---|
| 0.0 | 0.897 | 0.691 | 0.620 | 0.560 | 0.028 |
| 0.25 | 0.890 | 0.689 | 0.613 | 0.560 | 0.021 |
| 0.34 | 0.890 | 0.689 | 0.613 | 0.560 | 0.021 |
| 0.5 | 0.890 | 0.689 | 0.613 | 0.570 | 0.019 |
| 0.67 | 0.873 | 0.691 | 0.603 | 0.610 | 0.007 |
| 0.75 | 0.873 | 0.691 | 0.603 | 0.610 | 0.007 |
| 1.0 | 0.873 | 0.691 | 0.603 | 0.650 | 0.005 |

## Paired comparisons per group (cluster bootstrap over entities, McNemar)

| pair | group | metric | n | delta | 95% CI | McNemar p |
|---|---|---|---|---|---|---|
| e1_vs_e0 | G1 | exact_accuracy | 100 | +0.050 | [-0.032, +0.135] | 0.3018 |
| e1_vs_e0 | G2 | exact_accuracy | 100 | +0.360 | [+0.210, +0.465] | 0.0000 |
| e1_vs_e0 | G3 | exact_accuracy | 100 | +0.220 | [+0.067, +0.364] | 0.0000 |
| e1_vs_e0 | G4 | correct_abstention | 100 | -0.260 | [-0.375, -0.145] | 0.0000 |
| e1_vs_e0 | answerable | exact_accuracy | 300 | +0.210 | [+0.118, +0.297] | 0.0000 |
| e2_vs_e1 | G1 | exact_accuracy | 100 | -0.050 | [-0.118, +0.011] | 0.2266 |
| e2_vs_e1 | G2 | exact_accuracy | 100 | +0.160 | [+0.043, +0.373] | 0.0113 |
| e2_vs_e1 | G3 | exact_accuracy | 100 | +0.010 | [-0.083, +0.109] | 1.0000 |
| e2_vs_e1 | G4 | correct_abstention | 100 | -0.120 | [-0.192, -0.062] | 0.0042 |
| e2_vs_e1 | answerable | exact_accuracy | 300 | +0.040 | [-0.017, +0.103] | 0.1619 |
| e2_bm25_vs_e2 | G1 | exact_accuracy | 100 | -0.090 | [-0.167, -0.020] | 0.0225 |
| e2_bm25_vs_e2 | G2 | exact_accuracy | 100 | -0.070 | [-0.198, +0.011] | 0.2478 |
| e2_bm25_vs_e2 | G3 | exact_accuracy | 100 | -0.020 | [-0.086, +0.034] | 0.7744 |
| e2_bm25_vs_e2 | G4 | correct_abstention | 100 | +0.030 | [-0.049, +0.114] | 0.6776 |
| e2_bm25_vs_e2 | answerable | exact_accuracy | 300 | -0.060 | [-0.109, -0.019] | 0.0175 |
| e3_vs_e2 | G1 | exact_accuracy | 100 | -0.020 | [-0.083, +0.041] | 0.7539 |
| e3_vs_e2 | G2 | exact_accuracy | 100 | +0.070 | [-0.066, +0.273] | 0.3817 |
| e3_vs_e2 | G3 | exact_accuracy | 100 | +0.310 | [+0.152, +0.543] | 0.0001 |
| e3_vs_e2 | G4 | correct_abstention | 100 | +0.090 | [-0.104, +0.263] | 0.1877 |
| e3_vs_e2 | answerable | exact_accuracy | 300 | +0.120 | [+0.039, +0.205] | 0.0012 |
| e3_vs_e3a_exact | G1 | exact_accuracy | 100 | +0.010 | [+0.000, +0.032] | 1.0000 |
| e3_vs_e3a_exact | G2 | exact_accuracy | 100 | +0.010 | [-0.060, +0.147] | 1.0000 |
| e3_vs_e3a_exact | G3 | exact_accuracy | 100 | +0.000 | [-0.049, +0.064] | 1.0000 |
| e3_vs_e3a_exact | G4 | correct_abstention | 100 | +0.070 | [+0.000, +0.143] | 0.1185 |
| e3_vs_e3a_exact | answerable | exact_accuracy | 300 | +0.007 | [-0.026, +0.048] | 0.8642 |
| e3_vs_e3a | G1 | exact_accuracy | 100 | -0.060 | [-0.113, -0.010] | 0.0703 |
| e3_vs_e3a | G2 | exact_accuracy | 100 | +0.230 | [+0.039, +0.593] | 0.0032 |
| e3_vs_e3a | G3 | exact_accuracy | 100 | +0.390 | [+0.219, +0.603] | 0.0000 |
| e3_vs_e3a | G4 | correct_abstention | 100 | +0.000 | [-0.175, +0.153] | 1.0000 |
| e3_vs_e3a | answerable | exact_accuracy | 300 | +0.187 | [+0.090, +0.292] | 0.0000 |
| e4_vs_e3 | G1 | exact_accuracy | 100 | +0.000 | [+0.000, +0.000] | 1.0000 |
| e4_vs_e3 | G2 | exact_accuracy | 100 | -0.030 | [-0.077, +0.000] | 0.2500 |
| e4_vs_e3 | G3 | exact_accuracy | 100 | -0.020 | [-0.052, +0.000] | 0.5000 |
| e4_vs_e3 | G4 | correct_abstention | 100 | +0.090 | [+0.000, +0.217] | 0.0039 |
| e4_vs_e3 | answerable | exact_accuracy | 300 | -0.017 | [-0.034, -0.003] | 0.0625 |
| e3_vs_e0 | G1 | exact_accuracy | 100 | -0.020 | [-0.091, +0.054] | 0.7744 |
| e3_vs_e0 | G2 | exact_accuracy | 100 | +0.590 | [+0.481, +0.804] | 0.0000 |
| e3_vs_e0 | G3 | exact_accuracy | 100 | +0.540 | [+0.467, +0.696] | 0.0000 |
| e3_vs_e0 | G4 | correct_abstention | 100 | -0.290 | [-0.505, -0.083] | 0.0000 |
| e3_vs_e0 | answerable | exact_accuracy | 300 | +0.370 | [+0.302, +0.431] | 0.0000 |
