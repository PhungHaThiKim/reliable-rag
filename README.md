# Seminar 2. Đề 11. Software Supply Chain Security: LLM thuần, RAG, Hybrid RAG, KG-RAG, xác minh bằng chứng

Một hệ thống, một base LLM, hai bài toán nối nhau:
- **Task A, phát hiện package PyPI độc** (bám paper): E0-E4 trên `malllm`, `pyguard_obf`, `synthetic` và 6 test case kiểm soát C01-C06.
- **Task B, hỏi đáp supply chain** trên một công ty tổng hợp: 4 nhóm câu hỏi (hiểu mã, chính sách/ADR, tác động dependency nhiều bước,
  thiếu bằng chứng). **Mọi method trả lời cùng một bộ câu hỏi** có đáp án và bằng chứng chuẩn; không thiết kế để mỗi method thắng một nhóm.

Paper tham chiếu: *Detecting Malicious Source Code in PyPI Packages with LLMs: Does RAG Come in Handy?* ([mal-LLM](https://github.com/malexternalsc/mal-LLM)).
Tri thức hành vi lấy từ [PyGuard](https://github.com/lxyeternal/PyGuard).

**Trạng thái.** Code, 39 test và 5 notebook đã xong. Lần chạy thật đầu tiên: 2026-11-10 trên 3 pod RunPod một GPU, 2 reader
(Llama-3.1-8B-Instruct, Qwen2.5-Coder-7B-Instruct). Kết quả tổng hợp ở `results/cloud/` (chỉ `results.json`, `report.md`, notebook 00/01/04;
dữ liệu và dự đoán Task A chứa mã độc ở lại pod), đồ thị ở `results/figures/`, đọc kết quả ở [docs/PRESENTATION_FLOW.md](docs/PRESENTATION_FLOW.md).

**Đo được trong lần chạy 2026-11-02** (thay cho mọi ước lượng cũ):

| Pod | GPU | Việc | Thời gian chạy | GPU giây sinh (đo) |
|---|---|---|---|---|
| A | A100 80GB, 1.79 USD/giờ | Task A `pyguard_obf` (val 200 + test 310, 2 reader) | notebook 00: 3 phút; 01: 37 phút | test: 413 s (Llama), 328 s (Qwen) |
| B | L40S 48GB, 1.09 USD/giờ | `malllm` (tái lập + val + test 500), synthetic, cases, lặp lại | notebook 00: 3 phút; 01: 70 phút | test: 549 s (Llama), 455 s (Qwen) |
| C | L40S 48GB, 1.09 USD/giờ | Task B (dev 160 + test 400 câu, 8 method, 2 reader) | cài đặt 17 phút; notebook 04: 31 phút | test: 487 s (Llama), 411 s (Qwen) |

Phân tích tĩnh 1448 package `pyguard_obf` train: 14 giây (12 worker). Chi phí sinh mỗi 1000 mẫu mỗi method: 0.02-0.10 USD.

## Câu hỏi nghiên cứu và cấu hình

| RQ | Câu hỏi | So sánh |
|---|---|---|
| RQ1 | Truy hồi tri thức ngoài có giúp hơn LLM thuần? | E1 vs E0 |
| RQ2 | Hybrid (dense + BM25) có hơn chỉ dense? | E2 vs E1 |
| RQ3 | Truy hồi trên đồ thị hành vi có giúp hành vi mang tính quan hệ? | E3 vs E2, **E3 vs E3a** (cùng cạnh), E3 vs E3b |
| RQ4 | Xác minh bằng chứng có giảm claim không có chứng cứ mà không làm giảm phát hiện? | E4 vs E3 |
| RQ5 (Task B) | Với cùng reader và cùng câu hỏi, nhóm câu hỏi nào cần tài liệu, nhóm nào cần đồ thị, và hệ thống có biết từ chối? | 8 method x 4 nhóm |

| Method | Thêm gì vào prompt (sau khối package dùng chung) |
|---|---|
| `e0` LLM-only | không thêm gì |
| `e1` Simple RAG | top-5 tài liệu KB theo embedding |
| `e2` Hybrid RAG | dense + BM25, reciprocal-rank fusion; ablation `e2_bm25`, `e2_rerank` |
| `e3` KG-RAG hành vi | đường 1-3 bước trong đồ thị hành vi khớp pattern, có file:dòng và provenance từng cạnh |
| `e3a_exact` (E3a) | **đúng các cạnh của các đường E3 đã cho thấy**, mỗi cạnh một câu, xáo trộn có seed, cùng pattern K*, cùng ngân sách: chỉ bỏ cấu trúc đường |
| `e3a` (E3b) | fact của đồ thị viết thành câu, chọn bằng hybrid retrieval, không duyệt đồ thị |
| `e4` Verified KG-RAG | E3 + kiểm claim, bỏ claim không có chứng cứ, abstain khi thiếu/mâu thuẫn; ablation `e4_filter`, `e4_cite` |
| `e0_malllm`, `e1_malllm_yara`, `e1_malllm_code` | tái lập mal-LLM: 300 ký tự đầu `setup.py`, prompt gốc, top-2 YARA / top-1 mã độc |

Điều kiện chung (plan mục 6.2): cùng reader, cùng prompt hệ thống, decoding greedy, cùng khối package (metadata + đoạn mã `C*` quanh API nhạy cảm,
tối đa 3000 token) cho mọi method, cùng ngân sách 1500 token cho bằng chứng thêm vào. Không method nào thấy nhãn (`Sample.public_view()`).
So sánh luôn trên cùng sample id: paired bootstrap (cả F1) và McNemar.

## Dữ liệu

| Tên | Nguồn | Nội dung | Chia tách |
|---|---|---|---|
| `malllm` | mal-LLM `RQ_experiments/data/{train,test}_{malicious,benign}_packages_final.json` (Git LFS) | chỉ `setup.py` + danh sách file | test của paper giữ nguyên; val tách từ train theo nhóm; mẫu test trùng nhóm với train được đánh dấu |
| `pyguard_obf` | PyGuard `Dataset/PyPI/Experiment/Obfuscation/{Malware,Benign}/*.tgz` | package đầy đủ (khoảng 1041 độc / 1028 lành) | 70/15/15 sau khi gộp nhóm |
| `synthetic` | `pypi_rag/data/synthetic.py` | fixture không chạy được, có `gold_behaviors`, `gold_paths` | 70/15/15 theo nhóm |
| `cases` | `pypi_rag/data/cases.py` (C01-C06 của tài liệu test case, gọi `mock_runtime` giả) | `expected_verdict`, `required_evidence` (tính từ marker), `gold_function_path` | toàn bộ là test (`--strategy keep`) |
| Task B | `pypi_rag/supplychain/` (seed 1 = dev, seed 2 = test) | ~340 tài liệu ACME, KG parse từ chính các tài liệu, 4 x 100 câu hỏi | dev chỉ để tune E4 |

Nhóm = cùng tên package hoặc cùng family hoặc cùng hash hoặc near-duplicate (MinHash Jaccard >= 0.9). KB chỉ dựng từ **train** + tri thức ngoài;
kiểm tra rò rỉ loại tài liệu sinh từ val/test, tài liệu chứa nguyên văn file val/test, advisory nêu tên package val/test.

## Test case C01-C06 (Task A)

Sáu case của tài liệu test case chạy như một dataset (`prepare --spec cases: --strategy keep`), KB riêng `builtin + case_rules`
(PATTERN-EXFIL-001, MAL-001, BEN-001, BEN-002, RULE-001, cố định trước khi chạy). Hai cách hỏi: câu chuẩn (`q1`) và câu riêng của case
(`--use-sample-question`). Metric: `case/expected_verdict_accuracy`, `case/abstention_accuracy` (C06 phải `uncertain`),
`case/required_evidence_recall_visible|cited`, `case/answer_path_accuracy` (C05: claim quan hệ SUPPORTED và nêu đủ
`load_secret -> prepare_payload -> submit`). Đây là minh hoạ có kiểm soát (6 mẫu), không phải bằng chứng thống kê.

## Task B: hỏi đáp supply chain (4 nhóm, cùng bộ câu hỏi)

`make_org(seed)` sinh công ty ACME: 6 team, 30 service + 2 đã ngừng, ~95 package (public, nội bộ `acme-*`, typosquat) với cây phụ thuộc pin version,
7 advisory (4 typosquat, 3 bản phát hành bị chiếm quyền), policy bảo mật v1 (superseded) và v2 (hiện hành), 4 ADR ngoại lệ (có cái hết hạn),
hai policy token cùng hiện hành mâu thuẫn nhau, incident nhắc tên gây nhiễu. Tài liệu (`corpus.py`) là nguồn sự thật duy nhất:
text RAG đánh chỉ mục chúng, KG (`kg.py`) **parse chính các tài liệu đó** (mọi cạnh có `doc:line`).

| Nhóm | Loại câu hỏi | Bẫy (áp cho mọi method) |
|---|---|---|
| G1 hiểu mã | luồng nguồn -> mạng, host, biến môi trường (template + C01-C05) | đọc và gửi nhưng không nối nhau, luồng qua nhiều hàm |
| G2 chính sách | được phép deploy P==v?, hạn vá theo giờ, hạn token, ngoại lệ còn hiệu lực? | policy cũ, ADR hết hạn, ADR của service khác, dev được miễn |
| G3 tác động | service nào bị ảnh hưởng, đường phụ thuộc, version bắc cầu, team cần báo | dev-only, extra không bật, version ngoài khoảng, service đã ngừng |
| G4 thiếu bằng chứng | đáp án chuẩn `insufficient_evidence` | thực thể không tồn tại, version không pin, plugin theo config, hai policy mâu thuẫn, dynamic dispatch, mã thiếu |

Method: `e0, e1, e2, e2_bm25, e3, e3a_exact (E3a), e3a (E3b), e4`, cùng prompt hệ thống, cùng ngân sách 2000 token bằng chứng.
E4 = câu trả lời của E3 + kiểm chứng (bỏ phần tử không có trong bằng chứng, từ chối khi claim thiếu chứng cứ hoặc có tên bịa), ngưỡng tune trên org dev.
Metric theo nhóm: accuracy, partial F1, abstention accuracy (G4), false abstention (G1-G3), tỉ lệ tên bịa, claim không chứng cứ,
evidence recall/precision; so sánh cặp bằng bootstrap theo cụm thực thể + McNemar. Test kiểm tra đáp án G3 tính lại được chỉ từ KG parse tài liệu.

## Bộ metric: báo riêng theo task, không gộp một accuracy chung

Task A là phân loại nhị phân; Task B có câu trả lời dạng tập và câu không trả lời được. Hai task được báo cáo riêng; G4 không gộp với G1-G3.
Đánh đổi được diễn giải bằng đồ thị (`pypi_rag/evaluation/figures.py`): bảng nhiệt theo nhóm, risk-coverage, grounding theo coverage, chi phí theo chất lượng.

| Metric | Dùng để chứng minh | Task A | Task B |
|---|---|---|---|
| Malicious F1, macro F1 (strict: uncertain = sai) | chất lượng phát hiện | `strict/f1_malicious`, `strict/macro_f1`, FPR/FNR | - |
| Exact accuracy, set F1 | chất lượng câu trả lời | - | `G1-G3: exact_accuracy`, `set_f1` |
| Unsupported claim rate | chất lượng grounding | `evidence/unsupported_claim_rate` | `grounding_all/unsupported_claim_rate` |
| Unsupported / claim gốc, trên mọi mẫu | so E3 với E4 **cùng mẫu số** | `grounding_vs_e3`, `evidence/unsupported_over_original_claims` | `unsupported_over_original_claims` |
| Correct abstention rate | phản ứng khi thiếu bằng chứng | `case/abstention_accuracy` (C06) | `G4: correct_abstention_rate` |
| Answer coverage | vẫn đưa ra được đáp án | `coverage` | `answer_coverage` (G1-G3) |
| Risk-coverage, AURC | đánh đổi độ chính xác và từ chối | `aurc`, `rc_curve` | `answerable/aurc`, `rc_curve`, `policy_sweep` của E4 |
| Evidence recall | truy hồi có mang đủ bằng chứng không | `gold/evidence_coverage`, `case/required_evidence_recall_*` | `ev_recall`, `ev_recall_visible` |
| Latency, tokens, cost | chi phí | `tokens_*`, `latency_*`, khối `cost` | như Task A |

**So E3 với E4 cùng mẫu số.** E4 dùng lại câu trả lời của E3. Nếu chỉ tính trên các claim E4 còn giữ thì tỉ lệ claim không chứng cứ đẹp giả
(bỏ ca khó là xong). Vì vậy: tử số = claim không chứng cứ còn trong câu trả lời cuối; mẫu số = **claim gốc của E3** và **mọi mẫu / câu hỏi ban đầu**
(mẫu bị E4 chuyển thành từ chối tính là không có claim). Kèm theo luôn `claims_retained`, `supported_per_item` và `coverage` để thấy cái giá của việc lọc.
Ngưỡng E4 của Task B được tune trên org dev theo mục tiêu cân bằng `0.5 x exact accuracy (G1-G3) + 0.5 x từ chối đúng (G4)`, không theo một accuracy gộp.

## Kết quả (lần chạy 2026-11-10)

Nguồn: `results/cloud/pod_{A,B,C}/` (`results.json`, `report.md`), đồ thị ở `results/figures/`. Mọi khác biệt được gọi là "có ý nghĩa" đều có
CI 95% không chứa 0 (paired bootstrap theo cụm: nhóm package ở Task A, thực thể của câu hỏi ở Task B). Reader: Llama-3.1-8B-Instruct, Qwen2.5-Coder-7B-Instruct.

### Task A: strict malicious F1 (uncertain = sai), test

| Method | `pyguard_obf` Llama | `pyguard_obf` Qwen | `malllm` Llama | `malllm` Qwen |
|---|---|---|---|---|
| E0 LLM thuần | 0.754 | **0.953** | **0.822** | 0.866 |
| E1 dense | 0.727 | 0.925 | 0.674 | 0.934 |
| E2 hybrid | 0.731 | 0.914 | 0.709 | 0.945 |
| E2 BM25 | 0.729 | 0.919 | 0.706 | **0.946** |
| E2 rerank | 0.727 | 0.928 | 0.700 | 0.938 |
| E3 KG-RAG | 0.805 | 0.939 | 0.704 | 0.760 |
| E3a cùng cạnh, câu rời | **0.814** | 0.942 | 0.718 | 0.766 |
| E3b fact chọn bằng truy hồi | 0.722 | 0.928 | 0.699 | 0.865 |
| E4 E3 + xác minh (coverage) | 0.757 (0.79) | 0.887 (0.95) | 0.679 (0.97) | 0.743 (0.99) |

`pyguard_obf`: package đầy đủ, n = 310 (156 độc). `malllm`: chỉ setup.py, n = 500 lấy từ test của paper (134 độc).

- **Package đầy đủ, Llama:** E3 hơn E0 +0.052 [+0.030, +0.074] và hơn E2 +0.075 [+0.051, +0.100]; FPR 0.48 so với 0.75 của E2.
  E3a hơn E3 một chút (-0.008 [-0.018, -0.002]); E3 hơn E3b +0.083: lợi ích đến từ việc chọn fact bằng duyệt đồ thị, không từ trình bày đường.
- **Package đầy đủ, Qwen:** LLM thuần tốt nhất; E1, E2 kém E0 0.03-0.04 (có ý nghĩa); E3 so với E0 không khác có ý nghĩa.
- **Chỉ setup.py:** Llama: mọi truy hồi kém E0 0.11-0.15. Qwen: RAG văn bản hơn E0 +0.080 [+0.031, +0.168]; KG kém E0 -0.105 (đồ thị dựng từ một file ngắn dẫn sai).
- **E4:** F1 strict giảm 0.02-0.05 ở cả bốn cấu hình (có ý nghĩa), vì một phần package chuyển thành uncertain.

**Grounding, cùng mẫu số** (claim không chứng cứ còn lại / claim gốc của E3, trên mọi package):

| Cấu hình | E3 | E4 | Claim E4 giữ lại | Coverage E4 | F1 E3 -> E4 |
|---|---|---|---|---|---|
| `pyguard_obf` Llama | 0.366 | 0.000 | 0.62 | 0.79 | 0.805 -> 0.757 |
| `pyguard_obf` Qwen | 0.326 | 0.000 | 0.62 | 0.95 | 0.939 -> 0.887 |
| `malllm` Llama | 0.508 | 0.000 | 0.48 | 0.97 | 0.704 -> 0.679 |
| `malllm` Qwen | 0.416 | 0.000 | 0.55 | 0.99 | 0.760 -> 0.743 |

**Tái lập mal-LLM** (Llama, 300 ký tự đầu setup.py, prompt gốc): kết luận định tính của paper được giữ, RAG đơn giản không giúp.

| Cấu hình | F1 paper | F1 nhóm | Accuracy paper | Accuracy nhóm |
|---|---|---|---|---|
| Zero-shot | 0.459 | 0.559 | 0.708 | 0.830 |
| RAG trên luật YARA | 0.515 | 0.285 | 0.721 | 0.776 |
| RAG trên setup.py độc | 0.438 | 0.471 | 0.604 | 0.431 |

**Test case C01-C06** (6 mẫu, minh hoạ, không phải thống kê): C01, C02 đúng ở mọi method, cả hai reader. C04 (đọc file, gửi HTTP sau khi
người dùng đồng ý): Llama với câu hỏi chuẩn báo độc ở 6/7 method, với câu hỏi riêng của case thì 7/7 đúng. C05 (bí mật đi qua 3 file): với Qwen,
E0-E2 báo lành, các method có đồ thị báo độc; bằng chứng đồ thị cho thấy 5/6 dòng bắt buộc so với 2/6. C06 (dynamic dispatch): không method nào,
kể cả E4, trả lời uncertain. `case/answer_path_accuracy` = 0 ở mọi method (không câu trả lời nào nêu đủ `load_secret -> prepare_payload -> submit`
kèm claim quan hệ được hỗ trợ).

### Task B: theo nhóm câu hỏi, test (400 câu; G1-G3 exact accuracy, G4 từ chối đúng)

| Method | Qwen G1 | Qwen G2 | Qwen G3 | Qwen G4 | Qwen coverage | Llama G1 | Llama G2 | Llama G3 | Llama G4 | Llama coverage |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | 0.75 | 0.00 | 0.00 | **0.85** | 0.34 | 0.62 | 0.00 | 0.05 | 0.90 | 0.30 |
| E1 | **0.80** | 0.36 | 0.22 | 0.59 | 0.73 | **0.63** | 0.03 | 0.02 | 0.86 | 0.34 |
| E2 | 0.75 | 0.52 | 0.23 | 0.47 | 0.79 | 0.51 | 0.26 | 0.07 | 0.76 | 0.45 |
| E2 BM25 | 0.66 | 0.45 | 0.21 | 0.50 | 0.73 | 0.57 | 0.26 | 0.04 | 0.76 | 0.50 |
| E3 | 0.73 | **0.59** | **0.54** | 0.56 | **0.91** | 0.26 | **0.33** | **0.20** | **0.93** | 0.42 |
| E3a | 0.72 | 0.58 | **0.54** | 0.49 | **0.91** | 0.38 | **0.33** | 0.15 | 0.82 | 0.41 |
| E3b | 0.79 | 0.36 | 0.15 | 0.56 | 0.68 | 0.58 | 0.06 | 0.02 | 0.87 | 0.34 |
| E4 | 0.73 | 0.56 | 0.52 | 0.65 | 0.87 | 0.26 | **0.33** | **0.20** | **0.93** | 0.40 |

Coverage = tỉ lệ câu trả lời được (G1-G3) mà method đưa ra đáp án. E4 tune trên org dev: Qwen tau 1.0, Llama tau 0.0 (với Llama, E4 trùng E3).

- **G1 mã:** LLM thuần đủ; với Qwen, E1 và E3 so với E0 không khác có ý nghĩa. Với Llama, E3 kém E0 -0.36 (có ý nghĩa): KG báo
  "không có thực thể" cho câu hỏi về mã và reader từ chối dù đáp án nằm trong mã.
- **G2 chính sách:** không có tài liệu thì bằng 0. Qwen: E1 vs E0 +0.36, E2 vs E1 +0.16 (có ý nghĩa); E3 vs E2 +0.07 (không có ý nghĩa).
- **G3 phụ thuộc:** Qwen: E3 vs E2 +0.31, E3 vs E3b +0.39 (có ý nghĩa); E3 vs E3a 0.00. Set F1: E3 0.60, E2 0.25.
- **G4 thiếu bằng chứng:** thêm tài liệu làm Qwen trả lời cả câu lẽ ra phải từ chối (E1 vs E0 -0.26, có ý nghĩa); E4 vs E3 +0.09 đổi lấy
  -0.017 exact accuracy ở G1-G3.
- **Grounding cùng mẫu số (Qwen):** claim không chứng cứ / claim gốc: E0 0.788, E2 0.053, E3 0.030, E4 0.005.
- **Chi phí đo được (Qwen, USD / 1000 câu, L40S):** E0 0.018, E2 0.063, E3 0.033, E4 +0. KG dùng khoảng một nửa token của RAG văn bản.

**Theo bẫy (Qwen, accuracy):**

| Bẫy | n | E0 | E1 | E2 | E3 | E4 |
|---|---|---|---|---|---|---|
| nhiều bước (`multi_hop`) | 45 | 0.00 | 0.07 | 0.04 | **0.80** | **0.80** |
| extra chưa bật | 7 | 0.00 | 0.29 | 0.29 | **0.86** | **0.86** |
| policy cũ đã bị thay | 47 | 0.00 | 0.21 | 0.53 | **0.57** | 0.53 |
| version ngoài khoảng advisory | 18 | 0.00 | **0.78** | 0.67 | 0.00 | 0.00 |
| hai policy hiện hành mâu thuẫn (G4) | 17 | **1.00** | 0.82 | 0.35 | 0.00 | 0.00 |
| thực thể không tồn tại (G4) | 18 | **1.00** | 0.56 | 0.56 | 0.33 | 0.83 |

### Khung quyết định rút ra từ số đo

| Khi | Dùng | Vì số đo | Cẩn thận |
|---|---|---|---|
| Mã đã có trong input, reader mạnh về code | LLM thuần | Task A Qwen 0.953; G1 0.75, truy hồi không hơn | Reader yếu (Llama) cần fact đồ thị: +0.052 |
| Cần tri thức nội bộ (policy, ADR) | RAG hybrid hoặc KG | G2: 0.00 -> 0.52 (E2) / 0.59 (E3) | So version với khoảng advisory: KG 0/18, văn bản 0.78 |
| Câu hỏi nhiều bước qua phụ thuộc | KG-RAG có duyệt | G3: 0.54 so với 0.23 | Lợi ích từ chọn fact, không từ trình bày đường (E3a = E3) |
| Input ngắn, ít cấu trúc (setup.py) | LLM thuần hoặc RAG văn bản | Qwen: RAG +0.080, KG -0.105 | Đồ thị dựng từ một file dễ dẫn sai |
| Sai đắt, có thể thiếu bằng chứng | thêm xác minh (E4) | claim không chứng cứ 0.37 -> 0 (cùng mẫu số) | Coverage và F1 giảm; C06 vẫn không ai từ chối |

So với giả thuyết viết trước: G1, G2, G3 đúng hướng; G4 chỉ đúng một phần (thêm bằng chứng làm model từ chối ít đi, E4 chỉ bù một phần).

## Ánh xạ plan -> code

| Plan (mục 4) | File ở đây |
|---|---|
| `data/loaders, normalize, splitter, leakage_check` | [pypi_rag/data/](pypi_rag/data/) `loaders.py`, `safe_io.py`, `splitter.py`, `leakage_check.py`, `synthetic.py` |
| `analysis/ast_parser, api_extractor, code_slicer, provenance` | [pypi_rag/analysis/](pypi_rag/analysis/) `ast_parser.py` (call site, alias, data flow), `taxonomy.py`, `code_slicer.py`, `provenance.py` |
| `knowledge/taxonomy_adapter, pattern_adapter, documents, build_index` | [pypi_rag/knowledge/](pypi_rag/knowledge/) `pyguard_adapter.py`, `builtin.py`, `external.py` (YARA, advisories), `documents.py`, `build_index.py` |
| `retrieval/vector, bm25, hybrid, reranker` | [pypi_rag/retrieval/core.py](pypi_rag/retrieval/core.py) |
| `graph/schema, builder, entity_linker, traversal, graph_retriever` | [pypi_rag/graph/](pypi_rag/graph/) `schema.py`, `builder.py` (gồm entity linking), `traversal.py`, `retriever.py` |
| `generation/llm_client, prompt_builder, output_schema` | [pypi_rag/generation/](pypi_rag/generation/) `llm_client.py`, `prompts.py`, `output_schema.py` |
| `verification/citation_checker, claim_checker, contradiction, verdict_policy` | [pypi_rag/verification/verifier.py](pypi_rag/verification/verifier.py) |
| `pipelines/llm_only ... verified_kg_rag` | [pypi_rag/pipelines/runner.py](pypi_rag/pipelines/runner.py) |
| `evaluation/classification, retrieval, hallucination, statistical_tests, report` | [pypi_rag/evaluation/](pypi_rag/evaluation/) |
| `scripts/*.py` | [scripts/](scripts/) (gọi [pypi_rag/cli.py](pypi_rag/cli.py)) |
| `configs/*.yaml` | [configs/](configs/) (khoá YAML trùng tên flag CLI) |

## Cấu trúc folder

```
pypi_malware_rag/
  pypi_rag/
    schema.py            Sample, Evidence, PredictionResult (plan mục 5.2, mục 6.1)
    data/                đọc archive an toàn trong bộ nhớ, loader, chia tách theo nhóm, kiểm tra rò rỉ, dữ liệu tổng hợp
    analysis/            AST: call site + tên API chuẩn, hành vi, luồng dữ liệu, ngữ cảnh install/import; khối package dùng chung
    knowledge/           luật/pattern tự viết, ví dụ từ train, adapter PyGuard / YARA / advisories, dựng KB
    retrieval/           embedding, BM25, RRF, rerank
    graph/               đồ thị hành vi (NetworkX), duyệt 1-3 bước, khớp pattern, tuyến tính hoá đường; fact cho E3a
    generation/          prompt chung, client dummy / HF / OpenAI-compatible + cache, parse JSON
    verification/        nhãn claim, chính sách abstain, tune trên val
    pipelines/runner.py  E0..E4 + ablation + tái lập mal-LLM
    evaluation/          F1 strict/conservative/answered, risk-coverage, claim metric, recall@K, path recall, metric case,
                         bootstrap (theo nhóm), McNemar
    supplychain/         Task B: org.py, corpus.py, kg.py, questions.py, qa.py, evaluate.py, pipeline.py
    cli.py               prepare, pyguard-bundle, build-kb, build-graph, run, tune-policy, rescore, report, sc-generate, sc-run, sc-tune
  configs/  scripts/  tests/ (39 test CPU)
  notebooks/
    00_data_knowledge_colab.ipynb   clone + pin upstream, chuẩn hoá dữ liệu, split, KB, chẩn đoán đồ thị, đóng băng (CPU)
    01_experiments_colab.ipynb      tái lập mal-LLM, validation + tune E4, D1/D2/D3, D4 cases, chi phí, checklist (GPU)
    02_evidence_eval_colab.ipynb    nhãn claim, ablation xác minh, quét chính sách, lỗi theo nhóm, ví dụ, gán nhãn thủ công (CPU)
    03_demo_seminar.ipynb           bảng + dashboard cho phần trình bày, chỉ đọc kết quả (CPU)
    04_supplychain_qa_colab.ipynb   Task B: sinh org dev/test, 8 method x 4 nhóm, tune E4 trên dev, so sánh cặp, lỗi theo bẫy (GPU)
```

## Cách chạy

**Trên Colab.** Upload nguyên folder `pypi_malware_rag/` lên Drive tại `MyDrive/ThS/Research/colab/` (sửa `DRIVE_BASE` ở cell đầu nếu khác,
đường dẫn không được có dấu cách). Kết quả nằm ở `MyDrive/ThS/Research/colab/pypi_rag_work/`. Chạy lần lượt:

1. `00_data_knowledge_colab.ipynb` (CPU là đủ, ~30 phút lần đầu). Tạo `pypi_rag_work/{data,kb,graphs}/` và `FROZEN.json`.
   Tuỳ chọn trong cell: `USE_PYGUARD_OBF` (tải ~660 MB package đầy đủ), `YARA_FULL`, `FETCH_ADVISORIES` (thêm secret `GITHUB_TOKEN`).
2. `01_experiments_colab.ipynb` trên GPU. Mặc định 2 reader (`NousResearch/Meta-Llama-3.1-8B-Instruct` như paper, `Qwen/Qwen2.5-Coder-7B-Instruct`),
   500 package test và 200 package val mỗi bộ (`MAX_TEST`, `MAX_VAL`), backend `vllm` (tự cài; không cài được thì dùng `hf`).
   Thời gian chạy: **không dùng ước lượng**; xem mục "Chi phí: đo pilot trước" bên dưới.
   Mọi câu trả lời được cache, phiên bị ngắt thì chạy lại cell là tiếp tục.
3. `04_supplychain_qa_colab.ipynb` trên GPU: **Task B, bắt buộc cho kết luận seminar** (sinh org dev/test, 8 method x 4 nhóm, tune E4 trên dev).
4. `02_evidence_eval_colab.ipynb` (CPU). Xuất `review/claims_review.csv` để gán nhãn thủ công.
5. `03_demo_seminar.ipynb` (CPU). Lần đầu tạo `results/snapshot/`; sau đó tải folder về máy là chạy được không cần Drive.

**Trên máy (CPU).**

```
cd pypi_malware_rag
python -m pytest -q tests/
python -m pypi_rag.cli prepare --spec synthetic:150:0 --out /tmp/prd/data
python -m pypi_rag.cli build-kb --data /tmp/prd/data --out /tmp/prd/kb.jsonl
python -m pypi_rag.cli run --data /tmp/prd/data --kb /tmp/prd/kb.jsonl --methods e0 e1 e2 e3 e3a e4 \
    --llm dummy --embedder hash --out /tmp/prd/run
python -m pypi_rag.cli rescore --results /tmp/prd/run --derive-e4      # dựng E4 variants từ E3, không gọi LLM

# test case C01-C06
python -m pypi_rag.cli prepare --spec cases: --strategy keep --out /tmp/prd/cases
python -m pypi_rag.cli build-kb --data /tmp/prd/cases --out /tmp/prd/cases_kb.jsonl --sources builtin case_rules
python -m pypi_rag.cli run --data /tmp/prd/cases --kb /tmp/prd/cases_kb.jsonl --methods e0 e2 e3 e3a_exact e4 \
    --use-sample-question --strata-key case --llm dummy --out /tmp/prd/cases_run

# Task B
python -m pypi_rag.cli sc-generate --seed 1 --out /tmp/sc/dev --n-per-group 40
python -m pypi_rag.cli sc-generate --seed 2 --out /tmp/sc/test --n-per-group 100
python -m pypi_rag.cli sc-run --data /tmp/sc/dev --out /tmp/sc/run_dev --llm dummy
python -m pypi_rag.cli sc-tune --results /tmp/sc/run_dev --data /tmp/sc/dev --out /tmp/sc/policy.json
python -m pypi_rag.cli sc-run --data /tmp/sc/test --out /tmp/sc/run_test --llm dummy --policy /tmp/sc/policy.json
```

Kiểm tra notebook không cần Colab: đặt `PYPI_RAG_QUICK=1` và `PYPI_RAG_WORK=<thư mục tạm>` rồi chạy notebook từ `notebooks/`
(chỉ dữ liệu tổng hợp, reader `dummy`, không tải gì). Reader `dummy` là luật từ khoá để kiểm tra code, **số của nó không có ý nghĩa nghiên cứu**.
Các lệnh theo plan mục 14 cũng chạy được: `python scripts/run_experiment.py --method e0 e3 --split test --data ... --kb ... --out ...`.

## Chạy trên cloud GPU (RunPod, Lambda, ...)

Model 8B không cần ghép nhiều GPU (tensor parallel); mỗi GPU chạy một reader. Pod 2 GPU thường hết hàng ("no instances available"),
nên cách chạy chuẩn là **nhiều pod 1 GPU** (A100 80GB, L40S 48GB hoặc H100 đều đủ).

**Cấu hình chuẩn (mọi script giả định như vậy):**

| Mục | Giá trị | Ghi chú |
|---|---|---|
| Template | PyTorch, CUDA 12.8 trở lên | `runpod-torch-v280` |
| Container disk | 60 GB | mất khi stop: `cloud_setup.sh` cài lại vllm nếu thiếu |
| Lưu trữ | **100 GB gắn tại `/workspace`** | pod volume (mặc định); trọng số 2 model ~32 GB, dữ liệu + KB + kết quả ~10 GB |
| Network volume | tuỳ chọn, cũng gắn tại `/workspace` | chỉ khi pod nằm cùng data center với volume; dùng lại dữ liệu giữa các pod |

Mọi script dùng `ROOT=/workspace` (có thì dùng, không thì `$HOME`), nên hai kiểu lưu trữ chạy như nhau.
**Stop** giữ `/workspace` (tính phí lưu trữ khi dừng); **Terminate** xoá pod volume (network volume vẫn còn).
Script chỉ gọi `runpodctl stop`, không bao giờ terminate: tải kết quả tổng hợp về trước, rồi mới terminate bằng tay.

1. Tạo pod theo bảng trên; đưa folder `pypi_malware_rag/` vào `/workspace/code/` (`scp`, `runpodctl send`, hoặc `git clone`).
2. Một máy, chạy hết (Task A rồi Task B):
   ```
   cd /workspace/code/pypi_malware_rag
   bash scripts/cloud_setup.sh                       # cài vllm + deps, tải trọng số, chạy test
   tmux new -s run 'bash scripts/run_cloud.sh'        # notebook 00 -> 01 -> 04 (Task B) -> 02 -> 03
   ```
   Thoát tmux bằng `Ctrl-b d`, gắn lại bằng `tmux attach -t run`. Log ở `/workspace/pypi_rag_work/executed/*.log`.
3. **Chia việc cho nhiều pod** (mỗi pod tự cài, chạy, rồi tự `stop` sau `PYPI_RAG_GRACE` giây để kịp tải kết quả).
   Task B là pipeline bắt buộc: khi chia việc, **đúng một pod** chạy `PYPI_RAG_TASK=b`; `run_cloud.sh` cảnh báo nếu thiếu.
   ```
   PYPI_RAG_DATASETS=pyguard_obf PYPI_RAG_EXTRAS=0 bash scripts/launch_pod.sh   # pod 1: Task A trên pyguard_obf
   PYPI_RAG_DATASETS=malllm      PYPI_RAG_EXTRAS=1 bash scripts/launch_pod.sh   # pod 2: malllm + synthetic + cases + lặp lại
   PYPI_RAG_TASK=b bash scripts/launch_pod.sh                                    # pod 3: Task B (notebook 04)
   ```
   Biến `RUNPOD_*` (pod id, API key) chỉ có trong môi trường của tiến trình 1, không có trong phiên SSH: `launch_pod.sh` phải chạy với các biến
   này (xem `scripts/` hoặc đọc từ `/proc/1/environ`) thì mới tự dừng được.
4. Chỉ tải về **kết quả tổng hợp** (`results.json`, `report.md`, `policy.json`, notebook đã chạy). `data/` và `predictions_*` của Task A chứa
   mã độc dạng text: để lại trên pod; mọi tính lại metric của Task A chạy trên pod (`cli rescore`). Task B là dữ liệu tổng hợp, tải về được.

**Chi phí: đo pilot trước, không ước lượng chay.** Chạy một pilot nhỏ với reader thật (`--max-items 50` cho Task A, `--max-items 15` mỗi nhóm cho
Task B), rồi ngoại suy bằng thời gian GPU đo được:
```
python scripts/estimate_cost.py --pilot <thư mục kết quả pilot> --items 500 --methods 8 --readers 2 --runs 2 --price 1.79
```
Mỗi `results.json` có khối `cost`: GPU giây đo được (tổng phần của từng prompt trong batch sinh, trên các prompt khác nhau), chia cho từng method
theo token; E4 dùng lại lần sinh của E3 nên không tốn thêm GPU. `--gpu-price-hour` của `run`, `rescore`, `sc-run`, `sc-rescore` đổi ra USD.
Phân tích tĩnh chạy song song theo số CPU **của container** (đọc cgroup quota, không theo số nhân của máy chủ), cache theo hash mã + luật,
mỗi package có giới hạn 180 giây (quá thì chỉ quét regex và ghi là coverage chưa đủ): 1448 package `pyguard_obf` train mất 14 giây trên 12 worker.

Notebook 00 chỉ dùng CPU, nên GPU ngồi chờ trong lúc đó. Muốn tiết kiệm thì chạy notebook 00 trên Colab hoặc máy CPU
rẻ rồi copy `pypi_rag_work/` sang pod. CLI dùng được trực tiếp: `--llm vllm:<model>` (engine trong tiến trình) hoặc
`--llm openai:<model>` + `OPENAI_BASE_URL` trỏ tới một `vllm serve` (gửi song song `--concurrency` request).

## Đồ thị hành vi (E3)

Node: `Package, Module, Function, Operation` (một lời gọi), `API, Behavior, DataValue, FileResource, NetworkEndpoint, Pattern`.
Cạnh: `CONTAINS, DEFINES, CALLS, INSTANCE_OF, HAS_BEHAVIOR, PRODUCES, FLOWS_TO, READS, WRITES, SENDS_TO, FETCHES_FROM, PRECEDES, MATCHES_PATTERN`.
Mỗi cạnh có `source_file, line_start, line_end, extractor, confidence, evidence_status`.
`FLOWS_TO` chỉ được tạo khi AST cho thấy giá trị đi từ lời gọi này sang lời gọi kia: lồng lời gọi (`ast_nested`, 1.0), def-use trong hàm
(`intraproc_def_use`, 0.8), qua giá trị trả về / tham số của hàm trong package (`interproc_summary`, 0.6). Hai lời gọi cùng file mà không có
chuỗi def-use chỉ có `PRECEDES` (`inferred`). File không parse được (Python 2, obfuscation) chỉ có thao tác tìm bằng regex, không có cạnh luồng,
và prompt ghi rõ "thiếu đường không có nghĩa là không có luồng".

## An toàn

Package nghi độc chỉ được đọc tĩnh: archive đọc **trong bộ nhớ** (không giải nén ra đĩa), từ chối path traversal, symlink, file quá lớn,
tỉ lệ nén bất thường; mã chỉ được `ast.parse`, không bao giờ `pip install`, import hay chạy `setup.py`. Archive gốc nằm trên đĩa máy ảo Colab
(`/content/external`) hoặc `/workspace` của pod, không lên Drive. `pypi_rag_work/data` và predictions của Task A chứa mã độc **dạng text**:
không đồng bộ về máy cá nhân. File pickle của PyGuard không bao giờ được nạp. Fixture tổng hợp dùng TLD `.invalid` và dải IP tài liệu
`203.0.113.0/24`, payload vô hại.

**Cô lập và mạng.**
- Dữ liệu mã độc thật chỉ xử lý trong máy ảo / pod dùng riêng cho dự án, không chạy trên máy cá nhân.
- Mạng chỉ cần ở bước cài đặt và notebook 00 (tải repo upstream, trọng số). Từ notebook 01 trở đi `run_cloud.sh` và `run_taskb.sh` đặt
  `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`: model, embedding, reranker chạy từ bản đã tải, không gọi ra ngoài khi prompt chứa mã nguồn.
- Không gửi mã nguồn chưa kiểm duyệt cho dịch vụ ngoài: backend `openai:` từ chối mọi `OPENAI_BASE_URL` không phải localhost
  (chỉ dùng cho `vllm serve` cục bộ) trừ khi đặt `PYPI_RAG_ALLOW_REMOTE_LLM=1` cho dữ liệu được phép chia sẻ. Công cụ điều khiển pod (MCP)
  chỉ nhận lệnh hạ tầng, không nhận dữ liệu.
- Giới hạn: container RunPod không cho tự đặt firewall chặn egress; mức cô lập hiện tại là "không thực thi mã + không có lệnh mạng ở bước
  suy luận". Nếu cần chặt hơn, chạy notebook 01-04 trong máy ảo có chính sách egress chặn mọi kết nối ra ngoài (sau khi đã tải trọng số).

## Upstream

| Repo | Commit pin | License | Dùng gì |
|---|---|---|---|
| mal-LLM | `0491c6c7` (2025-03-16) | MIT | dữ liệu `*_packages_final.json`, prompt baseline, YARA mẫu |
| PyGuard | `f684d0ab` (2025-12-19) | **không có file license** | taxonomy, lexicon API -> action, pattern; package Obfuscation. Chỉ đọc lúc chạy, không copy vào repo này, phải trích dẫn |

Notebook 00 kiểm tra lại SHA, license và sự tồn tại của mọi file adapter dùng, ghi `upstream_audit.json`.

## Giới hạn đã biết

- `malllm` chỉ có `setup.py`: ít cơ hội cho hành vi nhiều file, nên RQ3 phải đọc chủ yếu trên `pyguard_obf`.
- Phân tích luồng là intraprocedural + tóm tắt một mức giữa các hàm; gọi động (`getattr` với chuỗi tính toán), class phức tạp, Python 2 làm mất cạnh.
- Nhãn claim là kiểm tra **nhất quán với phân tích tĩnh**, không phải sự thật; phải kiểm bằng tập gán nhãn thủ công (notebook 02 cell 06) trước khi trích.
- Tái lập mal-LLM không giống hệt: Llama chạy local bf16 thay vì HF Inference API, embedding `bge-small` thay vì OpenAI ada-002,
  luật YARA nguyên văn thay vì được gpt-4o-mini diễn giải, YARA mẫu trừ khi bật `YARA_FULL`. Số tham chiếu trong notebook 01 lấy từ output notebook
  trong repo mal-LLM; đối chiếu lại với PDF paper trước khi trích.
- Lexicon PyGuard do LLM của PyGuard sinh, khớp pattern PyGuard là heuristic (cạnh `inferred`). Mặc định lexicon không thay bộ phân tích tĩnh
  dùng chung (bật bằng `--pyguard-taxonomy` như một ablation).
- Một seed, greedy; chưa có fine-tune (plan không yêu cầu, paper gốc có).
- `pyguard_obf`: gate "không cặp near-duplicate nào vắt qua hai split" FAIL (các biến thể obfuscation gần giống nhau; cụm quá lớn bị bỏ
  khỏi ghép nhóm để boilerplate không dính cả dataset). So sánh cặp vì vậy bootstrap theo `meta.group_id`.
- C05: bằng chứng đồ thị cho thấy 5/6 dòng bắt buộc; dòng biến đổi bên trong `prepare_payload` (`processor.py:2`) không phải span của đường.
- Task B là tổ chức **tổng hợp** (đáp án biết trước, kiểm soát được bẫy), không phải dữ liệu thật của một công ty; tên package public là tên thật
  nhưng version và phụ thuộc là giả, nên tri thức sẵn có của model về các package đó có thể gây nhiễu (đó cũng là điều cần đo ở E0).
- Lần chạy 2026-11-10: Task B trên pod được tune E4 theo mục tiêu gộp cũ; E4 trong bảng kết quả được tune lại trên org dev theo mục tiêu cân bằng
  và dựng lại offline từ câu trả lời của E3 (`sc-tune` + `sc-rescore --policy`), không gọi lại LLM.
- E4 chưa từ chối được C06: chính sách tune trên validation không bật quy tắc từ chối khi có dynamic dispatch (`abstain_on_dispatch`).
- KG của Task B không nói rõ "version này nằm ngoài khoảng advisory" và đưa cả hai policy mâu thuẫn mà không gắn cờ; hai bẫy này KG làm kém
  RAG văn bản (bảng theo bẫy ở mục Kết quả).
- Mỗi cấu hình chỉ một lần chạy, một seed; chưa có nhãn người cho claim (file `claims_review_*.csv` để gán nhãn nằm trên pod).
