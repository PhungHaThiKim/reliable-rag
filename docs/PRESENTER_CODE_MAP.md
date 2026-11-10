# Seminar 2, Project 11: người trình bày, code phụ trách và kết quả

- Deck: `Seminar2_Project11_RAG_slides.pptx`, gồm 29 slide chính (slide 2 là agenda) và 25 slide backup (30-54). Deck và `Project11_Results_Curated_Critical_Analysis.md` nằm ngoài repo.
- Mọi số liệu lấy từ lần chạy ngày 2026-11-10.
- Đường dẫn:
  - code: tính từ thư mục `pypi_rag/`, trừ khi ghi rõ `configs/`, `notebooks/`, `scripts/` hoặc `tests/`;
  - kết quả: tính từ `results/cloud/`, có trên GitHub (mô tả ở `results/README.md`). Notebook đã chạy nằm ở `pod_*/executed/`.

## Lưu ý về code và kết quả

1. **Tên method trong code khác tên trên slide.** `predictions_e3a.jsonl` là **E3b**, không phải E3a.

   | Slide và báo cáo | Tên trong code |
   |---|---|
   | E0 LLM thuần | `e0` |
   | E1 dense | `e1` |
   | E2 hybrid (RRF) | `e2` |
   | E2 chỉ BM25 | `e2_bm25` |
   | E3 KG-RAG | `e3` |
   | E3a: đúng các cạnh E3 đã đưa, viết thành câu rời | `e3a_exact` |
   | E3b: fact KG chọn bằng truy hồi, không duyệt đồ thị | `e3a` |
   | E4 = E3 + kiểm claim | `e4` |

2. **Mỗi task có bộ method riêng, cùng ý nghĩa E0-E4.**
   - Task A dùng `pipelines/runner.py` (biến `METHODS`).
   - Task B dùng `supplychain/qa.py` (biến `METHODS`).
3. **Trường `paper_*` trong `pod_B/results/r01_repro_malllm_Meta-Llama-3.1-8B-Instruct/reproduction.json` là số của notebook trong repo mal-LLM**, không phải số công bố trong paper. Slide 4 tách riêng ba nguồn: paper, notebook, nhóm.
4. **Kết quả Task A chỉ lưu số liệu tổng hợp** (`results.json`, `report.md`, `stats.json`). Dữ liệu malware và dự đoán từng package chỉ nằm trên pod.
   - Không cài, không import, không chạy package nghi độc.

## 1. Phân công theo method

Mỗi người phụ trách trọn một method: code lấy bằng chứng, các biến thể và đối chứng, kết quả trên cả hai task và hai reader, demo, và các trường hợp method đó sai. E4 là bước xác minh chạy trên câu trả lời của E3; cả nhóm cùng làm, Phụng làm chính.

| Method | Người phụ trách | Biến thể và đối chứng | Câu hỏi nghiên cứu |
|---|---|---|---|
| E0 LLM thuần | Quỳnh Anh | Tái lập mal-LLM: `e0_malllm`, `e1_malllm_yara`, `e1_malllm_code` | RQ1 (cùng Thành) |
| E1 dense | Văn Thành | | RQ1, RQ2 (cùng Huy Anh) |
| E2 hybrid | Huy Anh | `e2_bm25`, `e2_rerank` | RQ2, RQ3 (phía E2) |
| E3 KG-RAG | Kim Phụng | E3a (`e3a_exact`), E3b (`e3a`) | RQ3 |
| E4 xác minh | Cả nhóm, chính: Kim Phụng | `e4_filter`, `e4_cite` | RQ4 |

RQ5 (method nào hợp với nhóm câu hỏi nào): cả nhóm.

### Code phụ trách

| Người | Method | Code của method | Phần chung kèm theo |
|---|---|---|---|
| Quỳnh Anh | E0 | `pipelines/runner.py` (e0, e*_malllm); `supplychain/qa.py` (e0); `analysis/code_slicer.py` (package_representation, malllm_representation); `generation/prompts.py` (malllm_messages); `knowledge/external.py` (YARA, advisories); `knowledge/build_index.py` (malllm_code_docs) | Lớp LLM: `generation/llm_client.py`, `generation/output_schema.py`. Dữ liệu Task A: `data/loaders.py`, `data/splitter.py`, `data/synthetic.py`. Bộ case C01-C06: `data/cases.py`, `knowledge/case_rules.py`. Notebook 00 |
| Văn Thành | E1 | `retrieval/core.py` (Embedder, VectorIndex, Retriever ở chế độ vector, build_queries); knowledge base dùng chung cho E1 và E2: `knowledge/build_index.py`, `documents.py`, `builtin.py`, `pyguard_adapter.py`; method e1 trong `pipelines/runner.py` và `supplychain/qa.py` (QAContext.retriever); `configs/retrieval.yaml` | Kiểm tra rò rỉ: `data/leakage_check.py` |
| Huy Anh | E2 | `retrieval/core.py` (code_tokens, BM25, rrf_fuse, rrf_scores, CrossEncoderReranker, Retriever ở chế độ rrf và bm25); method e2, e2_bm25, e2_rerank trong `pipelines/runner.py` và `supplychain/qa.py` | Đánh giá và thống kê: `evaluation/classification.py`, `summary.py`, `statistical_tests.py`, `figures.py`, `report.py`, `evaluation/evidence.py`; `supplychain/evaluate.py` (score_one, summarize, compare, evidence_scores, _grounding, cost_estimate); `scripts/estimate_cost.py`. Notebook 02 |
| Kim Phụng | E3, E3a, E3b; E4 (chính) | E3: `analysis/ast_parser.py`, `taxonomy.py`, `provenance.py`, `parallel.py`; `graph/*`; `supplychain/kg.py`; nhánh KG của `pipelines/runner.py` và `supplychain/qa.py`; `configs/graph.yaml`. E4: `verification/verifier.py` (check_claims, apply_policy, tune_policy); `supplychain/qa.py` (verify, apply_qa_policy, reapply_policy); `supplychain/evaluate.py` (tune_policy, policy_sweep); method e4, e4_filter, e4_cite | Khung chạy: `schema.py`, `cli.py`, `pipelines/runner.py`, `supplychain/pipeline.py`, `generation/prompts.py` (build_messages, fit_items), `data/safe_io.py`. Dữ liệu Task B: `supplychain/org.py`, `corpus.py`, `questions.py`. Hạ tầng: `scripts/*.sh`. Notebook 01 và 04 |
| Cả nhóm | E4 | | Thực nghiệm, thảo luận, kết luận, report, slides |

Test đi theo code:
- Kim Phụng: `test_analysis_graph.py`, `test_supplychain.py`, `test_perf.py`; trong `test_safe_io_and_data.py` là các test đọc archive an toàn; trong `test_pipeline_metrics.py` là `test_claim_labels_and_policy`, `test_cli_end_to_end_with_dummy`, `test_predict_all_orders_prompts_by_package`.
- Quỳnh Anh: `test_cases.py`; trong `test_safe_io_and_data.py` là các test loader và chia tập; trong `test_pipeline_metrics.py` là `test_output_parsing`, `test_cached_llm_dedupes_and_keeps_order_and_concurrent_client`.
- Văn Thành và Huy Anh: `test_rrf_and_bm25_and_hybrid`.
- Văn Thành: `test_kb_leakage_flags_eval_derived_docs_and_advisories`.
- Huy Anh: `test_classification_policies`.

### Kết quả chính theo method

| Method | Task A (F1 malicious) | Task B (Qwen) | Chỗ method sai |
|---|---|---|---|
| E0 (Quỳnh Anh) | PyGuard: Qwen 0.953, Llama 0.754. mal-LLM: Qwen 0.866, Llama 0.822. Tái lập: 0.559 / 0.285 / 0.471 (notebook repo 0.459 / 0.515 / 0.438) | G1 0.75; G2 0.00; G3 0.00; G4 0.85 | Không có tri thức nội bộ: từ chối 99% câu policy, 97% câu dependency. Llama gắn nhãn độc cho 66% package lành của PyGuard |
| E1 (Thành) | PyGuard: Qwen 0.925 (kém E0 0.028), Llama 0.727. mal-LLM: Qwen 0.934, Llama 0.674 | G1 0.80; G2 0.36 (hơn E0 0.36, có ý nghĩa); G3 0.22; G4 0.59. Recall bằng chứng 0.602 | Dense bỏ sót id chính xác như `POL-SEC-001`, version: câu `svc-status` không lấy được policy |
| E2 (Huy Anh) | PyGuard: Qwen 0.914 (kém E0 0.039), Llama 0.731. mal-LLM: Qwen 0.945 (hơn E0 0.080), Llama 0.709 | G1 0.75; G2 0.52 (hơn E1 0.16, có ý nghĩa); G3 0.23; G4 0.47. Recall bằng chứng 0.732 (chỉ BM25: 0.782). Version ngoài khoảng 0.67, trong khi E3 0/18 | Multi-hop 0.04; bẫy policy cũ 0.53; G4 thấp nhất trong các method |
| E3, E3a, E3b (Phụng) | PyGuard Llama: E3 0.805 (hơn E0 0.052), E3a 0.814, E3b 0.722. mal-LLM Qwen 0.760 (kém E0 0.105) | G2 0.59; G3 0.54 (E3a 0.54, E3b 0.15); multi-hop 0.80. Recall bằng chứng 0.989 nhưng chỉ đúng 0.62 | Version ngoài khoảng 0/18; policy mâu thuẫn 0/17. Llama từ chối 63% câu G1 (0.62 xuống 0.26) |
| E4 (cả nhóm, chính: Phụng) | Claim unsupported / claim gốc: E3 0.33-0.51, E4 0.000; F1 giảm 0.02-0.05 | Claim unsupported 0.030 xuống 0.005; correct abstention tăng 0.09, exact accuracy giảm 0.017; G4 0.65. Ngưỡng tau: Qwen 1.0, Llama 0 | Kiểm tra sự có mặt của id, không kiểm tra suy ra được hay không: câu sniffio vẫn SUPPORTED. Policy mâu thuẫn 0/17; C06 0/28 |

### Demo (theo thứ tự chạy)

Task A mở trong `results/cloud/pod_B/results/r07_cases_*` (trường `cases`). Task B tìm theo `qid` trong `results/cloud/pod_C/supplychain/runs/test_Qwen2.5-Coder-7B-Instruct/predictions_<method>.jsonl`.

| Thứ tự | Người | Demo | Điều demo cho thấy | Kết quả đã có |
|---|---|---|---|---|
| 1 | Quỳnh Anh (E0) | C01 `safe-math-utils`, C02 `quick-setup-tools` | Code đã nằm sẵn trong input thì LLM thuần là đủ | Mỗi case đúng 28/28 |
| | | Một câu G1 về luồng token ra mạng | Thêm tài liệu có thể làm sai đi | Qwen E0 đúng; E1, E2, BM25 sai. G1: E0 0.75 |
| | | E0 trên câu policy và dependency | LLM thuần không có tri thức nội bộ | G2, G3 = 0.00; từ chối 99% và 97% số câu |
| | | Tái lập mal-LLM | Kết luận giữ, số không khớp | F1 0.559 / 0.285 / 0.471, so với notebook 0.459 / 0.515 / 0.438 |
| 2 | Văn Thành (E1) | `s2-G2-0176` (thời hạn vá của `svc-status`) qua E1 | Dense bỏ sót policy | 8 tài liệu lấy về đều là manifest; E1 từ chối. G2: E0 0.00, E1 0.36 |
| | | C04 `report-uploader` | E1 sửa được false positive | Llama, cách hỏi chung: E0 báo độc, E1 trả lời lành (đúng) |
| | | C03 `token-sync-case` | Tài liệu tương tự có thể làm sai | Qwen, cách hỏi chung: E0 đúng, E1 trả lời lành (sai) |
| 3 | Huy Anh (E2) | `s2-G2-0176` qua E2 | BM25 bắt được id policy | Lấy được `POL-SEC-001` v2 và v1; trả lời 24 giờ, đúng. G2: E2 0.52 |
| | | `s2-G2-0100` (`sniffio` trên `svc-checkout`) | Bám đúng tài liệu nhưng vẫn sai | E2 trả lời "no", trích policy v1 đã superseded |
| | | `s2-G3-0222` (`svc-payouts` và `ADV-2026-025`) | Similarity không lấy được bước ở giữa | E2 có advisory và manifest, thiếu trang `paramiko`; trả lời "no" (sai) |
| | | Recall bằng chứng; bẫy version | Có tài liệu chưa chắc đã đủ; chỗ E2 hơn KG | Recall: E1 0.602, E2 0.732, BM25 0.782. Version ngoài khoảng: E2 0.67, E3 0/18 |
| 4 | Kim Phụng (E3) | `s2-G3-0222`: cùng câu với Huy Anh | Duyệt đồ thị lấy được bước ở giữa | E3 trả lời "yes", đúng. Multi-hop: E2 0.04, E3 0.80 |
| | | E3a và E3b trên cùng câu | Cái giúp là chọn fact, không phải dạng đường đi | G3: E3 0.54, E3a 0.54, E3b 0.15 |
| | | C05 `synthetic-dataflow-case` | Nối được luồng dữ liệu qua 3 file | Qwen: E0, E1, E2 sai; E3 trở đi đúng |
| | | Chỗ KG hỏng | Cạnh không mang khoảng version, không đánh dấu mâu thuẫn | Version ngoài khoảng 0/18; policy mâu thuẫn 0/17 |
| 5 | Kim Phụng (E4, cả nhóm) | `s2-G2-0100` đưa qua verifier | Có mặt id không có nghĩa là suy ra được | Verifier gắn SUPPORTED cho câu trả lời sai, vì mọi id công ty đều có trong tài liệu được trích |
| | | `s2-G4-0302` (`svc-thumbnails`, không có trong corpus) | E4 làm được | E2, E3 trả lời "no"; E4 từ chối (tỉ lệ claim được hỗ trợ 0.80, dưới ngưỡng 1.0) |
| | | `s2-G4-0338` (token của `svc-backoffice`, hai policy mâu thuẫn) | E4 không làm được | E2 trả lời 180, E3 và E4 trả lời 90; đáp án đúng: thiếu bằng chứng |
| | | C06 `plugin-runner-case`; tổng G4 | Biết khi nào nên dừng | C06: 0/28 trả `uncertain`. G4: E4 0.65, thấp hơn E0 0.85 (E0 từ chối gần như mọi câu) |

## 2. Theo slide: người trình bày, nội dung, code và kết quả

| Người trình bày | Slide | Nội dung slide | Người phụ trách | Code tương ứng | Kết quả và file |
|---|---|---|---|---|---|
| Quỳnh Anh | 1-3 | Tiêu đề; agenda (bảy phần, số slide và người trình bày). Bài toán: Task A phát hiện package độc, Task B hỏi đáp chuỗi cung ứng của ACME | Thiết kế hệ thống (Phụng) | `schema.py`; `generation/output_schema.py`; `data/safe_io.py` (chỉ parse tĩnh) | Đầu ra Task A: malicious, benign hoặc uncertain, claim trích file:line. Task B: yes/no, số, tập service hoặc insufficient evidence |
| Quỳnh Anh | 4 | Paper Ibiyo et al. (EASE 2025): fine-tuning thắng RAG | Phân tích paper (Quỳnh Anh) | Không có code của nhóm. Repo mal-LLM được pin trong `pod_B/upstream_audit.json`; notebook 00 mục 01 | Số công bố (F1 malicious): Llama-3.1-8B trước FT 0.32, sau FT 0.97; RAG YARA 0.52, advisories 0.48, setup.py 0.44; GPT-4o-mini 0.86 |
| Quỳnh Anh | 5 | Tái lập: kết luận giữ, số không khớp | Tái lập (Quỳnh Anh) | `data/loaders.py`; `analysis/code_slicer.py`; `generation/prompts.py`; `generation/output_schema.py`; `knowledge/external.py`; `knowledge/build_index.py`; `pipelines/runner.py` (e*_malllm); notebook 01 mục 01 | F1 malicious (zero-shot / YARA / setup.py): nhóm 0.559 / 0.285 / 0.471; notebook repo 0.459 / 0.515 / 0.438. Accuracy của nhóm 0.830 / 0.776 / 0.431. File `pod_B/results/r01_repro_malllm_Meta-Llama-3.1-8B-Instruct/reproduction.json` |
| Quỳnh Anh | 6 | Ba câu hỏi, năm RQ; knowledge availability và evidence utilization | Cả nhóm | `evaluation/summary.py` (RQ_PAIRS); `supplychain/evaluate.py` (danh sách cặp, compare); `configs/experiments.yaml` | Thông điệp: không method nào tốt nhất ở mọi nơi; loại câu hỏi quyết định |
| Quỳnh Anh | 7 | Hallucination: factuality và faithfulness; cách đánh giá | Đánh giá (Huy Anh); E4 (cả nhóm, chính: Phụng) | `evaluation/evidence.py` (claim_metrics); `evaluation/classification.py`; `supplychain/evaluate.py` (score_one, _quality, _grounding) | Ví dụ sniffio==1.13.0 với policy v1 đã superseded. Thước đo: exact accuracy và F1, tỉ lệ claim unsupported, correct abstention trên G4, coverage |
| Huy Anh | 8 | Kiến trúc chung E0, E1/E2, E3, E4; đối chứng E3a, E3b | Thiết kế hệ thống (Phụng) | Task A: `pipelines/runner.py` (METHODS, build_request, predict_all). Task B: `supplychain/qa.py` (METHODS, build_request, build_qa_messages, predict_qa). Chung: `generation/prompts.py` (fit_items); `generation/llm_client.py` | Cùng reader, prompt, câu hỏi và ngân sách bằng chứng cho mọi method |
| Huy Anh | 9 | Setup: hai task, một cách đánh giá | Thiết kế hệ thống, dữ liệu Task B (Phụng); dữ liệu Task A (Quỳnh Anh); kiểm tra rò rỉ (Thành); thống kê (Huy Anh) | Task A: `data/loaders.py`; `data/splitter.py`; `data/leakage_check.py`; `data/cases.py`. Task B: `supplychain/org.py`; `supplychain/corpus.py`; `supplychain/questions.py`; `supplychain/pipeline.py` (generate). Thống kê: `evaluation/statistical_tests.py`. `configs/*.yaml` | mal-LLM test 500 (134 malicious); PyGuard test 310 (156 malicious). ACME: 32 service, 96 package (270 phiên bản), 342 tài liệu, KG 423 nút và 949 cạnh. Dev 160 câu, test 400 câu. File `pod_B/data/malllm/stats.json`, `pod_A/data/pyguard_obf/stats.json`, `pod_C/supplychain/test/stats.json` |
| Huy Anh | 10 | LLM thuần chỉ dựa vào tham số và input | E0 (Quỳnh Anh) | `pipelines/runner.py` (e0); `analysis/code_slicer.py` (package_representation); `supplychain/qa.py` (e0); `generation/llm_client.py` | Qwen PyGuard F1 0.953. Qwen G2, G3: 0.00; từ chối 99% câu policy, 97% câu dependency. Llama gắn nhãn độc cho 66% package lành của PyGuard. Chi phí 0.018 USD / 1.000 câu Task B. File `pod_A/results/r04_test_pyguard_obf_*/results.json`, `pod_C/supplychain/runs/test_*/results.json` (cost) |
| Huy Anh | 11 | Demo: code đã nằm trong prompt (C01/C02, câu C03 thuộc G1) | E0 (Quỳnh Anh) | `data/cases.py`; `knowledge/case_rules.py`; `supplychain/questions.py` (g1_questions, code_snippet); notebook 01 mục 05b; notebook 03 Phần 1 | 28/28 đúng cho C01 và C02. G1: Qwen E0 0.75, không method nào cao hơn E0 có ý nghĩa thống kê. File `pod_B/results/r07_cases_q1_*`, `pod_B/results/r07_cases_qcase_*` |
| Huy Anh | 12 | Pipeline text RAG: E1 dense, BM25, E2 hybrid (RRF) | E1 (Thành); E2 (Huy Anh) | `retrieval/core.py`; `knowledge/build_index.py`; `knowledge/documents.py`; `configs/retrieval.yaml`; `supplychain/qa.py` (QAContext) | Encoder bge-small-en-v1.5; RRF k = 60. Task A: 10 + 10 ứng viên, giữ 5. Task B: 16 + 16 ứng viên, giữ 8 |
| Huy Anh | 13 | Cách dense, BM25 và hybrid xếp hạng cùng một câu hỏi; công thức cosine và RRF | E1 (Thành); E2 (Huy Anh) | `retrieval/core.py` (VectorIndex, BM25, rrf_fuse); `supplychain/qa.py` (QAContext.retriever: 16 ứng viên, giữ 8) | Câu `s2-G2-0176`, Qwen. Top-8 dense chỉ có manifest; BM25 xếp `POL-SEC-001` v2 hạng 1; RRF đưa v2 lên hạng 4. svc-status: dense hạng 1, BM25 hạng 6, 1/61 + 1/66 = 0.0315. E1 từ chối, chỉ BM25 trả lời 48 (sai), E2 trả lời 24 (đúng). File `pod_C/supplychain/runs/test_Qwen2.5-Coder-7B-Instruct/predictions_e1.jsonl`, `predictions_e2_bm25.jsonl`, `predictions_e2.jsonl` (trường trace.retrieved) |
| Huy Anh | 14 | Demo G2: câu hỏi policy cần tài liệu | E1 (Thành); E2 (Huy Anh) | `supplychain/corpus.py` (policy v1/v2, ADR); `supplychain/questions.py` (g2_questions); `supplychain/qa.py` (e0, e1, e2); notebook 04 mục B6 | G2 E0/E1/E2/E3: Qwen 0.00 / 0.36 / 0.52 / 0.59; Llama 0.00 / 0.03 / 0.26 / 0.33. Câu svc-status: `pod_C/supplychain/runs/test_Qwen2.5-Coder-7B-Instruct/predictions_e0.jsonl`, `predictions_e1.jsonl`, `predictions_e2.jsonl` |
| Huy Anh | 15 | Lỗi của text RAG: thiếu bước ở giữa | E2 (Huy Anh) | `supplychain/questions.py` (g3_questions, _trap_for, _path_docs); `supplychain/kg.py` | Multi-hop (45 câu, Qwen): E2 0.04, E3 0.80. Câu svc-payouts: `predictions_e2.jsonl` và `predictions_e3.jsonl` |
| Kim Phụng | 16 | Hai đồ thị, provenance trên mọi cạnh | E3 (Phụng) | Task A: `analysis/ast_parser.py`; `analysis/taxonomy.py`; `graph/schema.py`; `graph/builder.py`; `graph/traversal.py`. Task B: `supplychain/kg.py` (build_kg, KGIndex) | C05: 21 nút, 24 cạnh; cạnh FLOWS_TO liên thủ tục, độ tin cậy 0.6. KG ACME: 423 nút, 949 cạnh. File `pod_*/graphs/cases/graphs_test.summary.json` |
| Kim Phụng | 17 | Từ code Python đến đồ thị hành vi: FLOWS_TO và PRECEDES | E3 (Phụng) | `analysis/ast_parser.py` (def-use, flows); `graph/builder.py` (FLOWS_TO, PRECEDES); `graph/retriever.py` (graph_evidence) | Output thật của `analyze_package` và `build_behavior_graph` trên hai đoạn mã 5 dòng: `data=data` tạo FLOWS_TO (def-use, 0.8) và khớp pattern P04 Local file upload; `data="ping"` chỉ có PRECEDES (thứ tự dòng, 0.5, inferred), E3 báo không có luồng dữ liệu |
| Kim Phụng | 18 | E3 và hai đối chứng | E3, E3a, E3b (Phụng) | Task A: `graph/retriever.py` (graph_evidence là E3, flatten_paths là E3a, edge_facts là E3b). Task B: `supplychain/kg.py` (KGIndex.evidence, edge_sentence, fact_docs); `supplychain/qa.py` (e3, e3a_exact, e3a) | E3 / E3a / E3b: G3 Qwen 0.54 / 0.54 / 0.15; PyGuard Llama 0.805 / 0.814 / 0.722. E3 hơn E3b +0.39 (G3) và +0.083 (Task A), có ý nghĩa thống kê |
| Kim Phụng | 19 | Verifier E4: câu trả lời có trích dẫn vẫn có thể sai | E4 (cả nhóm, chính: Phụng) | Task A: `verification/verifier.py` (check_claims, apply_policy, tune_policy). Task B: `supplychain/qa.py` (verify, apply_qa_policy); `supplychain/evaluate.py` (tune_policy, policy_sweep) | Ngưỡng tau tune trên dev: Qwen 1.0, Llama 0 (`pod_C/supplychain/runs/dev_*/policy.json`). Ví dụ sniffio: verifier gắn SUPPORTED dù kết luận sai |
| Kim Phụng | 20 | Cách đo và khi nào chênh lệch là thật: F1, exact accuracy, coverage, correct abstention, CI 95% | Đánh giá và thống kê (Huy Anh) | `evaluation/classification.py` (strict F1); `supplychain/evaluate.py` (score_one, summarize); `evaluation/statistical_tests.py` (paired_bootstrap theo nhóm) | Bootstrap ghép cặp, 1.000 lần (Task A), 2.000 lần (Task B). PyGuard Llama E3 vs E0 +0.052 [+0.030, +0.074]; PyGuard Qwen E2 vs E0 -0.039 [-0.063, -0.018]; G3 Qwen E3 vs E2 +0.31 [+0.15, +0.54]; G2 Qwen E3 vs E2 +0.07 [-0.07, +0.27], không có ý nghĩa. File `results.json` (comparisons) |
| Kim Phụng | 21 | Result A (Task A) | Cả nhóm (thực nghiệm); đánh giá và thống kê (Huy Anh) | `evaluation/classification.py` (strict F1); `evaluation/summary.py` (summarize_method, comparisons); `evaluation/statistical_tests.py` (paired_f1_bootstrap, mcnemar); notebook 01 mục 03-04 | F1 malicious E0/E1/E2/E3: PyGuard Llama 0.754 / 0.727 / 0.731 / 0.805; PyGuard Qwen 0.953 / 0.925 / 0.914 / 0.939; mal-LLM Llama 0.822 / 0.674 / 0.709 / 0.704; mal-LLM Qwen 0.866 / 0.934 / 0.945 / 0.760. File `pod_A/results/r04_test_pyguard_obf_*/results.json`, `pod_B/results/r03_test_malllm_*/results.json` |
| Kim Phụng | 22 | Result B: kết quả phụ thuộc loại câu hỏi và reader | Cả nhóm (thực nghiệm); đánh giá và thống kê (Huy Anh) | `supplychain/evaluate.py` (score_one, summarize, compare); notebook 04 mục B3-B4 | Qwen E0/E1/E2/E3/E4: G1 0.75 / 0.80 / 0.75 / 0.73 / 0.73; G2 0.00 / 0.36 / 0.52 / 0.59 / 0.56; G3 0.00 / 0.22 / 0.23 / 0.54 / 0.52; G4 0.85 / 0.59 / 0.47 / 0.56 / 0.65. Llama G1 giảm từ 0.62 (E0) xuống 0.26 (E3). File `pod_C/supplychain/runs/test_*/results.json` (summaries) |
| Kim Phụng | 23 | Result C: bằng chứng có mặt không, có được dùng không | E1 (Thành); E2 (Huy Anh); E3 (Phụng) | `supplychain/evaluate.py` (evidence_scores) | G1-G3, Qwen, 300 câu, E0/E1/E2/E3/E4. Recall bằng chứng gold: 0.333 / 0.602 / 0.732 / 0.989 / 0.989. Exact accuracy: 0.25 / 0.46 / 0.50 / 0.62 / 0.603 |
| Văn Thành | 24 | Result D: graph thắng ở đâu, gãy ở đâu | E3 (Phụng) | `supplychain/questions.py` (gán bẫy); `supplychain/evaluate.py` (summarize, by_trap); notebook 04 mục B5 | Từ E2 lên E3: multi-hop 0.04 lên 0.80, extra-not-enabled 0.29 lên 0.86. Ở E3: version ngoài khoảng 0/18, policy mâu thuẫn 0/17. File `results.json` (summaries, by_trap) |
| Văn Thành | 25 | Result E: grounding, accuracy và coverage đánh đổi nhau | E4 (cả nhóm, chính: Phụng); đánh giá (Huy Anh) | `evaluation/evidence.py` (claim_metrics); `evaluation/summary.py` (grounding_vs_base); `supplychain/evaluate.py` (_grounding); notebook 02 | Task A: claim unsupported / claim gốc của E3 là 0.33-0.51, E4 còn 0.000; F1 giảm 0.02-0.05. Task B Qwen: E0 0.788, E2 0.053, E3 0.030, E4 0.005. So với E3, E4 tăng correct abstention 0.09, giảm exact accuracy 0.017 |
| Văn Thành | 26 | Khuyến nghị khởi đầu (khung quyết định) | Cả nhóm | Tổng hợp từ slide 21-25; `evaluation/report.py`; `evaluation/figures.py`; `supplychain/pipeline.py` (write_report) | Code trong prompt: LLM thuần. Policy nội bộ: hybrid text RAG. Multi-hop: KG. Thiếu hoặc mâu thuẫn bằng chứng: chưa giải quyết |
| Văn Thành | 27 | Giới hạn và hướng tiếp | Cả nhóm; kiểm tra rò rỉ (Thành) | `data/leakage_check.py` (split_leakage: PyGuard không qua cổng near-duplicate); `pod_B/results/r03_test_malllm_*_nodup` | Task B tổng hợp, gold chung bộ sinh với corpus và KG; nhãn claim tự động; E3a còn đổi thứ tự |
| Văn Thành | 28-29 | Kết luận; cảm ơn và hỏi đáp | Cả nhóm | `README.md`; `Project11_Results_Curated_Critical_Analysis.md` | Policy 0.00 lên 0.52; multi-hop 0.04 lên 0.80; lọc claim tốn 0.02-0.05 F1 |

## 3. Hỏi đáp: ai mở slide backup nào

Khi thầy hỏi, người phụ trách method trả lời và mở slide backup tương ứng.

| Người | Slide backup |
|---|---|
| Quỳnh Anh (E0) | 30 (bảng RQ đầy đủ), 33-34 (demo C01/C02, G1), 50 (case C01-C06) |
| Văn Thành (E1) | 35 (pipeline vector RAG), 38 (similarity không đủ) |
| Huy Anh (E2, đánh giá) | 31 (hallucination, RAG cơ bản), 36 (các biến thể), 37 (C04, policy superseded), 45-47 (bảng Task A, Task B, retrieval và AURC), 51 (chi phí) |
| Kim Phụng (E3, E4) | 32 (kiến trúc), 39-41 (hai đồ thị, đồ thị C05, đồ thị dependency), 42-43 (Algorithm 1-2), 44 (nhãn verifier), 48 (giá của xác minh), 49 (biết khi nào dừng), 52-53 (mẫu dữ liệu), 54 (Division of work) |
