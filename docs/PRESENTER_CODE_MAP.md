# Seminar 2, Project 11: người trình bày, code cần đọc và kết quả

- Deck: `Seminar2_Project11_RAG_slides.pptx`, gồm 25 slide chính và 25 slide backup (26-50). Deck và `Project11_Results_Curated_Critical_Analysis.md` nằm ngoài repo.
- Mọi số liệu lấy từ lần chạy ngày 2026-11-10.
- Đường dẫn:
  - code: tính từ thư mục `pypi_rag/`, trừ khi ghi rõ `configs/`, `notebooks/`, `scripts/` hoặc `tests/`;
  - kết quả: tính từ `results/cloud/`. Thư mục `results/` không có trên GitHub (bị `.gitignore` bỏ qua); bản kết quả của lần chạy lấy từ Phụng.

## Lưu ý trước khi đọc code

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
4. **Dữ liệu Task A trên máy chỉ có số liệu tổng hợp** (`results.json`, `report.md`, `stats.json`). Dữ liệu malware và dự đoán từng package nằm trên pod.
   - Khi đọc code, không cài, không import, không chạy package nghi độc.

## 0. Phân công code và demo (bản chốt)

Theo nhóm phương pháp:
- LLM thuần (E0): Quỳnh Anh.
- RAG văn bản (E1, E2): Văn Thành làm phía truy hồi, Huy Anh làm phía trả lời.
- KG-RAG (E3, E3a, E3b): Kim Phụng.

Các phần ngoài ba nhóm này được chia thêm như cột "Nhận thêm".

### Code

| Người | Phần việc gốc | Nhận thêm | Code phụ trách |
|---|---|---|---|
| Kim Phụng | KG-RAG: đồ thị hành vi, dependency KG, E3, E3a, E3b | Thiết kế hệ thống; bộ sinh dữ liệu Task B; hạ tầng chạy | `analysis/*`; `graph/*`; `supplychain/kg.py`; nhánh KG của `supplychain/qa.py`; `pipelines/runner.py`; `supplychain/pipeline.py`; `cli.py`; `schema.py`; `data/safe_io.py`; `supplychain/org.py`, `corpus.py`, `questions.py`; `scripts/*.sh`; notebook 01 và 04 |
| Quỳnh Anh | LLM thuần; paper và tái lập | Lớp LLM dùng chung; dữ liệu Task A; bộ case C01-C06 | `generation/llm_client.py`, `prompts.py`, `output_schema.py`; method `e0` trong `runner.py` và `qa.py`; tái lập: `analysis/code_slicer.py` (malllm_representation), `knowledge/external.py`, `knowledge/build_index.py` (malllm_code_docs), các method `e*_malllm`; dữ liệu: `data/loaders.py`, `data/splitter.py`, `data/synthetic.py`; `data/cases.py`, `knowledge/case_rules.py`; notebook 00 |
| Huy Anh | RAG phía trả lời; giảm hallucination | Xác minh E4; câu G4 thiếu bằng chứng; C06 | Nhánh text của `supplychain/qa.py` (build_request, build_qa_messages); `verification/verifier.py`; `supplychain/qa.py` (verify, apply_qa_policy, reapply_policy); `supplychain/evaluate.py` (tune_policy, policy_sweep, _grounding); `evaluation/evidence.py`; notebook 02 |
| Văn Thành | RAG phía truy hồi và các biến thể | Knowledge base; kiểm tra rò rỉ; đánh giá và thống kê; chi phí | `retrieval/core.py`; `configs/retrieval.yaml`; `knowledge/build_index.py`, `documents.py`, `builtin.py`, `pyguard_adapter.py`; `data/leakage_check.py`; `evaluation/classification.py`, `summary.py`, `statistical_tests.py`, `figures.py`, `report.py`; `supplychain/evaluate.py` (score_one, summarize, compare, evidence_scores, cost_estimate); `scripts/estimate_cost.py` |
| Cả nhóm | Thực nghiệm, thảo luận, kết luận, report, slides | | |

Test đi theo code: `test_analysis_graph.py`, `test_supplychain.py`, `test_perf.py` (Phụng); `test_cases.py`, phần dữ liệu của `test_safe_io_and_data.py` (Quỳnh Anh); `test_pipeline_metrics.py` (Thành).

### Demo (theo thứ tự chạy)

Task A mở trong `results/cloud/pod_B/results/r07_cases_*` (trường `cases`). Task B tìm theo `qid` trong `results/cloud/pod_C/supplychain/runs/test_Qwen2.5-Coder-7B-Instruct/predictions_<method>.jsonl`.

| Thứ tự | Người | Demo | Điều demo cho thấy | Kết quả đã có |
|---|---|---|---|---|
| 1 | Quỳnh Anh | C01 `safe-math-utils`, C02 `quick-setup-tools` | Code đã nằm sẵn trong input thì LLM thuần là đủ | Mỗi case đúng 28/28 |
| | | Một câu G1 về luồng token ra mạng | Thêm tài liệu có thể làm sai đi | Qwen E0 đúng; E1, E2, BM25 sai. G1: E0 0.75 |
| | | E0 trên câu policy và dependency | LLM thuần không có tri thức nội bộ | G2, G3 = 0.00; từ chối 99% và 97% số câu |
| | | Tái lập mal-LLM | Kết luận giữ, số không khớp | F1 0.559 / 0.285 / 0.471, so với notebook 0.459 / 0.515 / 0.438 |
| 2 | Văn Thành | `s2-G2-0176` (thời hạn vá của `svc-status`): tài liệu lấy về | Dense bỏ sót policy; BM25 bắt được id | E1: 8 tài liệu đều là manifest; E2: có `POL-SEC-001` v2 và v1 |
| | | `s2-G3-0222` (`svc-payouts` và `ADV-2026-025`): tài liệu lấy về | Similarity không lấy được bước ở giữa | E2 có advisory và manifest, thiếu trang `paramiko` |
| | | C03 `token-sync-case`; recall bằng chứng | Có tài liệu chưa chắc đã đủ | Recall: E1 0.602, E2 0.732, BM25 0.782, E3 0.989 |
| 3 | Huy Anh | `s2-G2-0176`: câu trả lời từng method | Câu hỏi policy cần tài liệu | E0, E1 từ chối; E2 trả lời 24 giờ, đúng. G2: 0.00 / 0.36 / 0.52 |
| | | `s2-G2-0100` (`sniffio` trên `svc-checkout`) | Bám đúng tài liệu nhưng vẫn sai | E2 trả lời "no", trích policy v1 đã superseded |
| | | C04 `report-uploader` | False positive | Llama: E0 báo độc, E1 sửa lại đúng, E2 vẫn báo độc |
| 4 | Kim Phụng | `s2-G3-0222`: cùng câu với Thành | Duyệt đồ thị lấy được bước ở giữa | E3 trả lời "yes", đúng. Multi-hop: E2 0.04, E3 0.80 |
| | | E3a và E3b trên cùng câu | Cái giúp là chọn fact, không phải dạng đường đi | G3: E3 0.54, E3a 0.54, E3b 0.15 |
| | | C05 `synthetic-dataflow-case` | Nối được luồng dữ liệu qua 3 file | Qwen: E0, E1, E2 sai; E3 trở đi đúng |
| | | Chỗ KG hỏng | Cạnh không mang khoảng version, không đánh dấu mâu thuẫn | Version ngoài khoảng 0/18; policy mâu thuẫn 0/17 |
| 5 | Huy Anh | `s2-G4-0302` (`svc-thumbnails`, không có trong corpus) | E4 làm được | E2, E3 trả lời "no"; E4 từ chối (tỉ lệ claim được hỗ trợ 0.80, dưới ngưỡng 1.0) |
| | | `s2-G4-0338` (token của `svc-backoffice`, hai policy mâu thuẫn) | E4 không làm được | E2 trả lời 180, E3 và E4 trả lời 90; đáp án đúng: thiếu bằng chứng |
| | | C06 `plugin-runner-case`; tổng G4 | Biết khi nào nên dừng | C06: 0/28 trả `uncertain`. G4: E4 0.65, thấp hơn E0 0.85 (E0 từ chối gần như mọi câu) |

## 1. Mỗi người cần đọc code nào

### Ha Thi Kim Phụng: thiết kế hệ thống Task A và Task B; đồ thị hành vi và dependency KG

Trình bày slide 14-19.

| Mục đích | File (hàm chính) |
|---|---|
| Khung chung | `schema.py`; `cli.py`; `pipelines/runner.py`; `supplychain/pipeline.py` |
| Đồ thị hành vi (Task A) | `analysis/ast_parser.py`; `analysis/taxonomy.py`; `analysis/provenance.py`; `analysis/parallel.py`; `graph/schema.py`; `graph/builder.py` (build_behavior_graph); `graph/traversal.py` (chains, match_patterns); `graph/retriever.py` (graph_evidence, flatten_paths, edge_facts); `configs/graph.yaml` |
| Công ty tổng hợp và dependency KG (Task B) | `supplychain/org.py` (make_org); `supplychain/corpus.py` (render_corpus); `supplychain/questions.py` (make_questions); `supplychain/kg.py` (build_kg, KGIndex); `supplychain/qa.py` |
| Verifier E4 | `verification/verifier.py` (check_claims, apply_policy, tune_policy); `supplychain/qa.py` (verify, apply_qa_policy); `supplychain/evaluate.py` (tune_policy, policy_sweep) |
| Để trình bày Result A-C | `evaluation/classification.py`; `evaluation/summary.py`; `evaluation/statistical_tests.py`; `supplychain/evaluate.py` |
| Test (ví dụ cách dùng) | `tests/test_analysis_graph.py`; `tests/test_cases.py`; `tests/test_supplychain.py` |

### An Nguyen Quỳnh Anh: LLM thuần; phân tích paper và tái lập; demo LLM thuần

Trình bày slide 1-6.

| Mục đích | File (hàm chính) |
|---|---|
| Tái lập mal-LLM | `data/loaders.py` (load_malllm); `analysis/code_slicer.py` (malllm_representation: 300 ký tự đầu của setup.py); `generation/prompts.py` (malllm_messages); `generation/output_schema.py` (parse_malllm_response); `knowledge/external.py` (load_yara); `knowledge/build_index.py` (malllm_code_docs); `pipelines/runner.py` (e0_malllm, e1_malllm_yara, e1_malllm_code) |
| LLM thuần E0 | `pipelines/runner.py` (method e0); `analysis/code_slicer.py` (package_representation); `generation/prompts.py` (build_messages); `generation/llm_client.py` (VLLMLLM, CachedLLM); `supplychain/qa.py` (method e0) |
| Demo | `data/cases.py` (C01-C06); `knowledge/case_rules.py`; `supplychain/questions.py` (g1_questions, code_snippet) |
| Để trình bày slide 5-6 | `evaluation/summary.py` (RQ_PAIRS); `evaluation/evidence.py` (claim_metrics); `supplychain/evaluate.py` (score_one, _quality, _grounding) |
| Notebook | `notebooks/01_experiments_colab.ipynb` mục 01 và 05b; `notebooks/03_demo_seminar.ipynb` Phần 1 |

### Đỗ Huy Anh: vector RAG và giảm hallucination; demo RAG

Trình bày slide 7-13.

| Mục đích | File (hàm chính) |
|---|---|
| Vector RAG | `retrieval/core.py` (Retriever); `supplychain/qa.py` (QAContext, build_request nhánh text) |
| Tài liệu cho demo | `supplychain/corpus.py` (policy v1 đã superseded, v2, ADR); `supplychain/questions.py` (g2_questions, g3_questions, _trap_for) |
| Giảm hallucination | `verification/verifier.py`; `supplychain/qa.py` (verify, apply_qa_policy); `evaluation/evidence.py` |
| Để trình bày slide 7-10 | `pipelines/runner.py` (METHODS, build_request); `generation/prompts.py` (fit_items: cùng ngân sách token cho mọi method); `data/loaders.py`; `data/splitter.py`; `data/leakage_check.py`; `data/cases.py`; `supplychain/org.py`; `evaluation/statistical_tests.py` |
| Notebook | `notebooks/04_supplychain_qa_colab.ipynb` mục B6 (ví dụ từng nhóm); `notebooks/02_evidence_eval_colab.ipynb` |

### Bùi Văn Thành: pipeline vector RAG và các biến thể

Trình bày slide 20-25.

| Mục đích | File (hàm chính) |
|---|---|
| Pipeline | `retrieval/core.py` (Embedder, VectorIndex, BM25, code_tokens, rrf_fuse, CrossEncoderReranker, Retriever, build_queries); `configs/retrieval.yaml` |
| Knowledge base | `knowledge/build_index.py` (build_kb); `knowledge/documents.py`; `knowledge/builtin.py`; `knowledge/external.py`; `knowledge/pyguard_adapter.py`; `data/leakage_check.py` (kb_leakage) |
| Biến thể | `pipelines/runner.py` (e1, e2, e2_bm25, e2_rerank); `supplychain/qa.py` (QAContext: 16 ứng viên mỗi retriever, giữ 8) |
| Để trình bày slide 20-24 | `supplychain/evaluate.py` (summarize: by_trap, _grounding, cost_estimate); `evaluation/summary.py` (grounding_vs_base, cost_estimate); `scripts/estimate_cost.py`; `evaluation/figures.py` |
| Test | `tests/test_pipeline_metrics.py` |

## 2. Bảng tổng hợp

| Người trình bày | Slide | Nội dung slide | Công việc chính (người phụ trách) | Code tương ứng | Kết quả và file |
|---|---|---|---|---|---|
| Quỳnh Anh | 1-2 | Tiêu đề. Bài toán: Task A phát hiện package độc, Task B hỏi đáp chuỗi cung ứng của ACME | Thiết kế hệ thống (Phụng) | `schema.py`; `generation/output_schema.py`; `data/safe_io.py` (chỉ parse tĩnh) | Đầu ra Task A: malicious, benign hoặc uncertain, claim trích file:line. Task B: yes/no, số, tập service hoặc insufficient evidence |
| Quỳnh Anh | 3 | Paper Ibiyo et al. (EASE 2025): fine-tuning thắng RAG | Phân tích paper (Quỳnh Anh) | Không có code của nhóm. Repo mal-LLM được pin trong `pod_B/upstream_audit.json`; notebook 00 mục 01 | Số công bố (F1 malicious): Llama-3.1-8B trước FT 0.32, sau FT 0.97; RAG YARA 0.52, advisories 0.48, setup.py 0.44; GPT-4o-mini 0.86 |
| Quỳnh Anh | 4 | Tái lập: kết luận giữ, số không khớp | Tái lập (Quỳnh Anh) | `data/loaders.py`; `analysis/code_slicer.py`; `generation/prompts.py`; `generation/output_schema.py`; `knowledge/external.py`; `knowledge/build_index.py`; `pipelines/runner.py` (e*_malllm); notebook 01 mục 01 | F1 malicious (zero-shot / YARA / setup.py): nhóm 0.559 / 0.285 / 0.471; notebook repo 0.459 / 0.515 / 0.438. Accuracy của nhóm 0.830 / 0.776 / 0.431. File `pod_B/results/r01_repro_malllm_Meta-Llama-3.1-8B-Instruct/reproduction.json` |
| Quỳnh Anh | 5 | Ba câu hỏi, năm RQ; knowledge availability và evidence utilization | Cả nhóm | `evaluation/summary.py` (RQ_PAIRS); `supplychain/evaluate.py` (danh sách cặp, compare); `configs/experiments.yaml` | Thông điệp: không method nào tốt nhất ở mọi nơi; loại câu hỏi quyết định |
| Quỳnh Anh | 6 | Hallucination: factuality và faithfulness; cách đánh giá | Hallucination (Huy Anh) | `evaluation/evidence.py` (claim_metrics); `evaluation/classification.py`; `supplychain/evaluate.py` (score_one, _quality, _grounding) | Ví dụ sniffio==1.13.0 với policy v1 đã superseded. Thước đo: exact accuracy và F1, tỉ lệ claim unsupported, correct abstention trên G4, coverage |
| Huy Anh | 7 | Kiến trúc chung E0, E1/E2, E3, E4; đối chứng E3a, E3b | Thiết kế hệ thống (Phụng) | Task A: `pipelines/runner.py` (METHODS, build_request, predict_all). Task B: `supplychain/qa.py` (METHODS, build_request, build_qa_messages, predict_qa). Chung: `generation/prompts.py` (fit_items); `generation/llm_client.py` | Cùng reader, prompt, câu hỏi và ngân sách bằng chứng cho mọi method |
| Huy Anh | 8 | Setup: hai task, một cách đánh giá | Thiết kế hệ thống (Phụng) | Task A: `data/loaders.py`; `data/splitter.py`; `data/leakage_check.py`; `data/cases.py`. Task B: `supplychain/org.py`; `supplychain/corpus.py`; `supplychain/questions.py`; `supplychain/pipeline.py` (generate). Thống kê: `evaluation/statistical_tests.py`. `configs/*.yaml` | mal-LLM test 500 (134 malicious); PyGuard test 310 (156 malicious). ACME: 32 service, 96 package (270 phiên bản), 342 tài liệu, KG 423 nút và 949 cạnh. Dev 160 câu, test 400 câu. File `pod_B/data/malllm/stats.json`, `pod_A/data/pyguard_obf/stats.json`, `pod_C/supplychain/test/stats.json` |
| Huy Anh | 9 | LLM thuần chỉ dựa vào tham số và input | LLM thuần (Quỳnh Anh) | `pipelines/runner.py` (e0); `analysis/code_slicer.py` (package_representation); `supplychain/qa.py` (e0); `generation/llm_client.py` | Qwen PyGuard F1 0.953. Qwen G2, G3: 0.00; từ chối 99% câu policy, 97% câu dependency. Llama gắn nhãn độc cho 66% package lành của PyGuard. Chi phí 0.018 USD / 1.000 câu Task B. File `pod_A/results/r04_test_pyguard_obf_*/results.json`, `pod_C/supplychain/runs/test_*/results.json` (cost) |
| Huy Anh | 10 | Demo: code đã nằm trong prompt (C01/C02, câu C03 thuộc G1) | Demo LLM thuần (Quỳnh Anh) | `data/cases.py`; `knowledge/case_rules.py`; `supplychain/questions.py` (g1_questions, code_snippet); notebook 01 mục 05b; notebook 03 Phần 1 | 28/28 đúng cho C01 và C02. G1: Qwen E0 0.75, không method nào cao hơn E0 có ý nghĩa thống kê. File `pod_B/results/r07_cases_q1_*`, `pod_B/results/r07_cases_qcase_*` |
| Huy Anh | 11 | Pipeline text RAG: E1 dense, BM25, E2 hybrid (RRF) | Pipeline và biến thể (Thành) | `retrieval/core.py`; `knowledge/build_index.py`; `knowledge/documents.py`; `configs/retrieval.yaml`; `supplychain/qa.py` (QAContext) | Encoder bge-small-en-v1.5; RRF k = 60. Task A: 10 + 10 ứng viên, giữ 5. Task B: 16 + 16 ứng viên, giữ 8 |
| Huy Anh | 12 | Demo G2: câu hỏi policy cần tài liệu | Demo RAG (Huy Anh) | `supplychain/corpus.py` (policy v1/v2, ADR); `supplychain/questions.py` (g2_questions); `supplychain/qa.py` (e0, e1, e2); notebook 04 mục B6 | G2 E0/E1/E2/E3: Qwen 0.00 / 0.36 / 0.52 / 0.59; Llama 0.00 / 0.03 / 0.26 / 0.33. Câu svc-status: `pod_C/supplychain/runs/test_Qwen2.5-Coder-7B-Instruct/predictions_e0.jsonl`, `predictions_e1.jsonl`, `predictions_e2.jsonl` |
| Huy Anh | 13 | Lỗi của text RAG: thiếu bước ở giữa | Demo RAG (Huy Anh); biến thể (Thành) | `supplychain/questions.py` (g3_questions, _trap_for, _path_docs); `supplychain/kg.py` | Multi-hop (45 câu, Qwen): E2 0.04, E3 0.80. Câu svc-payouts: `predictions_e2.jsonl` và `predictions_e3.jsonl` |
| Kim Phụng | 14 | Hai đồ thị, provenance trên mọi cạnh | Đồ thị hành vi, dependency KG (Phụng) | Task A: `analysis/ast_parser.py`; `analysis/taxonomy.py`; `graph/schema.py`; `graph/builder.py`; `graph/traversal.py`. Task B: `supplychain/kg.py` (build_kg, KGIndex) | C05: 21 nút, 24 cạnh; cạnh FLOWS_TO liên thủ tục, độ tin cậy 0.6. KG ACME: 423 nút, 949 cạnh. File `pod_*/graphs/cases/graphs_test.summary.json` |
| Kim Phụng | 15 | E3 và hai đối chứng | Thiết kế hệ thống (Phụng) | Task A: `graph/retriever.py` (graph_evidence là E3, flatten_paths là E3a, edge_facts là E3b). Task B: `supplychain/kg.py` (KGIndex.evidence, edge_sentence, fact_docs); `supplychain/qa.py` (e3, e3a_exact, e3a) | E3 / E3a / E3b: G3 Qwen 0.54 / 0.54 / 0.15; PyGuard Llama 0.805 / 0.814 / 0.722. E3 hơn E3b +0.39 (G3) và +0.083 (Task A), có ý nghĩa thống kê |
| Kim Phụng | 16 | Verifier E4: câu trả lời có trích dẫn vẫn có thể sai | Thiết kế hệ thống (Phụng); giảm hallucination (Huy Anh) | Task A: `verification/verifier.py` (check_claims, apply_policy, tune_policy). Task B: `supplychain/qa.py` (verify, apply_qa_policy); `supplychain/evaluate.py` (tune_policy, policy_sweep) | Ngưỡng tau tune trên dev: Qwen 1.0, Llama 0 (`pod_C/supplychain/runs/dev_*/policy.json`). Ví dụ sniffio: verifier gắn SUPPORTED dù kết luận sai |
| Kim Phụng | 17 | Result A (Task A) | Cả nhóm (thực nghiệm) | `evaluation/classification.py` (strict F1); `evaluation/summary.py` (summarize_method, comparisons); `evaluation/statistical_tests.py` (paired_f1_bootstrap, mcnemar); notebook 01 mục 03-04 | F1 malicious E0/E1/E2/E3: PyGuard Llama 0.754 / 0.727 / 0.731 / 0.805; PyGuard Qwen 0.953 / 0.925 / 0.914 / 0.939; mal-LLM Llama 0.822 / 0.674 / 0.709 / 0.704; mal-LLM Qwen 0.866 / 0.934 / 0.945 / 0.760. File `pod_A/results/r04_test_pyguard_obf_*/results.json`, `pod_B/results/r03_test_malllm_*/results.json` |
| Kim Phụng | 18 | Result B: kết quả phụ thuộc loại câu hỏi và reader | Cả nhóm | `supplychain/evaluate.py` (score_one, summarize, compare); notebook 04 mục B3-B4 | Qwen E0/E1/E2/E3/E4: G1 0.75 / 0.80 / 0.75 / 0.73 / 0.73; G2 0.00 / 0.36 / 0.52 / 0.59 / 0.56; G3 0.00 / 0.22 / 0.23 / 0.54 / 0.52; G4 0.85 / 0.59 / 0.47 / 0.56 / 0.65. Llama G1 giảm từ 0.62 (E0) xuống 0.26 (E3). File `pod_C/supplychain/runs/test_*/results.json` (summaries) |
| Kim Phụng | 19 | Result C: bằng chứng có mặt không, có được dùng không | Biến thể truy hồi (Thành); KG (Phụng) | `supplychain/evaluate.py` (evidence_scores) | G1-G3, Qwen, 300 câu, E0/E1/E2/E3/E4. Recall bằng chứng gold: 0.333 / 0.602 / 0.732 / 0.989 / 0.989. Exact accuracy: 0.25 / 0.46 / 0.50 / 0.62 / 0.603 |
| Văn Thành | 20 | Result D: graph thắng ở đâu, gãy ở đâu | KG (Phụng); cả nhóm | `supplychain/questions.py` (gán bẫy); `supplychain/evaluate.py` (summarize, by_trap); notebook 04 mục B5 | Từ E2 lên E3: multi-hop 0.04 lên 0.80, extra-not-enabled 0.29 lên 0.86. Ở E3: version ngoài khoảng 0/18, policy mâu thuẫn 0/17. File `results.json` (summaries, by_trap) |
| Văn Thành | 21 | Result E: grounding, accuracy và coverage đánh đổi nhau | Giảm hallucination (Huy Anh) | `evaluation/evidence.py` (claim_metrics); `evaluation/summary.py` (grounding_vs_base); `supplychain/evaluate.py` (_grounding); notebook 02 | Task A: claim unsupported / claim gốc của E3 là 0.33-0.51, E4 còn 0.000; F1 giảm 0.02-0.05. Task B Qwen: E0 0.788, E2 0.053, E3 0.030, E4 0.005. So với E3, E4 tăng correct abstention 0.09, giảm exact accuracy 0.017 |
| Văn Thành | 22 | Khuyến nghị khởi đầu (khung quyết định) | Cả nhóm | Tổng hợp từ slide 17-21; `evaluation/report.py`; `evaluation/figures.py`; `supplychain/pipeline.py` (write_report) | Code trong prompt: LLM thuần. Policy nội bộ: hybrid text RAG. Multi-hop: KG. Thiếu hoặc mâu thuẫn bằng chứng: chưa giải quyết |
| Văn Thành | 23 | Giới hạn và hướng tiếp | Cả nhóm | `data/leakage_check.py` (split_leakage: PyGuard không qua cổng near-duplicate); `pod_B/results/r03_test_malllm_*_nodup` | Task B tổng hợp, gold chung bộ sinh với corpus và KG; nhãn claim tự động; E3a còn đổi thứ tự |
| Văn Thành | 24-25 | Kết luận; cảm ơn và hỏi đáp | Cả nhóm | `README.md`; `Project11_Results_Curated_Critical_Analysis.md` | Policy 0.00 lên 0.52; multi-hop 0.04 lên 0.80; lọc claim tốn 0.02-0.05 F1 |

## 3. Hỏi đáp: ai mở slide backup nào

| Người | Slide backup |
|---|---|
| Quỳnh Anh | 26 (bảng RQ đầy đủ), 27 (hallucination, RAG cơ bản) |
| Huy Anh | 28 (kiến trúc), 29-30 (demo C01/C02, G1), 31-32 (pipeline, các biến thể), 33 (C04, policy superseded), 34 (similarity không đủ) |
| Kim Phụng | 35-37 (hai đồ thị, đồ thị C05, đồ thị dependency), 38-39 (Algorithm 1-2), 40 (nhãn verifier), 41-43 (bảng Task A, Task B, retrieval và AURC), 46 (case C01-C06), 48-49 (mẫu dữ liệu) |
| Văn Thành | 44 (giá của xác minh), 45 (biết khi nào dừng), 47 (chi phí), 50 (Division of work) |
