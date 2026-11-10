# results/: kết quả lần chạy 2026-11-10

Lần chạy thật trên 3 pod RunPod (mỗi pod một GPU), 2 reader: `NousResearch/Meta-Llama-3.1-8B-Instruct` và
`Qwen/Qwen2.5-Coder-7B-Instruct`, sinh bằng vLLM. Mọi số trong `README.md`, `docs/` và deck đều lấy từ thư mục này.

| Thư mục | Nội dung | GPU |
|---|---|---|
| `cloud/pod_A/` | Task A trên PyGuard (`pyguard_obf`): validation `r02`, test `r04` | A100 80GB |
| `cloud/pod_B/` | Task A trên mal-LLM (`malllm`): tái lập `r01`, validation `r02`, test `r03` (và bản `_nodup`), synthetic `r05`, lặp lại `r06`, sáu case C01-C06 `r07` | L40S |
| `cloud/pod_C/` | Task B (công ty tổng hợp ACME): câu hỏi dev và test, dự đoán của 8 method cho 2 reader, kết quả tổng hợp | L40S |
| `figures/` | Đồ thị vẽ từ các `results.json` ở trên | |

## Notebook đã chạy (có output)

| File | Notebook gốc | Nội dung |
|---|---|---|
| `cloud/pod_A/executed/00_data.ipynb` | `notebooks/00_data_knowledge_colab.ipynb` | Test và smoke, pin repo upstream, chia `pyguard_obf`, KB, đồ thị hành vi, `FROZEN.json` |
| `cloud/pod_A/executed/01_all.ipynb` | `notebooks/01_experiments_colab.ipynb` | Validation và test trên `pyguard_obf`, bảng F1, CI, kiểm RQ |
| `cloud/pod_B/executed/00_data.ipynb` | `notebooks/00_data_knowledge_colab.ipynb` | Như trên, cho `malllm`, `synthetic` và `cases` |
| `cloud/pod_B/executed/01_all.ipynb` | `notebooks/01_experiments_colab.ipynb` | Tái lập mal-LLM, validation, test `malllm` (cả bản bỏ near-duplicate), synthetic, C01-C06, chi phí và độ lặp lại |
| `cloud/pod_C/executed/04_taskb.ipynb` | `notebooks/04_supplychain_qa_colab.ipynb` | Sinh org dev (seed 1) và test (seed 2), tune E4 trên dev, chạy test, bảng theo nhóm câu hỏi, so sánh từng cặp method |

- Notebook 04 là bản chạy gốc (khoảng 30 phút), giữ nguyên toàn bộ print, kể cả log của vLLM.
- Notebook 01 của pod A và pod B được chạy lại sau bước chấm lại điểm. Mỗi bước in `[skip] ... đã có results.json` rồi đọc
  `results.json` để in bảng, CI và các dòng `PASS`/`FAIL` của từng RQ. Vì vậy hai notebook này không có dòng tiến độ sinh câu
  trả lời của LLM. `FAIL` nghĩa là giả thuyết của RQ không được dữ liệu ủng hộ, không phải lỗi khi chạy.
- `executed/*.log` là log của `jupyter nbconvert`; `launch.log` là log của script khởi chạy trên pod.

## Mỗi run có gì

Task A, thư mục `cloud/pod_{A,B}/results/rNN_<split>_<dataset>_<reader>/`:

- `results.json`: `summaries` (metric theo method), `comparisons` (bootstrap theo cụm: delta, CI 95%, p, McNemar), `strata`,
  `cases`, `cost`. `sample_ids` là id ẩn danh, không có mã nguồn;
- `report.md`: các bảng chính để đọc nhanh;
- `policy.json` (chỉ run validation): ngưỡng E4 tune trên validation, dùng lại cho test;
- `reproduction.json` (chỉ `r01`): số của nhóm đặt cạnh số trong notebook của repo mal-LLM.

Task B, thư mục `cloud/pod_C/supplychain/`:

- `dev/`, `test/`: `questions.jsonl` (câu hỏi, đáp án, bằng chứng chuẩn, loại bẫy) và `stats.json`;
- `runs/<split>_<reader>/results.json`: metric theo nhóm G1-G4, so sánh từng cặp method, chi phí, quét ngưỡng E4;
- `runs/<split>_<reader>/predictions_<method>.jsonl`: mỗi dòng là một câu hỏi, gồm câu trả lời, claim và nhãn của verifier,
  bằng chứng đưa vào prompt (`visible`), câu trả lời thô của model, số token, và `trace` (`trace.retrieved` là top-8 tài liệu
  của E1/E2, `trace.kg` là đường đi KG của E3);
- `runs/dev_<reader>/policy.json`: ngưỡng E4 tune trên dev.

Tên run: `r01` tái lập, `r02` validation, `r03` test `malllm`, `r04` test `pyguard_obf`, `r05` synthetic có gold, `r06` lặp lại
cùng cấu hình, `r07` case C01-C06 (`q1`: câu hỏi chung, `qcase`: câu hỏi riêng của case).

## Cố ý không có ở đây

- Dữ liệu Task A (package của mal-LLM và PyGuard), KB, đồ thị đầy đủ, dự đoán theo từng package và `llm_cache.jsonl`. Chúng chứa
  hoặc trích mã độc thật nên ở lại volume của pod.
- Notebook 02 và 03 đã chạy. Hai notebook này in ví dụ có dòng code của package độc, nên cũng ở lại pod; ở đây chỉ có log
  `executed/02_eval.log`, `executed/03_demo.log`.
- `snapshot/`: bản chạy local với reader dummy để thử notebook 03, không phải kết quả thật.
- `cloud/*.tgz`: bản nén trùng với các thư mục `pod_*`.
