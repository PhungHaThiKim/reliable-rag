# Seminar 2 · Đề 11 · RAG giảm hallucination mà vẫn giữ accuracy (phần KG-RAG)

Folder này chứa đúng phần code tạo ra các kết quả dùng trong bài trình bày Đề 11 "RAG: Concepts & Applications":
LLM thuần, RAG và biến thể, KG-RAG, đo trên cùng một bài toán hỏi đáp trên đồ thị tri thức. Phần nghiên cứu khác của repo gốc
(PRISM, các policy RL traversal, ablation retrieval, drift) đã được bỏ.

**Thông điệp.** Không phương pháp nào tốt nhất cho mọi trường hợp. Paper tham khảo (phát hiện mã độc PyPI) thấy fine-tune hơn RAG vì bài toán
là phân loại với tín hiệu nằm ngay trong đầu vào. Trên hỏi đáp tri thức thì ngược lại: phải truy hồi chứng cứ, và cách giảm bịa hiệu quả nhất
là ràng buộc câu trả lời vào chứng cứ kiểm chứng được trong đồ thị.

## Kết quả chính

WebQSP test, 1000 câu (cùng tập cho mọi cấu hình), pool ứng viên của GNN ReaRev. Mỗi ô ghi Qwen2.5-7B / Llama-3.1-8B.

| Cấu hình | llm_hit | grounded_correct | QHR (bịa) | strict_QHR | #đường |
|---|---|---|---|---|---|
| LLM thuần, không context | 0.474 / 0.588 | – | – | – | 0 |
| LLM thuần + fine-tune LoRA, không context | 0.412 / 0.446 | – | – | – | 0 |
| RAG dense top-5 trên đường KG, prompt chat | 0.755 / 0.768 | 0.709 / 0.704 | 0.118 / 0.150 | 0.254 / 0.289 | 3.46 |
| KG-RAG: GNN-RAG gốc (all, prompt chat) | 0.765 / 0.805 | 0.722 / 0.744 | 0.114 / 0.140 | 0.241 / 0.283 | 8.48 |
| KG-RAG: CGAR-G + reader LoRA + verify | 0.797 / 0.800 | 0.796 / 0.799 | 0.002 / 0.002 | 0.236 / 0.228 | 7.42 |

- `grounded_correct` là tỷ lệ câu trả lời đúng **và** có đường đi trong đồ thị làm chứng. Đây là chỉ số chính cho "đúng mà không bịa".
- `QHR` là tỷ lệ câu trả lời có ít nhất một entity không có chứng cứ trong đồ thị. `strict_QHR` tính thêm entity có trong đồ thị nhưng sai quan hệ.
- Từ GNN-RAG gốc sang CGAR-G: grounded_correct tăng 5.5 đến 7.4 điểm, QHR từ 0.11 đến 0.14 xuống 0.002, llm_hit giữ nguyên hoặc tăng, dùng 87% số đường.
- Fine-tune không có context kém hơn cả LLM gốc. Đây là phản ví dụ cho kết luận "fine-tune hơn RAG" khi đổi loại dữ liệu và câu hỏi.

Số liệu đầy đủ, CI 95% và phân tích ở [results/TONG_HOP_KET_QUA_P8_P9_2026-11-08.md](results/TONG_HOP_KET_QUA_P8_P9_2026-11-08.md).

## Ánh xạ phần trình bày sang code

| Phần trình bày | Cấu hình | Notebook, cell | Thư mục kết quả trên Drive (`graph_rag/results_gnn/`) |
|---|---|---|---|
| 1. LLM thuần | `no_retrieval` (zero-shot và LoRA) | 01 cell 08, 08d | `b08_reader_*`, `b08f_reader_*` |
| 2. RAG và biến thể | `sbert_top5`, `gnnrag_top5`; prompt `grounded`; verify | 01 cell 02, 08, 08c | `b02_baselines`, `b08_reader_*`, `b08g_reader_*`, `b08v_reader_*` |
| 3. KG-RAG | `gnnrag_all`, `cgar`, `cgar_esc`; prompt `cand` + verify; LoRA | 01 cell 03, 08, 08c, 08d | `b03_cgar`, `b06d_cgar_esc`, `b08c_reader_*`, `b08f_reader_*` |
| Đo hallucination | verifier theo đồ thị | 02 | `<thư mục reader>/hallucination/` |
| Bảng, ví dụ, kết luận | – | 03 | `seminar2_kgrag/results/headline_tables.csv` |

## Cấu trúc folder

```
seminar2_kgrag/
  kgqa_bench/            harness: dữ liệu, retriever, reader, verifier, metric
    data.py              BenchDataset, nạp GNN-RAG features (spec gnnrag:...)
    adapters.py          feature store chung cho mọi retriever
    methods.py           GNNRAGMethod (all / top-K), SBERTTopKMethod, CGARMethod, CGAREscalateMethod (CGAR-G)
    cgar_train.py        huấn luyện CGAR
    llm_reader.py        reader HF, prompt chat / grounded / cand, verify (ground_answer), LoRA adapter
    hallucination.py     verifier: claim SUPPORTED / UNSUPPORTED / UNCERTAIN theo đường đi trong đồ thị
    rescore.py           chấm lại offline (parser, verify) không cần GPU
    snapshot.py          trích bản kết quả gọn cho notebook demo (results/snapshot/)
    reader_ft.py         dựng dữ liệu SFT và fine-tune LoRA reader
    metrics.py report.py cli.py
  adagraph_cgar_v2.py    mô hình CGAR (relevance head, confidence head, loss)
  grl_rag.py             GNN feature store, embedder SBERT, trích đường ứng viên
  qa_text_eval.py        parser câu trả lời, nhận diện từ chối, hit / F1
  calibration.py         ECE, AURC, selective accuracy
  significance.py        paired bootstrap, McNemar
  notebooks/
    00_gnn_features_colab.ipynb        dựng gnn_features_*.pkl từ ReaRev (chỉ khi chưa có)
    01_pipeline_colab.ipynb            retrieval, CGAR, reader, CGAR-G, LoRA, checklist claim
    02_hallucination_eval_colab.ipynb  verifier, error taxonomy, risk-coverage, kiểm định cặp, gate
    03_demo_seminar.ipynb              bảng và ví dụ cho từng phần trình bày, không cần GPU
  tests/                 30 test CPU trên dữ liệu tổng hợp
  results/
    TONG_HOP_KET_QUA_P8_P9_2026-11-08.md   số liệu đầy đủ, CI 95%, phân tích
    runs/                bản notebook đã chạy thành công trên Colab A100, giữ nguyên output
      P8_pipeline_run_2026-11-09.ipynb       mọi bước retrieval, reader, CGAR-G, LoRA (có log train), ablation, checklist
      P9_hallucination_run_2026-11-09.ipynb  verifier trên 11 cấu hình, taxonomy, risk-coverage, kiểm định cặp, gate
    snapshot/            (tạo trên Colab) kết quả gọn để notebook 03 chạy không cần Drive
    headline_tables.csv  do notebook 03 ghi
```

## Cách chạy

**Xem kết quả đã chạy.** Mở hai notebook trong `results/runs/`: toàn bộ output gốc của lần chạy trên Colab A100, không cần chạy lại.

**Xem demo (không cần GPU).** Upload nguyên folder `seminar2_kgrag/` lên Drive, đặt cạnh `graph_rag/`
(mặc định `MyDrive/ThS/Research/colab/`, sửa `DRIVE_BASE` ở cell đầu nếu khác). Mở `notebooks/03_demo_seminar.ipynb` trên Colab và chạy hết.
Lần đầu, cell thứ hai tạo `results/snapshot/` từ `graph_rag/results_gnn/` trên Drive (summaries của mọi cấu hình, bản ghi từng câu của Qwen cho các ví dụ,
câu hỏi và đáp án). Sau đó tải folder về máy: notebook 03 đọc snapshot và chạy được ở bất kỳ đâu, không cần Drive. Có thể tạo snapshot bằng lệnh:

```
python -m kgqa_bench.snapshot --results <graph_rag>/results_gnn \
    --data-spec gnnrag:<graph_rag>/ckpt/gnn_features_webqsp_test.pkl:webqsp:test --out results/snapshot
```

**Tái lập số liệu.** Chạy `01_pipeline_colab.ipynb` rồi `02_hallucination_eval_colab.ipynb` trên Colab A100. Mọi cell bỏ qua thư mục đã có
`results.json`, nên lần chạy đầu chỉ kiểm tra và in lại bảng. Muốn chạy lại thật một bước thì xoá thư mục kết quả tương ứng.
Thời gian ước lượng khi chạy lại từ đầu: CGAR 20 phút, mỗi reader khoảng 1 giờ cho mỗi cấu hình prompt, LoRA khoảng 35 phút mỗi reader,
verifier khoảng 3 phút mỗi thư mục. Nếu chưa có `ckpt/gnn_features_*.pkl`, chạy `00_gnn_features_colab.ipynb` trước.

**Kiểm tra code trên máy (CPU).**

```
cd seminar2_kgrag
python -m pytest -q tests/
python -m kgqa_bench.cli run --train synthetic:16:44 --eval synthetic:8:45 \
    --methods gnnrag_all sbert_top5 cgar cgar_esc --embedder hash --device cpu \
    --reader dummy --reader-verify --no-retrieval-baseline --cgar-s1-epochs 1 --cgar-s2-steps 0 \
    --cgar-max-paths 2 --cgar-esc-ncand 0 --cgar-esc-frac 0.5 --out /tmp/s2_smoke --quiet
```

## Phương pháp tóm tắt

1. **GNN-RAG.** ReaRev chấm điểm thực thể ứng viên trên subgraph Freebase của câu hỏi. Hệ thống trích tối đa 3 đường quan hệ ngắn nhất
   từ thực thể chủ đề tới mỗi ứng viên, viết thành chữ và đưa cho LLM.
2. **CGAR.** Mạng nhỏ trên embedding SBERT chấm điểm từng đường, chọn tối đa 5 đường và trả về confidence cho câu hỏi.
   Confidence này xếp hạng câu dễ sai tốt hơn GNN-RAG ở cả 11 cấu hình reader và prompt; khi có verify, AURC-incorrect là 0.09 đến 0.13 so với 0.17 đến 0.21.
3. **CGAR-G.** Ba bước nhắm vào hallucination: mở rộng sang toàn bộ đường khi câu có hơn 10 ứng viên hoặc confidence dưới ngưỡng chọn trên dev;
   prompt liệt kê ứng viên là điểm cuối của các đường; verify chỉ giữ entity trùng một ứng viên, rỗng thì lấy ứng viên đầu.
4. **Reader LoRA.** Fine-tune Qwen và Llama trên tập train để chọn đúng các ứng viên là đáp án.
5. **Verifier.** Mỗi entity trong câu trả lời là một claim. SUPPORTED khi là điểm cuối của một đường có chuỗi quan hệ khớp proof tham chiếu
   hoặc là gold; UNSUPPORTED khi không có trong subgraph, có nhưng không được truy hồi, hoặc chỉ là node trung gian; UNCERTAIN khi là điểm
   cuối của một đường nhưng quan hệ khác proof.

## Đã bỏ so với repo nghiên cứu

| Thành phần | Lý do |
|---|---|
| Package `prism/` (PRISM: LAS, distill, RL, teacher LLM) | không có trong kết quả seminar |
| `GraphRAGFIMethod`, `RandomKMethod`, `baseline_graphrag_fi.py` | baseline retrieval không dùng trong phần reader |
| RL traversal và path-selection trong `grl_rag.py`, `adagraph_cgar.py`, `adagraph_controller.py` | thế hệ trước của CGAR |
| `escalation.py`, `analysis.py`, notebook cell 06, 06b, 06c, 07, 09, 10 | phân tích retrieval, ablation, drift; kết quả vẫn có trong file tổng hợp |
| LLM judge (P9 cell 10) | chưa dùng |
| Notebook P0 đến P7, các bản P8 cũ | lịch sử phát triển |

Parser câu trả lời và các hàm calibration được copy nguyên văn từ PRISM sang `qa_text_eval.py` và `calibration.py` để folder không phụ thuộc PRISM.
Logic của các hàm được giữ không đổi, nên số liệu chạy lại khớp với file tổng hợp.

## Giới hạn

- Nhãn hallucination là nhãn tự động theo đồ thị, chưa có nhãn người. Notebook 02 cell 02 đã xuất mẫu 200 câu để gán nhãn; cell 10 đo độ khớp khi có nhãn.
- Verify đưa QHR gần về 0 một phần vì dùng cùng luật với verifier. Phải đọc kèm grounded_correct và strict_QHR.
- Reader chạy trên 1000 câu WebQSP, một seed. Chưa có reader trên CWQ.
- Hai notebook trong `results/runs/` chạy bằng code của repo nghiên cứu trước khi rút gọn, nên còn các cell PRISM, ablation, drift; số trong đó trùng với code trong folder này.
- CGAR-G chưa đạt mục tiêu dưới 80% số đường của GNN-RAG (đang ở 87%). Với reader LoRA, CGAR-G kém GNN-RAG-all 1.0 điểm llm_hit ở Llama.
