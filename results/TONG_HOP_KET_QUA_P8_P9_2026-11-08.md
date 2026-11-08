# Tổng hợp kết quả P8 + P9 (CGAR / CGAR-G / hallucination), 08/11/2026

Tài liệu này gom toàn bộ số liệu đã chạy trên Colab (A100-SXM4-40GB) vào một chỗ, kèm phân tích. Mọi con số lấy nguyên từ output notebook, không có số placeholder.

## 0. Nguồn số liệu

| Phần | Notebook trong `run/` | Cell | Thư mục kết quả trên Drive (`results_gnn/`) |
|---|---|---|---|
| Dữ liệu, trần coverage | `P8_KGQA_Bench_Colab_(1) (1).ipynb` | 01 | – |
| Baseline retrieval | như trên | 02 | `b02_baselines` |
| Huấn luyện CGAR (gốc, v3) | như trên | 03 | `b03_cgar`, `b03_cgar_v3` |
| Bảng chính 3 seed + sweep | như trên | 06 | `b06_main` |
| Escalation offline | như trên | 06b | `b06_main` |
| Adaptivity, CI từng cặp, selective | như trên | 06c | `b06_main` |
| Calibration | như trên | 07 | `b06_main` |
| Reader chat / grounded / không retrieval | như trên | 08 | `b08_reader_*`, `b08g_reader_*` |
| CGAR-G retrieval + reader cand+verify | như trên | 08c | `b06d_cgar_esc`, `b08c_reader_*`, `b08v_reader_*` |
| Reader LoRA | như trên | 08d | `b12_train_ctx`, `b08f_reader_*`, `ckpt/lora_*_cand` |
| Ablation, drift | `P8_KGQA_Bench_Colab_final.ipynb` | 09, 10 | `b09_*`, `b10_drift` |
| Hallucination / groundedness | `P9_Hallucination_Eval_Colab_(3).ipynb` (cell 01 đến 11), `P9_Hallucination_Eval_Colab_final.ipynb` (cell 12) | 01 đến 12 | `<reader_dir>/hallucination/` |

Các bản notebook cũ hơn (`P8_KGQA_Bench_Colab.ipynb`, `_new`, `_(1)`, `_(3)`, `P9 ... .ipynb`, `(1)`, `(2)`) đã bị thay bằng các bản trên. Kết quả trên đĩa giống nhau vì các cell dùng lại `results.json`.

Thiết lập chung:

- **Pool ứng viên:** ReaRev (GNN-RAG) top-20, trung bình 5.1 đến 5.6 ứng viên mỗi câu. Đường đi là shortest relation path từ topic entity tới ứng viên, tối đa 3 đường mỗi cặp.
- **CGAR:** budget 5 đường, chỉ Stage 1 (supervised), checkpoint `b03_cgar/ckpt/cgar_s1_seed42.pt`. Bản v3 bị loại trên dev.
- **Reader:** Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct (mirror NousResearch). Mistral-7B-v0.3 chỉ có run chat cũ.
- **Tập reader:** 1000 câu WebQSP test (subset seed 0), 996 câu có đủ record cho mọi method. Một seed reader.
- **Kiểm định:** paired bootstrap 10000 lần, McNemar cho biến nhị phân. "sig" nghĩa là CI 95% không chứa 0.

## 1. Dữ liệu và trần coverage

| Tập | n | Đáp án trong subgraph | Số hop TB | Số triple TB | Số gold TB | Ứng viên TB |
|---|---|---|---|---|---|---|
| WebQSP train | 2848 | 88.6% | 1.30 | 4221 | 11.18 | 5.07 |
| WebQSP test | 1639 | 88.7% | 1.31 | 4306 | 10.13 | 5.60 |
| CWQ test | 3531 | 71.5% | 1.46 | 4273 | 1.89 | 5.23 |

- Đáp án nằm trong pool ứng viên ReaRev trên WebQSP test: **82.6%**. Pool heuristic cũ của Seminar 1 chỉ khoảng 33%.
- Mọi phương pháp chọn đường đều bị chặn trên bởi con số này. hit_ctx của gnnrag_all là 83.2%, gần chạm trần.

## 2. Retrieval

### 2.1 Bảng chính (cell 02, 03, 06)

WebQSP test, n = 1628:

| Method | F1 | Recall | Prec | hit_ctx | #paths | ECE | AURC | SelAcc@80 |
|---|---|---|---|---|---|---|---|---|
| gnnrag_all | 41.1 | 72.0 | 33.9 | 83.2 | 8.37 | 0.436 | 0.196 | 82.0 |
| gnnrag_top10 | 39.6 | 66.6 | 33.8 | 80.7 | 5.12 | 0.411 | 0.203 | 81.0 |
| gnnrag_top5 | 37.7 | 60.5 | 33.5 | 78.2 | 3.45 | 0.387 | 0.211 | 79.2 |
| gnnrag_top3 | 35.1 | 53.6 | 32.6 | 75.0 | 2.45 | 0.357 | 0.227 | 76.1 |
| sbert_top10 | 40.1 | 67.9 | 34.4 | 81.6 | 5.12 | 0.198 | 0.143 | 84.7 |
| sbert_top5 | 38.5 | 61.9 | 34.4 | 79.0 | 3.45 | 0.174 | 0.163 | 82.2 |
| graphrag_fi_k8 | 38.2 | 62.3 | 33.6 | 79.7 | 4.31 | 0.401 | 0.206 | 80.1 |
| random_k5 | 37.7 | 60.7 | 33.8 | 78.1 | 3.45 | – | – | – |
| **cgar** (3 seed) | **39.2 ± 0.03** | 62.9 | **35.6** | 79.6 | 3.45 | 0.221 | **0.118** | **84.8** |

CWQ test, n = 3531:

| Method | F1 | Recall | Prec | hit_ctx | #paths | ECE | AURC | SelAcc@80 |
|---|---|---|---|---|---|---|---|---|
| gnnrag_all | 21.6 | 62.1 | 14.6 | 65.8 | 11.84 | 0.298 | 0.288 | 65.6 |
| gnnrag_top10 | 21.8 | 57.8 | 14.9 | 62.1 | 5.73 | 0.264 | 0.295 | 64.4 |
| gnnrag_top5 | 21.8 | 52.9 | 15.1 | 58.0 | 3.78 | 0.232 | 0.306 | 62.1 |
| sbert_top10 | 21.9 | 59.3 | 14.9 | 63.5 | 5.73 | 0.050 | 0.322 | 65.9 |
| sbert_top5 | 22.2 | 55.1 | 15.4 | 60.0 | 3.78 | 0.049 | 0.353 | 62.4 |
| graphrag_fi_k8 | 21.7 | 55.7 | 15.0 | 60.6 | 4.91 | 0.251 | 0.298 | 63.6 |
| **cgar** (3 seed) | **22.65 ± 0.04** | 55.5 | **15.8** | 60.5 | 3.78 | 0.147 | **0.238** | **67.0** |

### 2.2 So sánh từng cặp (cell 06, 06c)

| Cặp | Tập | Chỉ số | Δ (pp) | CI 95% | sig |
|---|---|---|---|---|---|
| cgar − sbert_top5 | WebQSP | F1 | +0.75 | [+0.37, +1.12] | có |
| cgar − sbert_top5 | WebQSP | hit_ctx | +0.61 | [−0.43, +1.66] | không |
| cgar − gnnrag_top5 | WebQSP | F1 | +1.54 | [+1.17, +1.93] | có |
| cgar − gnnrag_top5 | WebQSP | hit_ctx | +1.41 | [+0.49, +2.40] | có |
| cgar − gnnrag_top10 | WebQSP | F1 | −0.36 | [−0.72, +0.01] | không |
| cgar − gnnrag_all | WebQSP | F1 | −1.88 | [−2.40, −1.36] | có |
| cgar − gnnrag_all | WebQSP | hit_ctx | −3.62 | [−4.55, −2.76] | có |
| cgar − gnnrag_all | CWQ | F1 | +1.11 | [+0.87, +1.35] | có |
| cgar − gnnrag_all | CWQ | hit_ctx | −5.24 | [−6.00, −4.50] | có |

### 2.3 Sweep cost của CGAR (cell 06, subset 594 / 600 câu)

| max_paths | rel_threshold | WebQSP cost | WebQSP F1 | WebQSP hit_ctx | CWQ cost | CWQ F1 |
|---|---|---|---|---|---|---|
| 3 | – | 2.45 | 36.1 | 75.8 | 2.64 | 22.8 |
| 5 | – | 3.42 | 37.8 | 78.1 | 3.76 | 22.7 |
| 10 | – | 5.00 | 39.3 | 80.6 | 5.67 | 21.9 |
| 20 | – | 6.71 | 39.9 | 82.0 | 8.01 | 21.5 |
| 20 | 0.0 | 3.50 | 39.7 | 75.9 | 4.35 | 23.1 |
| 20 | −1.0 | 3.94 | 40.3 | 77.3 | 4.94 | 22.9 |

β-sweep (0.50 và 0.99) phẳng: F1 37.5 đến 37.9 ở cùng cost. β không có tác dụng.

### 2.4 Adaptivity (cell 06c, A2)

- **Controller không bao giờ tự dừng sớm:** `controller_stopped_early = 0.0%`. 58.8% câu có pool ít hơn budget (`pool_limited`), 41.2% dùng đủ budget.
- Phân bố cost 1 đến 5 hoàn toàn do kích thước pool. Mọi method cũng sẽ có cùng phân bố. Không được claim "adaptive".
- Bản v3 có learned-stop thật: cost tự giảm từ 3.45 xuống 2.28. Nhưng dev F1 giảm từ 45.23 xuống 43.76 và ECE không cải thiện (0.194 so với 0.186), nên bị loại.

### 2.5 Hiệu năng theo kích thước pool (cell 06c, A2b, F1 / hit_ctx)

| Method | 1 ứng viên (444) | 2–3 (447) | 4–5 (230) | 6–10 (228) | >10 (279) |
|---|---|---|---|---|---|
| cgar | 46.0 / 79.3 | 44.7 / 85.7 | 42.6 / 81.7 | 34.5 / 74.1 | 20.7 / 73.1 |
| gnnrag_all | 45.9 / 79.3 | 44.2 / 85.9 | 40.6 / 83.0 | 35.4 / 80.7 | 33.4 / 87.5 |
| sbert_top5 | 46.0 / 79.3 | 44.7 / 85.7 | 41.2 / 78.7 | 32.6 / 71.9 | 19.1 / 73.8 |
| gnnrag_top5 | 46.0 / 79.3 | 44.5 / 85.9 | 40.5 / 79.1 | 30.6 / 71.1 | 17.0 / 69.2 |

Toàn bộ khoảng cách giữa CGAR và all nằm ở 17% câu có hơn 10 ứng viên: F1 20.7 so với 33.4, hit_ctx 73.1 so với 87.5. Ở nhóm 4–5 ứng viên, CGAR hơn all 2 điểm F1.

### 2.6 Calibration và selective answering (cell 07, 06c A4)

| WebQSP | ECE | Brier | AURC | SelAcc@50 | SelAcc@80 | overconf |
|---|---|---|---|---|---|---|
| gnnrag_all | 0.436 | 0.389 | 0.196 | 78.7 | 82.0 | 0.078 |
| sbert_top10 | 0.198 | 0.193 | 0.143 | 87.1 | 84.7 | 0.036 |
| sbert_top5 | 0.174 | 0.198 | 0.163 | 85.4 | 82.2 | 0.039 |
| cgar | 0.221 | 0.218 | **0.118** | **87.8** | **84.8** | 0.067 |

| CWQ | ECE | AURC | SelAcc@50 | SelAcc@80 |
|---|---|---|---|---|
| gnnrag_all | 0.298 | 0.288 | 67.8 | 65.6 |
| sbert_top10 | 0.050 | 0.322 | 68.5 | 65.9 |
| cgar | 0.147 | **0.238** | **76.7** | **67.0** |

hit_ctx theo coverage trên WebQSP: CGAR đạt 91.7% ở 20% câu tự tin nhất, gnnrag_all chỉ 80.4%. Score của gnnrag_all gần như không xếp hạng được câu đúng: hit_ctx ở 20% coverage thấp hơn ở 100% coverage.

### 2.7 Escalation offline CGAR → all (cell 06b, WebQSP)

| Escalate | Cost | F1 | hit_ctx |
|---|---|---|---|
| 0% (cgar) | 3.45 | 39.2 | 79.6 |
| 10% | 4.69 | 40.0 | 80.5 |
| 20% | 6.25 | 41.0 | 81.5 |
| 30% | 7.25 | 41.4 | 82.3 |
| 50% | 7.83 | 41.4 | 82.7 |
| 100% (all) | 8.37 | 41.1 | 83.2 |
| oracle 18% | 6.42 | **42.65** | 83.2 |

- Không có ngưỡng nào ở dưới 60% cost đạt F1 bằng all trừ 0.5 điểm: FAIL.
- Escalation 20% đạt F1 bằng all ở 75% cost. Oracle chỉ cần escalate 18% để vượt all 1.5 điểm. Khoảng trống giữa gate confidence và oracle là 1.7 điểm F1.

### 2.8 CGAR-G retrieval: `cgar_esc` có ngưỡng chọn trên dev (cell 08c, `b06d`)

Luật escalate: hơn 10 ứng viên hoặc confidence dưới τ. τ = 0.3148, là phân vị 20% trên dev 10% của train.

| Tập | Method | F1 | hit_ctx | Cost | Escalated |
|---|---|---|---|---|---|
| WebQSP | gnnrag_all | 41.1 | 83.2 | 8.37 | – |
| WebQSP | cgar | 39.1 | 79.5 | 3.45 | – |
| WebQSP | cgar_esc | **41.3** | 82.2 | 7.28 (87%) | 24.1% |
| CWQ | gnnrag_all | 21.6 | 65.8 | 11.84 | – |
| CWQ | cgar | 22.5 | 60.2 | 3.78 | – |
| CWQ | cgar_esc | 22.3 | 63.3 | 9.58 (81%) | 23.2% |

| cgar_esc − all | Δ (pp) | CI 95% | sig |
|---|---|---|---|
| WebQSP F1 | +0.23 | [−0.01, +0.45] | không |
| WebQSP hit_ctx | −0.98 | [−1.47, −0.55] | có |
| CWQ F1 | +0.69 | [+0.54, +0.85] | có |
| CWQ hit_ctx | −2.44 | [−2.95, −1.93] | có |

Gate: F1 PASS ở cả hai tập. hit_ctx PASS ở WebQSP, FAIL ở CWQ. Cost ≤ 80% FAIL ở cả hai tập. Cost thực tế cao hơn đường offline (7.28 so với 6.25 ở cùng khoảng 20% escalation) vì luật "hơn 10 ứng viên" chọn đúng nhóm câu có nhiều đường nhất.

### 2.9 Ablation và drift (cell 09, 10, `P8_KGQA_Bench_Colab_final.ipynb`, chạy lại 09/11/2026)

WebQSP test, n = 1628. Mỗi biến thể có train riêng được so với lần train S1 riêng của ablation (`b09_cgar_noRL`).

| Biến thể | F1 | Recall | Prec | hit_ctx | Cost | ECE | AURC | SelAcc@80 |
|---|---|---|---|---|---|---|---|---|
| CGAR chính (06, 3 seed) | 39.2 | 62.9 | 35.6 | 79.6 | 3.45 | 0.221 | 0.118 | 84.8 |
| S1 only (ablation) | 39.3 | 63.0 | 35.7 | 79.8 | 3.45 | 0.193 | 0.117 | 84.9 |
| S1 + RL (S2) | 39.2 | 62.8 | 35.5 | 79.4 | 3.45 | 0.205 | 0.115 | 83.9 |
| ngưỡng cố định 0 | 39.1 | 57.7 | 37.0 | 76.4 | 2.52 | 0.219 | 0.140 | 82.1 |
| ngưỡng cố định −1 | 39.3 | 58.6 | 37.0 | 77.0 | 2.68 | 0.219 | 0.137 | 82.5 |
| B1 learned-stop | 39.9 | 55.5 | **38.1** | 75.4 | 2.17 | 0.236 | 0.152 | 80.7 |
| B1, ngưỡng stop 0.3 | 39.4 | 53.9 | 38.1 | 74.9 | 1.99 | 0.237 | 0.156 | 80.1 |
| B1, ngưỡng stop 0.7 | **40.1** | 56.9 | 38.0 | 75.8 | 2.37 | 0.228 | 0.146 | 81.1 |
| B2 nhãn proof | 39.3 | 62.7 | 35.8 | 79.2 | 3.45 | **0.155** | 0.129 | 82.0 |
| B3 conf = hit | 38.6 | 62.2 | 34.4 | 79.4 | 3.44 | 0.173 | **0.109** | **85.5** |
| v3 = B1 + B2 + B3 | 38.2 | 54.1 | 35.6 | 73.4 | 2.28 | 0.194 | 0.127 | 82.6 |

| Biến thể − S1 only | Δ F1 (pp) | CI | Δ hit_ctx (pp) | CI |
|---|---|---|---|---|
| S1 + RL | −0.11 | [−0.28, +0.06] | −0.43 | [−0.92, 0.00] |
| B1 learned-stop | +0.54 | [−0.07, +1.16] | −4.42 | [−5.47, −3.38] |
| B2 nhãn proof | −0.03 | [−0.32, +0.27] | −0.55 | [−1.47, +0.37] |
| B3 conf = hit | −0.70 | [−0.98, −0.43] | −0.37 | [−1.11, +0.31] |
| v3 | −1.14 | [−1.77, −0.54] | −6.39 | [−7.74, −5.10] |

Các biến thể dùng chung checkpoint, so từng cặp với chính checkpoint đó:

| Biến thể − tham chiếu | Cost | Δ F1 (pp) | CI | Δ hit_ctx (pp) | CI |
|---|---|---|---|---|---|
| ngưỡng cố định 0 − CGAR chính | 2.52 vs 3.45 | −0.01 | [−0.53, +0.50] | −3.07 | [−3.99, −2.15] |
| ngưỡng cố định −1 − CGAR chính | 2.68 vs 3.45 | +0.21 | [−0.27, +0.68] | −2.46 | [−3.32, −1.66] |
| learned-stop ngưỡng 0.3 − ngưỡng 0.5 | 1.98 vs 2.17 | −0.43 | [−0.67, −0.19] | −0.49 | [−0.86, −0.18] |
| learned-stop ngưỡng 0.7 − ngưỡng 0.5 | 2.37 vs 2.17 | +0.23 | [0.00, +0.46] | +0.43 | [+0.12, +0.80] |

Gate: RL không cải thiện PASS; nhãn proof không giảm precision PASS; conf = hit giảm ECE PASS; learned-stop giảm cost mà giữ hit_ctx tốt hơn ngưỡng cố định FAIL (75.4 so với 76.4).

Đọc kết quả:

- **RL không cần thiết.** S1 + RL kém S1 0.11 điểm F1, không có ý nghĩa, và kém 0.43 điểm hit_ctx sát ngưỡng ý nghĩa.
- **Learned-stop (B1) là cách duy nhất làm controller dừng sớm thật.** Cost giảm 37% (3.45 xuống 2.17), precision tăng 2.4 điểm, F1 tăng 0.5 điểm (không có ý nghĩa). Đổi lại hit_ctx mất 4.4 điểm có ý nghĩa. Với reader, hit_ctx quan trọng hơn F1 retrieval, nên B1 không đáng dùng làm bản chính. Ngưỡng stop 0.7 cho F1 cao nhất bảng (40.1) ở cost 2.37.
- **Learned-stop không thắng ngưỡng cố định.** Ở cost gần nhau, ngưỡng cố định 0 (cost 2.52) giữ hit_ctx 76.4, còn learned-stop ngưỡng 0.7 (cost 2.37) giữ 75.8. Controller học dừng chưa tốt hơn một luật cắt đơn giản.
- **Mọi cách cắt cost đều đổi F1 lấy hit_ctx.** Ngưỡng cố định giảm cost 22 đến 27% mà F1 không đổi, nhưng mất 2.5 đến 3.1 điểm hit_ctx có ý nghĩa. F1 giữ được nhờ precision tăng, còn đáp án bị đẩy ra khỏi context ở nhiều câu hơn. Vì reader cần đáp án có trong context, F1 retrieval không đủ để đánh giá các biến thể giảm cost.
- **Ngưỡng learned-stop đi đúng hướng của một đường Pareto.** Từ ngưỡng 0.3 lên 0.7, cost tăng từ 1.98 lên 2.37, và F1 lẫn hit_ctx cùng tăng có ý nghĩa.
- **Nhãn proof (B2)** giảm ECE từ 0.193 xuống 0.155, không mất F1. Đây là cải tiến calibration rẻ nhất.
- **conf = hit (B3)** cho AURC thấp nhất bảng (0.109) và SelAcc@80 cao nhất (85.5), nhưng mất 0.7 điểm F1 có ý nghĩa. Nếu mục tiêu chính là selective answering thì B3 đáng cân nhắc.
- **Gộp cả ba (v3) tệ hơn từng phần**: mất 6.4 điểm hit_ctx. Tác dụng phụ của learned-stop lấn át lợi ích calibration. Việc cell 03 loại v3 trên dev là đúng.

Drift, 600 câu, checkpoint CGAR chính:

| Kiểu drift | 0% | 10% | 30% | 50% |
|---|---|---|---|---|
| relation_oov | 37.6 | 37.6 | 37.7 | 37.9 |
| score_noise | 37.6 | 37.6 | 37.4 | 37.3 |
| mixed | 37.6 | 37.4 | 37.5 | 37.6 |

F1 ở 50% quan hệ lạ còn tăng 0.28 điểm. Gate bền với drift PASS. Lý do: CGAR mã hoá quan hệ bằng SBERT trên tên quan hệ, nên không phụ thuộc từ điển quan hệ lúc train. Lưu ý: đường cong phẳng cũng cho thấy drift kiểu này quá nhẹ để phân biệt các phương pháp.

## 3. Reader end-to-end (cell 08, 08c, 08d)

1000 câu WebQSP test. "verify" nghĩa là chỉ giữ entity trùng ứng viên, rỗng thì lấy ứng viên đầu. "cand" là prompt có danh sách ứng viên.

### 3.1 Qwen2.5-7B

| Cấu hình | Method | Cost | llm_hit | llm_hit@1 | llm_f1 | abstain |
|---|---|---|---|---|---|---|
| Không retrieval, chat | – | 0 | 0.474 | 0.390 | 0.283 | 0.002 |
| Không retrieval, LoRA | – | 0 | 0.412 | 0.384 | 0.273 | – |
| chat | all | 8.48 | 0.765 | 0.716 | 0.625 | 0.027 |
| chat | cgar | 3.46 | 0.757 | 0.723 | 0.583 | 0.025 |
| chat | sbert_top5 | 3.46 | 0.755 | 0.694 | 0.577 | 0.018 |
| grounded | all | 8.48 | 0.640 | 0.606 | 0.532 | 0.287 |
| grounded | cgar | 3.46 | 0.620 | 0.602 | 0.493 | 0.306 |
| chat+verify | all | 8.48 | 0.770 | 0.728 | 0.627 | 0.027 |
| chat+verify | cgar | 3.46 | 0.764 | 0.737 | 0.589 | 0.025 |
| cand+verify | all | 8.48 | 0.789 | 0.725 | 0.632 | 0.001 |
| cand+verify | cgar | 3.46 | 0.773 | 0.738 | 0.594 | 0.001 |
| cand+verify | cgar_esc | 7.42 | 0.791 | 0.741 | 0.630 | 0.001 |
| LoRA cand+verify | all | 8.48 | **0.800** | **0.761** | **0.677** | 0.000 |
| LoRA cand+verify | cgar | 3.46 | 0.779 | 0.739 | 0.623 | 0.001 |
| LoRA cand+verify | cgar_esc | 7.42 | 0.797 | 0.751 | 0.666 | 0.001 |

### 3.2 Llama-3.1-8B

| Cấu hình | Method | Cost | llm_hit | llm_hit@1 | llm_f1 | abstain |
|---|---|---|---|---|---|---|
| Không retrieval, chat | – | 0 | 0.588 | 0.477 | 0.354 | 0.001 |
| Không retrieval, LoRA | – | 0 | 0.446 | 0.413 | 0.298 | – |
| chat | all | 8.48 | 0.805 | 0.735 | 0.630 | 0.018 |
| chat | cgar | 3.46 | 0.772 | 0.718 | 0.588 | 0.025 |
| chat | sbert_top5 | 3.46 | 0.768 | 0.680 | 0.574 | 0.025 |
| grounded | all | 8.48 | 0.751 | 0.611 | 0.588 | 0.135 |
| grounded | cgar | 3.46 | 0.727 | 0.581 | 0.539 | 0.161 |
| chat+verify | all | 8.48 | 0.798 | 0.749 | 0.639 | 0.018 |
| chat+verify | cgar | 3.46 | 0.769 | 0.738 | 0.599 | 0.025 |
| cand+verify | all | 8.48 | 0.804 | 0.702 | 0.636 | 0.024 |
| cand+verify | cgar | 3.46 | 0.785 | 0.731 | 0.592 | 0.028 |
| cand+verify | cgar_esc | 7.42 | 0.804 | 0.735 | 0.633 | 0.024 |
| LoRA cand+verify | all | 8.48 | **0.810** | **0.773** | **0.692** | 0.000 |
| LoRA cand+verify | cgar | 3.46 | 0.778 | 0.744 | 0.626 | 0.000 |
| LoRA cand+verify | cgar_esc | 7.42 | 0.800 | 0.761 | 0.676 | 0.000 |

Mistral-7B, chat, run cũ: all 0.806 / f1 0.609, cgar 0.800 / f1 0.564.

### 3.3 So sánh từng cặp ở mức reader (llm_hit)

| Cặp | Reader | Δ (pp) | CI 95% | sig |
|---|---|---|---|---|
| all chat − không retrieval chat | Qwen | +29.1 | [+25.7, +32.5] | có |
| all chat − không retrieval chat | Llama | +21.7 | [+18.4, +25.0] | có |
| cgar − all, chat | Qwen | −0.80 | [−2.31, +0.70] | không |
| cgar − all, chat | Llama | −3.31 | [−5.02, −1.71] | có |
| cgar − all, LoRA cand+verify | Qwen | −2.11 | [−3.31, −1.00] | có |
| cgar − all, LoRA cand+verify | Llama | −3.21 | [−4.52, −2.01] | có |
| cgar_esc − all, cand+verify | Qwen | +0.20 | [−0.70, +1.10] | không |
| cgar_esc − all, cand+verify | Llama | 0.00 | [−0.90, +0.90] | không |
| cgar_esc − all, LoRA cand+verify | Qwen | −0.30 | [−1.00, +0.40] | không |
| cgar_esc − all, LoRA cand+verify | Llama | −1.00 | [−1.71, −0.40] | có |
| cgar_esc cand+verify − all chat (GNN-RAG gốc) | Qwen | +2.61 | [+0.90, +4.32] | có |
| cgar_esc cand+verify − all chat | Llama | −0.10 | [−2.11, +2.01] | không |
| cgar_esc LoRA − all chat | Qwen | +3.21 | [+1.41, +5.02] | có |
| cgar_esc LoRA − all chat | Llama | −0.50 | [−2.51, +1.51] | không |

### 3.4 Huấn luyện LoRA (cell 08d)

- Dữ liệu SFT: 4981 ví dụ từ 2848 câu train, gồm 2499 ví dụ context của all và 2482 ví dụ context của cgar. Trung bình 2.85 đáp án mỗi ví dụ. Chỉ giữ ví dụ có ít nhất một gold trong danh sách ứng viên.
- Cấu hình: r 16, alpha 32, q/k/v/o, 2 epoch, lr 1e-4, batch 4 × grad-accum 4, 623 bước tối ưu, khoảng 35 phút mỗi model.
- Loss: Qwen từ 0.25 xuống 0.02, Llama từ 0.16 xuống 0.012. Cảnh báo OOM trong log là bộ cấp phát thử lại, không làm hỏng run.

## 4. Hallucination / groundedness (P9)

### 4.1 Định nghĩa ngắn

- **Claim** là một entity trong câu trả lời.
- **SUPPORTED:** entity là đích của một đường từ topic entity có chuỗi quan hệ khớp proof tham chiếu, hoặc là gold.
- **UNSUPPORTED:** không có trong subgraph (absent), có trong subgraph nhưng không được retrieve, hoặc có trong context nhưng chỉ là node trung gian.
- **UNCERTAIN:** là đích của một đường nhưng chuỗi quan hệ khác mọi proof.
- **QHR** (question hallucination rate): tỷ lệ câu đã trả lời có ít nhất một claim UNSUPPORTED. **strict-QHR** tính cả UNCERTAIN.
- **grounded-correct:** đúng và có đường proof. Đây là chỉ số "đúng mà không bịa".

### 4.2 Bảng chính theo cấu hình (method all và các method CGAR)

| Reader | Cấu hình | Method | QHR | strict-QHR | grounded-correct | AURC-halluc | AURC-incorrect |
|---|---|---|---|---|---|---|---|
| Qwen | chat | all | 0.114 | 0.241 | 0.722 | 0.136 | 0.157 |
| Qwen | chat | cgar | 0.120 | 0.236 | 0.714 | **0.082** | **0.105** |
| Qwen | grounded | all | 0.027 | 0.152 | 0.630 | 0.017 | 0.110 |
| Qwen | grounded | cgar | 0.014 | 0.123 | 0.617 | 0.015 | 0.068 |
| Qwen | chat+verify | all | 0.002 | 0.195 | 0.769 | ~0 | 0.165 |
| Qwen | chat+verify | cgar | 0.001 | 0.192 | 0.764 | ~0 | 0.089 |
| Qwen | cand+verify | all | 0.002 | 0.254 | 0.788 | ~0 | 0.210 |
| Qwen | cand+verify | cgar_esc | 0.002 | 0.245 | 0.790 | ~0 | 0.119 |
| Qwen | LoRA cand+verify | all | 0.002 | 0.234 | **0.799** | ~0 | 0.207 |
| Qwen | LoRA cand+verify | cgar | 0.001 | 0.248 | 0.779 | ~0 | 0.125 |
| Qwen | LoRA cand+verify | cgar_esc | 0.002 | 0.236 | 0.796 | ~0 | 0.120 |
| Llama | chat | all | 0.140 | 0.283 | 0.744 | 0.186 | 0.176 |
| Llama | chat | cgar | 0.152 | 0.275 | 0.713 | **0.084** | **0.119** |
| Llama | grounded | all | 0.186 | 0.299 | 0.629 | 0.183 | 0.128 |
| Llama | grounded | cgar | 0.194 | 0.313 | 0.593 | 0.174 | 0.085 |
| Llama | chat+verify | all | 0.002 | 0.240 | 0.796 | ~0 | 0.197 |
| Llama | chat+verify | cgar | 0.001 | 0.233 | 0.768 | ~0 | 0.106 |
| Llama | cand+verify | all | 0.001 | 0.261 | 0.804 | ~0 | 0.190 |
| Llama | cand+verify | cgar_esc | 0.001 | 0.257 | 0.804 | ~0 | 0.101 |
| Llama | LoRA cand+verify | all | 0.002 | 0.231 | **0.809** | ~0 | 0.204 |
| Llama | LoRA cand+verify | cgar | 0.001 | 0.242 | 0.778 | ~0 | 0.127 |
| Llama | LoRA cand+verify | cgar_esc | 0.002 | 0.228 | 0.799 | ~0 | 0.121 |
| Mistral | chat (cũ) | all | 0.316 | 0.423 | 0.616 | 0.277 | 0.177 |
| Mistral | chat (cũ) | cgar | 0.327 | 0.417 | 0.611 | 0.216 | 0.133 |

Mọi cấu hình có verify đều có QHR ≤ 0.002. Đây là hệ quả cấu trúc: verify và verifier dùng cùng luật đối chiếu ứng viên.

### 4.3 CGAR so với all, cùng reader và cùng cấu hình (P9 cell 07)

| Reader, cấu hình | Δ QHR (pp) | CI | Δ grounded-correct (pp) | CI |
|---|---|---|---|---|
| Qwen chat | +0.40 | [−1.20, +2.01] | −0.80 | [−2.41, +0.80] |
| Llama chat | +1.10 | [−0.70, +2.91] | −3.11 | [−4.82, −1.50] |
| Mistral chat | +1.10 | [−1.51, +3.71] | −0.50 | [−3.01, +2.01] |
| Qwen grounded | −0.90 | [−1.71, −0.10] | −1.20 | [−2.71, +0.30] |
| Llama grounded | +0.20 | [−1.81, +2.21] | −3.51 | [−5.72, −1.31] |
| Qwen chat+verify | −0.10 | [−0.30, 0.00] | −0.50 | [−1.81, +0.80] |
| Llama chat+verify | −0.10 | [−0.30, 0.00] | −2.81 | [−4.22, −1.41] |
| Qwen cand+verify | −0.10 | [−0.30, 0.00] | −1.51 | [−2.91, −0.10] |
| Llama cand+verify | 0.00 | [0.00, 0.00] | −1.91 | [−3.31, −0.60] |
| Qwen LoRA | −0.10 | [−0.30, 0.00] | −2.01 | [−3.21, −0.90] |
| Llama LoRA | −0.10 | [−0.30, 0.00] | −3.11 | [−4.42, −1.91] |
| **cgar_esc**, Qwen cand+verify | 0.00 | – | +0.20 | [−0.70, +1.10] |
| **cgar_esc**, Llama cand+verify | 0.00 | – | 0.00 | [−0.90, +0.90] |
| **cgar_esc**, Qwen LoRA | 0.00 | – | −0.30 | [−1.00, +0.40] |
| **cgar_esc**, Llama LoRA | 0.00 | – | −1.00 | [−1.71, −0.40] |

Tham chiếu: gnnrag_top5 − all, Llama chat, QHR +1.61 [+0.20, +3.01] có ý nghĩa. Cắt đường theo score GNN làm tăng bịa, CGAR thì không.

### 4.4 Phân loại claim theo nguồn chứng cứ (chat, P9 cell 03)

| Method (chat, gộp 3 reader) | SUPPORTED | UNCERTAIN | UNSUPPORTED absent | UNSUPPORTED trung gian | UNSUPPORTED không retrieve |
|---|---|---|---|---|---|
| gnnrag_all | 8120 | 1127 | 378 | 371 | 450 |
| cgar | 5214 | 848 | 486 | 369 | 475 |

- Proxy cũ "entity có trong context" sai ở 4415 claim: bỏ sót 1122 claim bịa nằm trong context, và báo oan 3293 claim đúng qua alias. Proxy cũ không dùng được để báo cáo.
- Sau verify, các nhóm UNSUPPORTED về khoảng 2 đến 8 claim mỗi method. UNCERTAIN giữ nguyên khoảng 800 đến 1030 claim.

### 4.5 Error taxonomy (số câu trên 996, P9 cell 05)

| Reader, cấu hình | Method | ok_grounded | missing_kg | cand_omission | path_selection | reasoning | correct_ungrounded | abstained |
|---|---|---|---|---|---|---|---|---|
| Qwen chat | all | 719 | 55 | 40 | 0 | 34 | 43 | 85 |
| Qwen chat | cgar | 711 | 56 | 36 | 13 | 29 | 43 | 93 |
| Qwen chat | gnnrag_top5 | 694 | 58 | 39 | 30 | 19 | 54 | 86 |
| Llama chat | all | 741 | 76 | 39 | 0 | 28 | 61 | 29 |
| Llama chat | cgar | 710 | 71 | 47 | 23 | 34 | 59 | 36 |
| Mistral chat | all | 614 | 83 | 40 | 0 | 25 | 189 | 5 |
| Qwen cand+verify | cgar | 770 | 109 | 57 | 30 | 27 | 0 | 3 |
| Qwen cand+verify | cgar_esc | 787 | 108 | 56 | 4 | 36 | 1 | 4 |
| Qwen LoRA | all | 796 | 108 | 58 | 0 | 33 | 1 | 0 |
| Qwen LoRA | cgar_esc | 793 | 109 | 58 | 4 | 30 | 1 | 1 |
| Llama LoRA | all | 806 | 109 | 58 | 0 | 22 | 1 | 0 |
| Llama LoRA | cgar_esc | 796 | 110 | 58 | 4 | 27 | 1 | 0 |

`missing_kg` là câu mà đáp án không có trong pool; `cand_omission` là đáp án có trong context nhưng reader bỏ sót. Dưới verify, `correct_ungrounded` (đúng nhờ kiến thức tham số) chuyển thành `missing_kg` vì entity ngoài ứng viên bị loại.

### 4.6 Risk gate theo confidence của retriever (P9 cell 06)

Tỷ lệ sai khi CGAR chỉ trả lời 50% câu tự tin nhất, so với all trả lời 100%:

| Reader, cấu hình | halluc CGAR@50% | halluc all@100% | incorrect CGAR@50% | incorrect all@100% |
|---|---|---|---|---|
| Qwen chat | 0.080 | 0.114 | 0.122 | 0.164 |
| Llama chat | 0.094 | 0.140 | 0.135 | 0.171 |
| Mistral chat | 0.222 | 0.316 | 0.145 | 0.190 |
| Qwen LoRA | 0.000 | 0.002 | 0.135 | 0.200 |
| Llama LoRA | 0.000 | 0.002 | 0.135 | 0.190 |

### 4.7 Escalation ở mức hallucination (P9 cell 09)

- Dưới prompt chat, escalation không giảm QHR. Qwen: 0.120 ở 0% escalate, 0.114 ở 100%. Llama: 0.152 xuống 0.140.
- Escalation tăng grounded-correct đơn điệu. Llama chat: từ 0.713 lên 0.744.
- Dưới LoRA cand+verify, grounded-correct đạt mức all khi escalate khoảng 50% (Qwen 0.798, cost 7.86) hoặc 70% (Llama 0.808, cost 8.34).

### 4.8 Gate tổng kết P9 (cell 12, `P9_Hallucination_Eval_Colab_final.ipynb`, 09/11/2026)

Với cấu hình có verify, QHR luôn ≈ 0 nên gate dùng strict-QHR (ngưỡng: không cao hơn all quá 1 điểm) và AURC-incorrect.

| Reader, cấu hình | Method | strict-QHR so với all | Gate strict | AURC-incorrect so với all | Δ grounded-correct (pp) |
|---|---|---|---|---|---|
| Llama cand+verify | cgar | 0.272 vs 0.261 | FAIL | 0.105 vs 0.190 | −1.91 sig |
| Llama cand+verify | cgar_esc | 0.257 vs 0.261 | PASS | 0.101 vs 0.190 | 0.00 |
| Qwen cand+verify | cgar | 0.252 vs 0.254 | PASS | 0.125 vs 0.210 | −1.51 sig |
| Qwen cand+verify | cgar_esc | 0.245 vs 0.254 | PASS | 0.119 vs 0.210 | +0.20 |
| Llama LoRA | cgar | 0.242 vs 0.231 | FAIL | 0.127 vs 0.204 | −3.11 sig |
| Llama LoRA | cgar_esc | 0.228 vs 0.231 | PASS | 0.121 vs 0.204 | −1.00 sig |
| Qwen LoRA | cgar | 0.248 vs 0.234 | FAIL | 0.125 vs 0.207 | −2.01 sig |
| Qwen LoRA | cgar_esc | 0.236 vs 0.234 | PASS | 0.120 vs 0.207 | −0.30 |
| Llama chat+verify | cgar | 0.233 vs 0.240 | PASS | 0.106 vs 0.197 | −2.81 sig |
| Qwen chat+verify | cgar | 0.192 vs 0.195 | PASS | 0.089 vs 0.165 | −0.50 |

Dòng tóm tắt CGAR-G so với GNN-RAG gốc (all, prompt chat), cùng reader:

| Reader | Cấu hình cgar_esc | grounded-correct | QHR | strict-QHR | llm_hit |
|---|---|---|---|---|---|
| Qwen | cand+verify | 0.790 vs 0.722 | 0.002 vs 0.114 | 0.245 vs 0.241 | 0.791 vs 0.765 |
| Qwen | LoRA cand+verify | 0.796 vs 0.722 | 0.002 vs 0.114 | 0.236 vs 0.241 | 0.797 vs 0.765 |
| Llama | cand+verify | 0.804 vs 0.744 | 0.001 vs 0.140 | 0.257 vs 0.283 | 0.804 vs 0.805 |
| Llama | LoRA cand+verify | 0.799 vs 0.744 | 0.002 vs 0.140 | 0.228 vs 0.283 | 0.800 vs 0.805 |

Đọc kết quả:

- **AURC-incorrect của CGAR thấp hơn all ở cả 10 dòng có verify**, khoảng 0.09 đến 0.13 so với 0.17 đến 0.21. Đây là gate duy nhất đạt ở mọi cấu hình và mọi method.
- **cgar không escalation có strict-QHR cao hơn all quá 1 điểm ở 3 trên 6 cấu hình có prompt ứng viên** (Llama cand+verify, Llama LoRA, Qwen LoRA). Khi danh sách ứng viên ngắn và thiếu đáp án đúng, reader vẫn phải chọn trong danh sách, nên chọn trúng một entity sai quan hệ. Với prompt chat+verify thì không có hiện tượng này, vì reader còn được trả lời ngoài danh sách rồi bị verify lọc.
- **cgar_esc đạt gate strict-QHR ở cả 4 cấu hình.** Escalation sửa được đúng nhóm câu mà danh sách 5 đường thiếu đáp án.
- **So với GNN-RAG gốc, CGAR-G tăng grounded-correct 5.5 đến 7.4 điểm** và đưa QHR về 0.002 ở cả hai reader. strict-QHR giảm rõ với Llama (0.283 xuống 0.228 khi dùng LoRA), còn với Qwen thì gần như không đổi (0.241 so với 0.236).
- **Tổng số gate FAIL là 11.** Có 7 gate grounded-correct của cgar, 1 gate grounded-correct của cgar_esc với Llama LoRA, và 3 gate strict-QHR của cgar.

## 5. Bảng claim: PASS / FAIL

| Claim | Kết quả | Chứng cứ |
|---|---|---|
| Retrieval bằng pool GNN giúp reader so với không retrieval | PASS | +29.1 pp (Qwen), +21.7 pp (Llama) llm_hit, CI không chứa 0 |
| CGAR tốt nhất trong các retriever budget 5 về F1 | PASS | +0.75 so với sbert_top5, +1.54 so với gnnrag_top5, sig |
| CGAR tốt hơn sbert_top5 về hit_ctx | FAIL | +0.61, CI chứa 0 |
| CGAR đạt chất lượng all ở cost thấp hơn (WebQSP) | FAIL | F1 −1.88, hit_ctx −3.62, sig |
| CGAR F1 cao hơn all trên CWQ | PASS (có lưu ý) | +1.11 sig, nhưng hit_ctx −5.24 |
| Controller adaptive thật | FAIL | dừng sớm 0%, cost do pool |
| Confidence CGAR xếp hạng rủi ro tốt nhất | PASS | AURC 0.118 so với 0.143 đến 0.211; lặp lại ở mọi reader và prompt |
| RL (Stage 2) cần thiết | FAIL | S1 + RL kém S1 0.11 pp F1, n.s. |
| Learned-stop giảm cost mà giữ hit_ctx tốt hơn ngưỡng cố định | FAIL | cost 2.17 nhưng hit_ctx −4.4 pp; thua ngưỡng 0 cùng cost |
| Ngưỡng cố định giảm cost mà không mất chất lượng | FAIL một phần | F1 không đổi nhưng hit_ctx −2.5 đến −3.1 pp có ý nghĩa ở 73 đến 78% cost |
| Nhãn proof cải thiện calibration | PASS | ECE 0.193 xuống 0.155, F1 không đổi |
| CGAR bền với quan hệ lạ | PASS | F1 +0.28 pp ở 50% OOV |
| CGAR không làm reader bịa nhiều hơn all | PASS | 11/11 cấu hình, Δ QHR không có ý nghĩa dương |
| CGAR làm reader bịa ít hơn all | FAIL | Δ QHR quanh 0 ở mọi cấu hình chưa verify |
| cgar (budget 5) không tăng strict-QHR quá 1 pp dưới prompt ứng viên | FAIL | 3/6 cấu hình vượt: 0.272 vs 0.261, 0.242 vs 0.231, 0.248 vs 0.234 |
| cgar_esc không tăng strict-QHR quá 1 pp | PASS | 4/4 cấu hình |
| AURC-incorrect của CGAR thấp hơn all ở mọi cấu hình verify | PASS | 10/10, 0.09 đến 0.13 so với 0.17 đến 0.21 |
| Prompt grounded giảm bịa mà không mất độ đúng | FAIL | Qwen mất 9 pp grounded-correct, abstain 29%; Llama QHR tăng |
| Verify hậu kỳ giảm bịa mà không mất độ đúng | PASS | QHR về 0.002, grounded-correct +4.7 (Qwen), +5.2 (Llama) |
| cgar_esc bằng all về chất lượng retrieval | PASS (WebQSP) | F1 +0.23 n.s., hit_ctx −0.98 |
| cgar_esc ở cost ≤ 80% all | FAIL | 87% WebQSP, 81% CWQ |
| cgar_esc bằng all cùng reader zero-shot cand+verify | PASS | Δ +0.20 và 0.00, CI ±0.9 |
| cgar_esc bằng all cùng reader LoRA | PASS Qwen, FAIL Llama | −0.30 n.s.; −1.00 sig |
| CGAR-G vượt GNN-RAG gốc (all, chat) về llm_hit | PASS Qwen, FAIL Llama | +3.21 sig; −0.50 n.s. |
| CGAR-G vượt GNN-RAG gốc về grounded-correct và QHR | PASS | 0.796 so với 0.722 (Qwen), 0.799 so với 0.744 (Llama); QHR 0.002 so với 0.11 đến 0.14 |
| LoRA reader giảm phần bịa còn lại (strict-QHR) | PASS | −2 đến −3 pp so với zero-shot cand+verify |

## 6. Phân tích chi tiết

### 6.1 Giới hạn của bài toán chọn đường

Pool ReaRev đã rất gọn, trung bình 5.6 ứng viên và 8.4 đường. 58.8% câu có ít đường hơn budget 5, nên ở các câu này mọi phương pháp chọn đường cho kết quả gần giống nhau. Phần khác biệt duy nhất nằm ở 17% câu có hơn 10 ứng viên. Ở đó cắt về 5 đường làm mất 14 điểm hit_ctx, và không phương pháp xếp hạng nào bù được: CGAR 73.1, sbert_top5 73.8, gnnrag_top5 69.2, so với all 87.5. Hệ quả là claim "ít đường hơn mà vẫn bằng" chỉ có thể đến từ escalation, không đến từ xếp hạng tốt hơn.

### 6.2 Giá trị thật của CGAR là confidence, không phải cost

Ba quan sát độc lập cùng chỉ về một hướng:

1. AURC retrieval của CGAR là 0.118, thấp nhất trong mọi method, trên cả WebQSP và CWQ.
2. Ở mức reader, AURC-incorrect của cgar và cgar_esc là 0.10 đến 0.13, còn của all là 0.16 đến 0.21. Kết quả này đúng cho cả 3 reader và cả 5 cấu hình prompt.
3. Ở mức hallucination dưới prompt chat, AURC-hallucination của CGAR là 0.08, của all là 0.14 đến 0.19. Trả lời 50% câu tự tin nhất theo CGAR có tỷ lệ bịa thấp hơn all trả lời 100% câu, ở cả 3 reader.

Score của GNN-RAG không phải xác suất (ECE 0.436) và gần như không xếp hạng được câu đúng. CGAR học được một tín hiệu "đường đã chọn có chứa đáp án không" dùng được cho selective answering.

Điểm này quan trọng hơn khi reader đã fine-tune. Reader LoRA không bao giờ abstain (abstention 0.000), nên toàn bộ việc kiểm soát rủi ro phải dựa vào tín hiệu bên ngoài reader. CGAR cung cấp đúng tín hiệu đó.

### 6.3 Escalation: gần đúng nhưng chưa rẻ

Đường offline cho thấy escalate 20% câu có confidence thấp nhất đã đạt F1 của all ở 75% cost. Oracle chỉ cần 18% để vượt all 1.5 điểm. Nhưng khi chọn ngưỡng trên dev và thêm luật "hơn 10 ứng viên", tỷ lệ escalate thực tế lên 24% và cost lên 87%. Lý do: câu nhiều ứng viên cũng là câu có nhiều đường nhất, nên mỗi lần escalate đắt hơn trung bình. Luật đếm ứng viên bắt đúng câu khó nhưng phá mục tiêu cost.

Khoảng cách 1.7 điểm F1 giữa gate confidence và oracle là dư địa cho một gate tốt hơn. Đây là hướng cải tiến có cơ sở nhất cho CGAR.

### 6.4 Reader càng mạnh, cắt đường càng thiệt

| Khoảng cách cgar − all (llm_hit, pp) | Qwen | Llama |
|---|---|---|
| Retrieval hit_ctx | −2.9 | −2.9 |
| Reader chat | −0.8 (n.s.) | −3.3 |
| Reader cand+verify | −1.6 | −1.9 |
| Reader LoRA cand+verify | −2.1 | −3.2 |

Với Qwen zero-shot, reader không tận dụng hết đường thừa nên cắt đường gần như miễn phí. Llama và mọi reader LoRA tận dụng được, nên khoảng cách quay lại bằng khoảng cách retrieval. Escalation thu hẹp lại còn −0.3 (Qwen) và −1.0 (Llama). Hệ quả cho bài báo: claim "ít đường hơn mà vẫn bằng" yếu dần khi reader mạnh lên, và cần nói rõ điều này.

### 6.5 Ba cách giảm bịa: chỉ một cách không tốn độ đúng

- **Prompt grounded** buộc reader nói "unknown". Qwen giảm QHR từ 0.114 xuống 0.027 nhưng abstain 29% và mất 9 điểm grounded-correct. Llama còn bịa nhiều hơn (0.140 lên 0.186): nó viết câu từ chối rồi chèn kiến thức tham số ("However, based on general knowledge..."). Không dùng được.
- **Verify hậu kỳ** loại entity không nằm trong ứng viên. QHR về 0.002 và grounded-correct tăng 4.7 đến 5.2 điểm, llm_hit gần như giữ nguyên. Đây là cách rẻ nhất, không cần GPU, áp dụng lại được cho mọi run cũ.
- **Reader LoRA** học chọn ứng viên. Nó tăng F1 4.5 đến 5.6 điểm và hit@1 (Llama từ 0.702 lên 0.773). Nó giảm strict-QHR 2 đến 3 điểm so với cand+verify zero-shot, vì reader liệt kê ít ứng viên sai quan hệ hơn. Với Qwen, strict-QHR thấp nhất lại là chat+verify (0.195), thấp hơn LoRA (0.234).

Prompt cand một mình tăng llm_hit 2.4 điểm cho Qwen, không tăng cho Llama, và làm strict-QHR tăng vì reader liệt kê nhiều ứng viên hơn (3.2 đến 3.6 claim mỗi câu).

### 6.6 Phần bịa còn lại sau verify

Strict-QHR vẫn 0.23 đến 0.26. Toàn bộ là claim UNCERTAIN: entity là đích của một đường thật trong KG nhưng chuỗi quan hệ khác proof tham chiếu. Ví dụ trong P9: "Politician", "Lawyer" cho câu "James K. Polk làm gì trước khi làm tổng thống", hay "Apache trout" cho "hoa bang Arizona". Một phần là bịa thật (sai quan hệ), một phần là gold thiếu hoặc proof tham chiếu quá hẹp. Chỉ nhãn người mới tách được hai loại này.

### 6.7 Kiến thức tham số và "đúng mà không có chứng cứ"

- Không retrieval: Llama 0.588, Qwen 0.474. Llama nhớ nhiều hơn.
- Câu đúng nhờ kiến thức tham số dưới prompt chat (`correct_ungrounded`): Qwen 43, Llama 61, Mistral 189. Mistral có llm_hit cao (0.806) nhưng grounded-correct chỉ 0.616, nghĩa là 19% câu "đúng" của Mistral không có đường đi nào làm chứng.
- llm_hit đơn thuần thưởng cho việc đoán đúng từ trí nhớ. Vì vậy grounded-correct mới là chỉ số chính cho đề tài giảm hallucination.
- Reader LoRA với prompt không có ứng viên chỉ đạt 0.41 đến 0.45, thấp hơn bản gốc. Fine-tune làm reader phụ thuộc vào danh sách ứng viên. Đây là đánh đổi chấp nhận được cho hệ RAG, nhưng không dùng con số này làm baseline "chỉ LLM".

### 6.8 Lỗi còn lại nằm ở pool, không ở reader

Dưới LoRA cand+verify, mỗi method còn khoảng 200 câu sai. Trong đó khoảng 109 câu là đáp án không có trong pool, 58 câu là đáp án có trong context nhưng reader bỏ sót, và 22 đến 33 câu là lỗi suy luận. Lỗi chọn đường của cgar_esc chỉ còn 4 câu. Muốn tăng tiếp phải cải thiện giai đoạn GNN hoặc subgraph: trần pool 82.6%, trần subgraph 88.7%.

## 7. Câu chuyện nên viết

1. **Đóng góp chính là pipeline giảm bịa cho GraphRAG:** verify theo ứng viên cộng reader LoRA. GNN-RAG từ grounded-correct 0.72 đến 0.74 lên 0.80 đến 0.81. QHR từ 0.11 đến 0.14 xuống 0.002. llm_hit không giảm. strict-QHR giảm 5.2 điểm với Llama nhưng chỉ 0.7 điểm với Qwen; với Qwen, chat+verify giảm strict-QHR nhiều hơn (4.6 điểm) nhưng grounded-correct thấp hơn LoRA 3 điểm.
2. **CGAR là bộ ước lượng rủi ro cho pipeline đó.** Nó không làm reader bịa nhiều hơn dù đưa ít evidence hơn. Confidence của nó xếp hạng câu sai tốt nhất ở mọi reader. Nó đặc biệt cần khi reader đã fine-tune không còn tự abstain.
3. **cgar_esc** đạt chất lượng của all ở 87% cost với reader zero-shot. Với reader LoRA thì bằng all ở Qwen và kém 1 điểm ở Llama. Trình bày như một điểm trên đường Pareto, không phải "vượt GNN-RAG". Khi dùng prompt ứng viên, escalation là bắt buộc: cgar budget 5 làm strict-QHR tăng quá 1 điểm ở 3 trên 6 cấu hình, còn cgar_esc thì không.

Không nên viết: "ít đường hơn mà tốt hơn GNN-RAG" khi so cùng reader; "CGAR giảm hallucination" theo nghĩa QHR; "controller adaptive".

## 8. Giới hạn

- Chưa có nhãn người. QHR và strict-QHR là nhãn tự động. UNCERTAIN chiếm 15 đến 20% claim.
- Reader chạy trên 1000 câu WebQSP, một seed. Chưa có reader trên CWQ.
- Mistral là run cũ, không có cấu hình verify, cand hay LoRA.
- QHR ≈ 0 dưới verify là hệ quả cấu trúc, không phải bằng chứng độc lập.

## 9. Việc còn lại

1. Gán nhãn người cho 200 dòng mẫu review × 2 người (Qwen chat), rồi chạy P9 cell 11.
2. Nếu muốn đạt cost 80%: thử fallback `budget:10` thay vì toàn bộ đường, hoặc bỏ luật "hơn 10 ứng viên" và chỉ dùng τ.
3. Reader LoRA trên CWQ để kiểm tra kết luận có chuyển sang câu hỏi đa bước không.

Checklist tự động ở P8 cell 11 (chạy 09/11/2026, đọc thẳng từ Drive) khớp với bảng claim ở Phần 5: 11 [OK], 7 [NO], 2 [--] cho nhánh PRISM không chạy.
