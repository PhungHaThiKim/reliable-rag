# Flow trình bày Seminar 2, Project 11: Software Supply Chain Security với LLM, RAG và KG-RAG

Số liệu thật lấy từ `results/cloud/` (kết quả tổng hợp tải từ pod, lần chạy 2026-11-10) và `results/figures/`.

## 1. Mạch chuyện (6 phần)

1. **Paper tham chiếu.** mal-LLM: RAG không cải thiện trên **một** cấu hình phân loại package độc (input 300 ký tự setup.py, top-1/2 tài liệu).
2. **Khoảng trống.** Tri thức ngoài và cấu trúc đồ thị hữu ích **trong điều kiện nào**? Không hỏi chung "RAG có giúp không".
3. **Kiến trúc thống nhất.** E0-E4 trên cùng reader, cùng prompt, cùng ngân sách token; hai bài toán: Task A (phát hiện) và Task B (tác động).
4. **Use case có kiểm soát.** Mã -> Chính sách -> Phụ thuộc -> Thiếu bằng chứng (G1 -> G2 -> G3 -> G4), mỗi nhóm có ví dụ case C01-C06.
5. **Kết quả.** Chất lượng (theo task, không gộp), grounding (cùng mẫu số E3/E4), coverage và risk-coverage, chi phí đo được.
6. **Khung quyết định.** Khi nào dùng LLM thuần, RAG văn bản, KG-RAG, có hay không bước xác minh.

| Phần | Slide | Thời gian |
|---|---|---|
| 1. Paper tham chiếu | 1-3 | 2.5 phút |
| 2. Khoảng trống nghiên cứu | 4-6 | 3 phút |
| 3. Kiến trúc thống nhất (gồm vấn đề khi xây) | 7-19 | 8.5 phút |
| 4. Use case có kiểm soát | 20-23 | 4 phút |
| 5. Kết quả | 24-29 | 5.5 phút |
| 6. Khung quyết định | 30 | 1.5 phút |
| Tổng | 30 slide | khoảng 25 phút + hỏi đáp |

## 2. Flow từng slide

### Phần 1. Paper tham chiếu

**Slide 1. Tiêu đề.** Software Supply Chain Security với LLM, RAG, KG-RAG; hai bài toán A (có độc không) và B (ảnh hưởng tới đâu); agenda 6 phần.

**Slide 2. Bài toán.** Mã độc vào công ty qua dependency: chạy lúc cài đặt/import, dùng cùng API với mã lành (Task A); typosquat, bản phát hành
bị chiếm quyền, phụ thuộc bắc cầu, policy và ngoại lệ (Task B). Khó nằm ở quan hệ.

**Slide 3. Paper mal-LLM.** Ibiyo et al., EASE 2025: zero-shot, Simple RAG (YARA, advisories, setup.py độc), fine-tune Llama-3.1-8B.
Kết luận của paper: RAG gần như không giúp, fine-tune giúp. **Nhưng** đó là một cấu hình: input 300 ký tự setup.py, truy hồi top-1/2 bằng embedding,
không cấu trúc, giải thích không được kiểm, chỉ một loại câu hỏi. Kết quả tái lập của nhóm đặt ở slide 25.

### Phần 2. Khoảng trống nghiên cứu

**Slide 4. Câu hỏi.** RQ1 truy hồi tài liệu (E1 vs E0), RQ2 hybrid (E2 vs E1), RQ3 cấu trúc đồ thị (E3 vs E2; E3 vs E3a cùng cạnh; E3 vs E3b fact
truy hồi), RQ4 xác minh (E4 vs E3, cùng mẫu số), RQ5 theo loại câu hỏi (G1-G4). Mọi method trả lời mọi câu.

**Slide 5. Giả thuyết viết trước.** G1 -> E0 đủ; G2 -> cần tài liệu; G3 -> đồ thị có lợi thế; G4 -> xác minh từ chối đúng hơn. Cột "điều có thể làm sai".

**Slide 6. Thiết kế benchmark.** Task A (malllm, pyguard_obf, synthetic), test case C01-C06, Task B (ACME tổng hợp, seed 1 dev, seed 2 test,
4 x 100 câu). Đáp án Task B tính từ dữ liệu sinh ra; tune chỉ trên dev.

### Phần 3. Kiến trúc thống nhất

**Slide 7. Kiến trúc.** Ba nguồn (package, tài liệu ACME, KB) -> bằng chứng có id (C*, K*, G*, F*) -> một LLM, một prompt -> verifier -> đánh giá theo task.
**Slide 8. Đọc an toàn.** Trong bộ nhớ, chỉ `ast.parse`; suy luận chạy offline; không gửi mã cho dịch vụ ngoài.
**Slide 9. Dữ liệu.** 5 bộ, chia theo nhóm, KB chỉ từ train; ACME 342 tài liệu, KG 949 cạnh, 400 câu.
**Slide 10. Phân tích tĩnh.** Luồng chỉ từ AST; dynamic dispatch; tăng tốc 1 giờ 14 phút -> 14 giây.
**Slide 11. Prompt chung và trích dẫn.**
**Slide 12. E1, E2: truy hồi văn bản, hybrid.**
**Slide 13. E3 Task A: đồ thị hành vi của mã.**
**Slide 14. E3 Task B: KG parse từ chính tài liệu.**
**Slide 15. E3 vs E3a vs E3b.**
**Slide 16. E4: xác minh, từ chối.**
**Slide 17. Hạ tầng.** 3 pod 1 GPU song song, cache, offline, chi phí đo được.
**Slide 18-19. Vấn đề khi xây hệ thống.** Bảng mục 3.

### Phần 4. Use case có kiểm soát (mã -> chính sách -> phụ thuộc -> thiếu bằng chứng)

**Slide 20. G1 mã.** C01, C02 + câu hỏi luồng dữ liệu (bẫy đọc và gửi nhưng không nối nhau). Kỳ vọng / kết quả theo method.
**Slide 21. G2 chính sách.** Policy v1 -> v2, ADR còn hạn / hết hạn, dev được miễn; C04 là phản ví dụ lành của Task A.
**Slide 22. G3 phụ thuộc.** Tác động nhiều bước, bẫy dev-only, extra chưa bật, version ngoài khoảng; C05 là đường nhiều bước ở mức mã.
**Slide 23. G4 thiếu bằng chứng.** C06 (dynamic dispatch), version không pin, plugin theo config, hai policy mâu thuẫn.

### Phần 5. Kết quả

**Slide 24. Bộ metric.** Bảng metric -> điều cần chứng minh (mục 4). Không gộp Task A với Task B, không gộp G4 với G1-G3.

**Slide 25. Kết quả Task A** (strict F1, FPR, coverage; tái lập mal-LLM). Đã có cho `pyguard_obf` test (n = 310):

| Reader | E0 | E2 | E3 | E3a | E3b | E4 | E3 vs E2 | E3 vs E3a |
|---|---|---|---|---|---|---|---|---|
| Llama-3.1-8B | 0.754 | 0.731 | 0.805 | 0.814 | 0.722 | 0.757 (cov 0.79) | +0.075 [+0.051, +0.100] | -0.008 [-0.018, -0.002] |
| Qwen2.5-Coder-7B | 0.953 | 0.914 | 0.939 | 0.942 | 0.928 | 0.887 (cov 0.95) | +0.025 [-0.003, +0.054] | -0.003 [-0.013, +0.006] |

| `malllm` (chỉ setup.py), n = 500 | E0 | E1 | E2 | E3 | E3a | E3b | E4 |
|---|---|---|---|---|---|---|---|
| Llama-3.1-8B | **0.822** | 0.674 | 0.709 | 0.704 | 0.718 | 0.699 | 0.679 |
| Qwen2.5-Coder-7B | 0.866 | 0.934 | **0.945** | 0.760 | 0.766 | 0.865 | 0.743 |

Đọc: với package đầy đủ và Llama, fact đồ thị **được chọn bằng duyệt** giúp rõ (E3 hơn E0 +0.052, hơn E2 +0.075, hơn E3b +0.083), nhưng
**cấu trúc đường** không giúp (E3a bằng hoặc hơn E3 một chút). Với Qwen (model chuyên code), LLM thuần đã mạnh nhất (0.953); RAG văn bản kém hơn
0.03-0.04. Với `malllm`: Llama thuần tốt nhất, mọi truy hồi làm giảm 0.11-0.15; Qwen thì RAG văn bản giúp (+0.08) còn đồ thị dựng từ một
file ngắn làm hại (-0.11). Tái lập mal-LLM (Llama, 300 ký tự): zero-shot F1 0.559 (paper 0.459), RAG-YARA 0.285 (paper 0.515),
RAG-malcode 0.471 (paper 0.438): kết luận định tính của paper được giữ.

**Slide 26. Kết quả Task B** (Qwen, 400 câu test; G1-G3 exact accuracy, G4 từ chối đúng):

| Method | G1 mã | G2 policy | G3 phụ thuộc | G4 từ chối đúng | coverage G1-G3 |
|---|---|---|---|---|---|
| E0 | 0.75 | 0.00 | 0.00 | 0.85 | 0.34 |
| E1 | 0.80 | 0.36 | 0.22 | 0.59 | 0.73 |
| E2 | 0.75 | 0.52 | 0.23 | 0.47 | 0.79 |
| E3 | 0.73 | 0.59 | 0.54 | 0.56 | 0.91 |
| E3a | 0.72 | 0.58 | 0.54 | 0.49 | 0.91 |
| E3b | 0.79 | 0.36 | 0.15 | 0.56 | 0.68 |
| E4 | 0.73 | 0.56 | 0.52 | 0.65 | 0.87 |

Có ý nghĩa (CI 95%): G2 E1 vs E0 +0.36, E2 vs E1 +0.16; G3 E3 vs E2 +0.31, E3 vs E3b +0.39; G4 E1 vs E0 -0.26. E3 vs E3a không khác ở nhóm nào.
Bẫy: version ngoài khoảng advisory E3 0/18 (E1 0.78); hai policy mâu thuẫn E3 0.00 (E1 0.82); nhiều bước E3 0.80 (E2 0.04).
Llama trả lời ít (coverage 0.30-0.50); G1 E3 0.26 so với E0 0.62 (KG báo "không có thực thể", reader từ chối).

**Slide 27. Đánh đổi: risk-coverage và grounding cùng mẫu số.** Task A, E3 -> E4 trên `pyguard_obf` test: claim không chứng cứ / claim gốc
0.366 -> 0.000 (Llama), 0.326 -> 0.000 (Qwen); cái giá: claim giữ lại 0.62, coverage 0.79 (Llama) và 0.95 (Qwen); strict F1 giảm khoảng 0.05.
Task B (Qwen): claim không chứng cứ / claim gốc E0 0.788, E2 0.053, E3 0.030, E4 0.005; risk-coverage: E3 thấp hơn E2 ở mọi mức coverage;
E4 (tau 1.0, tune trên dev theo mục tiêu cân bằng) +0.09 từ chối đúng ở G4, -0.017 exact accuracy ở G1-G3.

**Slide 28. Chi phí.** GPU giây đo được, chia theo token (Qwen): Task A `pyguard_obf` trên A100 (1.79 USD/giờ): E0 0.045, E2 0.080,
E3 0.053 USD / 1000 package; Task B trên L40S (1.09 USD/giờ): E0 0.018, E2 0.063, E3 0.033 USD / 1000 câu; E4 +0. KG dùng khoảng một nửa
token của RAG văn bản. Toàn bộ thực nghiệm chạy song song trên 3 pod một GPU, khoảng 1 giờ 20 phút. Hoá đơn RunPod đã ghi (tới 19:00Z):
3 pod mới 2.03 USD, cộng giờ cuối chưa ghi (ước khoảng 3.5 USD); pod H100 của lần chạy trước 6.56 USD.

**Slide 29. Giới hạn.**

### Phần 6. Khung quyết định

**Slide 30. Khi nào dùng gì** (rút từ số đo):

| Khi | Dùng | Vì số đo | Cẩn thận |
|---|---|---|---|
| Mã đã trong input, reader mạnh về code | LLM thuần | Task A Qwen 0.95; G1 0.75, truy hồi không hơn | Reader yếu (Llama) cần fact đồ thị: +0.05 |
| Cần tri thức nội bộ (policy, ADR) | RAG hybrid hoặc KG | G2: 0.00 -> 0.52 (E2) / 0.59 (E3) | So version với khoảng advisory: KG 0/18, văn bản 0.78 |
| Câu hỏi nhiều bước qua phụ thuộc | KG-RAG có duyệt | G3: 0.54 so với 0.23 | Lợi ích từ chọn fact, không từ trình bày đường (E3a = E3) |
| Input ngắn, ít cấu trúc (setup.py) | LLM thuần hoặc RAG văn bản | Qwen: RAG +0.08, KG -0.11 | Đồ thị dựng từ một file dễ dẫn sai |
| Sai đắt, có thể thiếu bằng chứng | thêm xác minh (E4) | claim không chứng cứ 0.37 -> 0 (cùng mẫu số) | Coverage giảm; C06 (dynamic dispatch) vẫn không ai từ chối |

So với giả thuyết slide 5: G1, G2, G3 đúng hướng; G4 chỉ đúng một phần (thêm bằng chứng làm model từ chối ít đi; E4 chỉ bù một phần).

## 3. Bảng vấn đề (slide 18-19)

| # | Vấn đề | Tại sao quan trọng | Cách xử lý | Còn tồn tại gì |
|---|---|---|---|---|
| 1 | Đồng xuất hiện không phải luồng dữ liệu | Báo nhầm SDK, telemetry lành | `FLOWS_TO` chỉ từ AST; còn lại là `PRECEDES` | Tóm tắt giữa hàm một mức |
| 2 | Phân tích chậm: 1 giờ 14 phút trên một nhân | GPU ngồi chờ, tốn tiền | Giới hạn tất định, song song theo cgroup quota, cache, timeout | Vượt giới hạn chỉ còn regex |
| 3 | Dynamic dispatch | Không biết đích thì không được nói "không có" | Ghi dispatch; verifier coi coverage chưa đủ | Đích nằm trong config |
| 4 | Rò rỉ nhãn và tri thức | F1 cao giả | Chia theo nhóm + MinHash; KB chỉ từ train | Near-duplicate `pyguard_obf` vắt split: bootstrap theo nhóm |
| 5 | So sánh công bằng | Thêm token tạo khác biệt giả | Cùng reader, prompt, ngân sách; E3a đúng các cạnh của E3 | Bằng token chưa chắc bằng thông tin |
| 6 | Mỗi phương pháp thắng một câu | Kết luận có sẵn | Cùng bộ câu hỏi; nhiều loại, nhiều bẫy mỗi nhóm; tune chỉ trên dev | Tổ chức tổng hợp |
| 7 | Gộp metric không đồng nhất | Accuracy chung che mất đánh đổi | Báo theo task và theo nhóm; G4 riêng; đồ thị đánh đổi | - |
| 8 | Tỉ lệ claim bịa của E4 đẹp giả | Lọc bỏ ca khó là "đẹp" | Cùng mẫu số: claim gốc của E3, mọi mẫu; kèm claim giữ lại và coverage | Verifier là kiểm nhất quán, chưa có nhãn người |
| 9 | Hết GPU, script lỗi khi chạy cloud | Không tạo được pod; pod dừng giữa chừng | Nhiều pod 1 GPU; LF cho script; sửa cell checklist; tự dừng có an toàn | Ước lượng chi phí phải từ pilot |
| 10 | An toàn | Chạy nhầm mã độc | Đọc tĩnh; suy luận offline; không gửi mã ra ngoài | RunPod không chặn egress được |

## 4. Bộ metric (slide 24)

| Metric | Dùng để chứng minh |
|---|---|
| Task A malicious F1 / macro F1 (strict) | chất lượng phát hiện |
| Task B exact accuracy / set F1 (G1-G3) | chất lượng câu trả lời |
| Unsupported claim rate | chất lượng grounding |
| Unsupported / claim gốc trên mọi mẫu | so E3 với E4 không bị "đẹp giả" do lọc |
| Correct abstention rate (G4, C06) | phản ứng khi thiếu bằng chứng |
| Answer coverage | khả năng vẫn đưa ra đáp án |
| Risk-coverage, AURC | đánh đổi độ chính xác và từ chối |
| Evidence recall | truy hồi có mang đủ bằng chứng không |
| Latency, tokens, cost | chi phí của phương pháp |

## 5. Hỏi đáp dự kiến

1. **Vì sao không có một accuracy chung?** Task A nhị phân, Task B có tập và câu không trả lời được; gộp lại che mất đánh đổi. Báo theo task, theo nhóm.
2. **E4 có "ăn gian" bằng cách từ chối?** Đo cùng mẫu số: claim gốc của E3 và mọi mẫu; luôn báo kèm coverage, claim giữ lại, strict F1.
3. **Vì sao không thiết kế mỗi nhóm cho một phương pháp?** Kết luận sẽ có trước khi đo. Mọi phương pháp trả lời mọi câu, có bẫy cho mọi phương pháp.
4. **KG của Task B có biết nhiều hơn text RAG?** Không: parse từ chính tài liệu; test tính lại đáp án chỉ từ KG.
5. **Cấu trúc đồ thị có giúp không?** Ở cả hai task: E3a (cùng cạnh, câu rời) bằng E3; cái giúp là chọn fact bằng duyệt (E3 hơn E3b:
   Task A Llama +0.083, Task B G3 +0.39).
6. **Có chạy mã độc không?** Không: đọc tĩnh, suy luận offline, không gửi mã ra ngoài.
7. **Chi phí?** Đo được, không ước lượng: xem slide 28 và `scripts/estimate_cost.py`.
