# ROADMAP: RL-based GraphRAG for Multi-hop QA

> Deadline: Submit paper thang 05/2026
> Env: Google Colab Pro+ (A100/V100, high-RAM runtime)
> Datasets: WebQSP, HotpotQA
> Target venues: EMNLP (thuc nghiem) / NeurIPS (thuat toan)

---

## Notebooks

| File | Phase | Noi dung |
|------|-------|---------|
| `P0_Setup_and_Data.ipynb` | 0 | Setup env, download data, build KG, utility functions |
| `P1a_Baseline_VectorRAG.ipynb` | 1 | Baseline: DPR/Contriever retrieval + LLM |
| `P1b_Baseline_HeuristicGraphRAG.ipynb` | 1 | Baseline: BFS, Beam Search, PPR tren KG |
| `P2_GNN_RAG_Reproduce.ipynb` | 2 | Reproduce GNN-RAG (ACL 2025), phan tich |
| `P3_KnowGPT_RL_Reproduce.ipynb` | 3 | Reproduce KnowGPT RL pipeline (NeurIPS 2024) |
| `P4a_Proposed_Architecture.ipynb` | 4 | Proposed method: components & architecture |
| `P4b_Proposed_Training.ipynb` | 4 | Training: Imitation Learning + REINFORCE |
| `P5_Experiments_Analysis.ipynb` | 5 | Full experiments, ablation, figures cho paper |
| `KnowGPT_v3.ipynb` | (ref) | Code KnowGPT da chay truoc do |

---

## Tong quan cac Phase

| Phase | Noi dung | Thoi gian du kien | Output |
|-------|----------|-------------------|--------|
| 0 | Setup & Data | Tuan 1 | Data pipeline, KG san sang |
| 1 | Reproduce Baselines | Tuan 2-3 | Baseline numbers (Vector RAG, Heuristic GraphRAG) |
| 2 | Reproduce GNN-RAG | Tuan 3-4 | GNN-RAG results on WebQSP |
| 3 | Reproduce KnowGPT (RL) | Tuan 4-5 | KnowGPT results, hieu RL pipeline |
| 4 | Xay dung Proposed Method | Tuan 5-8 | RL-based retriever moi |
| 5 | Experiments & Ablation | Tuan 8-10 | Full experiment tables |
| 6 | Writing & Submit | Tuan 10-12 | Paper hoan chinh |

---

## PHASE 0: Setup & Data Preparation

### 0.1 Setup moi truong Colab
- [ ] Tao notebook `00_setup_environment.ipynb`
- [ ] Cai dat cac thu vien can thiet (xem file `configs/requirements.txt`)
- [ ] Test ket noi Google Drive, cau hinh duong dan data

### 0.2 Download & Chuan bi Datasets

#### WebQSP (Knowledge Graph QA)
- [ ] Download WebQSP dataset: https://www.microsoft.com/en-us/download/details.aspx?id=52763
- [ ] Download Freebase subgraph (dung chung voi GNN-RAG)
- [ ] Format: question -> SPARQL -> answer entities on Freebase

#### HotpotQA (Multi-hop QA)
- [ ] Download HotpotQA: https://hotpotqa.github.io/
- [ ] Distractor setting (10 paragraphs, 2 gold)
- [ ] Xay Wikipedia entity graph tu supporting facts

#### ConceptNet (da co tu KnowGPT)
- [ ] Da co `conceptnet_en.csv` tren Drive
- [ ] Kiem tra lai format, so luong triples

### 0.3 Knowledge Graph Preparation
- [ ] Notebook `01_data_preparation.ipynb`
- [ ] Build graph structure (adjacency list, edge index) tu KG
- [ ] Tao entity/relation embeddings (pre-trained hoac train)
- [ ] Luu dang pickle/torch cho tai nhanh tren Colab

---

## PHASE 1: Reproduce Baselines

> Muc tieu: Co baseline numbers de so sanh

### 1.1 Vector-based RAG (DPR / Contriever)
- [ ] Notebook `notebooks/phase1_baselines/01_vector_rag.ipynb`
- [ ] Cai DPR hoac Contriever
- [ ] Index passages tu Wikipedia (cho HotpotQA) hoac KG triples (cho WebQSP)
- [ ] Retrieve top-k passages -> feed vao LLM -> do F1, EM
- [ ] Ghi results vao `results/baseline_vector_rag.json`

### 1.2 Heuristic GraphRAG
- [ ] Notebook `notebooks/phase1_baselines/02_heuristic_graph_rag.ipynb`
- [ ] Implement cac chien luoc:
  - BFS expansion (1-hop, 2-hop, 3-hop)
  - Beam search tren KG
  - Top-k expansion theo degree/PPR
- [ ] Verbalize subgraph -> prompt LLM -> do F1, EM
- [ ] So sanh cost (so node expanded, path length)
- [ ] Ghi results vao `results/baseline_heuristic_graphrag.json`

### Doc them - Ly thuyet can nam:
- [ ] **Graph algorithms**: BFS, DFS, Dijkstra, PPR - Sach: "Algorithm Design" (Kleinberg & Tardos), Chapter 3-4
- [ ] **Personalized PageRank**: Paper goc + implementation voi networkx/igraph

---

## PHASE 2: Reproduce GNN-RAG

> Muc tieu: Hieu GNN nhu "learned heuristic", co so de so sanh

### 2.1 Doc & Hieu paper GNN-RAG
- [ ] Doc ky paper: https://aclanthology.org/2025.findings-acl.856/
- [ ] Note lai:
  - GNN architecture (bao nhieu layer, loai GNN nao)
  - Cach score nodes/candidates
  - Cach extract shortest paths -> verbalize -> LLM
  - Retrieval Augmentation (RA) strategy
  - Routing mechanism

### 2.2 Run code GNN-RAG
- [ ] Notebook `notebooks/phase2_gnn_rag/01_gnn_rag_reproduce.ipynb`
- [ ] Clone repo GNN-RAG, setup dependencies
- [ ] Run tren WebQSP (dataset chinh cua paper)
- [ ] Verify reproduce duoc results gan voi paper
- [ ] Log: Hits@1, F1, so luong paths retrieved, avg path length

### 2.3 Phan tich GNN-RAG
- [ ] Notebook `notebooks/phase2_gnn_rag/02_gnn_rag_analysis.ipynb`
- [ ] Visualize subgraph duoc retrieve
- [ ] Phan tich failure cases: khi nao GNN-RAG sai?
- [ ] Do retrieval cost vs accuracy trade-off
- [ ] Nhan dien diem yeu: one-shot scoring, shortest path cung, noise khi union qua nhieu paths

### Doc them:
- [ ] **GNN fundamentals**: GCN (Kipf 2017), GAT (Velickovic 2018), GraphSAGE (Hamilton 2017)
- [ ] Sach: "Graph Representation Learning" (Hamilton) - Chapter 5,6
- [ ] **Message Passing framework**: hieu ro k-layer GNN = k-hop aggregation

---

## PHASE 3: Reproduce KnowGPT (RL-based Retrieval)

> Muc tieu: Nam vung RL pipeline tren KG, co nen tang de xay method moi

### 3.1 Review lai KnowGPT code da chay
- [ ] Review notebook `KnowGPT_v3.ipynb` (da co)
- [ ] Document lai:
  - MDP formulation (state, action, reward)
  - Policy network architecture
  - Training loop (REINFORCE / Policy Gradient)
  - Reward function: reachability + context-relatedness + conciseness
  - MAB prompt construction

### 3.2 Reproduce KnowGPT tren ConceptNet
- [ ] Notebook `notebooks/phase3_rl_retriever/01_knowgpt_reproduce.ipynb`
- [ ] Chay lai training policy network
- [ ] Evaluate tren CSQA/OBQA (nhu paper goc)
- [ ] Verify results

### 3.3 Chuyen KnowGPT sang WebQSP/HotpotQA
- [ ] Notebook `notebooks/phase3_rl_retriever/02_knowgpt_transfer.ipynb`
- [ ] Adapt KnowGPT pipeline cho Freebase KG (thay ConceptNet)
- [ ] Thay doi entity embedding phu hop
- [ ] Evaluate tren WebQSP -> so sanh voi GNN-RAG

### 3.4 Doc them papers RL on KG
- [ ] **MINERVA** (Das et al., ICLR 2018): "Go for a Walk and Arrive at the Answer"
  - MDP formulation kinh dien cho KG reasoning
  - REINFORCE voi reward shaping
- [ ] **DeepPath** (Xiong et al., EMNLP 2017): RL for KG path finding
- [ ] **MultiHopKG** (Lin et al., ICLR 2018): Reward shaping with pre-trained embeddings
- [ ] **LogicRAG** (AAAI 2026): Greedy retrieval strategy -> so sanh approach

### Doc them - Ly thuyet RL can nam:
- [ ] **RL co ban**: Sach "Reinforcement Learning: An Introduction" (Sutton & Barto)
  - Chapter 2: Multi-Armed Bandits
  - Chapter 3-4: MDP, Bellman Equations
  - Chapter 13: Policy Gradient Methods (REINFORCE)
- [ ] **Deep RL**: Policy Gradient, Actor-Critic, PPO
  - Khoa hoc: Sergey Levine CS285 (Berkeley) hoac David Silver RL course
- [ ] **Reward Shaping**: potential-based reward shaping (Ng et al., 1999)

---

## PHASE 4: Xay dung Proposed Method

> Muc tieu: Ket hop GNN (heuristic) + Greedy (pruning) + RL (sequential decision)

### 4.1 Thiet ke kien truc tong the
- [ ] Notebook `notebooks/phase4_proposed_method/01_architecture_design.ipynb`
- [ ] Dinh nghia ro MDP:
  ```
  State  s_t = (query_emb, current_node_emb, visited_path, GNN_scores_neighbors)
  Action a_t = chon node tiep theo HOAC dung (STOP action)
  Reward R   = f(answer_quality, retrieval_cost)
  ```
- [ ] Policy network architecture:
  - Input: concat(query_emb, node_emb, path_history_emb, GNN_heuristic_score)
  - Hidden: MLP hoac Transformer-based
  - Output: probability distribution over actions
- [ ] GNN module: pre-trained GNN tu Phase 2 lam heuristic feature
- [ ] Greedy pruning: top-K candidates tai moi buoc, budget constraint

### 4.2 Implement tung component
- [ ] `src/models/policy_network.py` - Policy network
- [ ] `src/models/gnn_heuristic.py` - GNN scoring module
- [ ] `src/retrieval/rl_retriever.py` - RL retrieval agent
- [ ] `src/retrieval/greedy_pruner.py` - Greedy pruning/budget control
- [ ] `src/retrieval/environment.py` - KG environment (MDP)
- [ ] `src/evaluation/metrics.py` - F1, EM, retrieval cost metrics

### 4.3 Training Pipeline
- [ ] Notebook `notebooks/phase4_proposed_method/02_training.ipynb`
- [ ] Implement REINFORCE with baseline
- [ ] Reward function:
  ```
  R = alpha * answer_reward (F1 cua LLM answer)
    + beta * efficiency_reward (penalty cho so buoc nhieu)
    + gamma * relevance_reward (GNN score cua path)
  ```
- [ ] Curriculum learning: bat dau voi 1-hop questions, tang dan do kho
- [ ] Training tricks:
  - Pre-train policy voi supervised signal (imitation learning tu GNN-RAG paths)
  - Reward shaping voi GNN scores
  - Action masking (khong quay lai node da tham)

### 4.4 Inference Pipeline
- [ ] Notebook `notebooks/phase4_proposed_method/03_inference.ipynb`
- [ ] Beam search voi learned policy (thay vi uniform)
- [ ] Budget-aware stopping: agent tu quyet dinh khi nao du thong tin
- [ ] Verbalize retrieved subgraph -> prompt LLM

### Doc them:
- [ ] **Imitation Learning**: DAgger algorithm, behavior cloning
- [ ] **Budget-constrained optimization**: Knapsack-style pruning
- [ ] **Monte Carlo Tree Search (MCTS)**: co the dung thay REINFORCE neu can

---

## PHASE 5: Experiments & Ablation Studies

### 5.1 Main Results
- [ ] Notebook `notebooks/phase5_analysis/01_main_results.ipynb`
- [ ] Bang so sanh chinh:

| Method | WebQSP Hits@1 | WebQSP F1 | HotpotQA F1 | HotpotQA EM | Avg Retrieval Cost |
|--------|--------------|-----------|-------------|-------------|-------------------|
| Vector RAG (DPR) | | | | | |
| Heuristic GraphRAG (BFS) | | | | | |
| Heuristic GraphRAG (Beam) | | | | | |
| GNN-RAG | | | | | |
| KnowGPT (RL) | | | | | |
| **Proposed (GNN+Greedy+RL)** | | | | | |

### 5.2 Ablation Studies
- [ ] Notebook `notebooks/phase5_analysis/02_ablation.ipynb`
- [ ] Ablation 1: Bo GNN heuristic -> RL alone
- [ ] Ablation 2: Bo RL -> GNN + Greedy only
- [ ] Ablation 3: Bo Greedy pruning -> GNN + RL (no budget)
- [ ] Ablation 4: Thay doi retrieval budget (k=3,5,10,20)
- [ ] Ablation 5: Thay doi reward weights (alpha, beta, gamma)

### 5.3 Analysis
- [ ] Notebook `notebooks/phase5_analysis/03_qualitative_analysis.ipynb`
- [ ] Case study: vi du cu the RL chon path tot hon heuristic
- [ ] Visualization: retrieval paths tren KG
- [ ] Error analysis: khi nao method fail?
- [ ] Efficiency analysis: retrieval cost vs accuracy curve
- [ ] Scalability: do tren KG sizes khac nhau

---

## PHASE 6: Writing & Submit

### 6.1 Paper Structure du kien
```
1. Introduction
2. Related Work
   2.1 GraphRAG
   2.2 RL for Knowledge Graph Reasoning
   2.3 Multi-hop QA
3. Methodology
   3.1 Problem Formulation (MDP on KG)
   3.2 GNN-based Heuristic Module
   3.3 RL-based Retrieval Policy
   3.4 Greedy Budget Control
   3.5 Training Procedure
4. Experiments
   4.1 Datasets & Baselines
   4.2 Main Results
   4.3 Ablation Studies
   4.4 Analysis
5. Conclusion
```

### 6.2 Timeline viet bai
- [ ] Tuan 10: Draft Introduction + Related Work
- [ ] Tuan 11: Draft Methodology + Experiments
- [ ] Tuan 11: Tao figures, tables
- [ ] Tuan 12: Revision, proofread
- [ ] Submit truoc deadline thang 05/2026

---

## Tai lieu tham khao & Hoc them

### Sach
| Sach | Chuong can doc | Muc dich |
|------|---------------|----------|
| Sutton & Barto - RL: An Introduction | Ch.2,3,4,13 | RL fundamentals, Policy Gradient |
| Hamilton - Graph Representation Learning | Ch.5,6 | GNN, Message Passing |
| Kleinberg & Tardos - Algorithm Design | Ch.3,4 | Graph algorithms |

### Khoa hoc Online
| Khoa hoc | Link | Muc dich |
|----------|------|----------|
| David Silver RL Course | YouTube | RL theory |
| CS285 Deep RL (Berkeley) | YouTube | Deep RL implementation |
| Stanford CS224W | YouTube | GNN & Graph ML |

### Papers quan trong (theo thu tu doc)
1. **MINERVA** (ICLR 2018) - RL tren KG, MDP formulation co ban
2. **DeepPath** (EMNLP 2017) - RL path finding tren KG
3. **GNN-RAG** (ACL 2025) - GNN-based retriever, baseline chinh
4. **KnowGPT** (NeurIPS 2024) - RL + MAB cho GraphRAG
5. **LogicRAG** (AAAI 2026) - Greedy retrieval, so sanh approach
6. **ReAct** (ICLR 2023) - LLM as agent, reasoning + acting
7. **AutoGraph-R1** - Graph construction voi RL

---

## Luu y khi chay tren Colab Pro+

1. **Luu checkpoint thuong xuyen** ve Drive (training RL co the mat nhieu gio)
2. **Su dung A100 GPU** khi train GNN va Policy Network
3. **High-RAM runtime** khi load KG lon (Freebase co the > 10GB)
4. **Chia nho experiment**: moi notebook lam 1 viec cu the, tranh notebook qua dai
5. **Version control**: luu cac notebook len Drive voi version date
