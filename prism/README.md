# PRISM — Predictive Retrieval with Inductive Subgraph Memory

Implementation prototype của paradigm: **retriever tự simulate câu trả lời của LLM** để dẫn dắt retrieval, thống nhất 4 ability (adaptive-K, path selection, calibration, diversity) qua một cơ chế thông tin duy nhất.

## Cấu trúc

```
prism/
├── config.py           # Hyperparameters
├── data.py             # Synthetic generator + WebQSP/CWQ hooks
├── encoder.py          # Inductive subgraph/path encoder (relation-typed)
├── las.py              # Lightweight Answer Simulator (transformer)
├── controller.py       # Information-theoretic stopping + counterfactual scoring
├── distill.py          # Stage 1: distill LAS từ teacher
├── rl.py               # Stage 2: REINFORCE fine-tune
├── metrics.py          # ECE, Brier, Hit, F1, Pareto
├── train.py            # Main training entrypoint
├── evaluate.py         # Evaluation entrypoint
└── prism_demo.ipynb    # Notebook chạy end-to-end (Colab-ready)
```

## Quick start

```bash
pip install -r requirements.txt

# 1. Train Stage 1 (distillation) on synthetic data
python -m COLAB.graph_rag.prism.train --stage distill --data synthetic --epochs 5

# 2. Train Stage 2 (RL) starting from Stage 1 checkpoint
python -m COLAB.graph_rag.prism.train --stage rl --data synthetic --steps 500

# 3. Evaluate
python -m COLAB.graph_rag.prism.evaluate --ckpt outputs/prism_rl_best.pt
```

Hoặc mở [`prism_demo.ipynb`](prism_demo.ipynb) chạy hết trong Colab.

## Switching to real data

Synthetic mode chạy được không cần dataset ngoài (validates pipeline). Có 2 cách lên real:

### Option A — RoGDataset (recommend, no GNN-RAG needed)

Dùng pre-extracted KG subgraphs từ HF `rmanluo/RoG-webqsp` hoặc `rmanluo/RoG-cwq`:

```python
from COLAB.graph_rag.prism.data import RoGDataset
from COLAB.graph_rag.prism.config import CFG
from COLAB.graph_rag.prism.las import LAS

# pip install datasets sentence-transformers
train_set = RoGDataset.from_hf(
    "rmanluo/RoG-webqsp", split="train",
    sbert_model="all-MiniLM-L6-v2",
    cache_dir="./rog_cache",
    cfg=CFG,
)

# IMPORTANT: update CFG vocab sizes before LAS init
CFG.n_relations = train_set.n_relations
CFG.n_entities  = train_set.n_entities

las = LAS(CFG)
```

→ BFS từ q_entity, score candidates bằng path frequency, không cần train upstream retriever.

### Option B — Real GNN-RAG features (.pkl)

1. Chuẩn bị `gnn_features.pkl` từ GNN-RAG repo (paths + scores per query)
2. Implement `WebQSPDataset.__getitem__` adapt schema
3. Set teacher LLM trong `data.py::TeacherLLM`: thay bằng huggingface model thật

## Key files chi tiết

- **`las.py`**: trái tim của PRISM. Distill từ frozen LLM, output P(answer | q, S).
- **`controller.py`**: derive 4 ability từ P_LAS — không thêm params nào.
- **`encoder.py`**: ULTRA-style inductive — chỉ relation type embeddings, OOV-safe.

## Roadmap (chưa làm trong v0)

- [ ] Real Llama teacher integration (hiện tại có TeacherLLM stub)
- [ ] Online closed-loop adaptation (đang ở stub trong `rl.py`)
- [ ] Cross-KG transfer experiment (cần CWQ + Wikidata loader)
- [ ] Graph-drift robustness experiment

## License

Research prototype. Inspect, modify, run.
