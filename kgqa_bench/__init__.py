"""kgqa_bench — one harness to run CGAR, PRISM and the baselines on ANY KGQA
dataset with ONE metric suite.

Why this exists
---------------
Before this package the two methods lived in different worlds:
  * CGAR consumed GNN-RAG feature pickles (text triples + SBERT) inside a
    notebook, and its metrics (coverage-F1, cost) lived in notebook cells;
  * PRISM consumed relation-ID tensors from `prism.data.RoGDataset` and
    reported LAS-internal Hit@1 / ECE from `prism.experiments`.
Numbers from the two pipelines were therefore computed on different
candidate pools, different path sets and different metric definitions and
could not be put in the same table.

`kgqa_bench` fixes that:

    BenchDataset  (dataset-agnostic question records)
        ├── adapters.to_gnn_features(ds)      → CGAR / GNN-RAG / GraphRAG-FI
        └── adapters.build_prism_samples(ds)  → PRISM
    methods.*                                 → per-question `records`
    metrics.summarize_records(records)        → one metric dict per (dataset, method)
    report.*                                  → markdown / LaTeX tables + Pareto

Supported dataset sources (see `data.py`):
    - GNN-RAG feature pickles       (`gnn_features_<name>_<split>.pkl`)
    - RoG Hugging Face datasets     (`rmanluo/RoG-webqsp`, `rmanluo/RoG-cwq`, ...)
    - generic JSONL                 ({id, question, q_entity, a_entity, graph})
    - synthetic                     (tests / CPU smoke runs)

Quick start (Colab, see `cli.py --help` for all flags):

    python -m kgqa_bench.cli run \
        --train rog:rmanluo/RoG-webqsp:train \
        --eval  rog:rmanluo/RoG-webqsp:test rog:rmanluo/RoG-cwq:test \
        --methods gnnrag_all gnnrag_top10 graphrag_fi cgar prism \
        --out results/bench_v1
    python -m kgqa_bench.cli report --results results/bench_v1
"""
from __future__ import annotations

import os
import sys

# Make the sibling top-level modules (grl_rag, adagraph_cgar_v2,
# baseline_graphrag_fi, significance, prism) importable regardless of cwd.
_GRAPH_RAG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _GRAPH_RAG_DIR not in sys.path:
    sys.path.insert(0, _GRAPH_RAG_DIR)

from .data import BenchDataset, BenchQuestion  # noqa: E402
from .metrics import summarize_records, stratify_records, compare_records  # noqa: E402

__all__ = [
    "BenchDataset", "BenchQuestion",
    "summarize_records", "stratify_records", "compare_records",
]
