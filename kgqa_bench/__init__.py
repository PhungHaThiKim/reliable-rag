"""kgqa_bench: one harness to run the GraphRAG retrievers (GNN-RAG all / top-K,
SBERT top-K, CGAR, CGAR-G) and an LLM reader on a KGQA dataset with ONE metric
suite, plus a graph-grounded hallucination verifier.

    BenchDataset  (dataset-agnostic question records, data.py)
        -> adapters.to_gnn_features(ds)       shared candidate ranking + shortest paths
    methods.*                                 per-question `records`
    llm_reader.attach_llm_answers             LLM answers (+ candidate prompt, answer grounding)
    metrics.summarize_records(records)        one metric dict per (dataset, method)
    hallucination.run                         claim-level SUPPORTED / UNSUPPORTED / UNCERTAIN
    rescore.rescore_dir                       offline re-scoring (parser, verify) without GPU
    reader_ft                                 LoRA fine-tuning of the reader

Quick start (Colab, see `cli.py --help` for all flags):

    python -m kgqa_bench.cli run \
        --train gnnrag:ckpt/gnn_features_train.pkl:webqsp:train \
        --eval  gnnrag:ckpt/gnn_features_webqsp_test.pkl:webqsp:test \
        --methods gnnrag_all gnnrag_top5 sbert_top5 cgar cgar_esc \
        --reader hf:Qwen/Qwen2.5-7B-Instruct:cand --reader-verify --out results/demo
    python -m kgqa_bench.cli report --results results/demo
"""
from __future__ import annotations

import os
import sys

# Make the sibling top-level modules (grl_rag, adagraph_cgar_v2, significance,
# qa_text_eval, calibration) importable regardless of cwd.
_GRAPH_RAG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _GRAPH_RAG_DIR not in sys.path:
    sys.path.insert(0, _GRAPH_RAG_DIR)

from .data import BenchDataset, BenchQuestion  # noqa: E402
from .metrics import summarize_records, stratify_records, compare_records  # noqa: E402

__all__ = [
    "BenchDataset", "BenchQuestion",
    "summarize_records", "stratify_records", "compare_records",
]
