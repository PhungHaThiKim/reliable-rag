"""Offline script — pre-compute teacher LLM logits for every (sample, subset).

Run ONCE per dataset to populate the teacher cache. Distillation training
then uses CachedTeacherLLM which only reads the cache file.

Usage in Colab:

    %cd /content/drive/MyDrive/ThS/Research/colab/graph_rag/prism
    !python -m prism.cache_teacher_logits \\
        --dataset rmanluo/RoG-webqsp \\
        --splits train,validation \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --out teacher_logits_webqsp.pt \\
        --subsets 3 \\
        --max_items 3000

Or programmatically from a notebook:

    from prism.cache_teacher_logits import main_programmatic
    main_programmatic(hf_name="rmanluo/RoG-webqsp",
                      out_path="teacher_logits_webqsp.pt",
                      max_items=3000)

Compute estimate (A100 80GB, Llama-3.1-8B-bf16, 1 forward / call):
    - 1 (sample, subset) pair ≈ 0.4 s
    - 3000 samples × 3 subsets = 9000 pairs
    - ≈ 60 minutes total
    - Saved as torch dict, ~50 MB
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

from .config import CFG
from .data import RoGDataset, SyntheticKGQA
from .teacher_llm import (LiveTeacherLLM, TeacherConfig,
                          build_teacher_cache)


def _build_id_lookups(rog_ds: RoGDataset):
    """Inverse maps: idx -> string. For prompt verbalisation."""
    ent_lookup = {idx: name for name, idx in rog_ds.ent2id.items()}
    rel_lookup = {idx: name for name, idx in rog_ds.rel2id.items()}
    return rel_lookup, ent_lookup


def _build_question_lookup(rog_ds: RoGDataset):
    """qid → question text."""
    out = {}
    for i in range(len(rog_ds)):
        item = rog_ds.raw[rog_ds._valid_indices[i]]
        qid = str(item.get("id", f"q_{i}"))
        out[qid] = item.get("question", "")
    return out


def main_programmatic(
    hf_name: str = "rmanluo/RoG-webqsp",
    splits: list = ("train",),
    out_path: str = "teacher_logits_webqsp.pt",
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    subsets_per_sample: int = 3,
    max_items: int = None,
    cache_dir: str = "./rog_cache",
    sbert_model: str = "all-MiniLM-L6-v2",
    dtype: str = "bf16",
):
    """Programmatic entry — call this from a notebook."""
    datasets = []
    rel_lookup, ent_lookup, question_lookup = None, None, {}

    for split in splits:
        print(f"=== loading split={split} from {hf_name} ===")
        ds = RoGDataset.from_hf(
            hf_name, split=split,
            sbert_model=sbert_model, cache_dir=cache_dir,
            cfg=CFG, max_items=max_items,
        )
        # Bump CFG once per dataset (idempotent)
        CFG.n_relations = max(CFG.n_relations, ds.n_relations)
        CFG.n_entities = max(CFG.n_entities, ds.n_entities)
        if rel_lookup is None:
            rel_lookup, ent_lookup = _build_id_lookups(ds)
        else:
            r2, e2 = _build_id_lookups(ds)
            rel_lookup.update(r2)
            ent_lookup.update(e2)
        question_lookup.update(_build_question_lookup(ds))
        datasets.append(ds)
        print(f"   {split}: {len(ds)} valid samples, "
              f"{ds.n_relations} relations, {ds.n_entities} entities")

    tcfg = TeacherConfig(model_name=model_name, dtype=dtype)
    teacher = LiveTeacherLLM(tcfg)

    cache = build_teacher_cache(
        *datasets,
        out_path=out_path,
        teacher=teacher,
        subsets_per_sample=subsets_per_sample,
        rel_lookup=rel_lookup,
        ent_lookup=ent_lookup,
        question_lookup=question_lookup,
    )
    print(f"\nfinal cache size: {len(cache):,} entries")
    return cache


def main_synthetic(
    out_path: str = "teacher_logits_synthetic.pt",
    n_train: int = 800,
    n_eval: int = 200,
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    subsets_per_sample: int = 3,
    dtype: str = "bf16",
):
    """Cache for synthetic dataset — for pipeline validation only."""
    print(f"=== synthetic train={n_train} eval={n_eval} ===")
    train_set = SyntheticKGQA(n_train, cfg=CFG, seed=CFG.seed)
    eval_set = SyntheticKGQA(n_eval, cfg=CFG, seed=CFG.seed + 1)

    tcfg = TeacherConfig(model_name=model_name, dtype=dtype)
    teacher = LiveTeacherLLM(tcfg)

    # Synthetic samples have no question text → we pass a placeholder
    question_lookup = {str(i): f"synthetic question {i}"
                       for i in range(n_train + n_eval)}

    cache = build_teacher_cache(
        train_set, eval_set,
        out_path=out_path,
        teacher=teacher,
        subsets_per_sample=subsets_per_sample,
        question_lookup=question_lookup,
    )
    print(f"\nfinal cache size: {len(cache):,} entries")
    return cache


def main_cli():
    ap = argparse.ArgumentParser(description="Pre-compute teacher LLM logits.")
    ap.add_argument("--dataset", default="rmanluo/RoG-webqsp",
                     help="HF dataset name OR 'synthetic'")
    ap.add_argument("--splits", default="train",
                     help="comma-separated list of splits")
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--subsets", type=int, default=3)
    ap.add_argument("--max_items", type=int, default=None)
    ap.add_argument("--cache_dir", default="./rog_cache")
    ap.add_argument("--dtype", default="bf16", choices=["fp16", "bf16", "fp32"])
    args = ap.parse_args()

    if args.dataset == "synthetic":
        main_synthetic(
            out_path=args.out, model_name=args.model,
            subsets_per_sample=args.subsets, dtype=args.dtype,
        )
    else:
        splits = [s.strip() for s in args.splits.split(",")]
        main_programmatic(
            hf_name=args.dataset, splits=splits, out_path=args.out,
            model_name=args.model, subsets_per_sample=args.subsets,
            max_items=args.max_items, cache_dir=args.cache_dir, dtype=args.dtype,
        )


if __name__ == "__main__":
    main_cli()
