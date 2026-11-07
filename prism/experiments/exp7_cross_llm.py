"""Experiment 7 — Cross-LLM Transfer Matrix (Concern A response).

Addresses reviewer's "self-distillation flaw" concern: PRISM should be useful
across DIFFERENT LLM families.

Matrix design:
    ╔═══════════════╤═══════════════╤═══════════════╗
    ║               ║ Reader: Llama  ║ Reader: Mistral║
    ╠═══════════════╪═══════════════╪═══════════════╣
    ║ Teacher: Llama║   diagonal     ║   cross        ║
    ║ Teacher:Mistral║   cross        ║   diagonal     ║
    ╚═══════════════╧═══════════════╧═══════════════╝

If PRISM transfers well, off-diagonal degradation < 5pp.
Optional: online adaptation recovery experiment.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import torch

from ..config import CFG
from ..controller_batched import deterministic_select as prism_select
from ..las import LAS


def evaluate_with_llm_reader(las: LAS, eval_set, llm_reader,
                                cfg=CFG, verbose=True) -> Dict:
    """Run PRISM, get selected paths, feed to llm_reader, compute Hit/F1.

    Args:
        llm_reader: object with .answer(question, paths_text) -> string
    """
    from ..metrics import f1_score, hit_at_k
    from tqdm import tqdm

    hits, f1s, n_paths_list = [], [], []
    device = next(las.parameters()).device

    iterator = tqdm(range(len(eval_set)), desc="cross-LLM eval") if verbose else range(len(eval_set))
    for i in iterator:
        sample = eval_set[i]
        for f in ["query_emb", "paths", "path_to_candidate",
                  "candidate_gnn_scores", "candidate_ids", "gold_entities"]:
            setattr(sample, f, getattr(sample, f).to(device))

        trace = prism_select(las, sample, cfg)
        # If llm_reader is None, use LAS's own answer (degenerate diagonal)
        if llm_reader is None:
            hit = hit_at_k(trace.final_probs, sample.candidate_ids,
                            sample.gold_entities, k=1)
            f1 = f1_score(trace.final_probs, sample.candidate_ids,
                            sample.gold_entities, 0.05)
        else:
            # BUGFIX: the reader used to receive placeholder strings
            # ("_path_3"); now it gets verbalised reasoning paths and is
            # scored against gold NAMES with GNN-RAG-style normalisation.
            path_texts = getattr(sample, "path_texts", None)
            if path_texts:
                paths_text = "\n".join(path_texts[idx] for idx in trace.selected_path_idx)
            else:
                paths_text = "\n".join(f"path {idx}" for idx in trace.selected_path_idx)
            question = getattr(sample, "question_text", f"q_{sample.qid}")
            answer = llm_reader.answer(question, paths_text)
            gold_strs = set(getattr(sample, "gold_names", None) or
                            [str(g.item()) if hasattr(g, 'item') else str(g)
                             for g in sample.gold_entities])
            from ..qa_text_eval import eval_hit, eval_f1, extract_answer_entities
            pred_list = extract_answer_entities(answer)
            hit = eval_hit(" ".join(pred_list), list(gold_strs))
            f1 = eval_f1(pred_list, list(gold_strs))[0]
        hits.append(int(hit))
        f1s.append(float(f1))
        n_paths_list.append(len(trace.selected_path_idx))

    return {
        "hit_at_1": float(sum(hits) / max(len(hits), 1)),
        "f1": float(sum(f1s) / max(len(f1s), 1)),
        "mean_paths": float(sum(n_paths_list) / max(len(n_paths_list), 1)),
        "n": len(hits),
    }


def run_exp7(las_dict: Dict[str, LAS], readers: Dict, eval_set,
              cfg=CFG, out_dir: str = "./results") -> Dict:
    """Run the full transfer matrix.

    Args:
        las_dict: {"teacher_name": trained_LAS}
                  e.g. {"llama3": las_l3, "mistral7": las_mistral}
        readers: {"reader_name": llm_reader_object}
                  reader_object must have .answer(q, paths) -> string

    Returns:
        {(teacher, reader): {hit_at_1, f1, mean_paths}}
    """
    os.makedirs(out_dir, exist_ok=True)
    matrix = {}
    for teacher_name, las in las_dict.items():
        for reader_name, reader in readers.items():
            print(f"\n=== Teacher={teacher_name} × Reader={reader_name} ===")
            key = f"{teacher_name}__{reader_name}"
            matrix[key] = evaluate_with_llm_reader(las, eval_set, reader, cfg)
            matrix[key]["teacher"] = teacher_name
            matrix[key]["reader"] = reader_name

    # Compute diagonal vs off-diagonal averages
    diag_hits = [v["hit_at_1"] for k, v in matrix.items()
                 if v["teacher"].lower().split("-")[0] == v["reader"].lower().split("-")[0]]
    offdiag_hits = [v["hit_at_1"] for k, v in matrix.items()
                    if v["teacher"].lower().split("-")[0] != v["reader"].lower().split("-")[0]]

    out = {
        "matrix": matrix,
        "diagonal_mean_hit": float(sum(diag_hits) / max(len(diag_hits), 1)),
        "off_diagonal_mean_hit": float(sum(offdiag_hits) / max(len(offdiag_hits), 1)),
        "transfer_gap_pp": float((sum(diag_hits) / max(len(diag_hits), 1) -
                                   sum(offdiag_hits) / max(len(offdiag_hits), 1)) * 100)
                            if diag_hits and offdiag_hits else None,
    }

    with open(os.path.join(out_dir, "exp7_cross_llm.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)

    _print_matrix(matrix)
    return out


def _print_matrix(matrix: Dict):
    print("\n=== Cross-LLM Transfer Matrix ===")
    teachers = sorted({v["teacher"] for v in matrix.values()})
    readers = sorted({v["reader"] for v in matrix.values()})
    header = " " * 18 + " | ".join(f"{r:>14}" for r in readers)
    print(header)
    print("-" * len(header))
    for t in teachers:
        row = [f"Teacher: {t:<10}"]
        for r in readers:
            key = f"{t}__{r}"
            if key in matrix:
                row.append(f"{matrix[key]['hit_at_1']*100:>12.2f}%")
            else:
                row.append("       —")
        print("  ".join(row))
