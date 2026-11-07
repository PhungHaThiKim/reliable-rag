"""Experiment 9 — Qualitative case studies (Paper §5.7).

Hand-pick 4 representative examples for narrative impact:
    1. Easy success: K=5 selected, high confidence, correct answer
    2. Medium success: K=10, medium confidence, correct
    3. Hard success: K=50, low confidence, correct (calibration discounts)
    4. Hard failure: K=50, LOW confidence (correctly flagged uncertain)

Output: markdown table + JSON dump for paper appendix.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

import torch
from tqdm import tqdm

from ..config import CFG
from ..controller_batched import deterministic_select as prism_select
from ..las import LAS
from ..metrics import hit_at_k


@torch.no_grad()
def collect_case_pool(las: LAS, eval_set, cfg=CFG, verbose=True) -> List[Dict]:
    """Collect rich trace info per sample for downstream selection."""
    device = next(las.parameters()).device
    cases = []
    iterator = tqdm(range(len(eval_set)), desc="collect cases") if verbose else range(len(eval_set))
    for i in iterator:
        sample = eval_set[i]
        for f in ["query_emb", "paths", "path_to_candidate",
                  "candidate_gnn_scores", "candidate_ids", "gold_entities"]:
            setattr(sample, f, getattr(sample, f).to(device))

        trace = prism_select(las, sample, cfg)
        hit = hit_at_k(trace.final_probs, sample.candidate_ids,
                        sample.gold_entities, k=1)
        top3 = trace.final_probs.topk(min(3, trace.final_probs.numel()))
        top3_ents = sample.candidate_ids[top3.indices].tolist()
        top3_probs = top3.values.tolist()

        cases.append({
            "qid": sample.qid,
            "question_text": getattr(sample, "question_text", f"q_{sample.qid}"),
            "selected_paths": list(trace.selected_path_idx),
            "n_paths": len(trace.selected_path_idx),
            "chosen_K": trace.chosen_K,
            "confidence": float(trace.confidence.item()),
            "top3_entities": top3_ents,
            "top3_probs": top3_probs,
            "gold": sample.gold_entities.tolist(),
            "hit": int(hit),
            "step_kl": trace.step_kl,
            "stopped_reason": trace.stopped_reason,
        })
    return cases


def pick_four_cases(cases: List[Dict]) -> Dict[str, Dict]:
    """Pick exemplars: easy-success, medium-success, hard-success, hard-failure."""
    cases_correct = [c for c in cases if c["hit"]]
    cases_wrong = [c for c in cases if not c["hit"]]

    out = {}
    # Easy success: small chosen_K, high confidence, correct
    if cases_correct:
        easy = sorted(cases_correct,
                       key=lambda c: (c["chosen_K"], -c["confidence"]))[0]
        out["easy_success"] = easy

    # Medium success: middle chosen_K, ~middle conf, correct
    if cases_correct:
        sorted_by_K = sorted(cases_correct, key=lambda c: c["chosen_K"])
        med = sorted_by_K[len(sorted_by_K) // 2]
        out["medium_success"] = med

    # Hard success: large chosen_K, correct (low conf or borderline)
    if cases_correct:
        hard = sorted(cases_correct,
                       key=lambda c: (-c["chosen_K"], c["confidence"]))[0]
        out["hard_success"] = hard

    # Hard failure: large chosen_K, LOW confidence, wrong → "abstention winner"
    if cases_wrong:
        flag = sorted(cases_wrong,
                       key=lambda c: (-c["chosen_K"], c["confidence"]))[0]
        out["calibrated_failure"] = flag

    return out


def render_markdown_table(picks: Dict[str, Dict]) -> str:
    """Markdown-ready table for paper appendix."""
    lines = ["| Case | qid | K | #paths | Conf | Gold | Pred-top1 | Hit |",
             "|---|---|---|---|---|---|---|---|"]
    for cname, c in picks.items():
        gold = c["gold"][:2]
        top1 = c["top3_entities"][0] if c["top3_entities"] else "—"
        lines.append(
            f"| {cname} | {c['qid'][:12]} | {c['chosen_K']} | "
            f"{c['n_paths']} | {c['confidence']:.3f} | "
            f"{gold} | {top1} | {c['hit']} |"
        )
    return "\n".join(lines)


def run_exp9(las: LAS, eval_set, cfg=CFG,
              out_dir: str = "./results") -> Dict:
    os.makedirs(out_dir, exist_ok=True)
    cases = collect_case_pool(las, eval_set, cfg)
    picks = pick_four_cases(cases)
    table_md = render_markdown_table(picks)

    out = {
        "picked_cases": picks,
        "markdown_table": table_md,
        "pool_size": len(cases),
    }
    with open(os.path.join(out_dir, "exp9_qualitative.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)

    with open(os.path.join(out_dir, "exp9_qualitative.md"), "w") as f:
        f.write("# PRISM Qualitative Case Studies\n\n")
        f.write(table_md)
        f.write("\n\n## Detailed cases\n\n")
        for cname, c in picks.items():
            f.write(f"### {cname.replace('_', ' ').title()}\n")
            f.write(f"- **Question:** {c['question_text']}\n")
            f.write(f"- **K chosen:** {c['chosen_K']} (paths selected: {c['n_paths']})\n")
            f.write(f"- **Confidence:** {c['confidence']:.3f}\n")
            f.write(f"- **Gold answers:** {c['gold']}\n")
            f.write(f"- **Top-3 predicted:** {list(zip(c['top3_entities'], c['top3_probs']))}\n")
            f.write(f"- **Hit:** {bool(c['hit'])}\n")
            f.write(f"- **Stop reason:** {c['stopped_reason']}\n")
            f.write(f"- **KL trace:** {c['step_kl'][:5]}{'...' if len(c['step_kl']) > 5 else ''}\n\n")
    print(f"ok Saved exp9 case studies → {out_dir}")
    print(f"\n{table_md}")
    return out
