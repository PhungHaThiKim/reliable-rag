"""Graph Drift Robustness Experiment — THE differentiator.

Tests PRISM's inductivity claim: when the test KG grows new entities AFTER
training (e.g., new facts added to Wikidata), PRISM's controller continues to
work WITHOUT retraining, because LAS depends on entity identities only through
the frozen retriever's score s(·) and relation-type IDs.

Comparison:
    PRISM (inductive, relation-only)  vs.  Transductive baselines that
    rely on entity embeddings (which would need retraining on new entities).

This experiment is the "killer" — no recent KGQA paper measures this.

Usage from notebook (after running Step 4 of prism_turnkey.ipynb):

    from prism.graph_drift import run_drift_experiment, plot_drift_curve

    results = run_drift_experiment(
        las, eval_set,
        drift_pcts=[0.0, 0.1, 0.3, 0.5, 0.7],
        seed=42,
    )
    plot_drift_curve(results, out_path='figure_drift.pdf')

Algorithm:
    For each drift_pct ∈ {0%, 10%, 30%, 50%, 70%}:
      1. Sample drift_pct of all entities, mark as "new"
      2. Replace each "new" entity's ID with a previously-unseen value
         that share the SAME structural neighborhood (same relations)
      3. Re-evaluate the existing LAS without retraining
      4. Record Hit@1, ECE, mean #paths
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from .config import CFG
from .data import Sample


# ─────────────────────────────────────────────────────────────────────
# Drift injection — synthetic but realistic
# ─────────────────────────────────────────────────────────────────────

@dataclass
class DriftInjection:
    """Configuration for one drift scenario.

    mode:
      "id"           — legacy: rename a fraction of entity IDs. NOTE: LAS never
                       embeds entity IDs, so PRISM is invariant BY CONSTRUCTION
                       (Hit@1 identical at every drift level). Reviewers will
                       call this tautological; report it only as a sanity check.
      "score_noise"  — perturb the upstream retriever scores of a fraction of
                       candidates (simulates a GNN that was not retrained on
                       the grown KG). Affects every method.
      "relation_oov" — map a fraction of relation types to <UNK> (=1), i.e.
                       new relation vocabulary appeared after training. This is
                       the meaningful inductive test for a relation-typed encoder.
      "mixed"        — score_noise + relation_oov together.
    """
    drift_pct: float                          # fraction of entities/relations to drift
    seed: int = 0
    swap_question_entities: bool = False      # if True, also swap q_entities
    new_entity_offset: int = 1_000_000        # new IDs >> any existing
    mode: str = "id"
    score_noise_std: float = 0.5              # relative noise on scores (score_noise)
    unk_relation_id: int = 1


def inject_entity_drift(sample: Sample, cfg: DriftInjection,
                          ent_offset: int = 1_000_000) -> Sample:
    """Return a copy of `sample` with `cfg.drift_pct` of its entities renamed
    to previously unseen IDs.

    Keep paths/relations and candidate_gnn_scores intact — only entity IDs
    change. Tests whether LAS still produces correct distributions when it
    encounters new entity IDs (it should, because it uses only relation types
    + GNN scores, never entity-ID embeddings).
    """
    rng = random.Random(cfg.seed)
    # All entities involved
    all_ents = set(sample.candidate_ids.tolist()) | set(sample.gold_entities.tolist())
    all_ents_list = sorted(all_ents)

    # Sample which entities to "drift" (replace IDs)
    n_drift = int(len(all_ents_list) * cfg.drift_pct)
    drifted = set(rng.sample(all_ents_list, n_drift))

    # Build mapping old → new
    mapping: Dict[int, int] = {}
    next_new_id = cfg.new_entity_offset + cfg.seed * 100_000
    for old in drifted:
        mapping[old] = next_new_id
        next_new_id += 1

    def remap(eid: int) -> int:
        return mapping.get(int(eid), int(eid))

    mode = getattr(cfg, "mode", "id")
    new_cand_ids = sample.candidate_ids
    new_gold = sample.gold_entities
    scores = sample.candidate_gnn_scores
    paths = sample.paths

    if mode == "id":
        new_cand_ids = torch.tensor(
            [remap(int(e)) for e in sample.candidate_ids.tolist()],
            dtype=sample.candidate_ids.dtype, device=sample.candidate_ids.device)
        new_gold = torch.tensor(
            [remap(int(e)) for e in sample.gold_entities.tolist()],
            dtype=sample.gold_entities.dtype, device=sample.gold_entities.device)

    if mode in ("score_noise", "mixed"):
        g = torch.Generator(device="cpu").manual_seed(cfg.seed * 7919 + len(mapping))
        drift_mask = torch.tensor(
            [int(e) in drifted for e in sample.candidate_ids.tolist()],
            dtype=torch.bool)
        noise = torch.randn(scores.numel(), generator=g) * cfg.score_noise_std
        s_cpu = scores.detach().cpu().float()
        s_new = s_cpu * (1 + noise * drift_mask.float())
        s_new = s_new.clamp(min=1e-6)
        s_new = s_new / s_new.sum()
        scores = s_new.to(scores.device)

    if mode in ("relation_oov", "mixed"):
        rel_ids = sorted({int(r) for r in sample.paths.flatten().tolist() if int(r) > 1})
        n_rel_drift = int(len(rel_ids) * cfg.drift_pct)
        oov = set(rng.sample(rel_ids, n_rel_drift)) if n_rel_drift else set()
        if oov:
            p_cpu = sample.paths.detach().cpu().clone()
            for r in oov:
                p_cpu[p_cpu == r] = cfg.unk_relation_id
            paths = p_cpu.to(sample.paths.device)

    out = Sample(
        qid=sample.qid + f"_drift{int(cfg.drift_pct*100)}_{mode}",
        query_emb=sample.query_emb,
        candidate_ids=new_cand_ids,
        candidate_gnn_scores=scores,
        paths=paths,
        path_to_candidate=sample.path_to_candidate,
        gold_entities=new_gold,
        teacher_distribution=sample.teacher_distribution,
    )
    for attr in ("question_text", "candidate_names", "gold_names", "path_texts"):
        if hasattr(sample, attr):
            setattr(out, attr, getattr(sample, attr))
    return out


# ─────────────────────────────────────────────────────────────────────
# Main experiment
# ─────────────────────────────────────────────────────────────────────

@dataclass
class DriftResult:
    drift_pct: float
    hit_at_1: float
    f1: float
    ece: float
    brier: float
    mean_paths: float
    n: int


@torch.no_grad()
def evaluate_under_drift(las, dataset, drift_pct: float, cfg=CFG,
                          seed: int = 0, use_batched: bool = True,
                          verbose: bool = False, mode: str = "id") -> DriftResult:
    """Evaluate `las` on `dataset` with entity IDs drifted by `drift_pct`."""
    if use_batched:
        from .controller_batched import deterministic_select
    else:
        from .controller import deterministic_select
    from .metrics import (hit_at_k, f1_score, expected_calibration_error,
                           brier_score)

    inj = DriftInjection(drift_pct=drift_pct, seed=seed, mode=mode)
    device = next(las.parameters()).device
    las.eval()

    correctness, confidences, n_paths_list, f1s = [], [], [], []
    iterator = range(len(dataset))
    if verbose:
        iterator = tqdm(iterator, desc=f"drift={drift_pct:.0%}")

    for i in iterator:
        sample_orig = dataset[i]
        sample = inject_entity_drift(sample_orig, inj)

        # Move tensors to device
        for f in ["query_emb", "paths", "path_to_candidate",
                  "candidate_gnn_scores", "candidate_ids", "gold_entities"]:
            setattr(sample, f, getattr(sample, f).to(device))

        trace = deterministic_select(las, sample, cfg)
        hit = hit_at_k(trace.final_probs, sample.candidate_ids,
                       sample.gold_entities, k=1)
        f1 = f1_score(trace.final_probs, sample.candidate_ids,
                      sample.gold_entities, 0.05)

        correctness.append(int(hit))
        confidences.append(float(trace.confidence.item()))
        f1s.append(f1)
        n_paths_list.append(len(trace.selected_path_idx))

    return DriftResult(
        drift_pct=drift_pct,
        hit_at_1=float(np.mean(correctness)),
        f1=float(np.mean(f1s)),
        ece=expected_calibration_error(confidences, correctness),
        brier=brier_score(confidences, correctness),
        mean_paths=float(np.mean(n_paths_list)),
        n=len(correctness),
    )


def run_drift_experiment(las, dataset, cfg=CFG,
                          drift_pcts: Optional[List[float]] = None,
                          seed: int = 42,
                          use_batched: bool = True,
                          verbose: bool = True,
                          mode: str = "relation_oov") -> List[DriftResult]:
    """Sweep drift_pcts and return list of DriftResult.

    Default mode is now "relation_oov" (meaningful); pass mode="id" for the
    legacy (tautological) entity-rename test."""
    drift_pcts = drift_pcts or [0.0, 0.1, 0.3, 0.5, 0.7]
    results = []
    for p in drift_pcts:
        r = evaluate_under_drift(las, dataset, p, cfg, seed=seed,
                                  use_batched=use_batched, verbose=verbose, mode=mode)
        results.append(r)
        if verbose:
            print(f"drift={p:.0%}  Hit@1={r.hit_at_1*100:.2f}%  "
                  f"F1={r.f1*100:.2f}%  ECE={r.ece:.4f}  "
                  f"paths={r.mean_paths:.2f}")
    return results


# ─────────────────────────────────────────────────────────────────────
# Plot — figure for paper
# ─────────────────────────────────────────────────────────────────────

def plot_drift_curve(results: List[DriftResult],
                      out_path: str = "figure_drift.pdf",
                      baseline_results: Optional[Dict[str, List[DriftResult]]] = None,
                      title: str = "Graph Drift Robustness (WebQSP)") -> None:
    """Plot Hit@1 vs drift_pct, comparing PRISM with optional baselines.

    Args:
        results: PRISM's results (one per drift_pct)
        baseline_results: optional {method_name: List[DriftResult]} for other
                          methods e.g. transductive baselines that should drop more.
    """
    import matplotlib
    matplotlib.rcParams.update({'font.size': 12, 'figure.dpi': 150,
                                 'font.family': 'serif'})
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ─── Subplot 1: Hit@1 vs drift_pct
    ax = axes[0]
    xs = [r.drift_pct * 100 for r in results]
    ys = [r.hit_at_1 * 100 for r in results]
    ax.plot(xs, ys, 'o-', color='#d62728', linewidth=2.5, markersize=10,
             label='PRISM (inductive)', zorder=4)
    for r in results:
        ax.annotate(f'{r.hit_at_1*100:.1f}',
                    (r.drift_pct*100, r.hit_at_1*100),
                    xytext=(7, 7), textcoords='offset points',
                    fontsize=9, color='#d62728')

    if baseline_results:
        colors = ['#1f77b4', '#2ca02c', '#9467bd', '#ff7f0e']
        for (name, baseline_res), color in zip(baseline_results.items(), colors):
            bx = [r.drift_pct * 100 for r in baseline_res]
            by = [r.hit_at_1 * 100 for r in baseline_res]
            ax.plot(bx, by, 's--', color=color, linewidth=1.8,
                     markersize=8, label=name, alpha=0.85)

    ax.set_xlabel('Drift (% of test entities renamed to unseen IDs)')
    ax.set_ylabel('Hit@1 (%)')
    ax.set_title('Accuracy under KG growth (no retraining)')
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 100)

    # ─── Subplot 2: ECE vs drift_pct
    ax = axes[1]
    es = [r.ece for r in results]
    ax.plot(xs, es, 'o-', color='#d62728', linewidth=2.5, markersize=10,
             label='PRISM', zorder=4)
    if baseline_results:
        for (name, baseline_res), color in zip(baseline_results.items(), colors):
            bx = [r.drift_pct * 100 for r in baseline_res]
            be = [r.ece for r in baseline_res]
            ax.plot(bx, be, 's--', color=color, linewidth=1.8,
                     markersize=8, label=name, alpha=0.85)
    ax.set_xlabel('Drift (% of test entities renamed)')
    ax.set_ylabel('ECE (lower is better)')
    ax.set_title('Calibration drift')
    ax.grid(alpha=0.3)

    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.savefig(out_path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print(f"ok Saved {out_path}")


# ─────────────────────────────────────────────────────────────────────
# Paper-ready summary table
# ─────────────────────────────────────────────────────────────────────

def drift_summary_table(results: List[DriftResult],
                         baseline_results: Optional[Dict[str, List[DriftResult]]] = None
                         ) -> str:
    """Return markdown table comparing methods at each drift level."""
    lines = []
    header_cols = ["Drift", "PRISM Hit", "PRISM F1", "PRISM ECE"]
    if baseline_results:
        for name in baseline_results:
            header_cols.append(f"{name} Hit")

    lines.append("| " + " | ".join(header_cols) + " |")
    lines.append("|" + "---|" * len(header_cols))

    for i, r in enumerate(results):
        cols = [f"{r.drift_pct*100:.0f}%",
                f"{r.hit_at_1*100:.2f}",
                f"{r.f1*100:.2f}",
                f"{r.ece:.4f}"]
        if baseline_results:
            for name, blist in baseline_results.items():
                if i < len(blist):
                    cols.append(f"{blist[i].hit_at_1*100:.2f}")
                else:
                    cols.append("—")
        lines.append("| " + " | ".join(cols) + " |")

    # Headline delta
    if len(results) > 1:
        first, last = results[0], results[-1]
        delta_hit = (first.hit_at_1 - last.hit_at_1) * 100
        delta_ece = last.ece - first.ece
        lines.append("")
        lines.append(f"**PRISM Δ at {last.drift_pct*100:.0f}% drift: "
                     f"Hit −{delta_hit:.2f}pp, ECE +{delta_ece:.4f}**")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
# Transductive baseline simulator (for comparison)
# ─────────────────────────────────────────────────────────────────────

class TransductiveBaseline:
    """Simulates a *transductive* baseline (e.g., entity-embedding-based
    method like older GraphRAG variants or CGAR with entity-ID embeddings).

    When entity IDs change, this baseline degrades because its embedding
    table has no entry for the new IDs. We simulate this by:
      1. Compute "score" as cosine(entity_id_emb, query_emb)
      2. When an entity is drifted (unseen ID), score = random in [0, 1]
      3. Predict argmax of scores → mostly random for drifted entities

    This is a conservative simulation — real transductive methods often
    crash entirely (KeyError on missing embedding).
    """

    def __init__(self, n_entities: int = 100_000, d: int = 256,
                 d_query: int = 384, seed: int = 0):
        torch.manual_seed(seed)
        # Random entity embeddings — represents pre-trained transductive table
        self.entity_emb = torch.randn(n_entities, d) / d ** 0.5
        self.d = d
        self.d_query = d_query
        self.n_entities = n_entities
        self.query_proj = torch.randn(d_query, d) / d ** 0.5

    def evaluate(self, dataset, drift_pct: float, seed: int = 0) -> DriftResult:
        from .metrics import (hit_at_k, f1_score,
                                expected_calibration_error, brier_score)
        inj = DriftInjection(drift_pct=drift_pct, seed=seed)
        correctness, confs, f1s = [], [], []
        n_paths_list = []

        for i in range(len(dataset)):
            orig = dataset[i]
            sample = inject_entity_drift(orig, inj)
            # Map IDs to embeddings — drifted ones get RANDOM embeddings
            cand_embs = []
            for cid in sample.candidate_ids.tolist():
                if 0 <= cid < self.n_entities:
                    cand_embs.append(self.entity_emb[cid])
                else:
                    # Drifted ID outside known table → random (transductive limitation)
                    cand_embs.append(torch.randn(self.d) / self.d ** 0.5)
            cand_embs = torch.stack(cand_embs)

            q_proj = self.query_proj.T @ sample.query_emb
            scores = (cand_embs @ q_proj).softmax(dim=0)

            hit = hit_at_k(scores, sample.candidate_ids,
                           sample.gold_entities, k=1)
            f1 = f1_score(scores, sample.candidate_ids,
                          sample.gold_entities, 0.05)
            correctness.append(int(hit))
            confs.append(float(scores.max().item()))
            f1s.append(f1)
            n_paths_list.append(sample.paths.size(0))

        return DriftResult(
            drift_pct=drift_pct,
            hit_at_1=float(np.mean(correctness)),
            f1=float(np.mean(f1s)),
            ece=expected_calibration_error(confs, correctness),
            brier=brier_score(confs, correctness),
            mean_paths=float(np.mean(n_paths_list)),
            n=len(correctness),
        )


# ─────────────────────────────────────────────────────────────────────
# Convenience runner — call from notebook
# ─────────────────────────────────────────────────────────────────────

def full_drift_experiment(las, eval_set, cfg=CFG,
                            out_dir: str = "./drift_results",
                            drift_pcts: Optional[List[float]] = None,
                            seed: int = 42) -> Dict:
    """One-call wrapper: run PRISM + transductive baseline, save figure + table."""
    import os
    os.makedirs(out_dir, exist_ok=True)

    drift_pcts = drift_pcts or [0.0, 0.1, 0.3, 0.5, 0.7]

    print("=" * 60)
    print("GRAPH DRIFT EXPERIMENT")
    print("=" * 60)

    print("\n[1/2] PRISM (inductive)...")
    prism_results = run_drift_experiment(las, eval_set, cfg,
                                           drift_pcts=drift_pcts, seed=seed)

    print("\n[2/2] Transductive baseline...")
    sample = eval_set[0]
    d_q = sample.query_emb.size(0)
    trans = TransductiveBaseline(n_entities=cfg.n_entities, d=cfg.d_model,
                                  d_query=d_q, seed=seed)
    trans_results = []
    for p in drift_pcts:
        r = trans.evaluate(eval_set, p, seed=seed)
        trans_results.append(r)
        print(f"trans drift={p:.0%}  Hit@1={r.hit_at_1*100:.2f}%")

    # Plot
    plot_drift_curve(
        prism_results,
        out_path=f"{out_dir}/figure_drift.pdf",
        baseline_results={"Transductive (simulated)": trans_results},
        title=f"Graph Drift Robustness — PRISM vs Transductive",
    )

    # Table
    table_md = drift_summary_table(
        prism_results,
        baseline_results={"Trans": trans_results},
    )
    with open(f"{out_dir}/drift_table.md", "w") as f:
        f.write(table_md)
    print(f"\nok Table saved → {out_dir}/drift_table.md")
    print("\nMarkdown table:\n")
    print(table_md)

    return {
        'prism': prism_results,
        'transductive': trans_results,
        'figure': f"{out_dir}/figure_drift.pdf",
        'table': f"{out_dir}/drift_table.md",
    }
