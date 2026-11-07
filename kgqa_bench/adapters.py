"""Adapters: BenchDataset → (a) GNN-RAG-style features for CGAR/baselines,
(b) PRISM `Sample`s.

Both adapters start from the SAME candidate ranking and the SAME shortest
paths, so the two methods are compared on identical inputs.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .data import BenchDataset, BenchQuestion, build_adjacency, shortest_relation_paths


# ─────────────────────────────────────────────────────────────────────
# (a) CGAR / GNN-RAG side: GNNHeuristic-compatible feature store
# ─────────────────────────────────────────────────────────────────────

def to_gnn_features(dataset: BenchDataset):
    """Return a `grl_rag.GNNHeuristic` whose `.features` is built from the
    BenchDataset (no pickle round-trip). Works with extract_candidate_paths,
    CGARRetriever, GraphRAGFIRetriever, GreedyGNNRetriever unchanged."""
    from grl_rag import GNNHeuristic
    gnn = GNNHeuristic.__new__(GNNHeuristic)   # skip the (noisy) file-loading __init__
    gnn.features = {q.qid: q.to_feature_dict() for q in dataset}
    return gnn


# ─────────────────────────────────────────────────────────────────────
# (b) PRISM side
# ─────────────────────────────────────────────────────────────────────

class PrismVocab:
    """Relation vocabulary shared across datasets. 0 = <PAD>, 1 = <UNK>.
    Fit on the TRAIN datasets only; unseen relations at eval → <UNK>, which
    is exactly the inductive setting PRISM claims to handle."""

    PAD, UNK = 0, 1

    def __init__(self):
        self.rel2id: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1}

    def fit(self, *datasets: BenchDataset) -> "PrismVocab":
        for ds in datasets:
            for q in ds:
                for _h, r, _t in q.triples:
                    self.rel2id.setdefault(r, len(self.rel2id))
                    self.rel2id.setdefault("inv:" + r, len(self.rel2id))
        return self

    def encode(self, rel: str) -> int:
        return self.rel2id.get(rel, self.UNK)

    def __len__(self):
        return len(self.rel2id)

    def state_dict(self):
        return dict(self.rel2id)

    @classmethod
    def from_state_dict(cls, d):
        v = cls(); v.rel2id = dict(d); return v


def _stable_int(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest()[:12], 16)


@dataclass
class PrismBuildConfig:
    n_candidates: int = 32
    max_paths: int = 50
    max_path_len: int = 3
    max_paths_per_cand: int = 3
    force_gold_in_candidates: bool = False   # TRAIN ONLY (label leakage at eval)
    teacher_mode: str = "oracle"             # "oracle" | "score_only"
    undirected: bool = True


def build_prism_samples(dataset: BenchDataset, vocab: PrismVocab, embedder,
                        build_cfg: Optional[PrismBuildConfig] = None,
                        verbose: bool = False) -> List["Sample"]:
    """BenchDataset → list of prism.data.Sample (+ text side-channel attrs).

    Candidate set = top-`n_candidates` of `top_candidates` (same ranking CGAR
    sees). Paths = shortest relation sequences from the topic entities to each
    candidate (same BFS as `heuristic_candidate_scores`). Entity IDs are only
    used for gold matching (never embedded) and are stable hashes of names.
    """
    from prism.data import Sample
    bc = build_cfg or PrismBuildConfig()
    samples: List[Sample] = []
    for q in dataset:
        adj = build_adjacency(q.triples, undirected=bc.undirected)
        paths_by_target = shortest_relation_paths(adj, q.q_entities, bc.max_path_len,
                                                  max_paths_per_target=bc.max_paths_per_cand)
        ranked = [c for c in q.top_candidates if c in paths_by_target]
        gold_set = set(q.answers)
        if bc.force_gold_in_candidates:
            forced = [g for g in q.answers if g in paths_by_target and g not in ranked]
            ranked = forced + ranked
        cands = ranked[: bc.n_candidates]
        if not cands:
            continue
        raw = torch.tensor([float(q.node_scores.get(c, 0.0)) for c in cands])
        raw = raw - raw.min() if raw.numel() else raw
        scores = raw / (raw.sum() + 1e-9) if raw.sum() > 0 else torch.full_like(raw, 1.0 / len(cands))

        paths_list, p2c, path_texts, path_triples = [], [], [], []
        for ci, c in enumerate(cands):
            for p in paths_by_target[c][: bc.max_paths_per_cand]:
                rel_ids = [vocab.encode(r) for (_h, r, _t) in p][: bc.max_path_len]
                paths_list.append(rel_ids + [vocab.PAD] * (bc.max_path_len - len(rel_ids)))
                p2c.append(ci)
                path_texts.append(" -> ".join([p[0][0]] + [x for (_h, r, t) in p for x in (r, t)]))
                path_triples.append([tuple(t) for t in p])
                if len(paths_list) >= bc.max_paths:
                    break
            if len(paths_list) >= bc.max_paths:
                break
        if not paths_list:
            continue

        cand_ids = torch.tensor([_stable_int(c) for c in cands], dtype=torch.long)
        gold_ids = torch.tensor([_stable_int(a) for a in q.answers], dtype=torch.long) \
            if q.answers else cand_ids[:1].clone()
        gold_in_cand = torch.tensor([1.0 if c in gold_set else 0.0 for c in cands])
        if bc.teacher_mode == "oracle" and gold_in_cand.sum() > 0:
            teacher = 0.7 * gold_in_cand / gold_in_cand.sum() + 0.3 * scores
        else:
            teacher = torch.softmax(scores * 3.0, dim=0)
        teacher = teacher / teacher.sum()

        s = Sample(
            qid=q.qid,
            query_emb=embedder.encode(q.question).float().clone(),
            candidate_ids=cand_ids,
            candidate_gnn_scores=scores.float(),
            paths=torch.tensor(paths_list, dtype=torch.long),
            path_to_candidate=torch.tensor(p2c, dtype=torch.long),
            gold_entities=gold_ids,
            teacher_distribution=teacher.float(),
        )
        s.question_text = q.question
        s.candidate_names = list(cands)
        s.gold_names = list(q.answers)
        s.path_texts = path_texts
        s.path_triples = path_triples
        s.meta = dict(q.meta)
        samples.append(s)
    if verbose:
        print(f"[prism adapter] {dataset.key}: {len(samples)}/{len(dataset)} samples "
              f"(vocab={len(vocab)})")
    return samples


class PrismSampleDataset:
    """List-like wrapper matching what prism.distill / rl / experiments expect."""

    def __init__(self, samples: List, name: str = ""):
        self.samples = samples
        self.name = name

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


def move_sample(sample, device):
    for f in ["query_emb", "paths", "path_to_candidate", "candidate_gnn_scores",
              "candidate_ids", "gold_entities", "teacher_distribution"]:
        setattr(sample, f, getattr(sample, f).to(device))
    return sample
