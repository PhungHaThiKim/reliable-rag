"""Data layer for PRISM.

Provides:
- SyntheticKGQA: self-contained generator that mimics GNN-RAG outputs.
- RoGDataset: real WebQSP/CWQ via Hugging Face `rmanluo/RoG-*` (no GNN-RAG needed).
- WebQSPDataset: stub loader for real GNN-RAG features (.pkl).
- Sample dataclass: unified format consumed by models.
- TeacherLLM: synthetic oracle producing P(answer | query, subgraph) used as
  Stage 1 distillation target. Replace with actual LLM logits for real runs.
"""
from __future__ import annotations
import os
import pickle
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import CFG


@dataclass
class Sample:
    qid: str
    query_emb: torch.Tensor                 # [d_query]
    candidate_ids: torch.Tensor             # [C]   entity indices
    candidate_gnn_scores: torch.Tensor      # [C]
    paths: torch.Tensor                     # [N_paths, max_path_len]  relation IDs (0 = pad)
    path_to_candidate: torch.Tensor         # [N_paths]  index into candidate_ids
    gold_entities: torch.Tensor             # [G]   entity indices
    teacher_distribution: torch.Tensor      # [C]   target P(answer | q, full S)


def _seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


class SyntheticKGQA(Dataset):
    """Procedurally generates KGQA examples.

    Each example: a question encoded as a random direction in ℝ^d_query, a set of
    candidate entities with GNN scores correlated to a hidden 'gold' projection,
    and shortest paths whose relation sequences embed signal about the gold entity.
    """

    def __init__(self, n: int, cfg=CFG, seed: Optional[int] = None):
        _seed(cfg.seed if seed is None else seed)
        self.cfg = cfg
        self.samples: List[Sample] = [self._gen(i) for i in range(n)]

    def _gen(self, qid: int) -> Sample:
        c = self.cfg
        rng = np.random.RandomState(c.seed + qid)

        query_emb = torch.tensor(rng.randn(c.d_query) / np.sqrt(c.d_query), dtype=torch.float32)

        n_cand = min(c.n_candidates, c.n_entities - 1)
        candidate_ids = torch.tensor(
            rng.choice(c.n_entities, size=n_cand, replace=False), dtype=torch.long
        )

        gold_idx_in_cand = int(rng.choice(min(8, n_cand)))
        gold_entity = candidate_ids[gold_idx_in_cand].item()
        n_gold = int(rng.randint(1, 4))
        extra_golds = rng.choice(n_cand, size=n_gold - 1, replace=False) if n_gold > 1 else []
        gold_indices = sorted({gold_idx_in_cand, *[int(x) for x in extra_golds]})
        gold_entities = candidate_ids[gold_indices]

        gold_signal = torch.zeros(n_cand)
        gold_signal[gold_indices] = 1.0
        noise = torch.tensor(rng.randn(n_cand) * 0.3, dtype=torch.float32)
        candidate_gnn_scores = torch.softmax(2.0 * gold_signal + noise, dim=0)

        paths_list, p2c_list = [], []
        min_paths = max(4, min(15, c.max_paths - 1))
        n_paths = int(rng.randint(min_paths, max(min_paths + 1, c.max_paths)))
        n_relations = c.n_relations
        gold_relation_set = set(rng.choice(n_relations - 1, size=4, replace=False) + 1)

        for _ in range(n_paths):
            cand_idx = int(rng.choice(n_cand))
            path_len = int(rng.randint(1, c.max_path_len + 1))

            if cand_idx in gold_indices:
                gold_rels = list(gold_relation_set)
                pool = gold_rels + list(rng.choice(n_relations - 1, size=path_len, replace=True) + 1)
            else:
                pool = list(rng.choice(n_relations - 1, size=path_len, replace=True) + 1)

            chosen_rels = list(rng.choice(pool, size=path_len, replace=False) if len(pool) >= path_len
                                else rng.choice(pool, size=path_len, replace=True))
            padded = chosen_rels + [0] * (c.max_path_len - path_len)
            paths_list.append(padded)
            p2c_list.append(cand_idx)

        paths = torch.tensor(paths_list, dtype=torch.long)
        path_to_candidate = torch.tensor(p2c_list, dtype=torch.long)

        teacher_distribution = self._teacher(gold_indices, candidate_gnn_scores, paths,
                                              path_to_candidate, gold_relation_set, n_cand)

        return Sample(
            qid=str(qid),
            query_emb=query_emb,
            candidate_ids=candidate_ids,
            candidate_gnn_scores=candidate_gnn_scores,
            paths=paths,
            path_to_candidate=path_to_candidate,
            gold_entities=gold_entities,
            teacher_distribution=teacher_distribution,
        )

    @staticmethod
    def _teacher(gold_indices, gnn_scores, paths, p2c, gold_rels, n_cand):
        """Synthetic oracle distribution. Real runs replace with LLM logits."""
        evidence = torch.zeros(n_cand)
        for i in range(paths.size(0)):
            cand = int(p2c[i])
            rels = paths[i].tolist()
            n_gold_rels = sum(1 for r in rels if r in gold_rels)
            evidence[cand] += n_gold_rels
        score = 1.5 * gnn_scores + 0.5 * (evidence / (evidence.max() + 1e-6))
        return torch.softmax(score * 3.0, dim=0)

    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]


# =============================================================================
# RoGDataset — real WebQSP/CWQ from Hugging Face, no GNN-RAG features needed.
# =============================================================================

class RoGDataset(Dataset):
    """Real KGQA via `rmanluo/RoG-webqsp` or `rmanluo/RoG-cwq` HF datasets.

    Each HF item has: id, question, q_entity, a_entity, graph (triples).
    We build:
      - relation/entity vocabs (cached to disk for reproducibility);
      - BFS adjacency from `graph`;
      - candidates = entities reachable within max_path_len hops from q_entity;
      - paths   = shortest-path relation sequences from q_entity to each candidate;
      - GNN-style scores = path-frequency softmax (proxy for GNN-RAG scores);
      - teacher distribution = heuristic gold + score blend (replace with real
        LLM later via TeacherLLM subclass).

    Usage:
        ds = RoGDataset.from_hf("rmanluo/RoG-webqsp", split="train",
                                 sbert_model="all-MiniLM-L6-v2",
                                 cache_dir="./rog_cache", cfg=CFG)
        # IMPORTANT: bump cfg before instantiating LAS:
        CFG.n_relations = ds.n_relations
        CFG.n_entities  = ds.n_entities
    """

    def __init__(
        self,
        hf_items: list,
        rel2id: Dict[str, int],
        ent2id: Dict[str, int],
        sbert,
        cfg=CFG,
        max_paths_per_cand: int = 5,
        force_gold_in_candidates: bool = False,
        teacher_mode: str = "oracle",
        cache_samples: bool = True,
    ):
        """
        force_gold_in_candidates: if True, reachable gold entities are
            inserted into the candidate set regardless of their score.
            WARN: This is LABEL LEAKAGE at evaluation time (the candidate pool
            is built from the answer). Only ever enable it for a TRAIN split
            (it raises the fraction of trainable questions). Default False.
        teacher_mode: "oracle" — heuristic teacher mixes the gold one-hot
            with the scores (supervised target; NOT an LLM simulation);
            "score_only" — gold-free teacher from scores (sanity baseline).
            Real-LLM runs replace this via CachedTeacherLLM.
        cache_samples: memoise built Samples (BFS + SBERT are the cost).
        """
        self.raw = hf_items
        self.rel2id = rel2id
        self.ent2id = ent2id
        self.sbert = sbert
        self.cfg = cfg
        self.max_paths_per_cand = max_paths_per_cand
        self.force_gold_in_candidates = force_gold_in_candidates
        self.teacher_mode = teacher_mode
        self.cache_samples = cache_samples
        self._sample_cache: Dict[int, Sample] = {}

        self._valid_indices = self._scan_valid()

    @classmethod
    def from_hf(
        cls,
        hf_name: str = "rmanluo/RoG-webqsp",
        split: str = "train",
        sbert_model: str = "all-MiniLM-L6-v2",
        cache_dir: Optional[str] = None,
        cfg=CFG,
        max_items: Optional[int] = None,
        force_gold_in_candidates: bool = False,
        teacher_mode: str = "oracle",
        rel2id: Optional[Dict[str, int]] = None,
        ent2id: Optional[Dict[str, int]] = None,
    ):
        """rel2id/ent2id: pass the TRAIN vocab when building eval/other-dataset
        splits so relation IDs are shared (needed for cross-dataset transfer
        and for the drift experiment). Unseen relations map to <UNK>=1."""
        from datasets import load_dataset

        ds = load_dataset(hf_name, split=split)
        if max_items is not None:
            ds = ds.select(range(min(max_items, len(ds))))

        items = list(ds)
        if rel2id is None or ent2id is None:
            rel2id, ent2id = cls._build_vocab(items, cache_dir, key=f"{hf_name.replace('/','_')}_{split}")
        else:
            # extend ENTITY vocab (entity ids are only used for gold matching,
            # never embedded) but keep the relation vocab fixed + <UNK>.
            rel2id = dict(rel2id); rel2id.setdefault("<UNK>", 1 if 1 not in rel2id.values() else len(rel2id))
            ent2id = dict(ent2id)
            for item in items:
                for triple in item.get("graph", []):
                    if len(triple) != 3: continue
                    h, _r, t = triple
                    ent2id.setdefault(h, len(ent2id)); ent2id.setdefault(t, len(ent2id))
                for e in (item.get("q_entity") or []) + (item.get("a_entity") or []):
                    ent2id.setdefault(e, len(ent2id))

        sbert = None
        if sbert_model:
            from sentence_transformers import SentenceTransformer
            sbert = SentenceTransformer(sbert_model)

        return cls(items, rel2id, ent2id, sbert, cfg=cfg,
                   force_gold_in_candidates=force_gold_in_candidates,
                   teacher_mode=teacher_mode)

    @staticmethod
    def _build_vocab(items, cache_dir, key):
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            cache_path = os.path.join(cache_dir, f"vocab_{key}.pkl")
            if os.path.exists(cache_path):
                with open(cache_path, "rb") as f:
                    return pickle.load(f)

        rel2id: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
        ent2id: Dict[str, int] = {}

        for item in items:
            for triple in item.get("graph", []):
                if len(triple) != 3:
                    continue
                h, r, t = triple
                if r not in rel2id:
                    rel2id[r] = len(rel2id)
                if h not in ent2id:
                    ent2id[h] = len(ent2id)
                if t not in ent2id:
                    ent2id[t] = len(ent2id)
            for e in item.get("q_entity", []) or []:
                if e not in ent2id:
                    ent2id[e] = len(ent2id)
            for e in item.get("a_entity", []) or []:
                if e not in ent2id:
                    ent2id[e] = len(ent2id)

        if cache_dir:
            with open(cache_path, "wb") as f:
                pickle.dump((rel2id, ent2id), f)

        return rel2id, ent2id

    @property
    def n_relations(self) -> int:
        return len(self.rel2id)

    @property
    def n_entities(self) -> int:
        return len(self.ent2id)

    def _scan_valid(self) -> List[int]:
        """Pre-filter items that yield ≥1 candidate path (avoid empty Samples)."""
        valid = []
        for i, item in enumerate(self.raw):
            if not item.get("q_entity") or not item.get("graph"):
                continue
            if not any(e in self.ent2id for e in item["q_entity"]):
                continue
            valid.append(i)
        return valid

    def __len__(self) -> int:
        return len(self._valid_indices)

    def _bfs(self, adj, sources, max_depth) -> Dict[int, List[List[int]]]:
        """Multi-source BFS. Returns candidate_id -> list of relation-id paths."""
        paths_to: Dict[int, List[List[int]]] = {}
        queue = deque((s, [], 0) for s in sources)
        sources_set = set(sources)
        seen_at_depth: Dict[Tuple[int, int], int] = {}

        while queue:
            node, path, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for r, t in adj.get(node, []):
                if t in sources_set and depth == 0:
                    continue
                new_path = path + [r]
                if t not in sources_set:
                    paths_to.setdefault(t, []).append(new_path)
                key = (t, depth + 1)
                if key not in seen_at_depth:
                    seen_at_depth[key] = 1
                    queue.append((t, new_path, depth + 1))
        return paths_to

    def __getitem__(self, idx: int) -> Sample:
        if self.cache_samples and idx in self._sample_cache:
            return self._sample_cache[idx]
        sample = self._build(idx)
        if self.cache_samples:
            self._sample_cache[idx] = sample
        return sample

    def _build(self, idx: int) -> Sample:
        item = self.raw[self._valid_indices[idx]]
        c = self.cfg

        if self.sbert is not None:
            q_emb_np = self.sbert.encode(item["question"], show_progress_bar=False)
            query_emb = torch.tensor(q_emb_np, dtype=torch.float32)
        else:
            rng = np.random.RandomState(abs(hash(item["question"])) % (2**31))
            query_emb = torch.tensor(rng.randn(c.d_query) / np.sqrt(c.d_query),
                                     dtype=torch.float32)

        adj: Dict[int, List[Tuple[int, int]]] = {}
        unk = self.rel2id.get("<UNK>", 1)
        for triple in item["graph"]:
            if len(triple) != 3:
                continue
            h, r, t = triple
            if h not in self.ent2id or t not in self.ent2id:
                continue
            adj.setdefault(self.ent2id[h], []).append((self.rel2id.get(r, unk), self.ent2id[t]))

        q_ents = [self.ent2id[e] for e in item["q_entity"] if e in self.ent2id]
        cand_paths = self._bfs(adj, q_ents, c.max_path_len)

        if not cand_paths:
            return self._fallback_sample(item, query_emb)

        # Optionally force reachable gold into the candidate set. Helps
        # TRAINING (more questions with a learnable target) but is label
        # leakage at EVAL time — gated by self.force_gold_in_candidates.
        gold_raw = item.get("a_entity", []) or []
        gold_ent_ids = [self.ent2id[e] for e in gold_raw if e in self.ent2id]
        if self.force_gold_in_candidates:
            forced = [(g, len(cand_paths[g])) for g in gold_ent_ids if g in cand_paths]
        else:
            forced = []
        forced_set = {g for g, _ in forced}

        # Top-K by path frequency, excluding already-forced golds
        scores = {cand: len(paths) for cand, paths in cand_paths.items()
                  if cand not in forced_set}
        n_remain = max(0, c.n_candidates - len(forced))
        top_non_gold = sorted(scores.items(), key=lambda x: -x[1])[: n_remain]

        # Final candidates: golds first, then top non-gold
        top = forced + top_non_gold
        top = top[: c.n_candidates]
        if not top:
            return self._fallback_sample(item, query_emb)

        candidate_ids = torch.tensor([x[0] for x in top], dtype=torch.long)
        raw_scores = torch.tensor([float(x[1]) for x in top], dtype=torch.float32)
        # Normalize instead of sharp softmax to keep teacher distribution diffuse.
        # Sharp softmax on raw path counts → near one-hot on the most-frequent
        # candidate, leaving zero learning signal for the rest.
        gnn_scores = raw_scores / (raw_scores.sum() + 1e-9)

        cand_id_to_idx = {int(c.item()): i for i, c in enumerate(candidate_ids)}
        paths_list, p2c_list = [], []
        for cand, p_list in cand_paths.items():
            if cand not in cand_id_to_idx:
                continue
            ci = cand_id_to_idx[cand]
            for p in p_list[: self.max_paths_per_cand]:
                truncated = list(p[: c.max_path_len])
                padded = truncated + [0] * (c.max_path_len - len(truncated))
                paths_list.append(padded)
                p2c_list.append(ci)
                if len(paths_list) >= c.max_paths:
                    break
            if len(paths_list) >= c.max_paths:
                break

        if not paths_list:
            return self._fallback_sample(item, query_emb)

        paths = torch.tensor(paths_list, dtype=torch.long)
        path_to_candidate = torch.tensor(p2c_list, dtype=torch.long)

        # Re-use gold_ent_ids computed earlier
        gold_entities = (torch.tensor(gold_ent_ids, dtype=torch.long)
                          if gold_ent_ids else candidate_ids[:1].clone())

        gold_set = set(gold_ent_ids)
        gold_in_cand = torch.tensor(
            [1.0 if int(cid.item()) in gold_set else 0.0 for cid in candidate_ids],
            dtype=torch.float32,
        )
        # Softer teacher: gold gets weighted bump rather than near-one-hot. Mix
        # with gnn_scores so non-gold candidates still carry some mass (gives
        # LAS something to learn beyond a trivial 1-of-N classifier).
        if self.teacher_mode == "oracle" and gold_in_cand.sum() > 0:
            gold_dist = gold_in_cand / gold_in_cand.sum()
            teacher_distribution = 0.7 * gold_dist + 0.3 * gnn_scores
        else:
            # gold-free: gnn_scores already normalized; sharpen slightly
            teacher_distribution = torch.softmax(gnn_scores * 3.0, dim=0)
        teacher_distribution = teacher_distribution / teacher_distribution.sum()

        sample = Sample(
            qid=str(item.get("id", idx)),
            query_emb=query_emb,
            candidate_ids=candidate_ids,
            candidate_gnn_scores=gnn_scores,
            paths=paths,
            path_to_candidate=path_to_candidate,
            gold_entities=gold_entities,
            teacher_distribution=teacher_distribution,
        )
        # Text side-channel for LLM readers / teacher prompts (not used by LAS).
        id2ent = getattr(self, "_id2ent", None)
        if id2ent is None:
            id2ent = self._id2ent = {v: k for k, v in self.ent2id.items()}
        id2rel = getattr(self, "_id2rel", None)
        if id2rel is None:
            id2rel = self._id2rel = {v: k for k, v in self.rel2id.items()}
        q_names = [id2ent.get(e, str(e)) for e in q_ents]
        sample.question_text = item.get("question", "")
        sample.candidate_names = [id2ent.get(int(e), str(int(e))) for e in candidate_ids.tolist()]
        sample.gold_names = list(gold_raw)
        sample.path_texts = []
        for p_rels, ci in zip(paths_list, p2c_list):
            rel_names = [id2rel.get(r, f"r{r}") for r in p_rels if r != 0]
            head = q_names[0] if q_names else "?"
            sample.path_texts.append(" -> ".join([head] + rel_names + [sample.candidate_names[ci]]))
        return sample

    def _fallback_sample(self, item, query_emb) -> Sample:
        """Minimal valid Sample when BFS yields nothing."""
        c = self.cfg
        candidate_ids = torch.zeros(2, dtype=torch.long)
        gnn_scores = torch.tensor([0.5, 0.5], dtype=torch.float32)
        paths = torch.zeros(1, c.max_path_len, dtype=torch.long)
        path_to_candidate = torch.zeros(1, dtype=torch.long)
        gold_entities = torch.zeros(1, dtype=torch.long)
        teacher_distribution = torch.tensor([0.5, 0.5], dtype=torch.float32)
        return Sample(
            qid=str(item.get("id", "fallback")),
            query_emb=query_emb,
            candidate_ids=candidate_ids,
            candidate_gnn_scores=gnn_scores,
            paths=paths,
            path_to_candidate=path_to_candidate,
            gold_entities=gold_entities,
            teacher_distribution=teacher_distribution,
        )


class WebQSPDataset(Dataset):
    """Loader for real GNN-RAG-pickled features.

    Expects pkl shape: list of dicts with keys
        qid, query_text, candidates [{id, score}], paths [{relations, candidate_idx}], gold_ids
    """

    def __init__(self, pkl_path: str, query_encoder, cfg=CFG):
        with open(pkl_path, "rb") as f:
            self.raw = pickle.load(f)
        self.query_encoder = query_encoder
        self.cfg = cfg

    def __len__(self): return len(self.raw)

    def __getitem__(self, i) -> Sample:
        raise NotImplementedError(
            "Adapt your gnn_features.pkl schema in WebQSPDataset.__getitem__. "
            "See SyntheticKGQA._gen for the target Sample layout."
        )


class TeacherLLM:
    """Stub for real LLM teacher.

    For synthetic mode: uses Sample.teacher_distribution directly.
    For real mode: subclass and override `score_subgraph` to call your LLM and
    return P(answer | query, subgraph_S) over candidate entities.
    """

    def score_subgraph(self, sample: Sample, selected_path_idx: List[int]) -> torch.Tensor:
        """Return P(a | q, S_selected) over sample.candidate_ids.

        Default behavior: re-weight `teacher_distribution` by mass concentrated on
        candidates touched by selected paths. For real LLM, replace this whole body.
        """
        if not selected_path_idx:
            uniform = torch.ones_like(sample.teacher_distribution)
            return uniform / uniform.sum()

        touched = sample.path_to_candidate[torch.tensor(selected_path_idx)]
        cand_mass = torch.zeros_like(sample.teacher_distribution)
        for c in touched.tolist():
            cand_mass[c] += 1.0
        cand_mass = cand_mass / cand_mass.sum()

        mixed = 0.7 * sample.teacher_distribution + 0.3 * cand_mass
        return mixed / mixed.sum()


def collate(batch: List[Sample]) -> List[Sample]:
    """No padding/stacking — variable shapes handled per-sample in models."""
    return batch


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool = True):
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate
    )
