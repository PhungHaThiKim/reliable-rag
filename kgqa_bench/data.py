"""Dataset-agnostic KGQA records + loaders.

A `BenchQuestion` is the lowest common denominator both CGAR and PRISM need:

    question text, topic entities, gold answers, a local subgraph (string
    triples), per-candidate retriever scores and a ranked candidate list.

Loaders
-------
load_gnnrag_pkl(path)              GNN-RAG features produced by P7v3 cell 2
load_rog_hf(name, split, ...)      Hugging Face `rmanluo/RoG-*` (needs `datasets`)
load_jsonl(path, ...)              generic {id, question, q_entity, a_entity, graph}
BenchDataset.synthetic(n, seed)    small random KGs with planted gold paths

When a source has no upstream GNN scores (RoG / JSONL), candidates are scored
with `heuristic_candidate_scores` (path-frequency within `max_hops`, optionally
blended with SBERT question-entity similarity). Document this in the paper:
those numbers are "GNN-free" and NOT comparable with GNN-RAG-scored runs.
"""
from __future__ import annotations

import json
import os
import pickle
import random
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

Triple = Tuple[str, str, str]


@dataclass
class BenchQuestion:
    qid: str
    question: str
    q_entities: List[str]
    answers: List[str]
    triples: List[Triple]                    # local subgraph (names, directed h-r-t)
    node_scores: Dict[str, float]            # candidate -> retriever score (higher = better)
    top_candidates: List[str]                # ranked candidates (prefix = top-K)
    meta: Dict = field(default_factory=dict) # hops, n_triples, source, ...

    # --- derived -------------------------------------------------------
    @property
    def n_triples(self) -> int:
        return len(self.triples)

    def to_feature_dict(self) -> Dict:
        """GNN-RAG feature-pickle schema consumed by grl_rag.GNNHeuristic."""
        return {
            "question": self.question,
            "node_scores": dict(self.node_scores),
            "top_candidates": list(self.top_candidates),
            "q_entities": list(self.q_entities),
            "answers": list(self.answers),
            "subgraph_tuples": [list(t) for t in self.triples],
            "gnn_f1": self.meta.get("gnn_f1", 0.0),
            "gnn_hit": self.meta.get("gnn_hit", 0.0),
        }


# ─────────────────────────────────────────────────────────────────────
# Graph helpers
# ─────────────────────────────────────────────────────────────────────

def build_adjacency(triples: Iterable[Triple], undirected: bool = True
                    ) -> Dict[str, List[Tuple[str, str]]]:
    """node -> [(relation, neighbour)]. Inverse edges get an 'inv:' prefix so
    the relation sequence of a path stays informative."""
    adj: Dict[str, List[Tuple[str, str]]] = {}
    for h, r, t in triples:
        h, r, t = str(h), str(r), str(t)
        adj.setdefault(h, []).append((r, t))
        if undirected:
            adj.setdefault(t, []).append(("inv:" + r, h))
    return adj


def bfs_hops(adj, sources: Sequence[str], targets: Sequence[str], max_hops: int = 4
             ) -> Optional[int]:
    """Minimum #hops from any source to any target (None if unreachable)."""
    targets_set = set(targets)
    if not sources or not targets_set:
        return None
    if any(s in targets_set for s in sources):
        return 0
    seen = set(sources)
    frontier = list(sources)
    for depth in range(1, max_hops + 1):
        nxt = []
        for node in frontier:
            for _r, nb in adj.get(node, []):
                if nb in seen:
                    continue
                if nb in targets_set:
                    return depth
                seen.add(nb)
                nxt.append(nb)
        if not nxt:
            break
        frontier = nxt
    return None


def shortest_relation_paths(adj, sources: Sequence[str], max_hops: int,
                            max_paths_per_target: int = 3
                            ) -> Dict[str, List[List[Tuple[str, str, str]]]]:
    """Multi-source BFS. target -> list of paths; each path is a list of
    (head, relation, tail) triples (first `max_paths_per_target` found at the
    shortest depth)."""
    out: Dict[str, List[List[Tuple[str, str, str]]]] = {}
    depth_of: Dict[str, int] = {s: 0 for s in sources}
    queue = deque((s, []) for s in sources)
    src_set = set(sources)
    while queue:
        node, path = queue.popleft()
        d = len(path)
        if d >= max_hops:
            continue
        for r, nb in adj.get(node, []):
            if nb in src_set:
                continue
            nd = d + 1
            if nb not in depth_of:
                depth_of[nb] = nd
            if depth_of[nb] != nd:
                continue  # longer than its shortest path
            lst = out.setdefault(nb, [])
            if len(lst) < max_paths_per_target:
                new_path = path + [(node, r, nb)]
                lst.append(new_path)
                if len(lst) == 1:  # expand each node once (first arrival)
                    queue.append((nb, new_path))
    return out


# ─────────────────────────────────────────────────────────────────────
# Heuristic candidate scoring (for sources without upstream GNN scores)
# ─────────────────────────────────────────────────────────────────────

def heuristic_candidate_scores(question: str, q_entities: Sequence[str],
                               triples: Sequence[Triple], max_hops: int = 3,
                               embedder=None, w_sim: float = 0.5,
                               n_top: int = 20) -> Tuple[Dict[str, float], List[str]]:
    """Score reachable entities by shortest-path frequency (+ optional SBERT
    similarity to the question). Returns (node_scores, ranked_top_candidates).
    """
    adj = build_adjacency(triples)
    paths = shortest_relation_paths(adj, list(q_entities), max_hops,
                                    max_paths_per_target=10)
    if not paths:
        return {}, []
    freq = {ent: len(p) for ent, p in paths.items()}
    max_f = max(freq.values()) or 1
    scores = {e: f / max_f for e, f in freq.items()}
    if embedder is not None and w_sim > 0:
        import torch
        q = embedder.encode(question)
        names = list(scores.keys())
        embs = embedder.encode_batch(names)
        sim = torch.nn.functional.cosine_similarity(embs, q.unsqueeze(0).expand_as(embs), dim=-1)
        sim = (sim - sim.min()) / (sim.max() - sim.min() + 1e-6)
        for n, s in zip(names, sim.tolist()):
            scores[n] = (1 - w_sim) * scores[n] + w_sim * s
    ranked = sorted(scores, key=lambda e: -scores[e])[:n_top]
    return scores, ranked


# ─────────────────────────────────────────────────────────────────────
# Dataset container
# ─────────────────────────────────────────────────────────────────────

class BenchDataset:
    def __init__(self, name: str, split: str, questions: List[BenchQuestion],
                 source: str = "unknown"):
        self.name = name
        self.split = split
        self.questions = questions
        self.source = source
        for q in self.questions:
            q.meta.setdefault("dataset", name)
            q.meta.setdefault("split", split)

    # --- basics ---------------------------------------------------------
    def __len__(self):
        return len(self.questions)

    def __iter__(self):
        return iter(self.questions)

    def __getitem__(self, i) -> BenchQuestion:
        return self.questions[i]

    @property
    def key(self) -> str:
        return f"{self.name}:{self.split}"

    def qids(self) -> List[str]:
        return [q.qid for q in self.questions]

    def subset(self, n: int, seed: int = 0) -> "BenchDataset":
        rng = random.Random(seed)
        qs = list(self.questions)
        rng.shuffle(qs)
        return BenchDataset(self.name, f"{self.split}[{n}]", qs[:n], self.source)

    def filter(self, fn: Callable[[BenchQuestion], bool]) -> "BenchDataset":
        return BenchDataset(self.name, self.split, [q for q in self.questions if fn(q)], self.source)

    def annotate_hops(self, max_hops: int = 4) -> "BenchDataset":
        """Fill meta['hops'] = shortest distance q_entity → any answer inside
        the local subgraph (None = answer not in subgraph: coverage ceiling)."""
        for q in self.questions:
            if "hops" in q.meta:
                continue
            adj = build_adjacency(q.triples)
            q.meta["hops"] = bfs_hops(adj, q.q_entities, q.answers, max_hops)
            q.meta["answer_in_subgraph"] = q.meta["hops"] is not None
        return self

    def stats(self) -> Dict:
        self.annotate_hops()
        n = len(self)
        reach = sum(1 for q in self if q.meta.get("answer_in_subgraph"))
        hops = [q.meta["hops"] for q in self if q.meta.get("hops") is not None]
        return {
            "dataset": self.key, "n": n, "source": self.source,
            "answer_in_subgraph_rate": reach / max(n, 1),
            "mean_hops": sum(hops) / max(len(hops), 1),
            "mean_triples": sum(q.n_triples for q in self) / max(n, 1),
            "mean_n_gold": sum(len(q.answers) for q in self) / max(n, 1),
            "mean_n_candidates": sum(len(q.top_candidates) for q in self) / max(n, 1),
        }

    # --- persistence ----------------------------------------------------
    def save(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"name": self.name, "split": self.split, "source": self.source,
                         "questions": [asdict(q) for q in self.questions]}, f)

    @classmethod
    def load(cls, path: str) -> "BenchDataset":
        with open(path, "rb") as f:
            d = pickle.load(f)
        qs = [BenchQuestion(**{**q, "triples": [tuple(t) for t in q["triples"]]})
              for q in d["questions"]]
        return cls(d["name"], d["split"], qs, d.get("source", "pickle"))

    # --- constructors ---------------------------------------------------
    @classmethod
    def from_gnnrag_pkl(cls, path: str, name: str, split: str = "test") -> "BenchDataset":
        return load_gnnrag_pkl(path, name, split)

    @classmethod
    def from_rog_hf(cls, hf_name: str, split: str, **kw) -> "BenchDataset":
        return load_rog_hf(hf_name, split, **kw)

    @classmethod
    def from_jsonl(cls, path: str, name: str, split: str, **kw) -> "BenchDataset":
        return load_jsonl(path, name, split, **kw)

    @classmethod
    def synthetic(cls, n: int = 40, seed: int = 0, name: str = "synthetic",
                  split: str = "test", n_entities: int = 60, n_relations: int = 12,
                  n_edges: int = 160, n_top: int = 20) -> "BenchDataset":
        """Random KG per question with a planted 1–3 hop gold path and noisy
        scores that favour gold. Deterministic in `seed`."""
        rng = random.Random(seed)
        qs = []
        rels = [f"rel_{i}" for i in range(n_relations)]
        for i in range(n):
            ents = [f"e{i}_{j}" for j in range(n_entities)]
            triples = set()
            for _ in range(n_edges):
                h, t = rng.sample(ents, 2)
                triples.add((h, rng.choice(rels), t))
            topic = ents[0]
            hops = rng.randint(1, 3)
            gold_chain = [topic] + rng.sample(ents[1:], hops)
            for a, b in zip(gold_chain[:-1], gold_chain[1:]):
                triples.add((a, rng.choice(rels), b))
            n_gold = rng.randint(1, 2)
            answers = [gold_chain[-1]]
            if n_gold == 2:
                extra = rng.choice(ents[1:])
                triples.add((gold_chain[-2], rng.choice(rels), extra))
                answers.append(extra)
            triples = sorted(triples)
            scores, ranked = heuristic_candidate_scores(f"q{i}", [topic], triples, max_hops=3,
                                                        n_top=n_top)
            for a in answers:   # plant signal: gold tends to score high but not always
                if a in scores:
                    scores[a] = min(1.0, scores[a] + rng.uniform(0.2, 0.8))
            for e in scores:
                scores[e] = max(0.0, scores[e] + rng.gauss(0, 0.15))
            ranked = sorted(scores, key=lambda e: -scores[e])[:n_top]
            qs.append(BenchQuestion(
                qid=f"{name}-{split}-{i}", question=f"synthetic question {i} about {topic}?",
                q_entities=[topic], answers=answers, triples=triples,
                node_scores=scores, top_candidates=ranked,
                meta={"planted_hops": hops, "source": "synthetic"}))
        return cls(name, split, qs, "synthetic")


# ─────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────

def load_gnnrag_pkl(path: str, name: str, split: str = "test", max_items: Optional[int] = None,
                    subset_seed: Optional[int] = None, **_ignored) -> BenchDataset:
    """GNN-RAG feature pickle (P7v3 cell 2 `load_dataset_features`).
    `max_items` keeps the first N questions, or a seeded random N with `subset_seed`."""
    with open(path, "rb") as f:
        feats = pickle.load(f)
    items = list(feats.items())
    if max_items is not None and max_items < len(items):
        if subset_seed is not None:
            idx = sorted(random.Random(subset_seed).sample(range(len(items)), max_items))
            items = [items[i] for i in idx]
        else:
            items = items[:max_items]
    qs = []
    for qid, d in items:
        triples = []
        for t in d.get("subgraph_tuples", []):
            if len(t) != 3:
                continue
            h, r, tt = t
            h = h["text"] if isinstance(h, dict) else h
            r = r["text"] if isinstance(r, dict) else r
            tt = tt["text"] if isinstance(tt, dict) else tt
            triples.append((str(h), str(r).strip(), str(tt)))
        answers = []
        for a in d.get("answers", []):
            answers.append(a.get("text", a.get("kb_id", str(a))) if isinstance(a, dict) else str(a))
        q_ents = [e.get("text", str(e)) if isinstance(e, dict) else str(e)
                  for e in d.get("q_entities", [])]
        node_scores = {str(k): float(v) for k, v in d.get("node_scores", {}).items()}
        top = [str(c) for c in d.get("top_candidates", [])]
        if not top and node_scores:
            top = sorted(node_scores, key=lambda e: -node_scores[e])[:20]
        qs.append(BenchQuestion(
            qid=str(qid), question=d.get("question", ""), q_entities=q_ents,
            answers=answers, triples=triples, node_scores=node_scores, top_candidates=top,
            meta={"gnn_f1": d.get("gnn_f1", 0.0), "gnn_hit": d.get("gnn_hit", 0.0),
                  "source": "gnnrag_pkl"}))
    return BenchDataset(name, split, qs, source=f"gnnrag_pkl:{os.path.basename(path)}")


def _items_to_dataset(items: Iterable[Dict], name: str, split: str, source: str,
                      max_hops: int = 3, embedder=None, w_sim: float = 0.0,
                      n_top: int = 20, max_triples: Optional[int] = None,
                      scorer: Optional[Callable] = None) -> BenchDataset:
    qs = []
    for i, item in enumerate(items):
        triples = [tuple(map(str, t)) for t in item.get("graph", []) if len(t) == 3]
        if max_triples is not None and len(triples) > max_triples:
            triples = triples[:max_triples]
        q_ents = [str(e) for e in (item.get("q_entity") or [])]
        answers = [str(a) for a in (item.get("a_entity") or item.get("answer") or [])]
        if not q_ents or not triples:
            continue
        if scorer is not None:
            scores, ranked = scorer(item, triples)
        elif "node_scores" in item:  # pre-scored JSONL
            scores = {str(k): float(v) for k, v in item["node_scores"].items()}
            ranked = sorted(scores, key=lambda e: -scores[e])[:n_top]
        else:
            scores, ranked = heuristic_candidate_scores(
                item.get("question", ""), q_ents, triples, max_hops=max_hops,
                embedder=embedder, w_sim=w_sim, n_top=n_top)
        if not ranked:
            continue
        qs.append(BenchQuestion(
            qid=str(item.get("id", f"{name}-{split}-{i}")), question=item.get("question", ""),
            q_entities=q_ents, answers=answers, triples=triples,
            node_scores=scores, top_candidates=ranked,
            meta={"source": source, "scorer": "upstream" if ("node_scores" in item or scorer) else
                  ("pathfreq+sbert" if (embedder is not None and w_sim > 0) else "pathfreq")}))
    return BenchDataset(name, split, qs, source=source)


def load_rog_hf(hf_name: str = "rmanluo/RoG-webqsp", split: str = "test",
                max_items: Optional[int] = None, max_hops: int = 3, embedder=None,
                w_sim: float = 0.0, n_top: int = 20, cache_dir: Optional[str] = None,
                name: Optional[str] = None, subset_seed: Optional[int] = None) -> BenchDataset:
    """`rmanluo/RoG-webqsp` / `rmanluo/RoG-cwq` (fields: id, question, q_entity,
    a_entity, graph). Candidates are scored heuristically (no GNN).
    `max_items` takes the FIRST N items unless `subset_seed` is given (random N)."""
    name = name or hf_name.split("/")[-1].replace("RoG-", "")
    cache_path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        tag = f"{name}_{split}_{max_items or 'all'}{'' if subset_seed is None else f'_s{subset_seed}'}_h{max_hops}_sim{w_sim}.pkl"
        cache_path = os.path.join(cache_dir, tag)
        if os.path.exists(cache_path):
            return BenchDataset.load(cache_path)
    from datasets import load_dataset
    ds = load_dataset(hf_name, split=split)
    if max_items is not None:
        k = min(max_items, len(ds))
        idx = sorted(random.Random(subset_seed).sample(range(len(ds)), k)) if subset_seed is not None else range(k)
        ds = ds.select(idx)
    bench = _items_to_dataset(list(ds), name, split, source=f"hf:{hf_name}",
                              max_hops=max_hops, embedder=embedder, w_sim=w_sim, n_top=n_top)
    if cache_path:
        bench.save(cache_path)
    return bench


def load_jsonl(path: str, name: str, split: str, max_items: Optional[int] = None,
               max_hops: int = 3, embedder=None, w_sim: float = 0.0, n_top: int = 20
               ) -> BenchDataset:
    """Generic JSONL, one question per line:
        {"id": ..., "question": ..., "q_entity": [...], "a_entity": [...],
         "graph": [[h, r, t], ...], "node_scores": {entity: score}  (optional)}
    Convert MetaQA / FreebaseQA / your own KG with this schema and it runs."""
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
            if max_items and len(items) >= max_items:
                break
    return _items_to_dataset(items, name, split, source=f"jsonl:{os.path.basename(path)}",
                             max_hops=max_hops, embedder=embedder, w_sim=w_sim, n_top=n_top)


# ─────────────────────────────────────────────────────────────────────
# Drift injection at DATASET level (applies identically to every method)
# ─────────────────────────────────────────────────────────────────────

def drift_dataset(ds: BenchDataset, pct: float, mode: str = "relation_oov", seed: int = 0,
                  score_noise_std: float = 0.5, unk_rel: str = "UNK_REL") -> BenchDataset:
    """Return a drifted copy of `ds`.

    relation_oov : a fraction `pct` of relation TYPES (sampled over the whole
                   dataset) is renamed to `unk_rel`, i.e. relation vocabulary
                   that appeared after training. PRISM maps it to <UNK>; CGAR
                   sees the literal string 'UNK_REL' through SBERT.
    score_noise  : node_scores of a fraction `pct` of candidates get
                   multiplicative Gaussian noise and top_candidates is re-ranked
                   (an upstream retriever that was not retrained).
    mixed        : both.
    """
    rng = random.Random(seed)
    rel_types = sorted({r for q in ds for (_h, r, _t) in q.triples})
    n_rel = int(len(rel_types) * pct)
    oov = set(rng.sample(rel_types, n_rel)) if (mode in ("relation_oov", "mixed") and n_rel) else set()
    out = []
    for q in ds:
        triples = [(h, (unk_rel if r in oov else r), t) for (h, r, t) in q.triples]
        scores = dict(q.node_scores); top = list(q.top_candidates)
        if mode in ("score_noise", "mixed") and scores:
            cands = list(scores)
            n_drift = int(len(cands) * pct)
            for c in rng.sample(cands, n_drift) if n_drift else []:
                scores[c] = max(1e-6, scores[c] * (1 + rng.gauss(0, score_noise_std)))
            top = sorted(scores, key=lambda e: -scores[e])[: len(q.top_candidates)]
        out.append(BenchQuestion(qid=q.qid, question=q.question, q_entities=list(q.q_entities),
                                 answers=list(q.answers), triples=triples, node_scores=scores,
                                 top_candidates=top, meta={**q.meta, "drift_pct": pct, "drift_mode": mode}))
    return BenchDataset(ds.name, f"{ds.split}+drift{int(pct*100)}{mode}", out, ds.source)


# ─────────────────────────────────────────────────────────────────────
# Spec parser for the CLI:  "gnnrag:/path/x.pkl:webqsp:test" | "rog:rmanluo/RoG-cwq:test"
#                           "jsonl:/path/x.jsonl:metaqa:test" | "synthetic:40:0"
# ─────────────────────────────────────────────────────────────────────

def load_spec(spec: str, **kw) -> BenchDataset:
    parts = spec.split(":")
    kind = parts[0]
    if kind == "gnnrag":
        _, path, name, split = (parts + ["test"])[:4]
        return load_gnnrag_pkl(path, name, split, **{k: v for k, v in kw.items() if k in ("max_items", "subset_seed")})
    if kind == "rog":
        _, hf_name, split = (parts + ["test"])[:3]
        return load_rog_hf(hf_name, split, **kw)
    if kind == "jsonl":
        _, path, name, split = (parts + ["test"])[:4]
        return load_jsonl(path, name, split, **{k: v for k, v in kw.items()
                                                if k in ("max_items", "max_hops", "embedder", "w_sim", "n_top")})
    if kind == "synthetic":
        n = int(parts[1]) if len(parts) > 1 else 40
        seed = int(parts[2]) if len(parts) > 2 else 0
        split = parts[3] if len(parts) > 3 else "test"
        return BenchDataset.synthetic(n, seed, name=f"synthetic{seed}", split=split)
    if kind == "pkl":
        return BenchDataset.load(parts[1])
    raise ValueError(f"unknown dataset spec: {spec}")
