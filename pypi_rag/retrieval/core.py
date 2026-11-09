"""Retrieval: dense vectors, BM25, reciprocal-rank fusion, optional cross-encoder reranking (plan 8-9).

    vector  E1 Simple RAG: dense top-k per query, merged by max cosine
    bm25    lexical only (E2 ablation)
    rrf     E2 Hybrid: per-query rankings of BOTH retrievers fused with RRF (+ optional rerank)
All modes share the same documents, queries, final_k and source diversity rule.
"""
from __future__ import annotations

import hashlib
import math
import os
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------------- tokenization
_TOK = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*|\d+")
_CAMEL = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+")
STOP = {"self", "def", "return", "import", "from", "the", "a", "an", "and", "or", "of", "to", "in", "is", "if",
        "else", "for", "with", "as", "none", "true", "false", "not", "lines", "line", "this", "that", "it", "be"}


def code_tokens(text: str) -> List[str]:
    out = []
    for t in _TOK.findall(text):
        lt = t.lower()
        if lt in STOP:
            continue
        out.append(lt)
        if "." in t:
            out.extend(p.lower() for p in t.split(".") if p and p.lower() not in STOP)
        for part in t.replace(".", "_").split("_"):
            if part and part.lower() != lt:
                out.extend(p.lower() for p in _CAMEL.findall(part) if len(p) > 1)
    return out


# ---------------------------------------------------------------------------------- embeddings
class Embedder:
    """sentence-transformers model, or `hash` (hashing trick, CPU, deterministic) for tests."""

    def __init__(self, model_name: str = "hash", device: Optional[str] = None, dim: int = 1024,
                 batch_size: int = 64, query_prefix: str = ""):
        self.model_name, self.dim, self.batch_size = model_name, dim, batch_size
        self.query_prefix = query_prefix
        self.model = None
        if model_name != "hash":
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name, device=device)
            if not query_prefix and "bge" in model_name.lower():
                self.query_prefix = "Represent this sentence for searching relevant passages: "

    def _hash(self, texts: Sequence[str]) -> np.ndarray:
        X = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok, c in Counter(code_tokens(t)).items():
                h = int(hashlib.md5(tok.encode()).hexdigest()[:8], 16)
                X[i, h % self.dim] += (1.0 if (h >> 31) & 1 else -1.0) * (1 + math.log(c))
        n = np.linalg.norm(X, axis=1, keepdims=True); n[n == 0] = 1
        return X / n

    def encode(self, texts: Sequence[str], is_query: bool = False) -> np.ndarray:
        if self.model is None:
            return self._hash(texts)
        texts = [(self.query_prefix + t) if is_query else t for t in texts]
        return np.asarray(self.model.encode(list(texts), batch_size=self.batch_size, normalize_embeddings=True,
                                            show_progress_bar=False), dtype=np.float32)


class VectorIndex:
    def __init__(self, texts: Sequence[str], embedder: Embedder, cache_path: Optional[str] = None):
        self.embedder = embedder
        key = hashlib.sha256(("\x00".join(texts) + embedder.model_name).encode()).hexdigest()[:16]
        if cache_path and os.path.exists(f"{cache_path}.{key}.npy"):
            self.X = np.load(f"{cache_path}.{key}.npy")
        else:
            self.X = embedder.encode(texts) if len(texts) else np.zeros((0, 1), np.float32)
            if cache_path:
                os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
                np.save(f"{cache_path}.{key}.npy", self.X)

    def search(self, queries: Sequence[str], k: int) -> List[List[Tuple[int, float]]]:
        if not len(self.X) or not queries:
            return [[] for _ in queries]
        Q = self.embedder.encode(queries, is_query=True)
        S = Q @ self.X.T
        out = []
        for row in S:
            idx = np.argsort(-row)[:k]
            out.append([(int(i), float(row[i])) for i in idx])
        return out


# ---------------------------------------------------------------------------------- BM25
class BM25:
    def __init__(self, texts: Sequence[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(code_tokens(t)) for t in texts]
        self.len = np.array([sum(d.values()) for d in self.docs], dtype=np.float32)
        self.avg = float(self.len.mean()) if len(self.len) else 1.0
        df = Counter()
        self.post: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        for i, d in enumerate(self.docs):
            for t, c in d.items():
                df[t] += 1; self.post[t].append((i, c))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> np.ndarray:
        s = np.zeros(len(self.docs), dtype=np.float32)
        for t in set(code_tokens(query)):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, c in self.post[t]:
                s[i] += idf * c * (self.k1 + 1) / (c + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return s

    def search(self, queries: Sequence[str], k: int) -> List[List[Tuple[int, float]]]:
        out = []
        for q in queries:
            s = self.scores(q)
            idx = np.argsort(-s)[:k]
            out.append([(int(i), float(s[i])) for i in idx if s[i] > 0])
        return out


def rrf_fuse(rankings: List[List[str]], k: int = 60) -> List[str]:
    """Plan section 9 verbatim."""
    scores: Dict[str, float] = {}
    for ranked_ids in rankings:
        for rank, doc_id in enumerate(ranked_ids, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda doc_id: scores[doc_id], reverse=True)


def rrf_scores(rankings: List[List[str]], k: int = 60) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for ranked in rankings:
        for rank, d in enumerate(ranked, start=1):
            scores[d] = scores.get(d, 0.0) + 1.0 / (k + rank)
    return scores


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base", device: Optional[str] = None, max_chars: int = 1500):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name, device=device, max_length=512)
        self.max_chars = max_chars

    def rerank(self, query: str, texts: Sequence[str]) -> List[float]:
        pairs = [(query[:self.max_chars], t[:self.max_chars]) for t in texts]
        return [float(x) for x in self.model.predict(pairs, show_progress_bar=False)]


# ---------------------------------------------------------------------------------- retriever
class Retriever:
    """Generic retriever over objects with `doc_id`, `source_type`, `search_text()`."""

    def __init__(self, docs: Sequence, mode: str = "rrf", embedder: Optional[Embedder] = None, reranker=None,
                 vector_k: int = 10, bm25_k: int = 10, final_k: int = 5, rrf_k: int = 60,
                 max_per_source: Optional[int] = 3, rerank_top_n: int = 20, cache_path: Optional[str] = None,
                 source_types: Optional[Sequence[str]] = None):
        if source_types:
            docs = [d for d in docs if d.source_type in set(source_types)]
        self.docs = list(docs)
        self.mode, self.reranker = mode, reranker
        self.vector_k, self.bm25_k, self.final_k, self.rrf_k = vector_k, bm25_k, final_k, rrf_k
        self.max_per_source, self.rerank_top_n = max_per_source, rerank_top_n
        texts = [d.search_text() for d in self.docs]
        self.vec = VectorIndex(texts, embedder or Embedder("hash"), cache_path) if mode in ("vector", "rrf") else None
        self.bm25 = BM25(texts) if mode in ("bm25", "rrf") else None

    def _diverse(self, ids: List[int], k: int) -> List[int]:
        out, per = [], Counter()
        for i in ids:
            st = self.docs[i].source_type
            if self.max_per_source and per[st] >= self.max_per_source:
                continue
            out.append(i); per[st] += 1
            if len(out) >= k:
                break
        if len(out) < k:                                  # fill if diversity left room
            out += [i for i in ids if i not in out][: k - len(out)]
        return out

    def retrieve(self, queries: Sequence[str], k: Optional[int] = None) -> List[Tuple[object, float, Dict]]:
        """-> [(doc, score, trace)] best first. trace keeps per-retriever ranks for diagnosis."""
        k = k or self.final_k
        queries = [q for q in queries if q and q.strip()] or [""]
        trace: Dict[int, Dict] = defaultdict(dict)
        if self.mode == "vector":
            best: Dict[int, float] = {}
            for hits in self.vec.search(queries, self.vector_k):
                for r, (i, s) in enumerate(hits, 1):
                    best[i] = max(best.get(i, -1e9), s)
                    trace[i]["vector_rank"] = min(trace[i].get("vector_rank", 1e9), r)
            ranked = sorted(best, key=lambda i: -best[i]); score = best
        elif self.mode == "bm25":
            best = {}
            for hits in self.bm25.search(queries, self.bm25_k):
                for r, (i, s) in enumerate(hits, 1):
                    best[i] = max(best.get(i, -1e9), s)
                    trace[i]["bm25_rank"] = min(trace[i].get("bm25_rank", 1e9), r)
            ranked = sorted(best, key=lambda i: -best[i]); score = best
        elif self.mode == "rrf":
            lists = []
            for name, hits_all in (("vector", self.vec.search(queries, self.vector_k)),
                                   ("bm25", self.bm25.search(queries, self.bm25_k))):
                for hits in hits_all:
                    lists.append([str(i) for i, _ in hits])
                    for r, (i, _s) in enumerate(hits, 1):
                        trace[i][f"{name}_rank"] = min(trace[i].get(f"{name}_rank", 1e9), r)
            sc = rrf_scores(lists, self.rrf_k)
            ranked = [int(x) for x in sorted(sc, key=lambda x: -sc[x])]; score = {int(x): v for x, v in sc.items()}
        else:
            raise ValueError(self.mode)
        if self.reranker is not None and ranked:
            top = ranked[: self.rerank_top_n]
            rs = self.reranker.rerank(" ".join(queries)[:3000], [self.docs[i].search_text() for i in top])
            for i, s in zip(top, rs):
                trace[i]["rerank_score"] = s
            ranked = [i for i, _ in sorted(zip(top, rs), key=lambda x: -x[1])] + ranked[self.rerank_top_n:]
            score = {i: trace[i].get("rerank_score", score.get(i, 0.0)) for i in ranked}
        chosen = self._diverse(ranked, k)
        return [(self.docs[i], float(score.get(i, 0.0)), dict(trace[i])) for i in chosen]


def build_queries(code_items, facts, n: int = 4, max_chars: int = 1500) -> List[str]:
    """Queries shared by every retrieval method: top code slices (line numbers stripped) + an API summary."""
    qs = []
    for it in code_items[:n]:
        body = "\n".join(l.split("| ", 1)[-1] for l in it.content.splitlines()[1:])
        qs.append((body + "\n" + " ".join(it.behaviors))[:max_chars])
    apis = sorted({c.api for c in facts.sensitive_calls()})[:40]
    beh = sorted(facts.behaviors())
    if apis or beh:
        qs.append(f"APIs: {' '.join(apis)}\nBehaviors: {' '.join(beh)}")
    return qs
