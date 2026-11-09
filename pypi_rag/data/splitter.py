"""Grouped splitting (plan 5.3): related samples never cross partitions.

Groups = union of: same normalized package name, same family_group, identical archive hash, and
near-duplicate code (MinHash-LSH on token shingles, Jaccard >= threshold). Strategies:
  grouped   label-stratified random assignment of whole groups (default 70/15/15)
  time      oldest groups -> train, newest -> test (needs collection_timestamp)
  keep      keep the split each loader set (e.g. the controlled cases, all test); empty -> test
  upstream  keep the upstream test set LOCKED (e.g. mal-LLM test); carve validation out of upstream
            train by group; samples of test that share a group with train are flagged
            meta.cross_split_dup = True (report both with and without them)
"""
from __future__ import annotations

import random
import re
import zlib
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..schema import Sample

_TOK = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|\S")
_P = np.uint64(4294967311)


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def norm_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name.lower()).strip("-")


def shingles(sample: Sample, k: int = 5, max_tokens: int = 20000) -> List[int]:
    toks: List[str] = []
    for f in sorted(sample.files, key=lambda f: f.path):
        if f.path.endswith((".py", ".pth")):
            toks.extend(_TOK.findall(f.text)[: max_tokens - len(toks)])
        if len(toks) >= max_tokens:
            break
    if len(toks) < k:
        return [zlib.crc32(" ".join(toks).encode())] if toks else []
    return list({zlib.crc32(" ".join(toks[i:i + k]).encode()) for i in range(len(toks) - k + 1)})


def minhash_signatures(samples: Sequence[Sample], n_perm: int = 64, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.integers(1, 2 ** 32 - 1, size=n_perm, dtype=np.uint64)
    b = rng.integers(0, 2 ** 32 - 1, size=n_perm, dtype=np.uint64)
    sig = np.full((len(samples), n_perm), np.iinfo(np.uint64).max, dtype=np.uint64)
    for i, s in enumerate(samples):
        sh = shingles(s)
        if not sh:
            continue
        x = np.asarray(sh, dtype=np.uint64)[:, None]
        sig[i] = ((a[None, :] * x + b[None, :]) % _P).min(axis=0)
    return sig


def near_duplicate_pairs(samples: Sequence[Sample], threshold: float = 0.9, n_perm: int = 64,
                         bands: int = 16) -> List[Tuple[int, int, float]]:
    sig = minhash_signatures(samples, n_perm)
    rows = n_perm // bands
    empty = (sig == np.iinfo(np.uint64).max).all(axis=1)
    cand = set()
    for bnd in range(bands):
        buckets: Dict[bytes, List[int]] = defaultdict(list)
        block = sig[:, bnd * rows:(bnd + 1) * rows]
        for i in range(len(samples)):
            if not empty[i]:
                buckets[block[i].tobytes()].append(i)
        for ids in buckets.values():
            if 1 < len(ids) <= 500:
                for x in range(len(ids)):
                    for y in range(x + 1, len(ids)):
                        cand.add((ids[x], ids[y]))
    out = []
    for i, j in cand:
        jac = float((sig[i] == sig[j]).mean())
        if jac >= threshold:
            out.append((i, j, jac))
    return out


def build_groups(samples: Sequence[Sample], near_dup: Optional[float] = 0.9, max_cluster_frac: float = 0.05
                 ) -> Tuple[List[int], Dict]:
    n = len(samples)
    uf = UnionFind(n)
    first: Dict[Tuple[str, str], int] = {}
    for i, s in enumerate(samples):
        for key in (("name", norm_name(s.package_name)), ("family", s.family_group), ("sha", s.sha256)):
            if not key[1]:
                continue
            if key in first:
                uf.union(first[key], i)
            else:
                first[key] = i
    pairs = near_duplicate_pairs(samples, near_dup) if near_dup else []
    # do not let boilerplate (e.g. cookiecutter setup.py) glue a large part of the dataset together
    deg = Counter([i for i, _, _ in pairs] + [j for _, j, _ in pairs])
    cap = max(5, int(max_cluster_frac * n))
    skipped = 0
    for i, j, _ in pairs:
        if deg[i] > cap or deg[j] > cap:
            skipped += 1; continue
        uf.union(i, j)
    gid = [uf.find(i) for i in range(n)]
    sizes = Counter(gid)
    return gid, {"n_groups": len(sizes), "largest_group": max(sizes.values()) if sizes else 0,
                 "n_near_dup_pairs": len(pairs), "near_dup_pairs_skipped_boilerplate": skipped}


def _assign_grouped(idx_by_group: Dict[int, List[int]], samples, ratios, rng) -> Dict[int, str]:
    names = ("train", "val", "test")
    by_label: Dict[str, List[int]] = defaultdict(list)
    for g, ids in idx_by_group.items():
        lab = Counter(samples[i].label for i in ids).most_common(1)[0][0]
        by_label[lab].append(g)
    out: Dict[int, str] = {}
    for lab, groups in by_label.items():
        rng.shuffle(groups)
        groups.sort(key=lambda g: -len(idx_by_group[g]))           # place big groups first
        total = sum(len(idx_by_group[g]) for g in groups)
        target = [r * total for r in ratios]
        have = [0, 0, 0]
        for g in groups:
            k = max(range(3), key=lambda j: (target[j] - have[j]) / max(target[j], 1e-9))
            out[g] = names[k]; have[k] += len(idx_by_group[g])
    return out


def assign_splits(samples: List[Sample], strategy: str = "grouped", ratios=(0.7, 0.15, 0.15), seed: int = 42,
                  near_dup: Optional[float] = 0.9) -> Dict:
    rng = random.Random(seed)
    gid, ginfo = build_groups(samples, near_dup)
    for s, g in zip(samples, gid):
        s.meta["group_id"] = int(g)
    idx_by_group: Dict[int, List[int]] = defaultdict(list)
    for i, g in enumerate(gid):
        idx_by_group[g].append(i)

    if strategy == "grouped":
        gs = _assign_grouped(idx_by_group, samples, ratios, rng)
        for i, s in enumerate(samples):
            s.split = gs[gid[i]]
    elif strategy == "time":
        def gtime(g):
            ts = [samples[i].collection_timestamp for i in idx_by_group[g] if samples[i].collection_timestamp]
            return min(ts) if ts else "9999"
        order = sorted(idx_by_group, key=gtime)
        total = len(samples); acc = 0
        for g in order:
            frac = acc / max(total, 1)
            sp = "train" if frac < ratios[0] else ("val" if frac < ratios[0] + ratios[1] else "test")
            for i in idx_by_group[g]:
                samples[i].split = sp
            acc += len(idx_by_group[g])
    elif strategy == "upstream":
        train_groups = {gid[i] for i, s in enumerate(samples) if s.meta.get("upstream_split") != "test"}
        tr_only = {g: [i for i in ids if samples[i].meta.get("upstream_split") != "test"] for g, ids in idx_by_group.items()}
        tr_only = {g: ids for g, ids in tr_only.items() if ids}
        r_val = ratios[1] / max(ratios[0] + ratios[1], 1e-9)
        gs = _assign_grouped(tr_only, samples, (1 - r_val, r_val, 0.0), rng)
        for i, s in enumerate(samples):
            if s.meta.get("upstream_split") == "test":
                s.split = "test"
                s.meta["cross_split_dup"] = gid[i] in train_groups
            else:
                s.split = gs[gid[i]]
    elif strategy == "keep":
        for s in samples:
            s.split = s.split or "test"
    else:
        raise ValueError(strategy)
    counts = Counter((s.split, s.label) for s in samples)
    rep = {"strategy": strategy, "seed": seed, "ratios": list(ratios), "near_dup_threshold": near_dup, **ginfo,
           "counts": {f"{sp}/{lab}": c for (sp, lab), c in sorted(counts.items())}}
    if strategy == "upstream":
        rep["test_cross_split_dup"] = sum(1 for s in samples if s.meta.get("cross_split_dup"))
    return rep


def grouped_kfold(samples: List[Sample], k: int = 5, seed: int = 42, near_dup: Optional[float] = 0.9) -> List[int]:
    """Fold id per sample (grouped, label-stratified) for small datasets."""
    rng = random.Random(seed)
    gid, _ = build_groups(samples, near_dup)
    groups = defaultdict(list)
    for i, g in enumerate(gid):
        groups[g].append(i)
    keys = list(groups); rng.shuffle(keys)
    keys.sort(key=lambda g: -len(groups[g]))
    fold_size = [0] * k; fold = [0] * len(samples)
    for g in keys:
        f = min(range(k), key=lambda j: fold_size[j])
        for i in groups[g]:
            fold[i] = f
        fold_size[f] += len(groups[g])
    return fold
