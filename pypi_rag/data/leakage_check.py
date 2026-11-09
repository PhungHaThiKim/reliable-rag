"""Leakage audit (plan 5.3 / acceptance checklist): split leakage and knowledge-base leakage."""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Sequence

from ..schema import Sample
from .splitter import near_duplicate_pairs, norm_name


def split_leakage(samples: Sequence[Sample], near_dup: float = 0.9, min_file_bytes: int = 200) -> Dict:
    by_split = defaultdict(list)
    for s in samples:
        by_split[s.split].append(s)
    rep: Dict = {"counts": {sp: dict(Counter(x.label for x in v)) for sp, v in by_split.items()}}

    def cross(key_fn):
        seen = defaultdict(set)
        for s in samples:
            for k in key_fn(s):
                if k:
                    seen[k].add(s.split)
        return sum(1 for v in seen.values() if len(v) > 1)

    rep["exact_archive_dup_across_splits"] = cross(lambda s: [s.sha256])
    rep["file_hash_dup_across_splits"] = cross(lambda s: [f.sha256 for f in s.files if len(f.text) >= min_file_bytes
                                                          and f.path.endswith(".py")])
    rep["package_name_across_splits"] = cross(lambda s: [norm_name(s.package_name)])
    rep["family_across_splits"] = cross(lambda s: [s.family_group])
    pairs = near_duplicate_pairs(list(samples), near_dup)
    rep["near_dup_pairs_across_splits"] = sum(1 for i, j, _ in pairs if samples[i].split != samples[j].split)
    rep["near_dup_pairs_total"] = len(pairs)
    rep["test_cross_split_dup_flagged"] = sum(1 for s in samples if s.meta.get("cross_split_dup"))
    return rep


def kb_leakage(docs: Iterable, samples: Sequence[Sample], eval_splits=("val", "test"), min_name_len: int = 5,
               name_check_types: Sequence[str] = ("advisory",)) -> Dict:
    """docs: knowledge.documents.KnowledgeDoc. Flags docs derived from eval samples, docs that contain an
    eval file verbatim, and analyst write-ups (advisories) that name an eval package. Names are not checked
    in rules / code examples, where popular names such as `requests` appear legitimately."""
    ev = [s for s in samples if s.split in eval_splits]
    ev_ids = {s.sample_id for s in ev}
    names = {norm_name(s.package_name) for s in ev if len(s.package_name) >= min_name_len}
    file_hashes = {f.sha256 for s in ev for f in s.files if len(f.text) >= 200}
    name_re = re.compile(r"(?<![\w-])(" + "|".join(sorted(map(re.escape, names), key=len, reverse=True)[:5000]) + r")(?![\w-])",
                         re.I) if names else None
    bad_origin, bad_name, bad_code = [], [], []
    n = 0
    for d in docs:
        n += 1
        if d.provenance.get("origin_sample_id") in ev_ids or d.provenance.get("split") in eval_splits:
            bad_origin.append(d.doc_id)
        if d.source_type in name_check_types and (
                (name_re is not None and name_re.search(d.text.replace("_", "-"))) or
                any(norm_name(p) in names for p in d.provenance.get("packages", []) or [])):
            bad_name.append(d.doc_id)
        if d.provenance.get("code_sha256") in file_hashes:
            bad_code.append(d.doc_id)
    return {"n_docs": n, "docs_from_eval_samples": bad_origin, "n_docs_from_eval_samples": len(bad_origin),
            "docs_mentioning_eval_package": bad_name, "n_docs_mentioning_eval_package": len(bad_name),
            "docs_with_eval_code": bad_code, "n_docs_with_eval_code": len(bad_code),
            "clean": not (bad_origin or bad_code or bad_name)}


def drop_leaky_docs(docs: List, report: Dict, drop_name_mentions: bool = True) -> List:
    """Remove every flagged doc (eval-derived, eval code verbatim, and optionally eval package names)."""
    bad = set(report.get("docs_from_eval_samples", [])) | set(report.get("docs_with_eval_code", []))
    if drop_name_mentions:
        bad |= set(report.get("docs_mentioning_eval_package", []))
    return [d for d in docs if d.doc_id not in bad]


def summarize(report: Dict, k: int = 10) -> Dict:
    """Same report with id lists cut to k examples (for printing / results.json)."""
    return {key: (v[:k] if isinstance(v, list) else v) for key, v in report.items()}
