"""Build the text knowledge base (E1/E2/E3a) from TRAIN data + external knowledge, then audit leakage.

Sources (select with `sources`):
  builtin     hand-written rules, benign counterexamples, behavior patterns (knowledge/builtin.py)
  examples    code slices of TRAIN packages: malicious examples and benign counterexamples that use
              sensitive APIs (package names scrubbed, provenance = origin sample id)
  malllm_code whole malicious TRAIN setup.py files (mal-LLM's third index)
  pyguard     PyGuard mined patterns (bundle from knowledge.pyguard_adapter)
  yara        YARA rules (.yar file or directory)
  advisories  GitHub advisories JSON (knowledge.external.fetch_advisories)
  case_rules  the records of the test-case document (PATTERN-EXFIL-001, MAL-001, BEN-001/002, RULE-001);
              only for the cases dataset (knowledge/case_rules.py)
"""
from __future__ import annotations

import random
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from ..analysis import analyze_package, package_representation
from ..data.leakage_check import drop_leaky_docs, kb_leakage, summarize
from ..schema import Sample, sha256_text
from . import builtin
from .documents import KnowledgeDoc, doc_hash, scrub

DEFAULT_SOURCES = ("builtin", "examples", "pyguard")


def builtin_docs() -> List[KnowledgeDoc]:
    docs = []
    for rid, title, beh, text, hint in builtin.RULES:
        docs.append(KnowledgeDoc(doc_id=f"builtin:{rid}", source_type="benign_rule" if hint == "benign_indicator" else "rule",
                                 title=title, text=text, behaviors=list(beh), apis=[],
                                 provenance={"origin": "builtin:v1", "split": "external", "label_hint": hint}))
    for p in builtin.BEHAVIOR_PATTERNS:
        docs.append(KnowledgeDoc(doc_id=f"builtin:{p.pattern_id}", source_type="pattern", title=p.title, text=p.text(),
                                 behaviors=sorted({b for s in p.steps for b in s}), apis=[],
                                 provenance={"origin": p.origin, "split": "external", "label_hint": p.label_hint,
                                             "pattern_id": p.pattern_id}))
    return docs


def example_docs(samples: Sequence[Sample], facts_of: Callable, max_per_label: int = 1500, slices_per_sample: int = 2,
                 slice_tokens: int = 450, seed: int = 0) -> List[KnowledgeDoc]:
    rng = random.Random(seed)
    by_label: Dict[str, List[KnowledgeDoc]] = {"malicious": [], "benign": []}
    seen = set()
    for s in samples:
        if s.split != "train" or s.label not in by_label:
            continue
        facts = facts_of(s)
        _h, items, _m = package_representation(s, facts, max_tokens=slice_tokens * slices_per_sample)
        items = [it for it in items if it.behaviors] or (items[:1] if s.label == "malicious" else [])
        for it in items[:slices_per_sample]:
            code = scrub(it.content, [s.package_name])
            key = sha256_text(code.split("\n", 1)[-1])
            if key in seen:
                continue
            seen.add(key)
            f = s.file(it.file_path)
            title = ("Code from a known-malicious training package" if s.label == "malicious"
                     else "Code from a benign training package that uses sensitive APIs")
            beh = ", ".join(it.behaviors) or "none detected"
            by_label[s.label].append(KnowledgeDoc(
                doc_id=f"ex:{s.sample_id}:{it.evidence_id}", source_type=f"{s.label}_example",
                title=f"{title} (behaviors: {beh})", text=code, behaviors=list(it.behaviors),
                apis=sorted({c.api for c in facts.sensitive_calls() if c.file == it.file_path
                             and it.start_line <= c.line <= it.end_line}),
                provenance={"origin": "train_split", "origin_sample_id": s.sample_id, "split": "train",
                            "label_hint": s.label, "code_sha256": f.sha256 if f else ""}))
    out = []
    for lab, docs in by_label.items():
        rng.shuffle(docs)
        out.extend(docs[:max_per_label])
    return out


def malllm_code_docs(samples: Sequence[Sample], max_chars: int = 3000) -> List[KnowledgeDoc]:
    docs, seen = [], set()
    for s in samples:
        sp = s.file("setup.py")
        if s.split != "train" or s.label != "malicious" or sp is None:
            continue
        txt = scrub(sp.text[:max_chars], [s.package_name])
        h = sha256_text(txt)
        if h in seen:
            continue
        seen.add(h)
        docs.append(KnowledgeDoc(doc_id=f"mlcode:{s.sample_id}", source_type="malicious_example",
                                 title="Malicious setup.py from the training set", text=txt, behaviors=[], apis=[],
                                 provenance={"origin": "mal-LLM:malicious_setup_py", "origin_sample_id": s.sample_id,
                                             "split": "train", "label_hint": "malicious", "code_sha256": sp.sha256,
                                             "subtype": "malllm_setup"}))
    return docs


def build_kb(samples: Sequence[Sample], facts_of: Callable, sources: Iterable[str] = DEFAULT_SOURCES,
             pyguard_bundle: Optional[Dict] = None, yara_path: Optional[str] = None,
             advisories_path: Optional[str] = None, max_examples_per_label: int = 1500, seed: int = 0,
             drop_name_mentions: bool = True) -> Tuple[List[KnowledgeDoc], Dict]:
    sources = list(sources)
    docs: List[KnowledgeDoc] = []
    counts: Dict[str, int] = {}
    if "builtin" in sources:
        d = builtin_docs(); docs += d; counts["builtin"] = len(d)
    if "examples" in sources:
        d = example_docs(samples, facts_of, max_per_label=max_examples_per_label, seed=seed); docs += d; counts["examples"] = len(d)
    if "malllm_code" in sources:
        d = malllm_code_docs(samples); docs += d; counts["malllm_code"] = len(d)
    if "pyguard" in sources and pyguard_bundle:
        from .pyguard_adapter import pattern_docs
        d = pattern_docs(pyguard_bundle["patterns"], pyguard_bundle.get("taxonomy", {}), pyguard_bundle.get("origin", "pyguard"))
        docs += d; counts["pyguard"] = len(d)
    if "yara" in sources and yara_path:
        from .external import load_yara
        d = load_yara(yara_path); docs += d; counts["yara"] = len(d)
    if "advisories" in sources and advisories_path:
        from .external import load_advisories
        d = load_advisories(advisories_path); docs += d; counts["advisories"] = len(d)
    if "case_rules" in sources:
        from .case_rules import case_rule_docs
        d = case_rule_docs(); docs += d; counts["case_rules"] = len(d)
    # dedupe identical content
    uniq, seen = [], set()
    for d in docs:
        h = doc_hash(d)
        if h not in seen:
            seen.add(h); uniq.append(d)
    leak = kb_leakage(uniq, samples)
    clean = drop_leaky_docs(uniq, leak, drop_name_mentions=drop_name_mentions)
    after = kb_leakage(clean, samples)
    report = {"sources": sources, "counts_before_audit": counts, "n_docs": len(clean), "n_dropped": len(uniq) - len(clean),
              "leakage_before": summarize(leak), "leakage_after": summarize(after),
              "by_source_type": _count(clean)}
    return clean, report


def _count(docs):
    c: Dict[str, int] = {}
    for d in docs:
        c[d.source_type] = c.get(d.source_type, 0) + 1
    return c
