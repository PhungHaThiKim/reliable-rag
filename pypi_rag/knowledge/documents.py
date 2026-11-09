"""Knowledge documents: one retrievable unit with source type and provenance (plan section 8)."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional

from ..schema import read_jsonl, sha256_text, write_jsonl

SOURCE_TYPES = ("rule", "benign_rule", "pattern", "malicious_example", "benign_example", "yara", "advisory",
                "pyguard_pattern")


@dataclass
class KnowledgeDoc:
    doc_id: str
    source_type: str
    title: str
    text: str
    behaviors: List[str] = field(default_factory=list)
    apis: List[str] = field(default_factory=list)
    provenance: Dict = field(default_factory=dict)   # origin, version, origin_sample_id, split, code_sha256, label_hint

    def render(self, max_chars: int = 1200) -> str:
        t = self.text if len(self.text) <= max_chars else self.text[:max_chars] + " ..."
        return f"[{self.source_type}] {self.title}\n{t}"

    def search_text(self) -> str:
        """Text indexed by BM25 / embeddings (title + body + api names + behavior names)."""
        return f"{self.title}\n{self.text}\n{' '.join(self.apis)}\n{' '.join(self.behaviors)}"

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        return KnowledgeDoc(**d)


def save_kb(path: str, docs: Iterable[KnowledgeDoc]) -> int:
    return write_jsonl(path, (d.to_dict() for d in docs))


def load_kb(path: str, source_types: Optional[Iterable[str]] = None) -> List[KnowledgeDoc]:
    st = set(source_types) if source_types else None
    return [KnowledgeDoc.from_dict(d) for d in read_jsonl(path) if st is None or d["source_type"] in st]


def scrub(text: str, names: Iterable[str]) -> str:
    """Replace package names (train examples) by <pkg> so retrieval cannot match on names."""
    for n in sorted({n for n in names if n and len(n) >= 3}, key=len, reverse=True):
        for v in {n, n.replace("-", "_"), n.replace("_", "-")}:
            text = re.sub(rf"(?<![\w]){re.escape(v)}(?![\w])", "<pkg>", text)
    return text


def doc_hash(d: KnowledgeDoc) -> str:
    return sha256_text(d.source_type + "\n" + d.text)
