"""Shared data contracts (plan section 5.2 and 6.1).

Every method sees a `Sample` WITHOUT its label (`Sample.public_view()`), and returns a
`PredictionResult`. Evidence items carry an id that the model must cite (C*, K*, G*, F*).
"""
from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, Iterator, List, Literal, Optional

Verdict = Literal["benign", "malicious", "uncertain"]
LABELS = ("benign", "malicious", "unknown")
POSITIVE = "malicious"


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_text(s: str) -> str:
    return sha256_bytes(s.encode("utf-8", errors="replace"))


@dataclass
class SourceFile:
    path: str                      # path inside the package (posix, no leading slash)
    sha256: str                    # hash of the ORIGINAL bytes
    text: str                      # decoded text (never executed)
    n_lines: int = 0
    truncated: bool = False
    encoding: str = "utf-8"

    def line(self, i: int) -> str:
        lines = self.text.splitlines()
        return lines[i - 1] if 1 <= i <= len(lines) else ""

    def span(self, start: int, end: int) -> str:
        lines = self.text.splitlines()
        return "\n".join(lines[max(start - 1, 0):max(end, 0)])


@dataclass
class Sample:
    sample_id: str
    package_name: str
    version: str
    label: str                                  # benign | malicious | unknown  (NEVER shown to methods)
    files: List[SourceFile] = field(default_factory=list)
    ecosystem: str = "pypi"
    label_source: str = ""
    sha256: str = ""                            # archive hash, or hash of the sorted file manifest
    split: str = ""
    source_ref: str = ""
    family_group: str = ""
    collection_timestamp: str = ""
    meta: Dict = field(default_factory=dict)    # parse warnings, archive info, ...
    gold: Dict = field(default_factory=dict)    # annotations (gold_behaviors, gold_paths); evaluation only

    # ---- views -------------------------------------------------------------
    def public_view(self) -> "Sample":
        """Copy without label / gold / family: what a method is allowed to see."""
        return Sample(self.sample_id, self.package_name, self.version, "unknown", self.files,
                      self.ecosystem, "", self.sha256, "", "", "", "", dict(self.meta), {})

    def file(self, path: str) -> Optional[SourceFile]:
        for f in self.files:
            if f.path == path:
                return f
        return None

    def manifest_record(self) -> Dict:
        """Normalized record of plan 5.2 (no source text)."""
        return {
            "sample_id": self.sample_id, "ecosystem": self.ecosystem, "package_name": self.package_name,
            "version": self.version, "label": self.label, "label_source": self.label_source,
            "sha256": self.sha256, "split": self.split,
            "file_manifest": [{"path": f.path, "sha256": f.sha256, "n_lines": f.n_lines} for f in self.files],
            "source_ref": self.source_ref, "family_group": self.family_group,
            "collection_timestamp": self.collection_timestamp,
        }

    def to_dict(self) -> Dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict) -> "Sample":
        d = dict(d)
        d["files"] = [SourceFile(**f) for f in d.get("files", [])]
        return Sample(**d)


@dataclass
class Evidence:
    evidence_id: str
    sample_id: str
    source_type: str                  # code_span | pattern | rule | example | graph_path | fact
    source_ref: str                   # doc id / file path / path id
    content: str
    score: Optional[float] = None
    file_path: Optional[str] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    behaviors: List[str] = field(default_factory=list)
    content_sha256: str = ""
    meta: Dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.content_sha256:
            self.content_sha256 = sha256_text(self.content)


@dataclass
class PredictionResult:
    sample_id: str
    method: str
    verdict: str                                       # benign | malicious | uncertain
    explanation: str = ""
    evidence: List[Evidence] = field(default_factory=list)
    claimed_evidence_ids: List[str] = field(default_factory=list)
    latency_ms: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    run_id: str = ""
    # extras kept for diagnosis / offline re-scoring
    confidence: Optional[float] = None
    claims: List[Dict] = field(default_factory=list)   # [{text, behavior, evidence:[ids], negated}]
    raw_response: str = ""
    prompt_sha256: str = ""
    parse_ok: bool = True
    parse_error: str = ""
    truncated: bool = False
    model: str = ""
    raw_verdict: str = ""                              # verdict before the E4 policy
    verification: Dict = field(default_factory=dict)
    extra: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict) -> "PredictionResult":
        d = dict(d)
        d["evidence"] = [Evidence(**e) for e in d.get("evidence", [])]
        return PredictionResult(**d)


# ---- jsonl helpers -------------------------------------------------------------

def _open(path: str, mode: str):
    return gzip.open(path, mode + "t", encoding="utf-8") if path.endswith(".gz") else open(path, mode, encoding="utf-8")


def write_jsonl(path: str, rows: Iterable[Dict]) -> int:
    n = 0
    with _open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=float) + "\n"); n += 1
    return n


def read_jsonl(path: str) -> Iterator[Dict]:
    with _open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def save_samples(path: str, samples: Iterable[Sample]) -> int:
    return write_jsonl(path, (s.to_dict() for s in samples))


def load_samples(path: str) -> List[Sample]:
    return [Sample.from_dict(d) for d in read_jsonl(path)]
