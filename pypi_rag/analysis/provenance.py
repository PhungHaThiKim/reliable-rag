"""Source provenance checks: does a cited span still exist with the same content?"""
from __future__ import annotations

from typing import Optional

from ..schema import Evidence, Sample, sha256_text


def span_text(sample: Sample, path: str, start: int, end: int) -> Optional[str]:
    f = sample.file(path)
    if f is None or start < 1 or end < start or start > f.n_lines:
        return None
    return "\n".join(f.text.splitlines()[start - 1:end])


def verify_code_evidence(sample: Sample, ev: Evidence) -> bool:
    """True when a code_span / graph evidence item points to existing lines whose hash matches."""
    if ev.file_path is None or ev.start_line is None:
        return ev.source_type not in ("code_span",)
    txt = span_text(sample, ev.file_path, ev.start_line, ev.end_line or ev.start_line)
    if txt is None:
        return False
    want = ev.meta.get("span_sha256")
    return want is None or want == sha256_text(txt)
