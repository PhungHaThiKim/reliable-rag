"""Package representation shared by EVERY method (plan section 7: same raw package representation).

`package_representation` returns a metadata header plus numbered code slices (evidence C1..Cn) chosen
by the static analyzer: windows around sensitive call sites, ranked by execution context
(install > import > runtime), critical behaviors and participation in a data flow, then cut to a
fixed token budget. Methods differ only in what they ADD after this block.

`malllm_representation` reproduces the mal-LLM baseline input (file list + first 300 characters of
setup.py) for the reproduction runs.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

from ..schema import Evidence, Sample
from . import taxonomy as T
from .ast_parser import PackageFacts

CTX_RANK = {"install": 3, "import": 2, "runtime": 1}


def approx_tokens(text: str) -> int:
    return len(text) // 3 + 1


def numbered(lines: List[str], start: int) -> str:
    return "\n".join(f"{start + i:>4}| {l[:240]}" for i, l in enumerate(lines))


def _merge_windows(wins: List[Tuple[int, int, float, set]]) -> List[Tuple[int, int, float, set]]:
    wins = sorted(wins)
    out: List[Tuple[int, int, float, set]] = []
    for s, e, p, b in wins:
        if out and s <= out[-1][1] + 1:
            ps, pe, pp, pb = out[-1]
            out[-1] = (ps, max(pe, e), max(pp, p), pb | b)
        else:
            out.append((s, e, p, set(b)))
    return out


def package_header(sample: Sample, max_files: int = 40) -> str:
    paths = [f.path for f in sample.files]
    shown = paths[:max_files]
    more = f" ... (+{len(paths) - max_files} more)" if len(paths) > max_files else ""
    return (f"Package: {sample.package_name} {sample.version}\n"
            f"Files ({len(paths)}): " + ", ".join(shown) + more)


def package_representation(sample: Sample, facts: PackageFacts, max_tokens: int = 3000, window: int = 3,
                           count_tokens: Callable[[str], int] = approx_tokens, max_slice_lines: int = 40
                           ) -> Tuple[str, List[Evidence], Dict]:
    header = package_header(sample)
    flow_sites = {f.src for f in facts.flows} | {f.dst for f in facts.flows}
    per_file: Dict[str, List[Tuple[int, int, float, set]]] = {}
    for c in facts.sensitive_calls():
        if c.kind == "hook" and c.api == "setup.py.toplevel":
            continue
        crit = len(set(c.behaviors) & set(T.CRITICAL))
        prio = CTX_RANK.get(c.context, 1) * 2 + crit + (2 if c.cid in flow_sites else 0) + 0.5 * len(c.behaviors) \
            + (1 if c.indicators else 0) - (1 if c.kind == "regex" else 0)
        lo = max(1, c.line - window)
        hi = min(c.end_line + window, c.line + max_slice_lines)
        per_file.setdefault(c.file, []).append((lo, hi, prio, set(c.behaviors)))
    cands: List[Tuple[float, str, int, int, set]] = []
    for path, wins in per_file.items():
        for s, e, p, b in _merge_windows(wins):
            cands.append((p, path, s, e, b))
    # small setup.py / __init__.py are always worth showing whole (install / import time code)
    for f in sample.files:
        if f.path.rsplit("/", 1)[-1] in ("setup.py", "__init__.py") and f.n_lines <= 60 and f.text.strip():
            if not any(c[1] == f.path for c in cands):
                cands.append((2.5 if f.path.endswith("setup.py") else 1.5, f.path, 1, f.n_lines, set()))
    if not cands:                                         # nothing sensitive: show file heads
        for f in sample.files[:8]:
            if f.text.strip():
                cands.append((0.1, f.path, 1, min(f.n_lines, 30), set()))
    cands.sort(key=lambda x: (-x[0], x[1], x[2]))

    items: List[Evidence] = []
    used = count_tokens(header)
    dropped = 0
    for p, path, s, e, b in cands:
        sf = sample.file(path)
        if sf is None:
            continue
        lines = sf.text.splitlines()[s - 1:e]
        content = f"{path} lines {s}-{s + len(lines) - 1}\n" + numbered(lines, s)
        t = count_tokens(content)
        if used + t > max_tokens:
            if not items and t > max_tokens:                # always show at least a cut first slice
                keep = max(5, int(len(lines) * max_tokens / t) - 2)
                lines = lines[:keep]
                content = f"{path} lines {s}-{s + len(lines) - 1}\n" + numbered(lines, s)
                t = count_tokens(content)
            else:
                dropped += 1
                continue
        used += t
        items.append(Evidence(evidence_id=f"C{len(items) + 1}", sample_id=sample.sample_id, source_type="code_span",
                              source_ref=f"{path}:{s}-{s + len(lines) - 1}", content=content, score=float(p),
                              file_path=path, start_line=s, end_line=s + len(lines) - 1, behaviors=sorted(b),
                              meta={"span_sha256": _span_sha(sf.text, s, s + len(lines) - 1)}))
    meta = {"n_candidates": len(cands), "n_shown": len(items), "n_dropped": dropped, "truncated": dropped > 0,
            "tokens": used, "parse_coverage": facts.parse_coverage()}
    return header, items, meta


def _span_sha(text: str, s: int, e: int) -> str:
    from ..schema import sha256_text
    return sha256_text("\n".join(text.splitlines()[s - 1:e]))


def malllm_representation(sample: Sample, n_chars: int = 300) -> Dict[str, str]:
    """Input of the mal-LLM setup.py baseline: package name, file list, first `n_chars` of setup.py."""
    sp = sample.file("setup.py")
    file_list = sample.meta.get("file_list") or [f.path for f in sample.files]
    if isinstance(file_list, str):
        file_list = file_list.split()
    return {"package_name": sample.package_name, "file_list": ", ".join(file_list[:200]),
            "snippet": (sp.text[:n_chars] if sp else "Not Available")}
