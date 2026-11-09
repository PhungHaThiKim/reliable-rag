"""Behavior-graph schema (plan 10.2, extended where noted).

Node types: Package, Module, Function, Operation (one call site; the plan's "file-read operation"),
            API, Behavior, DataValue, FileResource, NetworkEndpoint, Pattern
Edge types: CONTAINS, DEFINES, CALLS, INSTANCE_OF, HAS_BEHAVIOR, PRODUCES, FLOWS_TO, READS, WRITES,
            SENDS_TO, FETCHES_FROM, PRECEDES, MATCHES_PATTERN
(INSTANCE_OF, HAS_BEHAVIOR, PRODUCES and FETCHES_FROM are additions to the plan's list.)

Every edge carries: source_file, line_start, line_end, extractor, confidence, evidence_status.
evidence_status = ast_confirmed (read off the AST) | inferred (line order, regex fallback, heuristic
pattern mapping). FLOWS_TO edges exist ONLY where the analyzer found a def-use / nesting chain.
"""
from __future__ import annotations

NODE_TYPES = ("Package", "Module", "Function", "Operation", "API", "Behavior", "DataValue", "FileResource",
              "NetworkEndpoint", "Pattern")
EDGE_TYPES = ("CONTAINS", "DEFINES", "CALLS", "INSTANCE_OF", "HAS_BEHAVIOR", "PRODUCES", "FLOWS_TO", "READS",
              "WRITES", "SENDS_TO", "FETCHES_FROM", "PRECEDES", "MATCHES_PATTERN")
REQUIRED_EDGE_ATTRS = ("source_file", "line_start", "line_end", "extractor", "confidence", "evidence_status")
CONFIRMED_EXTRACTORS = ("ast", "ast_nested", "intraproc_def_use", "interproc_summary", "manifest", "taxonomy",
                        "mode_arg", "literal_arg")


def edge_attrs(etype: str, source_file: str, line_start: int, line_end: int, extractor: str, confidence: float,
               evidence_status: str = None, **extra) -> dict:
    assert etype in EDGE_TYPES, etype
    status = evidence_status or ("ast_confirmed" if extractor in CONFIRMED_EXTRACTORS else "inferred")
    d = dict(etype=etype, source_file=source_file, line_start=int(line_start), line_end=int(line_end),
             extractor=extractor, confidence=float(confidence), evidence_status=status)
    d.update(extra)
    return d


def check_edge(d: dict) -> bool:
    return all(k in d and d[k] is not None for k in REQUIRED_EDGE_ATTRS)
