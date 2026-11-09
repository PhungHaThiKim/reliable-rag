"""Adapter for PyGuard (github.com/lxyeternal/PyGuard) behavioral knowledge. Reads files of a local
(sparse) checkout at runtime; nothing from PyGuard is vendored here (the upstream repo has no license
file, so keep it as an external, pinned dependency and cite it).

Files used (paths relative to the repo root; verified against commit f684d0ab, 2025-12-19):
  Resources/Taxonomy/action_categories.json                      hierarchy of 327 behavior ids
  Core/ActionSequence/action_sequences/<subset>/*.json            LLM-mapped `api_name -> action id` per snippet
  Core/PatternGenerator/analysis_results/optimized_patterns/optimized_patterns_id.json   mined action-id patterns

What we take:
  * an API -> behavior lexicon (majority vote of PyGuard's per-snippet mappings), merged into the
    taxonomy with origin `pyguard@<sha>` (provenance is kept per rule);
  * PyGuard patterns as knowledge documents (E1/E2) and as Pattern nodes matched by action subsequence (E3).
PyGuard's pickled knowledge base (*.pkl) is never loaded: unpickling untrusted files can execute code.
"""
from __future__ import annotations

import glob
import json
import os
import re
import subprocess
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

from .documents import KnowledgeDoc

TAXONOMY = "Resources/Taxonomy/action_categories.json"
ACTION_SEQ_DIR = "Core/ActionSequence/action_sequences"
PATTERN_FILES = ("Core/PatternGenerator/analysis_results/optimized_patterns/optimized_patterns_id.json",
                 "Core/PatternGenerator/hierarchical_patterns/enhanced_patterns_id.json",
                 "Core/PatternGenerator/pattern_results/patterns_id.json")

_MAP = [
    (r"exfiltrat|upload|send_(http_)?post|post_request|webhook|smtp|send_email|telegram|discord|send_data|transmit", "network_send"),
    (r"download|http_get|fetch|retrieve_url|urlopen|recv|receive", "network_download"),
    (r"socket|connect|network_connection|open_connection", "network_connect"),
    (r"shell|subprocess|execute_command|run_command|os_system|popen|spawn|command_execution", "process_exec"),
    (r"exec_python|eval|exec_code|compile_code|dynamic_code|marshal|pickle|code_execution", "code_exec"),
    (r"base64|encode|decode|obfusc|encrypt|decrypt|compress|xor|hex", "obfuscation"),
    (r"environ|env_var|getenv|environment", "env_access"),
    (r"hostname|username|platform|system_info|mac_address|ip_address|information_gathering|fingerprint|get_user", "system_info"),
    (r"credential|password|token|cookie|browser|wallet|ssh_key|keychain|secret", "sensitive_data"),
    (r"read_file|open_file|file_read|list_dir|walk|glob|enumerate_files|read_content", "file_read"),
    (r"write_file|create_file|file_write|copy_file|move_file|save_file|write_content", "file_write"),
    (r"delete|remove_file|rmtree|unlink", "file_delete"),
    (r"persist|startup|registry|crontab|autorun|scheduled", "persistence"),
    (r"setuptools_install|install_hook|post_install|cmdclass|exec_setuptools", "install_hook"),
    (r"clipboard|keylog|screenshot|screen_capture", "surveillance"),
    (r"import_module|dynamic_import|__import__", "dynamic_import"),
    (r"ctypes|native|dll|shellcode", "native_code"),
]
_MAP_RE = [(re.compile(p, re.I), b) for p, b in _MAP]
_API_RE = re.compile(r"^[A-Za-z_]\w*(\.[A-Za-z_]\w*)*$")
SINGLE_BUILTINS = {"open", "exec", "eval", "compile", "__import__", "execfile", "getattr"}


def repo_sha(repo: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"], text=True).strip()[:12]
    except Exception:
        return "unknown"


def action_to_behaviors(action_id: str, path: Iterable[str] = ()) -> List[str]:
    s = " ".join([action_id] + list(path))
    out = []
    for rx, b in _MAP_RE:
        if rx.search(s) and b not in out:
            out.append(b)
    return out


def load_taxonomy(repo: str) -> Dict[str, Dict]:
    """action id -> {description, path:[category names/ids]}"""
    p = os.path.join(repo, TAXONOMY)
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    out: Dict[str, Dict] = {}

    def walk(node, path):
        if isinstance(node, dict):
            if "behaviors" in node and isinstance(node["behaviors"], list):
                for b in node["behaviors"]:
                    if isinstance(b, dict) and "id" in b:
                        out[b["id"]] = {"description": b.get("description", ""), "path": path + [node.get("id", "")]}
            for k, v in node.items():
                if isinstance(v, (list, dict)) and k != "behaviors":
                    walk(v, path + ([node.get("id") or node.get("name")] if node.get("id") or node.get("name") else []))
        elif isinstance(node, list):
            for x in node:
                walk(x, path)
    walk(data, [])
    return out


def _norm_api(api: str) -> Optional[str]:
    a = api.strip()
    a = a.split("(", 1)[0].strip().rstrip(".")
    if a in SINGLE_BUILTINS:
        return f"builtins.{a}"
    return a if _API_RE.match(a) and "." in a else None


def harvest_api_lexicon(repo: str, subsets: Iterable[str] = ("malware", "malware_fn", "benign_bandit4mal", "benign_guarddog"),
                        max_files: Optional[int] = None) -> Dict[str, Dict]:
    """api -> {action_id, n, behaviors, path}: majority vote of PyGuard's mapped_sequence entries."""
    votes: Dict[str, Counter] = defaultdict(Counter)
    paths: Dict[str, List[str]] = {}
    for sub in subsets:
        files = sorted(glob.glob(os.path.join(repo, ACTION_SEQ_DIR, sub, "*.json")))
        for fp in files[:max_files] if max_files else files:
            try:
                with open(fp, encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError, OSError):
                continue
            for entry in data if isinstance(data, list) else []:
                pa = entry.get("pattern_analysis") if isinstance(entry, dict) else None
                for m in (pa or {}).get("mapped_sequence", []) or []:
                    if not isinstance(m, dict):
                        continue
                    api = _norm_api(str(m.get("api_name", "")))
                    aid = m.get("id")
                    if api and aid:
                        votes[api][aid] += 1
                        paths[aid] = [str(m.get(k, "")) for k in ("first_id", "second_id", "third_id")]
    out = {}
    for api, c in votes.items():
        aid, n = c.most_common(1)[0]
        out[api] = {"action_id": aid, "n": int(sum(c.values())), "behaviors": action_to_behaviors(aid, paths.get(aid, [])),
                    "path": paths.get(aid, [])}
    return out


def lexicon_rules(lexicon: Dict[str, Dict], origin: str, min_count: int = 3) -> List[Tuple[str, List[str], str]]:
    return [(api, v["behaviors"], origin) for api, v in lexicon.items() if v["n"] >= min_count and v["behaviors"]]


def load_patterns(repo: str) -> List[Dict]:
    """[{pattern_id, steps:[action ids], type, support, source_file}] from the first pattern file found."""
    for rel in PATTERN_FILES:
        p = os.path.join(repo, rel)
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        found: List[Dict] = []

        def walk(node, parent_key=""):
            if isinstance(node, dict):
                pat = node.get("pattern")
                if isinstance(pat, (list, tuple)) and pat and all(isinstance(x, str) for x in pat):
                    found.append({"steps": list(pat), "type": str(node.get("type") or parent_key or "unknown"),
                                  "support": node.get("support"), "coverage": node.get("coverage")})
                    return
                for k, v in node.items():
                    if k in ("benign_sequences", "malware_sequences", "benign_cases", "malware_cases"):
                        continue
                    walk(v, k if isinstance(v, (list, dict)) else parent_key)
            elif isinstance(node, list):
                for x in node:
                    walk(x, parent_key)
        walk(data)
        seen = set(); out = []
        for d in found:
            key = (tuple(d["steps"]), d["type"])
            if key in seen:
                continue
            seen.add(key)
            d["pattern_id"] = f"PG{len(out) + 1:03d}"; d["source_file"] = rel
            out.append(d)
        return out
    raise FileNotFoundError(f"no PyGuard pattern file under {repo} (tried {PATTERN_FILES})")


def _kind(t: str) -> str:
    t = t.lower()
    if "malware" in t or "malicious" in t:
        return "malware_only" if "only" in t or "pure" in t else "malware"
    if "benign" in t:
        return "benign_only" if "only" in t or "pure" in t else "benign"
    return "distinction" if "distinct" in t else t


def pattern_docs(patterns: List[Dict], taxonomy: Dict[str, Dict], origin: str) -> List[KnowledgeDoc]:
    docs = []
    for p in patterns:
        kind = _kind(p["type"])
        steps = []
        beh: List[str] = []
        for aid in p["steps"]:
            t = taxonomy.get(aid, {})
            steps.append(f"{aid} ({t.get('description', '').strip() or 'no description'})")
            for b in action_to_behaviors(aid, t.get("path", [])):
                if b not in beh:
                    beh.append(b)
        meaning = {"malware_only": "observed only in malicious code snippets",
                   "benign_only": "observed only in benign code snippets"}.get(kind, f"type {kind}")
        text = (f"PyGuard mined action-sequence pattern, {meaning} (support {p.get('support')}). Sequence: "
                + " -> ".join(steps) + ".")
        docs.append(KnowledgeDoc(doc_id=f"pyguard:{p['pattern_id']}", source_type="pyguard_pattern",
                                 title=f"PyGuard pattern {p['pattern_id']} [{kind}]", text=text, behaviors=beh,
                                 apis=[], provenance={"origin": origin, "source_file": p["source_file"],
                                                      "label_hint": kind, "split": "external"}))
    return docs


class PyGuardMatcher:
    """Maps package call sites to PyGuard action ids (lexicon) and finds patterns whose action ids occur
    as a subsequence of a scope's ordered actions. Heuristic: edges it creates are `inferred`."""

    def __init__(self, lexicon: Dict[str, Dict], patterns: List[Dict], min_len: int = 2,
                 kinds: Iterable[str] = ("malware_only", "malware", "benign_only", "distinction")):
        self.lex = lexicon
        self.patterns = [p for p in patterns if len(p["steps"]) >= min_len and _kind(p["type"]) in set(kinds)]

    def match(self, facts) -> List[Dict]:
        by_scope: Dict[str, List[Tuple[int, str, str]]] = defaultdict(list)
        for c in facts.calls.values():
            v = self.lex.get(c.api)
            if v:
                by_scope[c.scope].append((c.line, v["action_id"], c.cid))
        out = []
        for scope, seq in by_scope.items():
            seq.sort()
            for p in self.patterns:
                i, hit = 0, []
                for _line, aid, cid in seq:
                    if i < len(p["steps"]) and aid == p["steps"][i]:
                        hit.append(cid); i += 1
                if i == len(p["steps"]):
                    out.append({"pattern_id": p["pattern_id"], "kind": _kind(p["type"]), "support": p.get("support"),
                                "steps": p["steps"], "ops": hit, "scope": scope})
        return out


def save_bundle(path: str, lexicon: Dict, patterns: List[Dict], taxonomy: Dict, origin: str):
    """Small JSON bundle (lexicon + patterns + used taxonomy entries) so later runs need no checkout."""
    used = {a for p in patterns for a in p["steps"]} | {v["action_id"] for v in lexicon.values()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"origin": origin, "lexicon": lexicon, "patterns": patterns,
                   "taxonomy": {k: v for k, v in taxonomy.items() if k in used}}, f)


def load_bundle(path: str) -> Dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
