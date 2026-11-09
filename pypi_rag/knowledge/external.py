"""External text knowledge used by the mal-LLM Simple RAG baseline: YARA rules and GitHub advisories.

mal-LLM indexed (1) YARA rules interpreted by gpt-4o-mini, (2) GitHub reviewed pip advisories and
(3) malicious train setup.py files. We index the raw YARA rule (name, tags, meta, strings, condition)
without an LLM rewrite, and advisories as `summary + description`. Advisories name packages, so
`data.leakage_check.kb_leakage` must drop every advisory that mentions a val/test package.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Dict, List, Optional

from ..analysis.taxonomy import behaviors_from_text
from .documents import KnowledgeDoc

_RULE_RE = re.compile(r"(?:private\s+|global\s+)*rule\s+(\w+)\s*(?::\s*([\w\s]+?))?\s*\{(.*?)\n\}", re.S)
_SEC_RE = re.compile(r"^\s*(meta|strings|condition)\s*:\s*$", re.M)


def parse_yara(text: str, origin: str) -> List[KnowledgeDoc]:
    docs = []
    for m in _RULE_RE.finditer(text):
        name, tags, body = m.group(1), (m.group(2) or "").split(), m.group(3)
        parts: Dict[str, str] = {}
        pos = [(mm.start(), mm.end(), mm.group(1)) for mm in _SEC_RE.finditer(body)]
        for k, (s, e, sec) in enumerate(pos):
            nxt = pos[k + 1][0] if k + 1 < len(pos) else len(body)
            parts[sec] = body[e:nxt].strip()
        meta = dict(re.findall(r'(\w+)\s*=\s*"([^"]*)"', parts.get("meta", "")))
        desc = meta.get("description", "")
        strings = re.findall(r'\$\w*\s*=\s*("[^"]*"|\{[^}]*\}|/[^/]*/)', parts.get("strings", ""))
        text_ = (f"YARA rule {name}" + (f" (tags: {', '.join(tags)})" if tags else "") + f". {desc}\n"
                 f"Strings: {'; '.join(s[:80] for s in strings[:15])}\nCondition: {parts.get('condition', '')[:300]}")
        docs.append(KnowledgeDoc(doc_id=f"yara:{name}", source_type="yara", title=f"YARA {name}", text=text_,
                                 behaviors=behaviors_from_text(desc + " " + " ".join(strings)), apis=[],
                                 provenance={"origin": origin, "rule": name, "author": meta.get("author", ""),
                                             "split": "external", "label_hint": "malicious_indicator"}))
    return docs


def load_yara(path: str) -> List[KnowledgeDoc]:
    docs: List[KnowledgeDoc] = []
    files = [path] if os.path.isfile(path) else [os.path.join(dp, f) for dp, _, fs in os.walk(path) for f in fs
                                                    if f.endswith((".yar", ".yara"))]
    for fp in sorted(files):
        with open(fp, encoding="utf-8", errors="replace") as f:
            docs.extend(parse_yara(f.read(), origin=f"yara:{os.path.basename(fp)}"))
    return docs


def fetch_advisories(out_json: str, ecosystem: str = "pip", types=("malware", "reviewed"), max_pages: int = 30,
                     token: Optional[str] = None) -> int:
    """GitHub global advisories REST API -> JSON list (Colab; needs network). Returns the count."""
    import requests
    hdr = {"Accept": "application/vnd.github+json"}
    if token or os.environ.get("GITHUB_TOKEN"):
        hdr["Authorization"] = f"Bearer {token or os.environ['GITHUB_TOKEN']}"
    rows = []
    for t in types:
        url = f"https://api.github.com/advisories?ecosystem={ecosystem}&type={t}&per_page=100"
        for _ in range(max_pages):
            r = requests.get(url, headers=hdr, timeout=30)
            if r.status_code != 200:
                print("advisories:", r.status_code, r.text[:200]); break
            rows.extend(r.json())
            nxt = r.links.get("next", {}).get("url")
            if not nxt:
                break
            url = nxt; time.sleep(0.5)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(rows, f)
    return len(rows)


def load_advisories(path: str) -> List[KnowledgeDoc]:
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    docs = []
    for a in rows:
        gid = a.get("ghsa_id") or a.get("id")
        summ, desc = a.get("summary") or "", (a.get("description") or "")[:2000]
        pkgs = [v.get("package", {}).get("name", "") for v in a.get("vulnerabilities") or []]
        docs.append(KnowledgeDoc(doc_id=f"adv:{gid}", source_type="advisory", title=f"Advisory {gid}: {summ[:100]}",
                                 text=f"Advisory summary: {summ}\nAdvisory description: {desc}",
                                 behaviors=behaviors_from_text(summ + " " + desc), apis=[],
                                 provenance={"origin": "github-advisories", "type": a.get("type"), "packages": pkgs,
                                             "published_at": a.get("published_at"), "split": "external",
                                             "label_hint": "malicious_indicator" if a.get("type") == "malware" else "advisory"}))
    return docs
