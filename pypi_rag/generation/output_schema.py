"""Parse and validate model output (plan 6.2: invalid JSON is recorded separately from wrong answers)."""
from __future__ import annotations

import json
import re
from typing import Dict, Iterable, List, Optional

from ..analysis.taxonomy import BEHAVIORS, behaviors_from_text

_ID = re.compile(r"\b([CKGF]\d{1,3})\b")


def _first_json_object(text: str) -> Optional[str]:
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        start = text.find("{", start + 1)
    return None


def _norm_verdict(v) -> Optional[str]:
    s = str(v).strip().lower()
    if s in ("malicious", "malware", "1", "true", "yes"):
        return "malicious"
    if s in ("benign", "clean", "0", "false", "no", "safe"):
        return "benign"
    if s in ("uncertain", "unknown", "unsure", "abstain"):
        return "uncertain"
    return None


def parse_json_response(text: str, valid_ids: Iterable[str] = ()) -> Dict:
    valid = set(valid_ids)
    out = {"verdict": "uncertain", "confidence": None, "claims": [], "explanation": "", "parse_ok": False,
           "parse_error": "", "cited_ids": []}
    blob = _first_json_object(text or "")
    obj = None
    if blob:
        try:
            obj = json.loads(blob)
        except json.JSONDecodeError:
            try:
                obj = json.loads(re.sub(r",\s*([}\]])", r"\1", blob.replace("'", '"')))
            except json.JSONDecodeError as e:
                out["parse_error"] = f"json: {e}"
    if isinstance(obj, dict):
        v = _norm_verdict(obj.get("verdict", obj.get("label", "")))
        if v is None:
            out["parse_error"] = f"bad verdict {obj.get('verdict')!r}"
        else:
            out["verdict"] = v; out["parse_ok"] = True
        try:
            c = obj.get("confidence")
            out["confidence"] = None if c is None else max(0.0, min(1.0, float(c) / (100.0 if float(c) > 1 else 1.0)))
        except (TypeError, ValueError):
            pass
        claims = []
        for cl in (obj.get("claims") or [])[:10]:
            if isinstance(cl, str):
                cl = {"text": cl}
            if not isinstance(cl, dict):
                continue
            txt = str(cl.get("text", ""))[:400]
            beh = str(cl.get("behavior", "")).strip()
            ev = cl.get("evidence") or []
            ev = [str(e).strip() for e in (ev if isinstance(ev, list) else [ev])]
            ev += [m for m in _ID.findall(txt) if m not in ev]
            claims.append({"text": txt, "behavior": beh if beh in BEHAVIORS else "",
                           "behavior_raw": beh, "inferred_behaviors": behaviors_from_text(txt + " " + beh),
                           "evidence": ev, "negated": bool(cl.get("negated", False)),
                           "invalid_ids": [e for e in ev if valid and e not in valid]})
        out["claims"] = claims
        out["explanation"] = str(obj.get("explanation", ""))[:1500]
    else:
        m = re.search(r"verdict\W+(malicious|benign|uncertain)", text or "", re.I) or \
            re.search(r"\b(malicious|benign)\b", text or "", re.I)
        if m:
            out["verdict"] = m.group(1).lower()
            out["parse_error"] = out["parse_error"] or "no JSON object; verdict from free text"
        else:
            out["parse_error"] = out["parse_error"] or "no JSON object"
        out["explanation"] = (text or "")[:1500]
    out["cited_ids"] = sorted({e for cl in out["claims"] for e in cl["evidence"]} |
                              set(_ID.findall(out["explanation"])))
    return out


def parse_malllm_response(text: str) -> Dict:
    """mal-LLM format: '**Predicted Classification**: 1/0' (also accepts Malicious/Benign)."""
    t = text or ""
    m = re.search(r"Predicted Classification\W*[:\-]?\W*(\d|malicious|benign)", t, re.I)
    v = None
    if m:
        g = m.group(1).lower()
        v = "malicious" if g in ("1", "malicious") else "benign" if g in ("0", "benign") else None
    if v is None:
        if re.search(r"don'?t know|cannot determine|unknown", t, re.I):
            v = "uncertain"
        else:
            m2 = re.search(r"\b(malicious|benign)\b", t, re.I)
            v = m2.group(1).lower() if m2 else "uncertain"
    ex = re.search(r"Explanation\W*[:\-]?\s*(.*)", t, re.I | re.S)
    return {"verdict": v, "confidence": None, "claims": [], "explanation": (ex.group(1) if ex else t)[:1500],
            "parse_ok": m is not None, "parse_error": "" if m else "no 'Predicted Classification' field", "cited_ids": []}
