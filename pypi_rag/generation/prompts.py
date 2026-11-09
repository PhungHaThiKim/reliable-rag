"""Prompt construction. ONE instruction block for every method (E0..E4); methods differ only in the
evidence sections appended after the shared package block. A fixed token budget applies to the
package block (same for all methods) and a second fixed budget to added evidence (E1..E4).

`malllm` prompts reproduce the mal-LLM baselines (setup.py first 300 chars; Simple RAG over YARA or
malicious code) for the reproduction runs; text adapted from mal-LLM `Zero_shot_baseline_llama3_1_8b/
simulate_test.py` and `RAG_experiments/Simple_RAG/simulate_yara_rag.py` (commit 0491c6c).
"""
from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple

from ..analysis.taxonomy import BEHAVIORS
from ..schema import Evidence, sha256_text

SYSTEM = (
    "You are a security analyst reviewing a Python package from PyPI for malicious behavior. You never run the code; "
    "you reason only over the evidence items given to you.\n"
    "Malicious means the package performs harmful actions the user would not expect: stealing data or credentials, "
    "downloading and executing payloads, backdoors, persistence, or hidden actions at install or import time. "
    "Packages that use sensitive APIs for their documented purpose are benign.\n"
    "Rules:\n"
    "- Every claim must cite the ids of the evidence items that show it (C* code, K* knowledge, G* graph paths, F* facts).\n"
    "- Do not claim behaviors that no evidence item shows.\n"
    "- Knowledge items (K*) describe typical behaviors in general; they are not evidence about this package by themselves.\n"
    "- Use only these behavior names in claims: " + ", ".join(BEHAVIORS) + ".\n"
    "Return ONLY one JSON object, no markdown:\n"
    '{"verdict": "malicious" or "benign", "confidence": <number 0-1>, '
    '"claims": [{"text": "<short claim>", "behavior": "<behavior name>", "evidence": ["C1"], "negated": false}], '
    '"explanation": "<two or three sentences>"}\n'
    "Use at most 5 claims. For a benign verdict you may state that a behavior is absent with \"negated\": true."
)

SECTION_TITLES = {"code_span": "Code excerpts (static extraction, line-numbered)",
                  "knowledge": "Retrieved knowledge (general, not about this package)",
                  "graph_path": "Behavioral graph evidence (static analysis of this package)",
                  "fact": "Behavioral facts (static analysis of this package)"}


def render_item(ev: Evidence, max_chars: int = 2400) -> str:
    c = ev.content if len(ev.content) <= max_chars else ev.content[:max_chars] + "\n..."
    return f"[{ev.evidence_id}] {c}"


def fit_items(items: Sequence[Evidence], budget: int, count_tokens: Callable[[str], int]) -> Tuple[List[Evidence], bool]:
    out, used, cut = [], 0, False
    for it in items:
        t = count_tokens(render_item(it))
        if used + t > budget:
            cut = True; continue
        out.append(it); used += t
    return out, cut


def build_messages(header: str, code_items: Sequence[Evidence], extra: Dict[str, Sequence[Evidence]],
                   count_tokens: Callable[[str], int], evidence_budget: int = 1500, question: str = None
                   ) -> Tuple[List[Dict], List[Evidence], Dict]:
    """extra: section key -> items ('knowledge' | 'graph_path' | 'fact'). Returns messages, visible items, meta."""
    parts = [f"## Package\n{header}", f"## {SECTION_TITLES['code_span']}\n" +
             ("\n\n".join(render_item(i) for i in code_items) if code_items else "(no source files available)")]
    visible = list(code_items)
    remaining = evidence_budget
    cut_any = False
    for key in ("graph_path", "fact", "knowledge"):
        its = list(extra.get(key, []))
        if not its:
            continue
        kept, cut = fit_items(its, remaining, count_tokens)
        cut_any |= cut
        remaining -= sum(count_tokens(render_item(i)) for i in kept)
        if kept:
            parts.append(f"## {SECTION_TITLES[key]}\n" + "\n\n".join(render_item(i) for i in kept))
            visible += kept
    if question:
        parts.append(f"## Question\n{question}\nAnswer it inside the explanation and claims of the JSON object.")
    parts.append("Decide whether this package is malicious. Answer with the JSON object only.")
    user = "\n\n".join(parts)
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    meta = {"evidence_cut": cut_any, "evidence_tokens": evidence_budget - remaining,
            "prompt_tokens_est": count_tokens(SYSTEM) + count_tokens(user), "prompt_sha256": sha256_text(SYSTEM + user)}
    return msgs, visible, meta


# ---------------------------------------------------------------------------------- mal-LLM reproduction
MALLLM_DEV = ("You are a cybersecurity expert analyzing potential malware in a Python package called {package_name}. "
              "You are provided with the names of the files in the package: {file_list}. Below is the first 300 "
              "characters from the setup.py file. Your task: - Determine if the package is **Malicious (1) or Benign (0)**. "
              "- Provide a **brief explanation** (max 2 sentences) of why it is classified this way.")
MALLLM_FORMAT = ("**Strict Response Format**: - **Filename**: {package_name} - **Predicted Classification**: "
                 "(1 for Malicious, 0 for Benign) - **Explanation**: (max 2 sentences)")
MALLLM_RAG_DEV = ("You are a cybersecurity expert analyzing potential malware in Python packages. Your task is to determine "
                  "if a package is malicious or benign based on the {ctx_name} context.")
MALLLM_RAG_DEV2 = ("The Python package **{package_name}** contains the following files:\n{file_list}\n\nUse the following "
                   "{ctx_name} context to verify if the snippet from setup.py can cause the entire package to be detected as "
                   "malicious.\n\n**If you don't know the answer, just say that you don't know.**")


def malllm_messages(rep: Dict[str, str], context: str = "", ctx_name: str = "YARA rule") -> List[Dict]:
    """Zero-shot (context == '') or Simple-RAG prompt in the mal-LLM format. `developer` is sent as `system`."""
    if not context:
        return [{"role": "system", "content": MALLLM_DEV.format(**rep) + "\n" + MALLLM_FORMAT.format(**rep)},
                {"role": "user", "content": f"setup.py snippet:\n{rep['snippet']}"}]
    return [{"role": "system", "content": MALLLM_RAG_DEV.format(ctx_name=ctx_name) + "\n" +
             MALLLM_RAG_DEV2.format(ctx_name=ctx_name, **rep) + "\n" + MALLLM_FORMAT.format(**rep)},
            {"role": "user", "content": f"**{ctx_name} Context:**\n{context}\n\n**Code Snippet:**\n{rep['snippet']}"}]
