"""End-to-end QA with an LLM reader on top of any method's selected paths.

`attach_llm_answers(records, dataset, reader)` adds per-record:
    llm_pred, llm_hit, llm_hit1, llm_f1, llm_acc, n_tokens_prompt
using the GNN-RAG prompt format (P6 cell 3) and GNN-RAG scoring
(qa_text_eval, ported from evaluate_results.py).

Readers
-------
HFReader(model_name, prompt_type)   any HF causal LM (Llama-2/3, Qwen, Mistral,
                                    gpt-oss-20b, ...) or seq2seq (Flan-T5)
DummyReader()                       lists every tail entity of the context —
                                    an "oracle extractor" used in tests
NoRetrievalReader(reader)           same reader, context dropped (hallucination
                                    baseline: how often is the answer right
                                    with NO evidence?)
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence

from qa_text_eval import eval_f1, eval_hit, eval_hit1, match, split_answer

SAQ_RULE_INSTRUCTION = ("Based on the reasoning paths, please answer the given question. "
                        "Please keep the answer as simple as possible and return all the "
                        "possible answers as a list.")
GROUNDED_INSTRUCTION = ("Answer the question using ONLY the reasoning paths below. "
                        "Return every entity that the paths support as a list, one per line, "
                        "copying entity names exactly as they appear in the paths. "
                        "If the paths do not contain enough evidence to answer, reply exactly: unknown.")
# CGAR-G (Nov 2026): candidate-constrained answering. The candidates are the entities the
# shown paths lead to, so every answer the reader can give is backed by a path it saw.
# No 'unknown' option: the grounded prompt showed that abstention costs far more correct
# answers than it removes fabrications (Qwen: 29% abstained, -9 pp grounded-correct).
CAND_INSTRUCTION = ("Based on the reasoning paths, answer the given question. "
                    "The candidate answers below are the entities the reasoning paths lead to. "
                    "Return every candidate that answers the question, one per line, copying the "
                    "names exactly as written. Do not add entities that are not in the candidate list "
                    "and do not explain.")
PROMPT_TEMPLATES = {
    "llama2": "[INST] <<SYS>>\n<</SYS>>\n{instruction}\n\n{input} [/INST]",
    "general": "{instruction}\n\n{input}",
    "chat": "{instruction}\n\n{input}",      # wrapped by tokenizer.apply_chat_template
    "grounded": "{instruction}\n\n{input}",  # chat-wrapped, GROUNDED_INSTRUCTION (strict answering + abstention)
    "cand": "{instruction}\n\n{input}",      # chat-wrapped, CAND_INSTRUCTION + candidate list (CGAR-G)
}
CHAT_WRAPPED = ("chat", "grounded", "cand")
INSTRUCTIONS = {"grounded": GROUNDED_INSTRUCTION, "cand": CAND_INSTRUCTION}


def path_tail(path_text: str) -> str:
    """Entity a path line 'h -> r -> t [-> r2 -> t2]' leads to (grl_rag.path_to_string format)."""
    parts = [x.strip() for x in str(path_text).split(" -> ")]
    return parts[-1] if len(parts) >= 3 else ""


def candidates_of(record: Dict) -> List[str]:
    """Answer candidates of a record = entities its selected paths lead to, in path order
    (CGAR: most relevant first; gnnrag: candidate rank). Stored as record['candidates']
    by make_record; derived from the path texts for older records."""
    c = record.get("candidates")
    if c:
        return list(c)
    out = []
    for t in record.get("selected_paths", []) or []:
        e = path_tail(t)
        if e and e not in out:
            out.append(e)
    return out


def build_prompt(question: str, reasoning_paths: Sequence[str], model_type: str = "general",
                 max_tokens: int = 4096, tokenize_fn=None, candidates: Optional[Sequence[str]] = None) -> str:
    """`candidates` (model_type 'cand' only): answer candidates listed after the paths; when None
    they are the tails of the paths that fit in the budget."""
    if not question.endswith("?"):
        question += "?"
    template = PROMPT_TEMPLATES.get(model_type, PROMPT_TEMPLATES["general"])
    instruction = INSTRUCTIONS.get(model_type, SAQ_RULE_INSTRUCTION)
    tokenize_fn = tokenize_fn or len
    question_part = f"Question:\n{question}"

    def body(paths):
        parts = [f"Reasoning Paths:\n{chr(10).join(paths)}"] if paths else []
        if model_type == "cand":
            cands = list(candidates) if candidates is not None else []
            if candidates is None:
                for p in paths:
                    e = path_tail(p)
                    if e and e not in cands:
                        cands.append(e)
            if cands:
                parts.append("Candidate answers:\n" + "\n".join(f"- {c}" for c in cands))
        parts.append(question_part)
        return "\n\n".join(parts)

    selected = []
    for p in reasoning_paths:
        test = template.format(instruction=instruction, input=body(selected + [p]))
        if tokenize_fn(test) > max_tokens:
            break
        selected.append(p)
    return template.format(instruction=instruction, input=body(selected))


def ground_answer(pred: str, candidates: Sequence[str], fallback: bool = True) -> Dict:
    """CGAR-G verification step: keep only answer entities that resolve to a candidate
    (an entity some shown path leads to); drop the rest. If the reader answered but
    nothing resolves, fall back to the first candidate (most relevant path). Abstentions
    are kept as abstentions. Returns {pred, kept, dropped, fallback, grounded}."""
    from .hallucination import EntityIndex
    ents, abstained = split_answer(pred)
    cands = [c for c in candidates if c]
    if not cands:
        return {"pred": pred, "kept": ents, "dropped": [], "fallback": 0, "grounded": 0, "abstained": int(abstained)}
    if abstained:
        return {"pred": pred, "kept": [], "dropped": [], "fallback": 0, "grounded": 1, "abstained": 1}
    idx = EntityIndex(cands)
    kept, dropped = [], []
    for e in ents:
        hits = idx.resolve(e)
        if hits:
            for h in hits[:1]:            # the candidate's own spelling, not the reader's
                if h not in kept:
                    kept.append(h)
        else:
            dropped.append(e)
    fb = 0
    if not kept and fallback and ents:
        kept, fb = [cands[0]], 1
    return {"pred": "\n".join(kept), "kept": kept, "dropped": dropped, "fallback": fb, "grounded": 1, "abstained": 0}


class DummyReader:
    """Oracle-extractor reader for tests: answers with the tail entity of
    every path line (so llm_hit == hit_ctx when answers are path tails)."""
    prompt_type = "general"

    def tokenize(self, text):
        return len(text.split())

    def answer(self, question: str, paths_text: str, candidates: Optional[Sequence[str]] = None) -> str:
        tails = []
        for line in paths_text.split("\n"):
            if "->" in line:
                tails.append(line.split("->")[-1].strip())
        return "\n".join(dict.fromkeys(tails)) if tails else ""


class NoRetrievalReader:
    """Wraps a reader and drops the context (no-RAG / hallucination baseline)."""

    def __init__(self, reader):
        self.reader = reader
        self.prompt_type = getattr(reader, "prompt_type", "general")

    def tokenize(self, text):
        return self.reader.tokenize(text)

    def answer(self, question, paths_text, candidates=None):
        return self.reader.answer(question, "")


class HFReader:
    """Hugging Face reader. prompt_type: 'llama2' (raw [INST] template, as in
    the GNN-RAG paper), 'chat' (tokenizer chat template: Llama-3, Qwen,
    Mistral-Instruct, gpt-oss), 'grounded' (chat template + strict grounded
    instruction with abstention), 'general' (seq2seq e.g. Flan-T5)."""

    def __init__(self, model_name: str, prompt_type: str = "chat", max_new_tokens: int = 256,
                 dtype: str = "bf16", max_context: int = 4096, seq2seq: bool = False,
                 adapter: Optional[str] = None):
        """adapter: directory of a LoRA adapter trained by kgqa_bench.reader_ft (merged at load)."""
        import torch
        from transformers import AutoTokenizer
        self.model_name, self.prompt_type, self.max_new_tokens = model_name, prompt_type, max_new_tokens
        self.max_context = max_context
        self.seq2seq = seq2seq
        self.adapter = adapter
        dt = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
        self.tok = AutoTokenizer.from_pretrained(model_name)
        if seq2seq:
            from transformers import AutoModelForSeq2SeqLM
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=dt, device_map="auto")
        else:
            from transformers import AutoModelForCausalLM
            self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dt, device_map="auto")
        if adapter:
            from peft import PeftModel
            disable_stale_torchao()
            self.model = PeftModel.from_pretrained(self.model, adapter).merge_and_unload()
        self.model.eval()
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token

    def tokenize(self, text):
        return len(self.tok.tokenize(text))

    def answer(self, question: str, paths_text: str, candidates: Optional[Sequence[str]] = None) -> str:
        import torch
        paths = [p for p in paths_text.split("\n") if p.strip()]
        prompt = build_prompt(question, paths, self.prompt_type, self.max_context - 64, self.tokenize,
                              candidates=candidates)
        if self.prompt_type in CHAT_WRAPPED:
            prompt = self.tok.apply_chat_template([{"role": "user", "content": prompt}],
                                                  tokenize=False, add_generation_prompt=True)
        enc = self.tok(prompt, return_tensors="pt", truncation=True, max_length=self.max_context).to(self.model.device)
        with torch.inference_mode():
            out = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                                      pad_token_id=self.tok.pad_token_id)
        gen = out[0] if self.seq2seq else out[0][enc["input_ids"].shape[1]:]
        return self.tok.decode(gen, skip_special_tokens=True).strip()


def disable_stale_torchao() -> bool:
    """Work around peft >= 0.17 on Colab: its LoRA dispatcher probes `is_torchao_available()` for every
    target Linear even without quantization, and that probe raises ImportError when the preinstalled
    torchao is older than peft wants (Colab ships 0.10.0). We are not quantizing, so make the probe
    answer False everywhere peft imported it. Returns True when a patch was applied."""
    import sys
    try:
        import peft.import_utils as piu
    except Exception:
        return False
    try:
        piu.is_torchao_available()
        return False
    except ImportError:
        pass
    fn = lambda *a, **k: False  # noqa: E731
    patched = False
    for name, mod in list(sys.modules.items()):
        if (name == "peft" or name.startswith("peft.")) and mod is not None and hasattr(mod, "is_torchao_available"):
            setattr(mod, "is_torchao_available", fn); patched = True
    return patched


def reader_from_spec(spec: str, adapter: Optional[str] = None):
    """'none' | 'dummy' | 'hf:<model>[:chat|grounded|cand|llama2|general][:seq2seq]'
    'grounded' = chat template + GROUNDED_INSTRUCTION (answer only from paths, 'unknown' allowed);
    'cand' = chat template + CAND_INSTRUCTION + candidate list (CGAR-G).
    adapter = LoRA directory from kgqa_bench.reader_ft (hf readers only)."""
    if spec in ("none", "", None):
        return None
    if spec == "dummy":
        return DummyReader()
    if spec.startswith("hf:"):
        parts = spec.split(":")
        model = parts[1]
        ptype = parts[2] if len(parts) > 2 else "chat"
        seq2seq = len(parts) > 3 and parts[3] == "seq2seq"
        return HFReader(model, prompt_type=ptype, seq2seq=seq2seq, adapter=adapter)
    raise ValueError(f"unknown reader spec: {spec}")


def attach_llm_answers(records: List[Dict], dataset, reader, cache_path: Optional[str] = None,
                       verbose: bool = True, prefix: str = "llm", verify: bool = False) -> List[Dict]:
    """Run the reader on every record's selected paths (in place).
    `cache_path` (jsonl) memoises (qid, method, paths-hash, prompt type) → prediction.
    verify=True (CGAR-G): keep the raw text as {prefix}_pred_raw and score the grounded answer
    (entities that resolve to a candidate the paths lead to; first candidate as fallback)."""
    by_qid = {q.qid: q for q in dataset}
    cache: Dict[str, str] = {}
    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as f:
            for line in f:
                d = json.loads(line); cache[d["key"]] = d["pred"]
    fcache = open(cache_path, "a") if cache_path else None
    it = records
    if verbose:
        try:
            from tqdm import tqdm; it = tqdm(records, desc=f"{prefix}-reader", leave=False)
        except ImportError:
            pass
    for r in it:
        q = by_qid.get(r["qid"])
        if q is None:
            continue
        paths_text = "\n".join(r.get("selected_paths", []))
        cands = candidates_of(r)
        ptype = getattr(reader, "prompt_type", "general")
        key = f"{r['qid']}|{r['method']}|{prefix}|{hash(paths_text)}" + (f"|{ptype}" if ptype == "cand" else "")
        if key in cache:
            pred = cache[key]
        else:
            pred = reader.answer(q.question, paths_text, candidates=cands)
            cache[key] = pred
            if fcache:
                fcache.write(json.dumps({"key": key, "pred": pred}) + "\n"); fcache.flush()
        r[f"{prefix}_pred_raw"] = pred
        if verify:
            g = ground_answer(pred, cands)
            pred = g["pred"]
            r[f"{prefix}_n_dropped"], r[f"{prefix}_fallback"], r[f"{prefix}_grounded"] = len(g["dropped"]), g["fallback"], g["grounded"]
        pred_list, abstained = split_answer(pred)      # same rule as kgqa_bench.rescore / hallucination
        pred_str = " ".join(pred_list)
        f1, _p, _r = eval_f1(pred_list, q.answers)
        r[f"{prefix}_pred"] = pred
        r[f"{prefix}_abstained"] = int(abstained)
        r[f"{prefix}_hit"] = eval_hit(pred_str, q.answers) if pred_list else 0
        r[f"{prefix}_hit1"] = eval_hit1(pred_list, q.answers)
        r[f"{prefix}_f1"] = f1
        r[f"{prefix}_acc"] = (sum(1 for a in q.answers if match(pred_str, a)) / len(q.answers)) if (q.answers and pred_list) else 0.0
        # unsupported answer = predicted entity that is NOT in the context (hallucination proxy)
        ctx = paths_text.lower()
        r[f"{prefix}_unsupported_rate"] = (sum(1 for p in pred_list if p.lower() not in ctx) / len(pred_list)) if pred_list else 0.0
        if hasattr(reader, "tokenize"):
            r["n_tokens_prompt"] = reader.tokenize(build_prompt(q.question, r.get("selected_paths", []),
                                                                ptype, candidates=cands))
    if fcache:
        fcache.close()
    return records
