"""End-to-end QA with an LLM reader on top of any method's selected paths.

`attach_llm_answers(records, dataset, reader)` adds per-record:
    llm_pred, llm_hit, llm_hit1, llm_f1, llm_acc, n_tokens_prompt
using the GNN-RAG prompt format (P6 cell 3) and GNN-RAG scoring
(prism.qa_text_eval, ported from evaluate_results.py).

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

from prism.qa_text_eval import eval_f1, eval_hit, eval_hit1, extract_answer_entities, match

SAQ_RULE_INSTRUCTION = ("Based on the reasoning paths, please answer the given question. "
                        "Please keep the answer as simple as possible and return all the "
                        "possible answers as a list.")
PROMPT_TEMPLATES = {
    "llama2": "[INST] <<SYS>>\n<</SYS>>\n{instruction}\n\n{input} [/INST]",
    "general": "{instruction}\n\n{input}",
    "chat": "{instruction}\n\n{input}",      # wrapped by tokenizer.apply_chat_template
}


def build_prompt(question: str, reasoning_paths: Sequence[str], model_type: str = "general",
                 max_tokens: int = 4096, tokenize_fn=None) -> str:
    if not question.endswith("?"):
        question += "?"
    template = PROMPT_TEMPLATES.get(model_type, PROMPT_TEMPLATES["general"])
    tokenize_fn = tokenize_fn or len
    question_part = f"Question:\n{question}"
    selected = []
    for p in reasoning_paths:
        cand = "\n".join(selected + [p])
        test = template.format(instruction=SAQ_RULE_INSTRUCTION,
                               input=f"Reasoning Paths:\n{cand}\n\n{question_part}")
        if tokenize_fn(test) > max_tokens:
            break
        selected.append(p)
    inp = f"Reasoning Paths:\n{chr(10).join(selected)}\n\n{question_part}" if selected else question_part
    return template.format(instruction=SAQ_RULE_INSTRUCTION, input=inp)


class DummyReader:
    """Oracle-extractor reader for tests: answers with the tail entity of
    every path line (so llm_hit == hit_ctx when answers are path tails)."""
    prompt_type = "general"

    def tokenize(self, text):
        return len(text.split())

    def answer(self, question: str, paths_text: str) -> str:
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

    def answer(self, question, paths_text):
        return self.reader.answer(question, "")


class HFReader:
    """Hugging Face reader. prompt_type: 'llama2' (raw [INST] template, as in
    the GNN-RAG paper), 'chat' (tokenizer chat template: Llama-3, Qwen,
    Mistral-Instruct, gpt-oss), 'general' (seq2seq e.g. Flan-T5)."""

    def __init__(self, model_name: str, prompt_type: str = "chat", max_new_tokens: int = 256,
                 dtype: str = "bf16", max_context: int = 4096, seq2seq: bool = False):
        import torch
        from transformers import AutoTokenizer
        self.model_name, self.prompt_type, self.max_new_tokens = model_name, prompt_type, max_new_tokens
        self.max_context = max_context
        self.seq2seq = seq2seq
        dt = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
        self.tok = AutoTokenizer.from_pretrained(model_name)
        if seq2seq:
            from transformers import AutoModelForSeq2SeqLM
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=dt, device_map="auto")
        else:
            from transformers import AutoModelForCausalLM
            self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dt, device_map="auto")
        self.model.eval()
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token

    def tokenize(self, text):
        return len(self.tok.tokenize(text))

    def answer(self, question: str, paths_text: str) -> str:
        import torch
        paths = [p for p in paths_text.split("\n") if p.strip()]
        prompt = build_prompt(question, paths, self.prompt_type, self.max_context - 64, self.tokenize)
        if self.prompt_type == "chat":
            prompt = self.tok.apply_chat_template([{"role": "user", "content": prompt}],
                                                  tokenize=False, add_generation_prompt=True)
        enc = self.tok(prompt, return_tensors="pt", truncation=True, max_length=self.max_context).to(self.model.device)
        with torch.inference_mode():
            out = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                                      pad_token_id=self.tok.pad_token_id)
        gen = out[0] if self.seq2seq else out[0][enc["input_ids"].shape[1]:]
        return self.tok.decode(gen, skip_special_tokens=True).strip()


def reader_from_spec(spec: str):
    """'none' | 'dummy' | 'hf:<model>[:chat|llama2|general][:seq2seq]'"""
    if spec in ("none", "", None):
        return None
    if spec == "dummy":
        return DummyReader()
    if spec.startswith("hf:"):
        parts = spec.split(":")
        model = parts[1]
        ptype = parts[2] if len(parts) > 2 else "chat"
        seq2seq = len(parts) > 3 and parts[3] == "seq2seq"
        return HFReader(model, prompt_type=ptype, seq2seq=seq2seq)
    raise ValueError(f"unknown reader spec: {spec}")


def attach_llm_answers(records: List[Dict], dataset, reader, cache_path: Optional[str] = None,
                       verbose: bool = True, prefix: str = "llm") -> List[Dict]:
    """Run the reader on every record's selected paths (in place).
    `cache_path` (jsonl) memoises (qid, method, paths-hash) → prediction."""
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
        key = f"{r['qid']}|{r['method']}|{prefix}|{hash(paths_text)}"
        if key in cache:
            pred = cache[key]
        else:
            pred = reader.answer(q.question, paths_text)
            cache[key] = pred
            if fcache:
                fcache.write(json.dumps({"key": key, "pred": pred}) + "\n"); fcache.flush()
        pred_list = extract_answer_entities(pred)
        pred_str = " ".join(pred_list)
        f1, _p, _r = eval_f1(pred_list, q.answers)
        r[f"{prefix}_pred"] = pred
        r[f"{prefix}_hit"] = eval_hit(pred_str, q.answers)
        r[f"{prefix}_hit1"] = eval_hit1(pred_list, q.answers)
        r[f"{prefix}_f1"] = f1
        r[f"{prefix}_acc"] = (sum(1 for a in q.answers if match(pred_str, a)) / len(q.answers)) if q.answers else 0.0
        # unsupported answer = predicted entity that is NOT in the context (hallucination proxy)
        ctx = paths_text.lower()
        r[f"{prefix}_unsupported_rate"] = (sum(1 for p in pred_list if p.lower() not in ctx) / len(pred_list)) if pred_list else 0.0
        if hasattr(reader, "tokenize"):
            r["n_tokens_prompt"] = reader.tokenize(build_prompt(q.question, r.get("selected_paths", []),
                                                                getattr(reader, "prompt_type", "general")))
    if fcache:
        fcache.close()
    return records
