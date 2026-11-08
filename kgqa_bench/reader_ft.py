"""LoRA fine-tuning of the reader on (question, retrieved paths) -> gold entity list (CGAR-G part 3).

GNN-RAG's published numbers come from a reader fine-tuned on WebQSP/CWQ train with the
retriever's paths in the prompt; our Nov-2026 readers (Qwen2.5-7B, Llama-3.1-8B) are
zero-shot. Zero-shot readers miss 4-7 pp of answers that ARE in the context and write
free text that the parser must clean. Fine-tuning fixes both: the reader learns to
return the candidate entities that answer the question, nothing else.

Pipeline
  1. retrieval records for the TRAIN split (no reader), e.g.
       python -m kgqa_bench.cli run --eval gnnrag:ckpt/gnn_features_train.pkl:webqsp:train \
           --methods gnnrag_all cgar --cgar-ckpt ... --cgar-max-paths 5 --out results_gnn/b12_train_ctx --quiet
  2. SFT examples (prompt exactly as the reader sees it; target = gold entities among the candidates):
       python -m kgqa_bench.reader_ft build --records results_gnn/b12_train_ctx/records_webqsp_train_cgar.jsonl \
           results_gnn/b12_train_ctx/records_webqsp_train_gnnrag_all.jsonl \
           --eval gnnrag:ckpt/gnn_features_train.pkl:webqsp:train --prompt cand --out results_gnn/b12_train_ctx/sft_cand.jsonl
     Mixing contexts from several retrievers is deliberate: the reader must not be tuned to one
     retriever's context length, or the all-vs-CGAR comparison would be unfair.
  3. LoRA:
       python -m kgqa_bench.reader_ft train --sft results_gnn/b12_train_ctx/sft_cand.jsonl \
           --model Qwen/Qwen2.5-7B-Instruct --out ckpt/lora_qwen_cand --epochs 2
  4. use it:  --reader hf:Qwen/Qwen2.5-7B-Instruct:cand --reader-adapter ckpt/lora_qwen_cand --reader-verify

Examples whose candidates contain no gold entity are skipped (the reader cannot learn
anything true from them); --keep-empty keeps them with the first candidate as target
(teaches "pick the most plausible", matching the verify-step fallback).
"""
from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, List, Optional, Sequence

from qa_text_eval import normalize

from .data import load_spec
from .llm_reader import CHAT_WRAPPED, build_prompt, candidates_of, disable_stale_torchao


# ─────────────────────────────────────────────────────────────────────
# 1. examples
# ─────────────────────────────────────────────────────────────────────

def gold_in_candidates(answers: Sequence[str], candidates: Sequence[str]) -> List[str]:
    """Candidates (their own spelling) whose normalised name equals a gold answer."""
    gold = {normalize(a) for a in answers}
    return [c for c in candidates if normalize(c) in gold]


def build_sft_examples(records: Sequence[Dict], dataset, prompt_type: str = "cand",
                       keep_empty: bool = False, max_tokens: int = 4096, tokenize_fn=None,
                       seed: int = 0, max_per_qid: Optional[int] = None) -> List[Dict]:
    """One example per record: {qid, method, prompt, target, n_target}. `prompt` is the user
    message (before the chat template); `target` the assistant reply (one entity per line)."""
    by_qid = {q.qid: q for q in dataset}
    out: List[Dict] = []
    per_qid: Dict[str, int] = {}
    for r in records:
        q = by_qid.get(r["qid"])
        if q is None:
            continue
        if max_per_qid and per_qid.get(q.qid, 0) >= max_per_qid:
            continue
        cands = candidates_of(r)
        target = gold_in_candidates(q.answers, cands)
        if not target:
            if not keep_empty or not cands:
                continue
            target = [cands[0]]
        prompt = build_prompt(q.question, r.get("selected_paths", []), prompt_type, max_tokens, tokenize_fn,
                              candidates=cands if prompt_type == "cand" else None)
        out.append({"qid": q.qid, "method": r.get("method"), "prompt": prompt,
                    "target": "\n".join(target), "n_target": len(target)})
        per_qid[q.qid] = per_qid.get(q.qid, 0) + 1
    random.Random(seed).shuffle(out)
    return out


def _load_records(path: str) -> List[Dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def cmd_build(a):
    load_kw = {}
    if a.max_items:
        load_kw["max_items"] = a.max_items
    ds = load_spec(a.eval, **load_kw)
    recs = [r for p in a.records for r in _load_records(p)]
    ex = build_sft_examples(recs, ds, prompt_type=a.prompt, keep_empty=a.keep_empty, seed=a.seed,
                            max_per_qid=a.max_per_qid)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        for e in ex:
            f.write(json.dumps(e) + "\n")
    by_m: Dict[str, int] = {}
    for e in ex:
        by_m[e["method"]] = by_m.get(e["method"], 0) + 1
    print(f"{len(ex)} examples from {len(recs)} records ({len(ds)} questions) -> {a.out}; per method {by_m}; "
          f"mean targets/example {sum(e['n_target'] for e in ex) / max(len(ex), 1):.2f}")


# ─────────────────────────────────────────────────────────────────────
# 2. LoRA training (transformers + peft, plain torch loop, completion-only loss)
# ─────────────────────────────────────────────────────────────────────

def encode_example(tok, prompt: str, target: str, prompt_type: str, max_len: int):
    """Token ids + labels (-100 on the prompt part) for one example."""
    if prompt_type in CHAT_WRAPPED:
        p_text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                         add_generation_prompt=True)
    else:
        p_text = prompt
    p_ids = tok(p_text, add_special_tokens=False)["input_ids"]
    t_ids = tok(target, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    if len(p_ids) + len(t_ids) > max_len:                 # keep the whole target, trim the prompt head
        p_ids = p_ids[-(max_len - len(t_ids)):]
    ids = p_ids + t_ids
    labels = [-100] * len(p_ids) + t_ids
    return ids, labels


def train_lora(sft_path: str, model_name: str, out_dir: str, prompt_type: str = "cand", epochs: int = 2,
               lr: float = 1e-4, r: int = 16, alpha: int = 32, dropout: float = 0.05, batch: int = 4,
               grad_accum: int = 4, max_len: int = 2048, max_examples: Optional[int] = None,
               seed: int = 0, dtype: str = "bf16", target_modules: Optional[Sequence[str]] = None,
               log_every: int = 20) -> Dict:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(seed); random.seed(seed)
    dt = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dt, device_map="auto")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none", task_type="CAUSAL_LM",
                     target_modules=list(target_modules or ("q_proj", "k_proj", "v_proj", "o_proj")))
    disable_stale_torchao()
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()

    with open(sft_path) as f:
        ex = [json.loads(l) for l in f if l.strip()]
    if max_examples:
        ex = ex[:max_examples]
    enc = [encode_example(tok, e["prompt"], e["target"], prompt_type, max_len) for e in ex]
    enc.sort(key=lambda x: len(x[0]))                          # length-bucketed batches, then shuffled
    batches = [enc[i:i + batch] for i in range(0, len(enc), batch)]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    n_steps = epochs * max(len(batches) // grad_accum, 1)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, int(0.03 * n_steps))) *
                                              max(0.0, 1 - s / max(n_steps, 1)))
    dev = next(model.parameters()).device
    history, step, micro = [], 0, 0
    model.train()
    for ep in range(epochs):
        random.shuffle(batches)
        run_loss = 0.0
        for b in batches:
            L = max(len(x[0]) for x in b)
            ids = torch.full((len(b), L), tok.pad_token_id, dtype=torch.long)
            lab = torch.full((len(b), L), -100, dtype=torch.long)
            att = torch.zeros((len(b), L), dtype=torch.long)
            for i, (x, y) in enumerate(b):
                ids[i, :len(x)] = torch.tensor(x); lab[i, :len(y)] = torch.tensor(y); att[i, :len(x)] = 1
            out = model(input_ids=ids.to(dev), attention_mask=att.to(dev), labels=lab.to(dev))
            (out.loss / grad_accum).backward()
            run_loss += out.loss.item(); micro += 1
            if micro % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
                if step % log_every == 0:
                    history.append({"epoch": ep, "step": step, "loss": run_loss / (grad_accum * log_every)})
                    print(f"  ep {ep} step {step}/{n_steps} loss {history[-1]['loss']:.4f}")
                    run_loss = 0.0
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    info = {"model": model_name, "sft": sft_path, "n_examples": len(ex), "epochs": epochs, "lr": lr, "r": r,
            "alpha": alpha, "batch": batch, "grad_accum": grad_accum, "max_len": max_len, "prompt": prompt_type,
            "steps": step, "history": history}
    with open(os.path.join(out_dir, "train_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"adapter saved -> {out_dir} ({step} optimizer steps)")
    return info


def cmd_train(a):
    train_lora(a.sft, a.model, a.out, prompt_type=a.prompt, epochs=a.epochs, lr=a.lr, r=a.r, alpha=a.alpha,
               dropout=a.dropout, batch=a.batch, grad_accum=a.grad_accum, max_len=a.max_len,
               max_examples=a.max_examples, seed=a.seed, dtype=a.dtype)


def build_parser():
    p = argparse.ArgumentParser(prog="kgqa_bench.reader_ft", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--records", nargs="+", required=True, help="records_*.jsonl of the TRAIN split (one or more retrievers)")
    b.add_argument("--eval", required=True, help="dataset spec of those records (train split)")
    b.add_argument("--prompt", default="cand", help="cand | chat")
    b.add_argument("--out", required=True)
    b.add_argument("--keep-empty", action="store_true")
    b.add_argument("--max-per-qid", type=int, default=None)
    b.add_argument("--max-items", type=int, default=None)
    b.add_argument("--seed", type=int, default=0)
    b.set_defaults(fn=cmd_build)
    t = sub.add_parser("train")
    t.add_argument("--sft", required=True)
    t.add_argument("--model", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--prompt", default="cand")
    t.add_argument("--epochs", type=int, default=2)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--r", type=int, default=16)
    t.add_argument("--alpha", type=int, default=32)
    t.add_argument("--dropout", type=float, default=0.05)
    t.add_argument("--batch", type=int, default=4)
    t.add_argument("--grad-accum", type=int, default=4)
    t.add_argument("--max-len", type=int, default=2048)
    t.add_argument("--max-examples", type=int, default=None)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--dtype", default="bf16")
    t.set_defaults(fn=cmd_train)
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
