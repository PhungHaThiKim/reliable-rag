"""LLM clients with ONE generation config for every method (plan 6.2).

    dummy                        deterministic heuristic reader (CPU tests; NOT a model)
    hf:<model id>                transformers, chat template, greedy decoding, left-padded batches (simple, slow)
    vllm:<model id>              vLLM offline engine: continuous batching + prefix caching (fastest on one GPU)
    openai:<model>               OpenAI-compatible endpoint, concurrent requests (OPENAI_BASE_URL, e.g. vllm serve)
`CachedLLM` stores every raw response keyed by sha256(model + messages + decoding config) in a JSONL
file, so an interrupted Colab run resumes and E4 reuses E3's generations without a second LLM call.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from ..analysis.taxonomy import BEHAVIORS


@dataclass
class GenOut:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    truncated: bool = False
    cached: bool = False


class BaseLLM:
    name = "base"
    max_new_tokens = 512
    temperature = 0.0

    def count_tokens(self, text: str) -> int:
        return len(text) // 3 + 1

    def generate(self, batch: Sequence[List[Dict]]) -> List[GenOut]:
        raise NotImplementedError

    def config(self) -> Dict:
        return {"name": self.name, "max_new_tokens": self.max_new_tokens, "temperature": self.temperature}


# ---------------------------------------------------------------------------------- dummy
class DummyLLM(BaseLLM):
    """Reads the evidence text with keyword rules. Always emits valid JSON with citations. Every 5th
    malicious answer (by prompt hash) adds an unsupported claim so verification code paths are exercised."""
    name = "dummy"
    STRONG = re.compile(r"\b(P0[1-35-79]|P1[05])\b")
    ITEM = re.compile(r"^\[([CKGF]\d+)\] ", re.M)

    def _items(self, user: str) -> Dict[str, str]:
        pos = [(m.start(), m.group(1)) for m in self.ITEM.finditer(user)]
        out = {}
        for k, (s, iid) in enumerate(pos):
            e = pos[k + 1][0] if k + 1 < len(pos) else len(user)
            out[iid] = user[s:e]
        return out

    def _qa_answer(self, user: str) -> str:
        """Task B (supply-chain QA): keyword heuristics over the question and evidence, valid JSON, citations."""
        items = self._items(user)
        fmt = re.search(r"Answer format: (.*)", user)
        fmt = fmt.group(1) if fmt else ""
        question = user.split("## Question\n", 1)[-1].split("\n", 1)[0]
        code = re.search(r"```python\n(.*?)```", user, re.S)
        code = code.group(1) if code else ""
        ev = {k: v for k, v in items.items() if not k.startswith("C")}
        ev_text = "\n".join(ev.values())
        cite = [next(iter(items))] if items else []
        ans, ins = None, False
        if code:
            if "importlib" in code or re.search(r"from \.\w+ import", code):
                ins = True
            elif "yes" in fmt and "no" in fmt:
                ans = "yes" if re.search(r"requests\.post|urlopen\(.*data|params=|send_mock_http", code) and "CACHE" not in code else "no"
            elif "host" in fmt:
                h = re.search(r"https://([^/\"]+)", code); ans = h.group(1) if h else None
            else:
                ans = re.findall(r'environ\.get\("([A-Z_]+)"', code)
        elif not ev:
            ins = True
        elif "yes" in fmt and "no" in fmt:
            names = re.findall(r"svc-[a-z0-9-]+", question)
            ans = "yes" if all(n in ev_text for n in names) and "applies to this version" in ev_text else "no"
        elif "service names" in fmt:
            ans = sorted({m for line in ev.values() if "applies to this version" in line
                          for m in re.findall(r"^\[\w+\] (svc-[a-z0-9-]+)", line)})
        elif "team names" in fmt:
            ans = sorted(set(re.findall(r"team ([a-z-]+)\)", ev_text)))
        elif "hours" in fmt or "days" in fmt:
            m = re.search(r"(\d+) " + ("hours" if "hours" in fmt else "days"), ev_text); ans = int(m.group(1)) if m else None
            ins = ans is None
        elif "version" in fmt:
            m = re.search(r"==(\d+\.\d+\.\d+)", ev_text); ans = m.group(1) if m else None
            ins = ans is None
        elif "package names" in fmt:
            ans = re.findall(r"-> ([a-z0-9-]+)==", ev_text.split("\n", 1)[0])
        claims = [] if ins else [{"text": "The answer follows from the cited item.", "evidence": cite}]
        return json.dumps({"answer": ans, "insufficient_evidence": ins, "confidence": 0.6, "claims": claims})

    def _answer(self, msgs: List[Dict]) -> str:
        user = msgs[-1]["content"]
        if "software supply chain of ACME" in msgs[0]["content"]:
            return self._qa_answer(user)
        items = self._items(user)
        h = int(hashlib.md5(user.encode()).hexdigest()[:6], 16)
        claims, score = [], 0.0
        rules = [("data_exfiltration", r"FLOWS_TO path: P0[1-5]|Data produced by .*(environ|gethostname|getuser|open).* flows to .*(post|urlopen|requests\.get|send)"),
                 ("download_exec", r"FLOWS_TO path: P0[67]|flows to .*(exec|Popen|system)"),
                 ("code_exec", r"\bexec\(|\beval\(|marshal\.loads"),
                 ("obfuscation", r"b64decode|zlib\.decompress|fromhex"),
                 ("install_hook", r"\[install\]|setup\.py:\d+ .*(urlopen|requests|exec|system)"),
                 ("network_send", r"requests\.post|urlopen\(.*data|\.send\(|webhook"),
                 ("sensitive_data", r"\.ssh|\.aws|credentials|Login Data|wallet"),
                 ("persistence", r"\.bashrc|crontab|CurrentVersion\\\\Run|Startup"),
                 ("env_access", r"os\.environ|getenv")]
        weights = {"data_exfiltration": 3, "download_exec": 3, "code_exec": 1, "obfuscation": 1, "install_hook": 1,
                   "network_send": 0.5, "sensitive_data": 1.5, "persistence": 2, "env_access": 0.2}
        for beh, rx in rules:
            for iid, txt in items.items():
                if iid.startswith("K"):
                    continue
                if re.search(rx, txt):
                    score += weights[beh]
                    claims.append({"text": f"The package shows {beh.replace('_', ' ')}.", "behavior": beh,
                                   "evidence": [iid]})
                    break
        if re.search(r"exec\(.*b64decode|b64decode.*\n.*exec\(", user):
            score += 2
        if "P16" in user and score < 3:
            score -= 0.5
        mal = score >= 2.5
        if mal and h % 5 == 0:
            claims.append({"text": "It also installs a keylogger.", "behavior": "surveillance", "evidence": ["C1"]})
        if not mal:
            claims = [{"text": "No data flow from local sources to the network or to code execution was found.",
                       "behavior": "data_exfiltration", "evidence": [next(iter(items), "C1")], "negated": True}]
        return json.dumps({"verdict": "malicious" if mal else "benign", "confidence": round(min(0.95, 0.5 + abs(score - 2.5) / 8), 2),
                           "claims": claims[:5], "explanation": f"Heuristic score {score:.1f}."})

    def generate(self, batch):
        out = []
        for msgs in batch:
            t0 = time.time()
            txt = self._answer(msgs)
            out.append(GenOut(txt, sum(self.count_tokens(m["content"]) for m in msgs), self.count_tokens(txt),
                              (time.time() - t0) * 1000))
        return out


# ---------------------------------------------------------------------------------- shared chat formatting
def _load_tokenizer(model_name: str):
    try:
        from transformers import AutoTokenizer
        kw = {"token": os.environ["HF_TOKEN"]} if os.environ.get("HF_TOKEN") else {}
        return AutoTokenizer.from_pretrained(model_name, **kw)
    except Exception as e:                                   # noqa: BLE001
        print(f"[llm] tokenizer {model_name} unavailable ({type(e).__name__}); token counts are approximate")
        return None


class _ChatFormat:
    """Chat template + truncation shared by the HF, vLLM and OpenAI-compatible backends, so that the
    same messages give the same tokens (and the same prompt budget) whatever the engine."""
    tok = None
    max_input_tokens = 8192

    def count_tokens(self, text: str) -> int:
        if self.tok is None:
            return len(text) // 3 + 1
        return len(self.tok(text, add_special_tokens=False)["input_ids"])

    def _template(self, msgs: List[Dict]) -> str:
        if self.tok is None:
            return "\n\n".join(m["content"] for m in msgs)
        try:
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:                                   # templates without a system role
            merged = [{"role": "user", "content": "\n\n".join(m["content"] for m in msgs)}]
            return self.tok.apply_chat_template(merged, tokenize=False, add_generation_prompt=True)

    def _fit(self, msgs: List[Dict]):
        """-> (messages, templated text, n tokens, truncated). Cuts the user message, keeps the final instruction."""
        text = self._template(msgs)
        n = self.count_tokens(text)
        if n <= self.max_input_tokens:
            return msgs, text, n, False
        user = msgs[-1]["content"]
        tail = user[user.rfind("\n\n"):]
        keep = int(len(user) * (self.max_input_tokens - 64) / n)
        msgs = msgs[:-1] + [{"role": "user", "content": user[:max(keep - len(tail), 200)] + "\n...[truncated]" + tail}]
        text = self._template(msgs)
        return msgs, text, self.count_tokens(text), True


# ---------------------------------------------------------------------------------- Hugging Face
class HFLLM(_ChatFormat, BaseLLM):
    """transformers.generate with static left-padded batches. Simple, but the GPU idles while the longest
    answer of each batch finishes: prefer `vllm:` for long runs. The model loads on the first cache miss."""

    def __init__(self, model_name: str, max_new_tokens: int = 512, max_input_tokens: int = 8192, dtype: str = "bf16",
                 batch_size: int = 4, adapter: Optional[str] = None):
        self.name, self.max_new_tokens, self.max_input_tokens = model_name, max_new_tokens, max_input_tokens
        self.batch_size, self.adapter, self.dtype = batch_size, adapter, dtype
        self.chunk = batch_size * 8
        self.tok = _load_tokenizer(model_name)
        self.tok.padding_side = "left"
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = None

    def _ensure_model(self):
        if self.model is None:
            import torch
            from transformers import AutoModelForCausalLM
            dt = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[self.dtype]
            kw = {"token": os.environ["HF_TOKEN"]} if os.environ.get("HF_TOKEN") else {}
            self.model = AutoModelForCausalLM.from_pretrained(self.name, torch_dtype=dt, device_map="auto", **kw)
            if self.adapter:
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(self.model, self.adapter).merge_and_unload()
            self.model.eval()
        return self.model

    def generate(self, batch):
        import torch
        model = self._ensure_model()
        fitted = [self._fit(m) for m in batch]
        order = sorted(range(len(batch)), key=lambda i: -fitted[i][2])
        res: Dict[int, GenOut] = {}
        for s in range(0, len(order), self.batch_size):
            idx = order[s:s + self.batch_size]
            texts = [fitted[i][1] for i in idx]
            enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            t0 = time.time()
            with torch.inference_mode():
                out = model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                                     pad_token_id=self.tok.pad_token_id)
            dt = (time.time() - t0) * 1000 / len(idx)
            for j, i in enumerate(idx):
                gen = out[j][enc["input_ids"].shape[1]:]
                txt = self.tok.decode(gen, skip_special_tokens=True).strip()
                res[i] = GenOut(txt, fitted[i][2], int((gen != self.tok.pad_token_id).sum()), dt, fitted[i][3])
        return [res[i] for i in range(len(batch))]

    def config(self):
        return {**super().config(), "backend": "hf", "max_input_tokens": self.max_input_tokens, "adapter": self.adapter,
                "decoding": "greedy"}


# ---------------------------------------------------------------------------------- vLLM (offline engine)
class VLLMLLM(_ChatFormat, BaseLLM):
    """vLLM offline engine: continuous batching, paged attention, prefix caching. Prompts are passed as token
    ids built with the same chat template as HFLLM (no double BOS). Greedy decoding. The engine starts on
    the first cache miss, so a fully cached rerun never loads the model. Latency is amortized per prompt."""

    def __init__(self, model_name: str, max_new_tokens: int = 512, max_input_tokens: int = 8192,
                 gpu_memory_utilization: float = 0.85, dtype: str = "bfloat16", seed: int = 42, chunk: int = 1024):
        self.name, self.max_new_tokens, self.max_input_tokens = model_name, max_new_tokens, max_input_tokens
        self.gpu_mem, self.dtype, self.seed, self.chunk = gpu_memory_utilization, dtype, seed, chunk
        self.tok = _load_tokenizer(model_name)
        self._llm = self._sp = None

    def _engine(self):
        if self._llm is None:
            from vllm import LLM, SamplingParams
            self._llm = LLM(model=self.name, dtype=self.dtype, max_model_len=self.max_input_tokens + self.max_new_tokens,
                            gpu_memory_utilization=self.gpu_mem, enable_prefix_caching=True, seed=self.seed)
            self._sp = SamplingParams(temperature=0.0, max_tokens=self.max_new_tokens)
            if self.tok is None:
                self.tok = self._llm.get_tokenizer()
        return self._llm

    def generate(self, batch):
        llm = self._engine()
        fitted = [self._fit(m) for m in batch]
        prompts = [{"prompt_token_ids": self.tok(f[1], add_special_tokens=False)["input_ids"]} for f in fitted]
        t0 = time.time()
        outs = llm.generate(prompts, self._sp, use_tqdm=False)
        dt = (time.time() - t0) * 1000 / max(len(batch), 1)
        return [GenOut(o.outputs[0].text.strip(), f[2], len(o.outputs[0].token_ids), dt, f[3]) for o, f in zip(outs, fitted)]

    def config(self):
        return {**super().config(), "backend": "vllm", "max_input_tokens": self.max_input_tokens, "decoding": "greedy",
                "dtype": self.dtype}


# ---------------------------------------------------------------------------------- OpenAI-compatible
class OpenAICompatLLM(_ChatFormat, BaseLLM):
    """OpenAI API or any compatible server (e.g. `vllm serve`, set OPENAI_BASE_URL). Requests are sent
    concurrently so the server can batch them. Token counting uses the HF tokenizer when `tokenizer` (or the
    model id) names one, so prompt budgets match the local backends."""

    def __init__(self, model: str, max_new_tokens: int = 512, seed: int = 42, concurrency: int = 64,
                 max_input_tokens: int = 8192, tokenizer: Optional[str] = None):
        from urllib.parse import urlparse

        from openai import OpenAI
        base = os.environ.get("OPENAI_BASE_URL") or ""
        host = urlparse(base).hostname or ""
        if host not in ("localhost", "127.0.0.1", "::1") and os.environ.get("PYPI_RAG_ALLOW_REMOTE_LLM") != "1":
            # prompts contain source code of suspect packages: never send it to an outside service by default
            raise RuntimeError("openai: backend is restricted to a local server (OPENAI_BASE_URL=http://localhost:8000/v1); "
                               "set PYPI_RAG_ALLOW_REMOTE_LLM=1 only for data you are allowed to share")
        self.name, self.max_new_tokens, self.seed, self.max_input_tokens = model, max_new_tokens, seed, max_input_tokens
        self.concurrency, self.chunk = concurrency, max(64, concurrency * 4)
        self.client = OpenAI(base_url=base or None,
                             api_key=os.environ.get("OPENAI_API_KEY") or "EMPTY", timeout=900, max_retries=5)
        tk = tokenizer or (model if "/" in model else None)
        self.tok = _load_tokenizer(tk) if tk else None

    def _one(self, msgs):
        msgs, _text, n, trunc = self._fit(msgs)
        t0 = time.time()
        r = self.client.chat.completions.create(model=self.name, messages=msgs, temperature=0.0, seed=self.seed,
                                                max_tokens=self.max_new_tokens)
        u = r.usage
        return GenOut((r.choices[0].message.content or "").strip(), getattr(u, "prompt_tokens", 0) or n,
                      getattr(u, "completion_tokens", 0) or 0, (time.time() - t0) * 1000, trunc)

    def generate(self, batch):
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.concurrency) as ex:
            return list(ex.map(self._one, batch))

    def config(self):
        return {**super().config(), "backend": "openai", "max_input_tokens": self.max_input_tokens, "decoding": "greedy"}


# ---------------------------------------------------------------------------------- cache
class CachedLLM(BaseLLM):
    def __init__(self, base: BaseLLM, cache_path: Optional[str]):
        self.base, self.cache_path = base, cache_path
        self.name, self.max_new_tokens = base.name, base.max_new_tokens
        self.cache: Dict[str, Dict] = {}
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, encoding="utf-8") as f:
                for line in f:
                    try:
                        d = json.loads(line); self.cache[d["key"]] = d
                    except (json.JSONDecodeError, KeyError):
                        pass

    def count_tokens(self, text):
        return self.base.count_tokens(text)

    def key(self, msgs) -> str:
        return hashlib.sha256(json.dumps({"cfg": self.base.config(), "m": msgs}, sort_keys=True).encode()).hexdigest()

    def generate(self, batch, chunk: Optional[int] = None):
        """Generates only unseen prompts (each distinct prompt once, in the given order), appends them to the
        JSONL cache after every chunk (resumable), and returns outputs in input order."""
        chunk = chunk or getattr(self.base, "chunk", 16)
        keys = [self.key(m) for m in batch]
        todo, seen = [], set()
        for k, m in zip(keys, batch):
            if k not in self.cache and k not in seen:
                seen.add(k); todo.append((k, m))
        t0, step = time.time(), max(chunk, len(todo) // 20 or 1)
        for s in range(0, len(todo), chunk):
            part = todo[s:s + chunk]
            outs = self.base.generate([m for _, m in part])
            rows = []
            for (k, _), o in zip(part, outs):
                d = {"key": k, "text": o.text, "input_tokens": o.input_tokens, "output_tokens": o.output_tokens,
                     "latency_ms": o.latency_ms, "truncated": o.truncated}
                self.cache[k] = d; rows.append(d)
            if self.cache_path:
                os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
                with open(self.cache_path, "a", encoding="utf-8") as f:
                    for d in rows:
                        f.write(json.dumps(d) + "\n")
            done = s + len(part)
            if len(todo) > chunk and (done // step != s // step or done == len(todo)):
                el = time.time() - t0
                print(f"[llm] {done}/{len(todo)} new generations | {el / 60:.1f} min | "
                      f"eta {el / done * (len(todo) - done) / 60:.1f} min", flush=True)
        return [GenOut(self.cache[k]["text"], self.cache[k]["input_tokens"], self.cache[k]["output_tokens"],
                       self.cache[k]["latency_ms"], self.cache[k].get("truncated", False), cached=k not in seen)
                for k in keys]

    def config(self):
        return self.base.config()


def llm_from_spec(spec: str, **kw) -> BaseLLM:
    """dummy | hf:<model> | vllm:<model> | openai:<model>"""
    pick = lambda *names: {k: v for k, v in kw.items() if k in names and v is not None}  # noqa: E731
    if spec == "dummy":
        return DummyLLM()
    if spec.startswith("hf:"):
        return HFLLM(spec[3:], **pick("max_new_tokens", "max_input_tokens", "batch_size", "adapter"))
    if spec.startswith("vllm:"):
        if kw.get("adapter"):
            raise ValueError("--adapter is only supported with hf: models")
        return VLLMLLM(spec[5:], **pick("max_new_tokens", "max_input_tokens", "gpu_memory_utilization"))
    if spec.startswith("openai:"):
        return OpenAICompatLLM(spec[7:], **pick("max_new_tokens", "max_input_tokens", "concurrency", "tokenizer"))
    raise ValueError(f"unknown llm spec {spec}")
