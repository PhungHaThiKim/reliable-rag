"""Real-LLM teacher for PRISM distillation.

Replaces the heuristic `TeacherLLM` in data.py with an actual frozen language
model (Llama-3.1-8B-Instruct by default). Outputs P(answer | q, S) by reading
the LLM's next-token logits at the candidate answer positions.

Critical performance optimisations:
1. **First-token shortcut**: only score the first token of each candidate string,
   not full multi-token decoding. ~50x cheaper than autoregressive scoring.
2. **Batched prompts**: when computing teacher for multiple (q, S) pairs,
   pad-and-batch them through the LLM in one forward.
3. **Disk cache**: every (qid, subset_key) → teacher_distribution mapping
   persists in a pickle so Stage 1 distill never re-calls the LLM.

Usage:

    # One-time offline caching (~75 min for WebQSP train on A100)
    from prism.teacher_llm import CachedTeacherLLM, build_teacher_cache
    build_teacher_cache(
        train_set, eval_set,
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        cache_path="teacher_logits_webqsp.pt",
        subsets_per_sample=3,
    )

    # During distillation
    teacher = CachedTeacherLLM(cache_path="teacher_logits_webqsp.pt")
    # plug into distill.py without changing distill_step signature
"""
from __future__ import annotations

import hashlib
import os
import pickle
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .config import CFG
from .data import Sample


# ─────────────────────────────────────────────────────────────────────
# Subset hashing — deterministic key for cache lookup
# ─────────────────────────────────────────────────────────────────────

def subset_key(qid: str, selected_path_idx: Sequence[int]) -> str:
    """Deterministic short key for a (qid, subset) pair."""
    sorted_idx = sorted(int(i) for i in selected_path_idx)
    raw = f"{qid}|{','.join(str(i) for i in sorted_idx)}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


# ─────────────────────────────────────────────────────────────────────
# Cached teacher — main interface for distill.py
# ─────────────────────────────────────────────────────────────────────

class CachedTeacherLLM:
    """Drop-in replacement for `data.TeacherLLM`. Reads pre-computed
    teacher distributions from disk; falls back to uniform if missing.

    Cache format: Dict[subset_key -> Tensor[C]] saved via torch.save.
    """

    def __init__(self, cache_path: Optional[str] = None,
                 strict: bool = False, verbose: bool = False):
        """Args:
            cache_path: path to torch.save'd Dict[str, Tensor].
            strict: if True, raise on missing key. Else return uniform.
            verbose: print cache stats on init.
        """
        self.cache_path = cache_path
        self.strict = strict
        self.verbose = verbose
        self.cache: Dict[str, torch.Tensor] = {}
        self.n_hits = 0
        self.n_miss = 0

        if cache_path and os.path.exists(cache_path):
            self.cache = torch.load(cache_path, map_location="cpu",
                                     weights_only=False)
            if verbose:
                print(f"[teacher cache] loaded {len(self.cache):,} entries "
                      f"from {cache_path}")
        elif strict:
            raise FileNotFoundError(f"Teacher cache not found: {cache_path}")
        else:
            if verbose:
                print(f"[teacher cache] empty (will return uniform); "
                      f"path={cache_path}")

    def score_subgraph(self, sample: Sample,
                        selected_path_idx: Sequence[int]) -> torch.Tensor:
        """Return P_M(a | q, S) as a tensor of shape [C]."""
        key = subset_key(sample.qid, selected_path_idx)
        if key in self.cache:
            self.n_hits += 1
            t = self.cache[key]
            if t.numel() != sample.candidate_ids.size(0):
                # Shape mismatch (e.g., re-extracted samples) → fallback
                self.n_miss += 1
                return self._uniform(sample)
            return t.float()
        self.n_miss += 1
        if self.strict:
            raise KeyError(f"missing teacher cache key for {sample.qid}")
        return self._uniform(sample)

    @staticmethod
    def _uniform(sample: Sample) -> torch.Tensor:
        c = sample.candidate_ids.size(0)
        return torch.full((c,), 1.0 / max(c, 1), dtype=torch.float32)

    def stats(self) -> Dict[str, float]:
        total = self.n_hits + self.n_miss
        return {
            "cached": len(self.cache),
            "hits": self.n_hits,
            "misses": self.n_miss,
            "hit_rate": self.n_hits / max(total, 1),
        }


# ─────────────────────────────────────────────────────────────────────
# Live LLM teacher — used during offline cache build
# ─────────────────────────────────────────────────────────────────────

@dataclass
class TeacherConfig:
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    # NousResearch mirror for users without HF token
    fallback_model: str = "NousResearch/Meta-Llama-3.1-8B-Instruct"
    dtype: str = "bf16"          # fp16 | bf16 | fp32
    max_context: int = 4096
    instruction: str = (
        "Based on the reasoning paths, please answer the given question by "
        "selecting one of the listed candidates. Answer with only the "
        "candidate name."
    )
    # When candidates are entity-ID strings (m.0xxx), provide a names map
    # via TeacherConfig.id_to_name (Dict[str, str]). Optional.
    id_to_name: Optional[Dict[str, str]] = None

    # ─── Issue #2 fix: scoring mode ───────────────────────────────────────
    # "first_token": original — score each candidate's FIRST BPE token.
    #                BROKEN on Freebase MIDs (all start with 'm') and on
    #                names sharing common prefixes (length bias).
    # "letter":      MCQ-style — append "A) cand1\nB) cand2\n..." and score
    #                next-token = letter. Constant cost (max 26 candidates).
    #                Recommended for cache build. (S-Path-RAG style.)
    # "sum_logprob": Full multi-token scoring. ~5x slower but principled.
    #                Use for validation experiment (200 samples) to show
    #                "letter" ≈ "sum_logprob".
    scoring_mode: str = "letter"


class LiveTeacherLLM:
    """Wraps a frozen LLM. Use ONLY during offline cache build.

    For each (q, S), constructs a prompt, runs ONE forward pass, and reads the
    candidate first-token logits.
    """

    def __init__(self, cfg: Optional[TeacherConfig] = None,
                 device: Optional[str] = None):
        self.tcfg = cfg or TeacherConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model()

    def _load_model(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
        torch_dtype = dtype_map.get(self.tcfg.dtype, torch.bfloat16)

        for model_name in (self.tcfg.model_name, self.tcfg.fallback_model):
            try:
                self.tok = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name, torch_dtype=torch_dtype, device_map="auto",
                )
                self.model.eval()
                self.loaded_model_name = model_name
                print(f"[teacher] loaded {model_name} ({self.tcfg.dtype})")
                return
            except Exception as e:
                print(f"[teacher] failed to load {model_name}: {e}")
        raise RuntimeError("could not load any teacher model")

    def _candidate_name(self, cand_id: int,
                         id_lookup: Optional[Dict[int, str]] = None) -> str:
        if id_lookup is not None and cand_id in id_lookup:
            name = id_lookup[cand_id]
            if self.tcfg.id_to_name:
                name = self.tcfg.id_to_name.get(name, name)
            return str(name)
        return str(cand_id)

    def _verbalise_paths(self, sample: Sample,
                          selected_path_idx: Sequence[int],
                          rel_lookup: Optional[Dict[int, str]] = None,
                          ent_lookup: Optional[Dict[int, str]] = None) -> str:
        """Turn selected paths into natural-language reasoning chains.

        rel_lookup / ent_lookup are optional ID→string maps. Falls back to
        printing IDs if absent.
        """
        if not selected_path_idx:
            return "(no paths)"
        lines = []
        for idx in selected_path_idx:
            rels = sample.paths[idx].tolist()
            cand_id = int(sample.path_to_candidate[idx].item())
            cand_name = self._candidate_name(cand_id, ent_lookup)
            rel_names = [rel_lookup.get(r, f"r{r}") if rel_lookup else f"r{r}"
                         for r in rels if r != 0]
            lines.append(" → ".join(rel_names + [cand_name]))
        return "\n".join(lines)

    @torch.inference_mode()
    def score_subgraph(self, sample: Sample,
                        selected_path_idx: Sequence[int],
                        question_text: Optional[str] = None,
                        rel_lookup: Optional[Dict[int, str]] = None,
                        ent_lookup: Optional[Dict[int, str]] = None,
                        ) -> torch.Tensor:
        """Returns P_M(a | q, S) as a Tensor[C] (CPU, float32).

        Scoring mode is controlled by self.tcfg.scoring_mode:
            "letter"      — MCQ-style, robust to MID prefixes (RECOMMENDED)
            "first_token" — score 1st BPE of name (BROKEN on Freebase)
            "sum_logprob" — full multi-token, expensive but principled
        """
        candidate_names = [self._candidate_name(int(c), ent_lookup)
                           for c in sample.candidate_ids.tolist()]
        ctx = self._verbalise_paths(sample, selected_path_idx,
                                    rel_lookup, ent_lookup)
        q_text = question_text or getattr(sample, "question_text", f"q_{sample.qid}")

        mode = getattr(self.tcfg, "scoring_mode", "letter")
        if mode == "letter":
            return self._score_letter(q_text, ctx, candidate_names)
        elif mode == "sum_logprob":
            return self._score_sum_logprob(q_text, ctx, candidate_names)
        else:  # "first_token" (legacy; keep for ablation)
            return self._score_first_token(q_text, ctx, candidate_names)

    # ─────────────────────────────────────────────────────────────
    # Scoring strategies — Issue #2 fix
    # ─────────────────────────────────────────────────────────────

    def _build_messages(self, q_text: str, ctx: str, candidate_block: str,
                          answer_hint: str):
        user_content = (
            f"{self.tcfg.instruction}\n\n"
            f"Question: {q_text}\n\n"
            f"Reasoning paths:\n{ctx}\n\n"
            f"Candidate answers:\n{candidate_block}\n\n"
            f"{answer_hint}"
        )
        messages = [{"role": "user", "content": user_content}]
        return self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

    @torch.inference_mode()
    def _score_letter(self, q_text: str, ctx: str,
                        candidate_names: List[str]) -> torch.Tensor:
        """MCQ-style scoring — no length-bias, robust to MID prefixes.

        Prompt: "A) cand1\\nB) cand2\\n..."  Score next-token = each letter.
        Constant cost (one LLM forward). Best for cache build.
        """
        n = len(candidate_names)
        # Generate letter labels: A, B, ..., Z, AA, AB, ... (rare to exceed 26)
        letters = []
        for i in range(n):
            if i < 26:
                letters.append(chr(ord('A') + i))
            else:
                letters.append(chr(ord('A') + (i // 26) - 1) + chr(ord('A') + (i % 26)))

        candidate_block = "\n".join(f"{lt}) {nm}" for lt, nm in
                                    zip(letters, candidate_names))
        prompt = self._build_messages(
            q_text, ctx, candidate_block,
            answer_hint="Answer (single letter only):",
        )
        enc = self.tok(prompt, return_tensors="pt",
                       truncation=True, max_length=self.tcfg.max_context)
        input_ids = enc["input_ids"].to(self.model.device)
        out = self.model(input_ids)
        last_logits = out.logits[0, -1].float()

        cand_logits = torch.empty(n)
        for i, lt in enumerate(letters):
            # Letter tokens are usually single-piece. Use " A", "A", etc. for
            # max robustness across tokenizers.
            ids = self.tok(lt, add_special_tokens=False)["input_ids"]
            if not ids:
                cand_logits[i] = -1e9
                continue
            # Score the first (and usually only) token of the letter
            cand_logits[i] = float(last_logits[ids[0]].item())

        return torch.softmax(cand_logits, dim=0)

    @torch.inference_mode()
    def _score_first_token(self, q_text: str, ctx: str,
                             candidate_names: List[str]) -> torch.Tensor:
        """LEGACY: first-token of each candidate name. BROKEN on Freebase MIDs.
        Kept for ablation purposes only."""
        candidate_block = "\n".join(f"- {c}" for c in candidate_names)
        prompt = self._build_messages(
            q_text, ctx, candidate_block,
            answer_hint="Your answer (exactly one candidate from the list):",
        )
        enc = self.tok(prompt, return_tensors="pt",
                       truncation=True, max_length=self.tcfg.max_context)
        input_ids = enc["input_ids"].to(self.model.device)
        out = self.model(input_ids)
        last_logits = out.logits[0, -1].float()

        cand_logits = torch.empty(len(candidate_names))
        for i, c in enumerate(candidate_names):
            toks = self.tok(c, add_special_tokens=False)["input_ids"]
            cand_logits[i] = (float(last_logits[toks[0]].item())
                              if toks else -1e9)
        return torch.softmax(cand_logits, dim=0)

    @torch.inference_mode()
    def _score_sum_logprob(self, q_text: str, ctx: str,
                              candidate_names: List[str]) -> torch.Tensor:
        """Full multi-token scoring — sum log P(token_t | prefix) over all
        tokens of each candidate name. ~5× slower than first-token but
        gives true P_M(c | q, S) up to length normalisation.

        Use this in validation experiment (Exp 2.b) to show letter ≈ sum_logprob.
        """
        candidate_block = "\n".join(f"- {c}" for c in candidate_names)
        prompt = self._build_messages(
            q_text, ctx, candidate_block,
            answer_hint="Your answer:",
        )
        # Score each candidate by concatenation + log-prob sum
        prompt_ids = self.tok(prompt, return_tensors="pt",
                               truncation=True,
                               max_length=self.tcfg.max_context).input_ids
        prompt_ids = prompt_ids.to(self.model.device)

        cand_logprobs = torch.empty(len(candidate_names))
        for i, c in enumerate(candidate_names):
            cand_ids = self.tok(" " + c, add_special_tokens=False).input_ids
            if not cand_ids:
                cand_logprobs[i] = -1e9
                continue
            cand_t = torch.tensor([cand_ids], device=self.model.device)
            full = torch.cat([prompt_ids, cand_t], dim=1)
            out = self.model(full)
            # Sum log-prob over the candidate tokens
            log_probs = torch.log_softmax(out.logits[0].float(), dim=-1)
            start = prompt_ids.shape[1] - 1
            score = 0.0
            for t, tok_id in enumerate(cand_ids):
                score += float(log_probs[start + t, tok_id].item())
            # Length-normalise to mitigate length bias
            cand_logprobs[i] = score / max(len(cand_ids), 1)

        return torch.softmax(cand_logprobs, dim=0)


# ─────────────────────────────────────────────────────────────────────
# Offline cache builder
# ─────────────────────────────────────────────────────────────────────

def _pick_subsets(n_paths: int, n_subsets: int = 6, rng=None,
                    stratified: bool = True) -> List[List[int]]:
    """For each sample, pick a SIZE-STRATIFIED collection of subsets.

    Issue #1 fix (reviewer): the original 3-mode {empty, random, full} is
    O(log) coverage of the size spectrum. With |P|=50, controller queries
    intermediate |S_t| (e.g. 4) are off-manifold.

    Stratified: target sizes log-spaced across [0, n_paths], so the LAS
    learns to interpolate over all subset sizes the controller may visit.

    Defaults to 6 stratified subsets:
        sizes ≈ {0, 2, 5, 10, 20, n_paths}
    All cost the same as 3-mode (no extra LLM calls beyond the 6).
    """
    import random as _r
    rng = rng or _r.Random()
    if not stratified:
        # Legacy behaviour: empty + random middles + full
        base: List[List[int]] = [[], list(range(n_paths))]
        for _ in range(n_subsets - 2):
            k = rng.randint(1, max(1, n_paths - 1))
            base.append(sorted(rng.sample(range(n_paths), k)))
        return base

    # Stratified: log-spaced target sizes across the full range
    n = max(n_paths, 1)
    # Default targets — paper Exp 1.1 (subset-size ablation)
    target_sizes = [0, 2, 5, 10, 20, n]
    # If user requested fewer, sample log-spaced from the default 6
    if n_subsets < len(target_sizes):
        import math
        step = len(target_sizes) / n_subsets
        target_sizes = [target_sizes[int(i * step)] for i in range(n_subsets)]
    # Clamp to [0, n]
    target_sizes = sorted({min(n, max(0, t)) for t in target_sizes})

    subsets: List[List[int]] = []
    for k in target_sizes:
        if k == 0:
            subsets.append([])
        elif k >= n:
            subsets.append(list(range(n)))
        else:
            subsets.append(sorted(rng.sample(range(n), k)))
    return subsets


def build_teacher_cache(
    *datasets,
    out_path: str,
    teacher: Optional[LiveTeacherLLM] = None,
    teacher_cfg: Optional[TeacherConfig] = None,
    subsets_per_sample: int = 3,
    rel_lookup: Optional[Dict[int, str]] = None,
    ent_lookup: Optional[Dict[int, str]] = None,
    question_lookup: Optional[Dict[str, str]] = None,
    flush_every: int = 200,
    skip_existing: bool = True,
) -> Dict[str, torch.Tensor]:
    """Pre-compute teacher distributions for every (sample, subset) pair.

    Args:
        datasets: one or more Dataset objects (train, eval, ...).
        out_path: where to torch.save the dict.
        teacher: pre-loaded LiveTeacherLLM; constructed on demand if None.
        subsets_per_sample: how many subsets to cache per sample.
        rel_lookup / ent_lookup: ID→str maps for prompt verbalisation.
        question_lookup: qid → question text. Required if Sample lacks
            `.question_text`.
        flush_every: dump partial cache to disk every N entries (crash-safe).
        skip_existing: if True, load existing cache file and skip its keys.

    Returns:
        The full cache dict (also persisted to out_path).
    """
    import random as _r
    teacher = teacher or LiveTeacherLLM(teacher_cfg)

    # Resume
    cache: Dict[str, torch.Tensor] = {}
    if skip_existing and os.path.exists(out_path):
        cache = torch.load(out_path, map_location="cpu", weights_only=False)
        print(f"[teacher cache] resuming with {len(cache):,} existing entries")

    rng = _r.Random(CFG.seed)
    t_start = time.time()
    n_done = 0
    n_target = 0

    for ds in datasets:
        for i in range(len(ds)):
            sample: Sample = ds[i]
            n_paths = sample.paths.size(0)
            subsets = _pick_subsets(n_paths, subsets_per_sample, rng)

            q_text = None
            if question_lookup and sample.qid in question_lookup:
                q_text = question_lookup[sample.qid]
            elif hasattr(sample, "question_text"):
                q_text = sample.question_text

            for sub in subsets:
                key = subset_key(sample.qid, sub)
                if key in cache:
                    continue
                n_target += 1
                probs = teacher.score_subgraph(
                    sample, sub,
                    question_text=q_text,
                    rel_lookup=rel_lookup,
                    ent_lookup=ent_lookup,
                )
                cache[key] = probs.cpu().float()
                n_done += 1

                if n_done % flush_every == 0:
                    torch.save(cache, out_path)
                    rate = n_done / max(time.time() - t_start, 1e-3)
                    print(f"[teacher cache] {n_done:,} new entries "
                          f"({len(cache):,} total)  rate={rate:.1f}/s  "
                          f"target={n_target:,}")

    torch.save(cache, out_path)
    print(f"[teacher cache] DONE  total={len(cache):,}  new={n_done:,}  "
          f"elapsed={time.time()-t_start:.0f}s  out={out_path}")
    return cache


# ─────────────────────────────────────────────────────────────────────
# Convenience: load cache + wrap to match old TeacherLLM API
# ─────────────────────────────────────────────────────────────────────

def make_teacher(cache_path: Optional[str] = None,
                  fallback_uniform: bool = True) -> CachedTeacherLLM:
    """Build a teacher object usable directly in distill.py.

    If `cache_path` is None or missing and fallback_uniform=True, returns a
    uniform-fallback teacher (useful for smoke tests).
    """
    return CachedTeacherLLM(cache_path=cache_path,
                             strict=not fallback_uniform,
                             verbose=True)
