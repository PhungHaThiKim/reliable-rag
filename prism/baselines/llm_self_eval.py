"""LLM self-evaluation confidence baseline (Tian et al. 2023).

LLM directly reports its own confidence: "On a scale of 0-1, how confident
are you that the answer is correct?". A strong baseline because it uses the
SAME LLM whose answer we're calibrating.

Used in PRISM Experiment 4 to demonstrate that PRISM's *retrieval* confidence
adds value over what the LLM itself can introspect.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch


class LLMSelfEvalConfidence:
    """Wraps a frozen LLM to extract self-reported confidence per (q, S, ans).

    Modes:
        "verbal":     Ask "How confident? 0-1". Parse the number from output.
        "token_prob": Score next token after "Yes" vs "No" to question
                      "Is this answer correct?". Default — more reliable.
        "entropy":    Token-level entropy of the answer span. Negative-mapped
                      via 1 - normalised_entropy. Tian et al. 2023 variant.
    """

    def __init__(self, llm_wrapper, mode: str = "token_prob"):
        """Args:
            llm_wrapper: object with .model and .tok attributes
                          (e.g., LiveTeacherLLM from teacher_llm.py)
            mode: "verbal" | "token_prob" | "entropy"
        """
        self.llm = llm_wrapper
        self.mode = mode

    @torch.inference_mode()
    def score(self, question: str, paths_text: str,
                answer: str) -> float:
        """Returns confidence ∈ [0,1] from LLM self-evaluation."""
        if self.mode == "token_prob":
            return self._score_yes_no(question, paths_text, answer)
        elif self.mode == "verbal":
            return self._score_verbal(question, paths_text, answer)
        elif self.mode == "entropy":
            return self._score_entropy(question, paths_text, answer)
        raise ValueError(f"unknown mode: {self.mode}")

    @torch.inference_mode()
    def _score_yes_no(self, question: str, paths_text: str,
                        answer: str) -> float:
        """P(Yes | "Is '{answer}' correct given context?")"""
        prompt_msg = (
            f"Question: {question}\n\n"
            f"Reasoning paths:\n{paths_text}\n\n"
            f"Proposed answer: {answer}\n\n"
            f"Is the proposed answer correct given the reasoning paths? "
            f"Answer with one word: Yes or No.\n\n"
            f"Answer:"
        )
        messages = [{"role": "user", "content": prompt_msg}]
        prompt = self.llm.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        enc = self.llm.tok(prompt, return_tensors="pt", truncation=True,
                            max_length=getattr(self.llm.tcfg, "max_context", 4096))
        input_ids = enc["input_ids"].to(self.llm.model.device)
        out = self.llm.model(input_ids)
        last_logits = out.logits[0, -1].float()

        yes_ids = self.llm.tok(" Yes", add_special_tokens=False)["input_ids"]
        no_ids = self.llm.tok(" No", add_special_tokens=False)["input_ids"]
        if not yes_ids or not no_ids:
            return 0.5
        # First token only
        yes_logit = float(last_logits[yes_ids[0]].item())
        no_logit = float(last_logits[no_ids[0]].item())
        # Soft-sigmoid: P(Yes) = exp(yes) / (exp(yes) + exp(no))
        m = max(yes_logit, no_logit)
        p_yes = float(
            torch.exp(torch.tensor(yes_logit - m)) /
            (torch.exp(torch.tensor(yes_logit - m)) +
             torch.exp(torch.tensor(no_logit - m)))
        )
        return p_yes

    @torch.inference_mode()
    def _score_verbal(self, question: str, paths_text: str,
                        answer: str) -> float:
        """Ask for a number 0-1. Parse from LLM output."""
        prompt_msg = (
            f"Question: {question}\n\n"
            f"Reasoning paths:\n{paths_text}\n\n"
            f"Proposed answer: {answer}\n\n"
            f"On a scale from 0.00 to 1.00, how confident are you that the "
            f"proposed answer is correct? Respond with just the number.\n\n"
            f"Confidence:"
        )
        messages = [{"role": "user", "content": prompt_msg}]
        prompt = self.llm.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        enc = self.llm.tok(prompt, return_tensors="pt", truncation=True,
                            max_length=4096)
        input_ids = enc["input_ids"].to(self.llm.model.device)
        out = self.llm.model.generate(input_ids, max_new_tokens=8,
                                       do_sample=False,
                                       pad_token_id=self.llm.tok.eos_token_id)
        text = self.llm.tok.decode(out[0, input_ids.shape[1]:],
                                    skip_special_tokens=True)
        # Parse first float from text
        import re
        m = re.search(r"([0-9]*\.?[0-9]+)", text)
        if not m:
            return 0.5
        try:
            val = float(m.group(1))
            return max(0.0, min(1.0, val))
        except ValueError:
            return 0.5

    @torch.inference_mode()
    def _score_entropy(self, question: str, paths_text: str,
                        answer: str) -> float:
        """1 - normalised token entropy of the LLM's answer generation."""
        prompt_msg = (
            f"Question: {question}\n\n"
            f"Reasoning paths:\n{paths_text}\n\n"
            f"Answer:"
        )
        messages = [{"role": "user", "content": prompt_msg}]
        prompt = self.llm.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        enc = self.llm.tok(prompt, return_tensors="pt", truncation=True,
                            max_length=4096)
        input_ids = enc["input_ids"].to(self.llm.model.device)
        # Get token-by-token logits for `answer`
        ans_ids = self.llm.tok(" " + answer, add_special_tokens=False)["input_ids"]
        if not ans_ids:
            return 0.5
        ans_t = torch.tensor([ans_ids], device=self.llm.model.device)
        full = torch.cat([input_ids, ans_t], dim=1)
        out = self.llm.model(full)
        # Compute per-token entropy
        log_probs = torch.log_softmax(out.logits[0, input_ids.shape[1] - 1:-1].float(), dim=-1)
        # Average entropy
        ent = -(log_probs.exp() * log_probs).sum(dim=-1).mean().item()
        max_ent = float(torch.log(torch.tensor(log_probs.size(-1), dtype=torch.float32)).item())
        norm_ent = ent / max(max_ent, 1e-9)
        return float(1.0 - min(1.0, max(0.0, norm_ent)))
