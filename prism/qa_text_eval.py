"""Text-level QA scoring, ported from GNN-RAG `evaluate_results.py`
(identical to P6_LLM_Evaluation cell 4). Used whenever an LLM reader
produces free text that must be matched against gold entity names."""
from __future__ import annotations

import re
import string
from typing import List, Sequence, Tuple


def normalize(s: str) -> str:
    s = str(s).lower()
    exclude = set(string.punctuation)
    s = ''.join(ch for ch in s if ch not in exclude)
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = re.sub(r'\b(<pad>)\b', ' ', s)
    return ' '.join(s.split())


def match(prediction: str, answer: str) -> bool:
    return normalize(answer) in normalize(prediction)


_SKIP = [r'^based on', r'^according to', r'^from the reasoning', r'^the reasoning path',
         r'^please note', r'^note:', r'^therefore', r'^hence', r'^in conclusion',
         r'^looking at', r'^considering', r'^given that', r'^i can see', r'^we can see',
         r'^it (is|appears|seems)',
         # Nov 2026 (D2): Llama-3.1 / Qwen chat openers that are NOT answers
         r'^here (is|are)', r'^the (possible |final |correct |likely )?answers?( to (the|this|your) question)?( (is|are|include|would be))?:?$',
         r'^(possible |final )?answers?:?$', r'^(the )?answers?:\s*\S', r'^to answer', r'^the question (asks|is)', r'^these are',
         r'^(so|thus|overall|in summary)\b']


def extract_answer_entities(raw_prediction: str) -> List[str]:
    """Pull entity answers out of verbose LLM output (lists, 'the answer is X')."""
    lines = [l.strip() for l in str(raw_prediction).split('\n') if l.strip()]
    if not lines:
        return [str(raw_prediction).strip()]
    entities: List[str] = []
    for line in lines:
        cleaned = re.sub(r'^[\d]+[\.\)]\s*', '', line.replace('**', ''))            # numbering, markdown bold
        cleaned = re.sub(r'^[-*\u2022]\s*', '', cleaned).strip()
        cleaned = cleaned.strip('"\'*_').rstrip('.').strip()                             # quotes / italics / final period
        if cleaned.endswith(':') and len(cleaned.split()) <= 8:                        # header line ("Possible answers:")
            continue
        if any(re.match(p, cleaned.lower()) for p in _SKIP):
            m = re.search(r'(?:answers?\s+(?:is|are)|answers?:)\s*(.+?)\.?$', cleaned, re.I)
            if m:
                parts = re.split(r',\s*(?:and\s+)?|\s+and\s+', m.group(1).strip())
                entities.extend([p.strip() for p in parts if p.strip()])
            continue
        if cleaned:
            entities.append(cleaned)
    return entities if entities else lines


def eval_hit(prediction: str, answers: Sequence[str]) -> int:
    return int(any(match(prediction, a) for a in answers)) if answers else 0


def eval_hit1(prediction_list: Sequence[str], answers: Sequence[str]) -> int:
    if not prediction_list or not answers:
        return 0
    return int(any(match(prediction_list[0], a) for a in answers))


def eval_f1(prediction_list: Sequence[str], answers: Sequence[str]) -> Tuple[float, float, float]:
    if not prediction_list or not answers:
        return 0.0, 0.0, 0.0
    pred_str = ' '.join(prediction_list)
    matched = sum(1 for a in answers if match(pred_str, a))
    precision = matched / len(prediction_list)
    recall = matched / len(answers)
    if precision + recall == 0:
        return 0.0, precision, recall
    return 2 * precision * recall / (precision + recall), precision, recall
