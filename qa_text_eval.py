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
         r'^(so|thus|overall|in summary)\b',
         # Nov 2026 (P9 review): 'The answer is: X' and meta-comments about the paths
         r'^(the )?(\w+ )?answers? (to (the|this|your) question )?(is|are|would be|include)\b']
# meta-comments about the context: never an answer, dropped outright (not even as fallback)
_META = [r'^the (given |provided )?(reasoning )?paths? (only |also )?(mention|provide|contain|list|include|show|say)',
         r'^they (only |also )?(mention|provide|contain|list|include)\b']


# Abstention phrases ("unknown", "the paths do not provide ...", "I couldn't find ...").
# Used per line/sentence by extract_answer_entities and on the whole text by split_answer.
ABSTAIN_RE = re.compile(
    r"\b(unknown|not enough (information|evidence)|insufficient (information|evidence)"
    r"|can ?not (be )?(determined|answered|found)|can ?not (determine|answer|find)|can'?t (determine|answer|find)"
    r"|unable to (determine|answer|find)|could ?n[o']?t (determine|answer|find)"
    r"|no (sufficient |relevant |direct |specific |explicit )?(information|evidence|mention)"
    r"|(does|do|did) ?n[o']?t (contain|provide|mention|include|specify|say|give)"
    r"|not (directly |explicitly )?(provided|mentioned|available|specified|stated))\b", re.I)
_SENT_SPLIT = re.compile(r"(?<=[a-z0-9\)][.!?])\s+(?=[A-Z])")   # "Salmon P. Chase" is not split
_ANSWER_IS = re.compile(r"(?:answers?\s+(?:is|are|would be)|answers?)\s*:?\s*(.+?)\.?$", re.I)


def _has_word(s: str) -> bool:
    return bool(re.search(r"\w", s))


def _clean_item(s: str) -> str:
    """Strip list markers / quotes; reduce a copied path 'A -> r -> B' to its tail entity."""
    s = re.sub(r'^[\d]+[\.\)]\s*', '', s.replace('**', ''))                       # numbering, markdown bold
    s = re.sub(r'^[-*\u2022]\s*', '', s).strip()
    s = s.strip('"\'*_`').rstrip('.,;').strip()                                    # quotes / italics / final period
    if '->' in s:
        parts = [x.strip() for x in s.split('->') if x.strip()]
        if len(parts) >= 3:                                                         # "A -> rel -> B [-> rel -> C]"
            s = parts[-1]
    return s


def _json_list(line: str):
    """Items of a (possibly truncated) JSON-style list line: ["a", "b", ...]."""
    if not line.startswith('['):
        return None
    try:
        import json
        val = json.loads(line)
        if isinstance(val, list):
            return [str(x) for x in val]
    except Exception:
        pass
    items = re.findall(r'"([^"]+)"', line) or re.findall(r"'([^']+)'", line)
    return items or None


def extract_answer_entities(raw_prediction: str) -> List[str]:
    """Pull entity answers out of verbose LLM output (lists, 'the answer is X').

    Nov 2026 (P9 review) fixes: a lead-in such as 'the answer is:' no longer yields
    the claim ':'; any line ending with ':' is a header; copied paths 'A -> r -> B'
    give their tail entity; JSON-style list lines are split into items; sentences
    that only state missing evidence ('the paths do not provide ...') are dropped,
    so an answer made only of such sentences returns [] (an abstention)."""
    lines = [l.strip() for l in str(raw_prediction).split('\n') if l.strip()]
    entities: List[str] = []
    fallback: List[str] = []          # opener-only lines (e.g. 'Based on the paths, Paris is ...') kept if nothing else
    for line in lines:
        items = _json_list(line)
        if items is not None:
            entities.extend(c for c in (_clean_item(x) for x in items) if _has_word(c) and not ABSTAIN_RE.search(c))
            continue
        cleaned = _clean_item(line)
        if not _has_word(cleaned) or cleaned.endswith(':'):                         # punctuation / header line
            continue
        sents = _SENT_SPLIT.split(cleaned) if len(cleaned.split()) >= 8 else [cleaned]
        for sent in sents:
            sent = sent.strip()
            if not _has_word(sent) or sent.endswith(':') or ABSTAIN_RE.search(sent):
                continue
            sent = _clean_item(sent)
            if any(re.match(p, sent.lower()) for p in _META):
                continue
            if any(re.match(p, sent.lower()) for p in _SKIP):
                m = _ANSWER_IS.search(sent)
                if m:
                    parts = re.split(r',\s*(?:and\s+)?|\s+and\s+', m.group(1).strip())
                    entities.extend(c for c in (_clean_item(x) for x in parts) if _has_word(c))
                else:
                    fallback.append(sent)
                continue
            entities.append(sent)
    return entities if entities else fallback


def split_answer(raw_prediction: str) -> Tuple[List[str], bool]:
    """(answer entities, abstained). Abstained = no entity left AND the text states
    missing evidence; an abstention sentence followed by a parametric answer
    ('...do not provide... However, it is X') is an ANSWER, not an abstention."""
    ents = extract_answer_entities(raw_prediction)
    return ents, (not ents) and bool(ABSTAIN_RE.search(str(raw_prediction)))


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
