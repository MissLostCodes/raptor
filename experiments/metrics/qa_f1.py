"""QASPER answer token-F1 (SQuAD/QASPER style).

Reference: Dasigi et al. (2021), QASPER; SQuAD F1 normalization (Rajpurkar
et al., 2016). The official QASPER evaluator uses these exact normalization
and token-F1 definitions for answer scoring.
"""

import re
import string
from collections import Counter
from typing import List


def normalize_answer(s: str) -> str:
    """Lowercase, remove punctuation, remove articles, collapse whitespace."""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def token_f1(prediction: str, ground_truth: str) -> float:
    """F1 over the multiset of normalized tokens.

    Mirrors the SQuAD/QASPER ``compute_f1``: if both normalized strings are
    empty, they are considered to match (1.0); if exactly one is empty, 0.0;
    if there is no token overlap, 0.0.
    """
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()

    # If either is empty, F1 is 1 iff both are empty (mirrors official eval).
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return float(pred_tokens == gold_tokens)

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


def qasper_answer_f1(
    prediction: str,
    gold_answers: List[str],
    is_unanswerable: bool = False,
) -> float:
    """QASPER answer F1.

    For unanswerable questions (flagged or with no gold answers) the model is
    rewarded for abstaining: 1.0 if the (normalized) prediction is empty or
    contains the token 'unanswerable', otherwise 0.0. For answerable
    questions, returns the max token-F1 over the gold answers.
    """
    if is_unanswerable or len(gold_answers) == 0:
        norm = normalize_answer(prediction)
        if norm == "" or "unanswerable" in norm:
            return 1.0
        return 0.0

    return max(token_f1(prediction, gold) for gold in gold_answers)
