"""QuALITY multiple-choice accuracy.

Reference: Pang et al. (2022), QuALITY. Models emit free text; we map that to a
0-based option index via (a) an explicit letter, (b) an explicit option number,
or (c) normalized token-overlap with the options as a fallback.
"""

import re
import string
from collections import Counter
from typing import List


def _normalize_tokens(s: str) -> List[str]:
    s = s.lower()
    s = "".join(ch if ch not in string.punctuation else " " for ch in s)
    return s.split()


def extract_choice(model_output: str, options: List[str]) -> int:
    """Map a free-text answer to a 0-based option index, or -1 if no match."""
    if model_output is None:
        return -1
    text = model_output.strip()
    if text == "":
        return -1

    n = len(options)

    # (a) Explicit letter A/B/C/D. Accept bare "B", "(B)", "Answer: C",
    # "The answer is D". Prefer an explicit answer-prefix, then a leading
    # letter token, then any standalone letter in range.
    upper = text.upper()

    m = re.search(
        r"\bANSWER\s*(?:IS|:)?\s*\(?\s*([A-Z])\s*\)?", upper
    )
    if m:
        idx = ord(m.group(1)) - ord("A")
        if 0 <= idx < n:
            return idx

    m = re.match(r"^\s*\(?\s*([A-Za-z])\s*\)?[\.\):\s]*$", text)
    if m:
        idx = ord(m.group(1).upper()) - ord("A")
        if 0 <= idx < n:
            return idx

    m = re.match(r"^\s*\(?\s*([A-Za-z])\s*\)[\.:\s]", text)
    if m:
        idx = ord(m.group(1).upper()) - ord("A")
        if 0 <= idx < n:
            return idx

    # (b) Explicit option number 1..n.
    m = re.search(r"\b([0-9]+)\b", text)
    if m:
        num = int(m.group(1))
        if 1 <= num <= n:
            return num - 1

    # (c) Best normalized token-overlap with each option.
    out_tokens = Counter(_normalize_tokens(text))
    if out_tokens:
        best_idx = -1
        best_score = 0
        for i, opt in enumerate(options):
            opt_tokens = Counter(_normalize_tokens(opt))
            overlap = sum((out_tokens & opt_tokens).values())
            if overlap > best_score:
                best_score = overlap
                best_idx = i
        if best_score > 0:
            return best_idx

    return -1


def accuracy(predictions: List[int], golds: List[int]) -> float:
    """Fraction of predictions equal to gold. 0.0 if empty."""
    if len(predictions) == 0:
        return 0.0
    correct = sum(1 for p, g in zip(predictions, golds) if p == g)
    return correct / len(predictions)


def accuracy_on_subset(
    predictions: List[int], golds: List[int], mask: List[bool]
) -> float:
    """Accuracy over items where ``mask`` is True. 0.0 if the subset is empty."""
    sub_pred = [p for p, m in zip(predictions, mask) if m]
    sub_gold = [g for g, m in zip(golds, mask) if m]
    if len(sub_pred) == 0:
        return 0.0
    correct = sum(1 for p, g in zip(sub_pred, sub_gold) if p == g)
    return correct / len(sub_pred)
