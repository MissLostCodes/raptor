"""QASPER evidence-F1 (set-based over evidence paragraphs).

Reference: Dasigi et al. (2021), QASPER. Evidence selection is scored as a
set-overlap F1 between predicted and gold evidence paragraphs, with the
convention that two empty sets match (1.0) and a single empty set scores 0.0.
"""

from typing import List


def evidence_f1(predicted: List[str], gold: List[str]) -> float:
    """Set-based P/R/F1 over evidence paragraph strings."""
    pred_set = set(predicted)
    gold_set = set(gold)

    if len(pred_set) == 0 and len(gold_set) == 0:
        return 1.0
    if len(pred_set) == 0 or len(gold_set) == 0:
        return 0.0

    overlap = len(pred_set & gold_set)
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_set)
    recall = overlap / len(gold_set)
    return (2 * precision * recall) / (precision + recall)


def evidence_f1_max(predicted: List[str], gold_sets: List[List[str]]) -> float:
    """Max evidence-F1 over multiple candidate gold evidence sets."""
    if len(gold_sets) == 0:
        # No annotated evidence sets: treat as a single empty gold set.
        return evidence_f1(predicted, [])
    return max(evidence_f1(predicted, gold) for gold in gold_sets)
