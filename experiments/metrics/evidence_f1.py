"""QASPER evidence-F1 (set-based over evidence paragraphs).

Reference: Dasigi et al. (2021), QASPER. Evidence selection is scored as a
set-overlap F1 between predicted and gold evidence paragraphs, with the
convention that two empty sets match (1.0) and a single empty set scores 0.0.
"""

from typing import List, Optional

from experiments.metrics.qa_f1 import normalize_answer


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


def gold_evidence_coverage(
    retrieved_context: str, gold_evidence: List[str]
) -> Optional[float]:
    """Token-recall proxy for retrieval evidence.

    Paragraph-set :func:`evidence_f1` requires discrete retrieved evidence
    paragraphs that string-match the gold paragraphs. A hierarchical-RAG
    answerer instead returns a single concatenated *context* built from chunks,
    so set-overlap is inapplicable. Instead we ask: what fraction of the
    (normalized, deduplicated) gold-evidence content tokens does the retrieved
    context surface? This is the gold-evidence token-coverage proxy used in the
    leaf-granularity study, applied here end-to-end.

    Returns ``None`` when there is no gold evidence to recover (so callers omit
    rather than report a misleading 0); ``0.0`` when gold evidence exists but
    none of its tokens appear in the context; otherwise recall in ``(0, 1]``.
    """
    gold_tokens = set()
    for paragraph in gold_evidence:
        gold_tokens.update(normalize_answer(paragraph).split())
    if not gold_tokens:
        return None
    context_tokens = set(normalize_answer(retrieved_context).split())
    covered = len(gold_tokens & context_tokens)
    return covered / len(gold_tokens)
