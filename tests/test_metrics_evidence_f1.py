import math

from experiments.metrics import (
    evidence_f1,
    evidence_f1_max,
    gold_evidence_coverage,
)


def test_evidence_f1_perfect_overlap():
    assert evidence_f1(["p1", "p2"], ["p2", "p1"]) == 1.0


def test_evidence_f1_partial():
    # pred = {p1, p2, p3}, gold = {p2, p3, p4}
    # shared = {p2, p3} = 2; p = 2/3, r = 2/3 => F1 = 2/3
    assert math.isclose(
        evidence_f1(["p1", "p2", "p3"], ["p2", "p3", "p4"]), 2 / 3, rel_tol=1e-9
    )


def test_evidence_f1_uneven_partial():
    # pred = {p1, p2}, gold = {p1, p2, p3, p4}
    # shared = 2; p = 2/2 = 1.0, r = 2/4 = 0.5 => F1 = 2*1*0.5/1.5 = 2/3
    assert math.isclose(evidence_f1(["p1", "p2"], ["p1", "p2", "p3", "p4"]), 2 / 3, rel_tol=1e-9)


def test_evidence_f1_both_empty():
    assert evidence_f1([], []) == 1.0


def test_evidence_f1_one_empty_pred():
    assert evidence_f1([], ["p1"]) == 0.0


def test_evidence_f1_one_empty_gold():
    assert evidence_f1(["p1"], []) == 0.0


def test_evidence_f1_disjoint():
    assert evidence_f1(["p1"], ["p2"]) == 0.0


def test_evidence_f1_max_picks_best():
    pred = ["p1", "p2"]
    gold_sets = [["p3", "p4"], ["p1", "p2"]]
    assert evidence_f1_max(pred, gold_sets) == 1.0


def test_evidence_f1_max_partial_best():
    pred = ["p1", "p2", "p3"]
    gold_sets = [["x"], ["p1", "p2", "p3", "p4"]]
    # second: shared 3; p=3/3=1, r=3/4=0.75 => F1 = 2*1*0.75/1.75 = 6/7
    assert math.isclose(evidence_f1_max(pred, gold_sets), 6 / 7, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# gold_evidence_coverage: token-recall proxy for retrieval evidence.
#
# Paragraph-set evidence-F1 is not computable from a concatenated retrieved
# context (retrieved units are chunks, not gold paragraphs), so we measure the
# fraction of gold-evidence content tokens that the retrieved context surfaces.
# ---------------------------------------------------------------------------
def test_coverage_full_when_context_contains_all_gold_tokens():
    ctx = "The model uses a transformer encoder trained on QASPER papers."
    gold = ["transformer encoder", "QASPER papers"]
    assert gold_evidence_coverage(ctx, gold) == 1.0


def test_coverage_zero_when_no_gold_tokens_present():
    ctx = "completely unrelated narrative about a dog"
    gold = ["transformer encoder architecture"]
    assert gold_evidence_coverage(ctx, gold) == 0.0


def test_coverage_partial_is_token_recall():
    # gold content tokens (articles removed by normalize): {transformer, encoder,
    # gradient, descent} -> context covers transformer+encoder = 2/4 = 0.5
    ctx = "we describe the transformer encoder here"
    gold = ["transformer encoder gradient descent"]
    assert math.isclose(gold_evidence_coverage(ctx, gold), 0.5, rel_tol=1e-9)


def test_coverage_none_when_no_gold_evidence():
    # No annotated evidence -> nothing to recover -> None (omit, don't report 0).
    assert gold_evidence_coverage("any context", []) is None
    assert gold_evidence_coverage("any context", ["", "   "]) is None


def test_coverage_zero_when_context_empty_but_gold_present():
    assert gold_evidence_coverage("", ["transformer encoder"]) == 0.0


def test_coverage_is_normalization_robust():
    # Case, punctuation, and articles must not reduce coverage: uses the same
    # SQuAD normalize_answer as answer-F1 (lowercase, strip punctuation, drop
    # a/an/the). (Note: punctuation is deleted, so hyphens JOIN tokens -- a
    # deliberate consequence of matching the answer-F1 tokenizer.)
    ctx = "The TRANSFORMER encoder, trained well."
    gold = ["transformer encoder"]
    assert gold_evidence_coverage(ctx, gold) == 1.0
