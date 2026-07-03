"""Offline unit tests for the PURE metric functions in granularity_sweep.

Covers ``normalize``, ``paragraph_covered``, ``evidence_coverage`` and
``best_size_per_doc`` only — the SBERT/FAISS sweep driver is HEAVY and runs on
Colab, not here. These tests are fully deterministic (no model loads, no
network).
"""

import math

from types import SimpleNamespace

from experiments.granularity_sweep import (
    normalize,
    paragraph_covered,
    evidence_coverage,
    best_size_per_doc,
    question_meta,
)


# --------------------------------------------------------------------------- #
# normalize
# --------------------------------------------------------------------------- #
def test_normalize_lowercases_and_splits():
    assert normalize("Hello World FOO") == ["hello", "world", "foo"]


def test_normalize_strips_and_collapses_whitespace():
    assert normalize("  a\tb\n c  ") == ["a", "b", "c"]


def test_normalize_drops_punctuation_only_tokens():
    # "-" and "()" are punctuation-only and dropped; word-attached punct kept.
    assert normalize("the model - works ( )") == ["the", "model", "works"]


def test_normalize_empty():
    assert normalize("") == []
    assert normalize("   ") == []


# --------------------------------------------------------------------------- #
# paragraph_covered
# --------------------------------------------------------------------------- #
def test_paragraph_covered_exact_substring_present():
    gold = "the quick brown fox"
    context = "before text the quick brown fox after text"
    assert paragraph_covered(gold, context, threshold=0.8) is True


def test_paragraph_covered_absent():
    gold = "the quick brown fox"
    context = "completely unrelated sentence about turtles"
    assert paragraph_covered(gold, context, threshold=0.8) is False


def test_paragraph_covered_threshold_flip():
    # 5 gold tokens; context contains exactly 4 of them -> recall 0.8.
    gold = "alpha beta gamma delta epsilon"
    context = "alpha beta gamma delta omega"  # missing 'epsilon'
    # recall = 4/5 = 0.8
    assert paragraph_covered(gold, context, threshold=0.8) is True
    # Just above 0.8 must now fail (0.8 < 0.81).
    assert paragraph_covered(gold, context, threshold=0.81) is False


def test_paragraph_covered_partial_below_threshold():
    # Only 1 of 4 gold tokens present -> recall 0.25 < 0.8.
    gold = "one two three four"
    context = "one nope nope nope"
    assert paragraph_covered(gold, context, threshold=0.8) is False


def test_paragraph_covered_empty_gold_is_trivially_covered():
    assert paragraph_covered("", "anything here") is True


# --------------------------------------------------------------------------- #
# evidence_coverage
# --------------------------------------------------------------------------- #
def test_evidence_coverage_all_present():
    gold_paras = ["the quick brown fox", "jumps over"]
    context = "the quick brown fox jumps over the lazy dog"
    assert evidence_coverage(gold_paras, context, threshold=0.8) == 1.0


def test_evidence_coverage_none_present():
    gold_paras = ["alpha beta gamma", "delta epsilon zeta"]
    context = "totally different content with no overlap whatsoever"
    assert evidence_coverage(gold_paras, context, threshold=0.8) == 0.0


def test_evidence_coverage_fraction_covered():
    # First paragraph fully present, second absent -> 1/2 = 0.5.
    gold_paras = ["the quick brown fox", "nonexistent unrelated phrase tokens"]
    context = "intro the quick brown fox outro"
    assert math.isclose(
        evidence_coverage(gold_paras, context, threshold=0.8), 0.5, rel_tol=1e-9
    )


def test_evidence_coverage_empty_gold_returns_none():
    assert evidence_coverage([], "some context") is None
    # Whitespace-only / falsy paragraphs collapse to empty too.
    assert evidence_coverage(["", "   "], "some context") is None


# --------------------------------------------------------------------------- #
# best_size_per_doc
# --------------------------------------------------------------------------- #
def test_best_size_per_doc_picks_argmax():
    records = [
        {"doc_id": "A", "size": 50, "mean_evidence_coverage": 0.4},
        {"doc_id": "A", "size": 100, "mean_evidence_coverage": 0.9},
        {"doc_id": "A", "size": 200, "mean_evidence_coverage": 0.7},
        {"doc_id": "B", "size": 50, "mean_evidence_coverage": 0.8},
        {"doc_id": "B", "size": 100, "mean_evidence_coverage": 0.3},
    ]
    best = best_size_per_doc(records)
    assert best == {"A": 100, "B": 50}


def test_best_size_per_doc_tie_prefers_smaller_size():
    records = [
        {"doc_id": "A", "size": 100, "mean_evidence_coverage": 0.7},
        {"doc_id": "A", "size": 200, "mean_evidence_coverage": 0.7},
    ]
    assert best_size_per_doc(records) == {"A": 100}


def test_best_size_per_doc_ignores_none_coverage():
    records = [
        {"doc_id": "A", "size": 50, "mean_evidence_coverage": None},
        {"doc_id": "A", "size": 100, "mean_evidence_coverage": 0.6},
        {"doc_id": "B", "size": 50, "mean_evidence_coverage": None},
    ]
    best = best_size_per_doc(records)
    # B has only None records -> omitted entirely.
    assert best == {"A": 100}


# --------------------------------------------------------------------------- #
# question_meta — per-question identity + metadata for the headroom decomposition
# --------------------------------------------------------------------------- #
def test_question_meta_extracts_qid_and_evidence_multiplicity():
    q = SimpleNamespace(
        question_id="q7",
        evidence=["para one", "para two", "para three"],
    )
    meta = question_meta(q)
    assert meta["qid"] == "q7"
    assert meta["m"] == 3  # evidence multiplicity = number of gold paragraphs


def test_question_meta_zero_evidence_is_multiplicity_zero():
    q = SimpleNamespace(question_id="q0", evidence=[])
    assert question_meta(q)["m"] == 0


def test_question_meta_tolerates_missing_evidence_attr():
    """Answer-recall corpora (QuALITY) carry no evidence -> m defaults to 0."""
    q = SimpleNamespace(question_id="qX")
    assert question_meta(q) == {"qid": "qX", "m": 0}
