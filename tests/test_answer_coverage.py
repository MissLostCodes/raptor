"""Tests for the answer-recall coverage proxy (Phase 6, second corpus).

QASPER provides gold-evidence paragraphs, but QuALITY (MC) and NarrativeQA
(free-form) do not. To generalize the no-LLM leaf-size sweep to a second corpus
we score retrieval by *answer recall*: the best token-recall of any gold answer
string within the retrieved context. These are the pure helpers behind that;
the heavy SBERT sweep is unchanged and not exercised here.
"""

import math

from experiments.granularity_sweep import (
    answer_coverage,
    paragraph_covered,
    _token_recall,
    _answer_coverage_q,
    _evidence_coverage_q,
)
from experiments.datasets.base import QAExample


# ---------------------------------------------------------------------------
# _token_recall (shared core)
# ---------------------------------------------------------------------------
def test_token_recall_full_and_partial():
    assert _token_recall("alpha beta", "alpha beta gamma") == 1.0
    assert math.isclose(_token_recall("alpha beta", "alpha only here"), 0.5)


def test_token_recall_empty_gold_is_one():
    assert _token_recall("", "anything") == 1.0


# paragraph_covered must keep its exact behavior after refactor to _token_recall.
def test_paragraph_covered_still_works():
    assert paragraph_covered("alpha beta", "alpha beta gamma", threshold=0.8) is True
    assert paragraph_covered("alpha beta", "alpha only", threshold=0.8) is False


# ---------------------------------------------------------------------------
# answer_coverage
# ---------------------------------------------------------------------------
def test_answer_coverage_takes_best_reference():
    # Two references; the context fully contains the second -> recall 1.0.
    cov = answer_coverage(["totally absent words", "the answer is here"],
                          "and the answer is here indeed")
    assert cov == 1.0


def test_answer_coverage_partial():
    cov = answer_coverage(["alpha beta"], "alpha only")
    assert math.isclose(cov, 0.5)


def test_answer_coverage_none_when_no_usable_answer():
    assert answer_coverage([], "ctx") is None
    assert answer_coverage(["", "   "], "ctx") is None


# ---------------------------------------------------------------------------
# per-question coverage_fn adapters (what sweep_document calls)
# ---------------------------------------------------------------------------
def test_answer_coverage_q_reads_gold_answers():
    q = QAExample(question_id="q", question="?", gold_answers=["alpha beta"])
    assert math.isclose(_answer_coverage_q(q, "alpha beta gamma", 0.8), 1.0)
    q2 = QAExample(question_id="q", question="?", gold_answers=[])
    assert _answer_coverage_q(q2, "ctx", 0.8) is None


def test_evidence_coverage_q_default_path_unchanged():
    q = QAExample(question_id="q", question="?", evidence=["alpha beta"])
    assert _evidence_coverage_q(q, "alpha beta gamma", 0.8) == 1.0
    q2 = QAExample(question_id="q", question="?", evidence=[])
    assert _evidence_coverage_q(q2, "ctx", 0.8) is None
