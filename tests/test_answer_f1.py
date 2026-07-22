"""Self-check for the downstream-QA answer-F1 scorer (offline, no SBERT/network).

Guards the new QA path in granularity_sweep: token-F1 semantics, max-over-golds,
None-on-no-gold (so unscored questions are skipped like the coverage proxies), and
that a fake qa_model actually flows through sweep_document into the records.

Run: python -m pytest tests/test_answer_f1.py   (or: python tests/test_answer_f1.py)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.granularity_sweep import _token_f1, answer_f1, sweep_document


def test_token_f1_basics():
    assert _token_f1("the cat sat", "the cat sat") == 1.0          # exact
    assert _token_f1("", "") == 1.0                                # empty vs empty
    assert _token_f1("cat", "") == 0.0                             # empty gold
    assert _token_f1("dog", "cat") == 0.0                          # no overlap
    # half precision (2/4 tokens right), full recall (2/2) -> F1 = 2*.5*1/1.5
    f1 = _token_f1("the cat on mat", "the cat")
    assert abs(f1 - (2 * 0.5 * 1.0 / 1.5)) < 1e-9
    # punctuation-only tokens are dropped by normalize -> ignored
    assert _token_f1("yes .", "yes") == 1.0


def test_answer_f1_max_over_golds_and_none():
    assert answer_f1("blue", ["red", "blue", "green"]) == 1.0      # best gold wins
    assert answer_f1("anything", []) is None                      # no gold -> skip
    assert answer_f1("anything", ["", "  "]) is None              # blank golds -> skip


class _FakeQA:
    """Echoes a fixed answer; proves sweep_document routes ctx->answer->score."""

    def __init__(self, answer):
        self._a = answer

    def answer_question(self, context, question, *a, **k):
        return self._a


class _Q:
    def __init__(self, question, gold_answers, evidence):
        self.question = question
        self.gold_answers = gold_answers
        self.evidence = evidence
        self.question_id = question


class _Doc:
    doc_id = "d0"

    def __init__(self, questions):
        self.questions = questions
        self.text = "the cat sat on the mat. " * 50


def test_sweep_document_threads_qa_model(monkeypatch):
    import experiments.granularity_sweep as gs

    # Stub the retriever so the test stays offline (no SBERT/FAISS): context is the
    # gold answer verbatim, so a perfect QA model must score answer_f1 == 1.0.
    monkeypatch.setattr(
        gs, "build_flat_retriever",
        lambda chunks, budget, emb=None: type("R", (), {"retrieve": staticmethod(lambda q: "blue")})(),
    )
    doc = _Doc([_Q("color?", ["blue"], ["the sky is blue"])])
    recs = sweep_document(doc, [100], qa_model=_FakeQA("blue"))
    assert recs[0]["mean_answer_f1"] == 1.0
    assert recs[0]["per_question"][0]["answer_f1"] == 1.0

    # Without a qa_model the record must NOT carry answer fields (back-compat).
    recs2 = sweep_document(doc, [100])
    assert "mean_answer_f1" not in recs2[0]
    assert "answer_f1" not in recs2[0]["per_question"][0]


if __name__ == "__main__":  # allow running without pytest
    test_token_f1_basics()
    test_answer_f1_max_over_golds_and_none()
    print("pure-metric self-checks passed (run pytest for the sweep_document test)")
