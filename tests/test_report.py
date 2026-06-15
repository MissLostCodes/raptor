"""Offline tests for experiments.report aggregation + rendering."""

from pytest import approx

from experiments import report


def test_aggregate_includes_evidence_coverage():
    records = [
        {"arm": "token", "dataset": "qasper", "doc_id": "p1", "question_id": "q1",
         "answer_f1": 0.5, "evidence_coverage": 0.8, "blocked": False, "error": None},
        {"arm": "token", "dataset": "qasper", "doc_id": "p1", "question_id": "q2",
         "answer_f1": 0.5, "evidence_coverage": 0.6, "blocked": False, "error": None},
        # A None coverage (no gold evidence) must be skipped, not crash.
        {"arm": "token", "dataset": "qasper", "doc_id": "p2", "question_id": "q3",
         "answer_f1": 1.0, "evidence_coverage": None, "blocked": False, "error": None},
    ]
    agg = report.aggregate(records)
    cov = agg["token"]["qasper"]["evidence_coverage"]
    assert cov["n"] == 2
    assert cov["mean"] == approx(0.7)


def test_bootstrap_ci_deterministic_and_bracketing():
    values = [0.0, 0.5, 1.0, 0.25, 0.75]
    lo1, hi1 = report.bootstrap_ci(values, seed=0)
    lo2, hi2 = report.bootstrap_ci(values, seed=0)
    assert (lo1, hi1) == (lo2, hi2)  # deterministic
    mean = sum(values) / len(values)
    assert lo1 <= mean <= hi1
    # different seed may differ but still brackets reasonably.
    lo3, hi3 = report.bootstrap_ci(values, seed=1)
    assert lo3 <= mean <= hi3


def test_bootstrap_ci_single_value():
    assert report.bootstrap_ci([0.7]) == (0.7, 0.7)


def test_aggregate_mean_std_n():
    records = [
        {"arm": "token", "dataset": "qasper", "answer_f1": 1.0},
        {"arm": "token", "dataset": "qasper", "answer_f1": 0.0},
        {"arm": "structure", "dataset": "qasper", "answer_f1": 0.5},
    ]
    agg = report.aggregate(records)
    tok = agg["token"]["qasper"]["answer_f1"]
    assert tok["n"] == 2
    assert tok["mean"] == 0.5
    assert agg["structure"]["qasper"]["answer_f1"]["n"] == 1
    assert agg["structure"]["qasper"]["answer_f1"]["mean"] == 0.5


def test_aggregate_quality_accuracy_and_hard_subset():
    records = [
        {"arm": "token", "dataset": "quality", "correct": 1, "is_hard": True},
        {"arm": "token", "dataset": "quality", "correct": 0, "is_hard": True},
        {"arm": "token", "dataset": "quality", "correct": 1, "is_hard": False},
        {"arm": "token", "dataset": "quality", "correct": 1, "is_hard": False},
    ]
    agg = report.aggregate(records)
    q = agg["token"]["quality"]
    assert q["accuracy"]["n"] == 4
    assert q["accuracy"]["mean"] == 0.75
    # hard subset: 1 of 2 correct.
    assert q["accuracy_hard"]["n"] == 2
    assert q["accuracy_hard"]["mean"] == 0.5


def test_aggregate_narrativeqa_metrics():
    records = [
        {"arm": "flat", "dataset": "narrativeqa", "rouge_l": 0.8, "bleu_1": 0.6,
         "bleu_4": 0.2, "meteor": 0.5},
        {"arm": "flat", "dataset": "narrativeqa", "rouge_l": 0.4, "bleu_1": 0.4,
         "bleu_4": 0.0, "meteor": None},
    ]
    agg = report.aggregate(records)
    nqa = agg["flat"]["narrativeqa"]
    assert nqa["rouge_l"]["n"] == 2
    assert nqa["rouge_l"]["mean"] == approx(0.6)
    # meteor None is dropped.
    assert nqa["meteor"]["n"] == 1
    assert nqa["meteor"]["mean"] == 0.5


def test_routing_rate():
    records = [
        # ahc doc d1 -> structure (2 questions, same doc counted once)
        {"arm": "ahc", "dataset": "qasper", "doc_id": "d1", "routing": {"route": "structure"}},
        {"arm": "ahc", "dataset": "qasper", "doc_id": "d1", "routing": {"route": "structure"}},
        # ahc doc d2 -> token
        {"arm": "ahc", "dataset": "qasper", "doc_id": "d2", "routing": {"route": "token"}},
        # non-ahc ignored
        {"arm": "token", "dataset": "qasper", "doc_id": "d3", "routing": {}},
    ]
    rates = report.routing_rate(records)
    assert rates["qasper"]["n_docs"] == 2
    assert rates["qasper"]["structure_rate"] == 0.5


def test_to_markdown_nonempty_with_names():
    records = [
        {"arm": "token", "dataset": "qasper", "answer_f1": 1.0},
        {"arm": "ahc", "dataset": "qasper", "answer_f1": 0.5,
         "doc_id": "d1", "routing": {"route": "structure"}},
    ]
    agg = report.aggregate(records)
    routing = report.routing_rate(records)
    md = report.to_markdown(agg, routing)
    assert isinstance(md, str) and md.strip()
    assert "token" in md
    assert "qasper" in md
    assert "structure_rate" in md
