"""Tests for Phase-0 target diagnostics (E1/E2/E3) in granularity_calibration.

These pure helpers measure *what optimal leaf size actually responds to* before
we guess at predictive features:

* E1 ``coverage_curve_shape`` -- per-doc oracle curve peak/flatness (the
  flat-curve confound: size-insensitive docs dilute any correlation).
* E2 ``gold_evidence_span_lengths`` -- mean tokens per gold-evidence paragraph,
  the hypothesized latent driver of optimal size.
* E3 ``evidence_dispersion`` -- how scattered a question's gold spans are.

All stdlib-only and deterministic.
"""

import math

from experiments.granularity_calibration import (
    coverage_curve_shape,
    gold_evidence_span_lengths,
    evidence_dispersion,
    rank_corr,
)


# A doc with a sharp peak and a doc with a flat curve.
RECORDS = [
    {"doc_id": "peak", "size": 50, "n_questions": 3, "mean_evidence_coverage": 0.50},
    {"doc_id": "peak", "size": 100, "n_questions": 3, "mean_evidence_coverage": 0.80},
    {"doc_id": "peak", "size": 200, "n_questions": 3, "mean_evidence_coverage": 0.60},
    {"doc_id": "flat", "size": 50, "n_questions": 3, "mean_evidence_coverage": 0.70},
    {"doc_id": "flat", "size": 100, "n_questions": 3, "mean_evidence_coverage": 0.71},
    {"doc_id": "flat", "size": 200, "n_questions": 3, "mean_evidence_coverage": 0.70},
    # A None record must be ignored.
    {"doc_id": "flat", "size": 400, "n_questions": 0, "mean_evidence_coverage": None},
]


# ---------------------------------------------------------------------------
# E1 -- coverage curve shape
# ---------------------------------------------------------------------------
def test_curve_shape_peak_and_flatness():
    shapes = coverage_curve_shape(RECORDS, flat_eps=0.05)
    peak = shapes["peak"]
    assert peak["peak_size"] == 100
    assert peak["peak_cov"] == 0.80
    assert math.isclose(peak["flatness"], 0.30)
    assert peak["is_flat"] is False


def test_curve_shape_flat_doc_flagged():
    shapes = coverage_curve_shape(RECORDS, flat_eps=0.05)
    flat = shapes["flat"]
    assert math.isclose(flat["flatness"], 0.01)  # 0.71 - 0.70
    assert flat["is_flat"] is True


def test_curve_shape_ignores_none_records():
    shapes = coverage_curve_shape(RECORDS)
    # The None record at size 400 must not appear as the peak.
    assert shapes["flat"]["peak_size"] in (50, 100, 200)


def test_curve_shape_empty():
    assert coverage_curve_shape([]) == {}


# ---------------------------------------------------------------------------
# E2 -- gold-evidence span lengths
# ---------------------------------------------------------------------------
def test_gold_span_lengths_mean_tokens():
    evidence = {
        "d1": ["a b c", "d e"],        # 3 and 2 tokens -> mean 2.5
        "d2": ["w x y z"],             # 4 tokens
        "d3": ["", "  "],              # all empty -> 0.0
    }
    out = gold_evidence_span_lengths(evidence)
    assert math.isclose(out["d1"], 2.5)
    assert math.isclose(out["d2"], 4.0)
    assert out["d3"] == 0.0


def test_gold_span_lengths_empty_doc_omitted_or_zero():
    out = gold_evidence_span_lengths({"d1": []})
    assert out["d1"] == 0.0


def test_gold_span_lengths_feeds_rank_corr():
    # Sanity: E2 output is a {doc: float} map usable with the existing rank_corr.
    spans = gold_evidence_span_lengths({"a": ["x x"], "b": ["y y y y"]})
    opt = {"a": 50, "b": 200}
    keys = sorted(set(spans) & set(opt))
    r = rank_corr([spans[k] for k in keys], [opt[k] for k in keys])
    assert r == 1.0  # longer spans -> larger optimal size, monotone


# ---------------------------------------------------------------------------
# E3 -- evidence dispersion
# ---------------------------------------------------------------------------
def test_evidence_dispersion_mean_span():
    # Per-question gold paragraph indices within the doc.
    indices = {
        "d1": [[0, 1], [5, 9]],   # spans of 1 and 4 -> mean 2.5
        "d2": [[3]],              # single index -> dispersion 0
    }
    out = evidence_dispersion(indices)
    assert math.isclose(out["d1"], 2.5)
    assert out["d2"] == 0.0


def test_evidence_dispersion_empty():
    assert evidence_dispersion({}) == {}
    assert evidence_dispersion({"d1": []})["d1"] == 0.0
