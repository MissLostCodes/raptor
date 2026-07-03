"""Offline unit tests for the nested-oracle headroom decomposition.

Everything under test is stdlib-only (no SBERT / FAISS / sklearn / network). The
decomposition consumes *per-question* coverage "cells" and attributes the oracle
headroom to named components (document / query-type / interaction / within-doc)
via a single primitive :func:`oracle_by_key`.

Cell schema (one dict per ``(doc_id, qid, size)``)::

    {"doc_id": str, "qid": str, "size": int, "coverage": float, ...metadata}

``qid`` need only be unique *within* a document; identity is ``(doc_id, qid)``.
"""

from experiments.headroom_decomposition import (
    oracle_by_key,
    decompose,
    cells_from_records,
)


# Two docs, one question each, two sizes. All headroom is DOCUMENT-level:
#   A peaks at size 100, B peaks at size 200.
DOC_LEVEL = [
    {"doc_id": "A", "qid": "a1", "size": 100, "coverage": 0.9},
    {"doc_id": "A", "qid": "a1", "size": 200, "coverage": 0.1},
    {"doc_id": "B", "qid": "b1", "size": 100, "coverage": 0.1},
    {"doc_id": "B", "qid": "b1", "size": 200, "coverage": 0.9},
]


# One doc, two questions that DISAGREE on the best size (within-document headroom):
#   q1 peaks at 100, q2 peaks at 200. No feature of the document can serve both.
WITHIN_DOC = [
    {"doc_id": "A", "qid": "q1", "size": 100, "coverage": 0.8},
    {"doc_id": "A", "qid": "q1", "size": 200, "coverage": 0.2},
    {"doc_id": "A", "qid": "q2", "size": 100, "coverage": 0.2},
    {"doc_id": "A", "qid": "q2", "size": 200, "coverage": 0.8},
]


def test_oracle_by_key_constant_is_best_global_fixed():
    """A constant key = O0: one size for everything, scored over all questions.

    At size 100 the two questions average (0.9 + 0.1)/2 = 0.5; at 200 also 0.5.
    Tie -> smaller size (100). Coverage is the mean over questions at that size.
    """
    cov, size_by_key = oracle_by_key(DOC_LEVEL, lambda c: "all")
    assert cov == 0.5
    assert size_by_key == {"all": 100}  # tie broken toward the smaller size


def test_oracle_per_document_captures_doc_level_headroom():
    """O1 = per-document key: A->100, B->200, each question hits its peak."""
    cov, size_by_doc = oracle_by_key(DOC_LEVEL, lambda c: c["doc_id"])
    assert cov == 0.9
    assert size_by_doc == {"A": 100, "B": 200}


def test_per_document_oracle_cannot_beat_within_doc_disagreement():
    """O1 must pick ONE size for a doc whose questions disagree -> stuck at 0.5.

    mean at 100 = (0.8+0.2)/2 = 0.5; at 200 = 0.5; tie -> 100. q1=0.8, q2=0.2.
    """
    cov, size_by_doc = oracle_by_key(WITHIN_DOC, lambda c: c["doc_id"])
    assert cov == 0.5
    assert size_by_doc == {"A": 100}


def test_per_question_oracle_is_the_ceiling():
    """O4 = per-(doc,qid) key: each question hits its own peak -> 0.8 here.

    The gap O4 - O1 = 0.8 - 0.5 = 0.3 is within-document headroom no document
    feature can ever capture.
    """
    cov, _ = oracle_by_key(WITHIN_DOC, lambda c: (c["doc_id"], c["qid"]))
    assert cov == 0.8


# Planted "all headroom is QUERY-TYPE": across BOTH docs, lookup questions peak
# small (100) and aggregation questions peak large (200). A per-document oracle is
# helpless (each doc has one of each, so its questions disagree); a per-query-type
# oracle captures everything. This is the §4.4 estimator-validation dataset.
QUERY_TYPE = [
    {"doc_id": "A", "qid": "L", "qtype": "lookup", "size": 100, "coverage": 0.9},
    {"doc_id": "A", "qid": "L", "qtype": "lookup", "size": 200, "coverage": 0.1},
    {"doc_id": "A", "qid": "G", "qtype": "agg", "size": 100, "coverage": 0.1},
    {"doc_id": "A", "qid": "G", "qtype": "agg", "size": 200, "coverage": 0.9},
    {"doc_id": "B", "qid": "L", "qtype": "lookup", "size": 100, "coverage": 0.9},
    {"doc_id": "B", "qid": "L", "qtype": "lookup", "size": 200, "coverage": 0.1},
    {"doc_id": "B", "qid": "G", "qtype": "agg", "size": 100, "coverage": 0.1},
    {"doc_id": "B", "qid": "G", "qtype": "agg", "size": 200, "coverage": 0.9},
]


def test_decompose_recovers_planted_query_type_headroom():
    """The ledger attributes ~all headroom to the query-type component."""
    led = decompose(QUERY_TYPE, type_fn=lambda c: c["qtype"])
    assert led["O0_global"] == 0.5
    assert led["O1_document"] == 0.5
    assert led["O2_query_type"] == 0.9
    assert led["O4_per_question"] == 0.9
    comp = led["components"]
    assert comp["document"] == 0.0
    assert abs(comp["query_type"] - 0.4) < 1e-12
    assert abs(comp["interaction"] - 0.0) < 1e-12
    assert abs(comp["within_doc"] - 0.0) < 1e-12
    assert abs(comp["total"] - 0.4) < 1e-12


# Enriched sweep records (as emitted by an enriched sweep_document): each carries a
# ``per_question`` list of {qid, coverage, m, ...}. cells_from_records explodes them
# into the flat (doc_id, qid, size, coverage, +metadata) cells the oracles consume.
ENRICHED = [
    {
        "doc_id": "A", "size": 100, "n_questions": 2, "mean_evidence_coverage": 0.5,
        "per_question": [
            {"qid": "q1", "coverage": 0.8, "m": 1},
            {"qid": "q2", "coverage": 0.2, "m": 3},
        ],
    },
    {
        "doc_id": "A", "size": 200, "n_questions": 2, "mean_evidence_coverage": 0.5,
        "per_question": [
            {"qid": "q1", "coverage": 0.2, "m": 1},
            {"qid": "q2", "coverage": 0.8, "m": 3},
        ],
    },
]


def test_cells_from_records_explodes_per_question_with_metadata():
    cells = cells_from_records(ENRICHED)
    assert len(cells) == 4
    c = next(x for x in cells if x["qid"] == "q1" and x["size"] == 100)
    assert c["doc_id"] == "A"
    assert c["coverage"] == 0.8
    assert c["m"] == 1  # metadata carried through for query-type grouping


def test_cells_from_records_skips_records_without_per_question():
    """Legacy records (no per_question) and None-coverage cells are dropped."""
    legacy = [
        {"doc_id": "Z", "size": 100, "mean_evidence_coverage": None},  # unscored
        {"doc_id": "Z", "size": 100, "mean_evidence_coverage": 0.4},   # no per_question
    ]
    assert cells_from_records(legacy) == []


def test_cells_from_records_dedupes_resumed_records_last_wins():
    """A resumed/merged sweep can emit the same (doc, size) twice; the later record
    wins per (doc_id, qid, size), mirroring coverage_matrix's overwrite semantics."""
    dup = [
        {"doc_id": "A", "size": 100, "per_question": [{"qid": "q1", "coverage": 0.3}]},
        {"doc_id": "A", "size": 100, "per_question": [{"qid": "q1", "coverage": 0.9}]},
    ]
    cells = cells_from_records(dup)
    assert len(cells) == 1
    assert cells[0]["coverage"] == 0.9  # last occurrence wins


def test_cells_from_records_roundtrips_into_the_decomposition():
    """The exploded cells reproduce the WITHIN_DOC ledger end-to-end."""
    led = decompose(cells_from_records(ENRICHED), type_fn=lambda c: c["m"])
    assert led["O0_global"] == 0.5
    assert led["O4_per_question"] == 0.8


def test_decompose_recovers_planted_within_doc_headroom():
    """With no query-type signal, all headroom lands in the within-doc residual."""
    led = decompose(WITHIN_DOC, type_fn=lambda c: "const")
    assert led["O0_global"] == 0.5
    assert led["O1_document"] == 0.5
    assert led["O4_per_question"] == 0.8
    comp = led["components"]
    assert comp["document"] == 0.0
    assert comp["query_type"] == 0.0
    assert abs(comp["within_doc"] - 0.3) < 1e-12
    assert abs(comp["total"] - 0.3) < 1e-12
