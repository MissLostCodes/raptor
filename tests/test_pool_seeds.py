"""Offline tests for across-seed pooling (mean +/- std over >=3 seeds)."""

import json
import statistics

from pytest import approx

from experiments.pool_seeds import (
    pool_seed_aggregates,
    pool_results,
    load_per_seed_records,
)


def test_pool_seed_aggregates_mean_std_over_seeds():
    per_seed_agg = {
        0: {"token": {"qasper": {"answer_f1": {"mean": 0.6}}}},
        1: {"token": {"qasper": {"answer_f1": {"mean": 0.8}}}},
        2: {"token": {"qasper": {"answer_f1": {"mean": 0.7}}}},
    }
    pooled = pool_seed_aggregates(per_seed_agg)
    cell = pooled["token"]["qasper"]["answer_f1"]
    assert cell["n_seeds"] == 3
    assert cell["mean"] == approx(0.7)
    assert cell["std"] == approx(statistics.stdev([0.6, 0.8, 0.7]))
    assert cell["seed_means"] == [0.6, 0.8, 0.7]  # ordered by seed
    assert cell["seeds"] == [0, 1, 2]


def test_pool_seed_aggregates_skips_metric_missing_in_a_seed():
    per_seed_agg = {
        0: {"token": {"qasper": {"answer_f1": {"mean": 0.6}}}},
        1: {"token": {"qasper": {}}},  # this seed lacks the metric
    }
    pooled = pool_seed_aggregates(per_seed_agg)
    cell = pooled["token"]["qasper"]["answer_f1"]
    assert cell["n_seeds"] == 1
    assert cell["mean"] == approx(0.6)
    assert cell["std"] == 0.0  # single seed -> 0 std (matches report._std)


def test_pool_seed_aggregates_skips_nan_means():
    nan = float("nan")
    per_seed_agg = {
        0: {"token": {"qasper": {"answer_f1": {"mean": nan}}}},
        1: {"token": {"qasper": {"answer_f1": {"mean": 0.5}}}},
    }
    pooled = pool_seed_aggregates(per_seed_agg)
    cell = pooled["token"]["qasper"]["answer_f1"]
    assert cell["n_seeds"] == 1
    assert cell["mean"] == approx(0.5)


def test_pool_results_from_raw_per_seed_records():
    per_seed_records = {
        0: [{"arm": "token", "dataset": "qasper", "doc_id": "d", "question_id": "q",
             "answer_f1": 0.4, "blocked": False, "error": None}],
        1: [{"arm": "token", "dataset": "qasper", "doc_id": "d", "question_id": "q",
             "answer_f1": 0.6, "blocked": False, "error": None}],
    }
    pooled = pool_results(per_seed_records)
    cell = pooled["token"]["qasper"]["answer_f1"]
    assert cell["n_seeds"] == 2
    assert cell["mean"] == approx(0.5)


def test_load_per_seed_records_reads_existing_only(tmp_path):
    (tmp_path / "results_0.json").write_text(
        json.dumps({"config": {}, "records": [{"arm": "token"}]})
    )
    # seed 1 file intentionally absent.
    out = load_per_seed_records(str(tmp_path), seeds=[0, 1])
    assert set(out.keys()) == {0}
    assert out[0] == [{"arm": "token"}]
