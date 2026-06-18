"""Smoke + behavioral tests for the signal-recovery power analysis.

These guard the module's use of the calibration API and the central scientific
invariant it relies on: when a real signal is present, recovery improves with more
questions per document (the positive control). Deterministic via seeded RNG.
"""

import random

from experiments import signal_recovery_power as srp


def test_synth_dataset_shapes():
    rng = random.Random(0)
    records, feats, opt = srp.synth_dataset(n_docs=20, questions_per_doc=5, sigma_q=0.2, rng=rng)
    assert len(records) == 20 * len(srp.SWEPT_SIZES)
    assert set(feats) == {f"d{i}" for i in range(20)}
    # every doc has all six features and a defined optimal size on the swept grid
    for d, fv in feats.items():
        assert set(srp.FEATURE_NAMES) <= set(fv)
        assert opt[d] in srp.SWEPT_SIZES


def test_evaluate_returns_winrate_in_range():
    rng = random.Random(1)
    records, feats, opt = srp.synth_dataset(40, 10, 0.2, rng)
    mean_cov, std, wins, n = srp.evaluate(records, feats, opt, n_splits=10)
    assert 0 <= wins <= n <= 10
    assert 0.0 <= mean_cov <= 1.0
    assert std >= 0.0


def test_positive_control_recovery_improves_with_more_questions():
    # The core claim: with a planted signal, both the recovered correlation strength
    # and the win-rate vs fixed are larger at high questions/doc than at low.
    rows = srp.sweep_questions([3, 40], n_docs=50, sigma_q=0.25, n_datasets=8, n_splits=30)
    low, high = rows[0], rows[1]
    assert abs(high["recovered_spearman"]) > abs(low["recovered_spearman"])
    assert high["winrate"] > low["winrate"]
