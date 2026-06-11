"""Offline unit tests for the PURE functions in granularity_calibration.

Everything under test is stdlib-only (no SBERT / FAISS / sklearn / network), so
these run instantly and deterministically. The learned sklearn regressor and the
SBERT sweep live in the notebook / sweep driver and are not exercised here.
"""

import math

from experiments.granularity_calibration import (
    coverage_matrix,
    mean_coverage_at_size,
    best_global_fixed,
    oracle_per_doc,
    headroom,
    coverage_under_sizes,
    rank_corr,
    fit_rho_to_size,
    predict_size,
    train_test_split_docs,
    feature_target_correlations,
    select_and_fit,
)


# A small synthetic sweep where the optimum is genuinely doc-dependent:
#   A peaks at 100, B peaks at 200, C peaks at 50.
SYNTH = [
    {"doc_id": "A", "size": 50, "n_questions": 3, "mean_evidence_coverage": 0.40},
    {"doc_id": "A", "size": 100, "n_questions": 3, "mean_evidence_coverage": 0.90},
    {"doc_id": "A", "size": 200, "n_questions": 3, "mean_evidence_coverage": 0.70},
    {"doc_id": "B", "size": 50, "n_questions": 2, "mean_evidence_coverage": 0.30},
    {"doc_id": "B", "size": 100, "n_questions": 2, "mean_evidence_coverage": 0.50},
    {"doc_id": "B", "size": 200, "n_questions": 2, "mean_evidence_coverage": 0.80},
    {"doc_id": "C", "size": 50, "n_questions": 4, "mean_evidence_coverage": 0.85},
    {"doc_id": "C", "size": 100, "n_questions": 4, "mean_evidence_coverage": 0.60},
    {"doc_id": "C", "size": 200, "n_questions": 4, "mean_evidence_coverage": 0.20},
]


# --------------------------------------------------------------------------- #
# coverage_matrix
# --------------------------------------------------------------------------- #
def test_coverage_matrix_shape_and_values():
    m = coverage_matrix(SYNTH)
    assert set(m) == {"A", "B", "C"}
    assert set(m["A"]) == {50, 100, 200}
    assert m["A"][100] == 0.90
    assert m["B"][200] == 0.80
    assert m["C"][50] == 0.85


def test_coverage_matrix_drops_none_coverage():
    recs = [
        {"doc_id": "A", "size": 50, "mean_evidence_coverage": None},
        {"doc_id": "A", "size": 100, "mean_evidence_coverage": 0.6},
        {"doc_id": "B", "size": 50, "mean_evidence_coverage": None},
    ]
    m = coverage_matrix(recs)
    assert m == {"A": {100: 0.6}}  # B fully dropped, A's None size dropped


def test_coverage_matrix_last_write_wins_on_duplicate():
    recs = [
        {"doc_id": "A", "size": 50, "mean_evidence_coverage": 0.1},
        {"doc_id": "A", "size": 50, "mean_evidence_coverage": 0.9},
    ]
    assert coverage_matrix(recs) == {"A": {50: 0.9}}


# --------------------------------------------------------------------------- #
# mean_coverage_at_size
# --------------------------------------------------------------------------- #
def test_mean_coverage_at_size():
    # size 50: (0.40 + 0.30 + 0.85) / 3
    assert math.isclose(mean_coverage_at_size(SYNTH, 50), (0.40 + 0.30 + 0.85) / 3)
    # size 200: (0.70 + 0.80 + 0.20) / 3
    assert math.isclose(mean_coverage_at_size(SYNTH, 200), (0.70 + 0.80 + 0.20) / 3)


def test_mean_coverage_at_size_only_counts_docs_with_that_size():
    recs = [
        {"doc_id": "A", "size": 50, "mean_evidence_coverage": 0.4},
        {"doc_id": "A", "size": 100, "mean_evidence_coverage": 0.8},
        {"doc_id": "B", "size": 50, "mean_evidence_coverage": 0.6},
        # B has no size-100 record -> only A contributes at 100.
    ]
    assert math.isclose(mean_coverage_at_size(recs, 100), 0.8)
    assert math.isclose(mean_coverage_at_size(recs, 50), 0.5)


def test_mean_coverage_at_size_absent_size_is_zero():
    assert mean_coverage_at_size(SYNTH, 999) == 0.0


# --------------------------------------------------------------------------- #
# best_global_fixed
# --------------------------------------------------------------------------- #
def test_best_global_fixed_picks_right_size():
    # Means: 50 -> 0.5167, 100 -> 0.6667, 200 -> 0.5667. Best is 100.
    size, cov = best_global_fixed(SYNTH)
    assert size == 100
    assert math.isclose(cov, (0.90 + 0.50 + 0.60) / 3)


def test_best_global_fixed_tie_prefers_smaller_size():
    recs = [
        {"doc_id": "A", "size": 100, "mean_evidence_coverage": 0.7},
        {"doc_id": "A", "size": 200, "mean_evidence_coverage": 0.7},
    ]
    size, cov = best_global_fixed(recs)
    assert size == 100
    assert math.isclose(cov, 0.7)


def test_best_global_fixed_empty():
    assert best_global_fixed([]) == (0, 0.0)


# --------------------------------------------------------------------------- #
# oracle_per_doc
# --------------------------------------------------------------------------- #
def test_oracle_per_doc_argmax_and_mean():
    mean_cov, best = oracle_per_doc(SYNTH)
    assert best == {"A": 100, "B": 200, "C": 50}
    # mean of per-doc bests: (0.90 + 0.80 + 0.85) / 3
    assert math.isclose(mean_cov, (0.90 + 0.80 + 0.85) / 3)


def test_oracle_per_doc_ge_best_global_fixed_always():
    # The oracle can always reproduce any single fixed size, so it dominates.
    _, best_fixed_cov = best_global_fixed(SYNTH)
    oracle_cov, _ = oracle_per_doc(SYNTH)
    assert oracle_cov >= best_fixed_cov

    # Also holds on a degenerate sweep where every doc peaks at the same size.
    flat = [
        {"doc_id": "A", "size": 50, "mean_evidence_coverage": 0.5},
        {"doc_id": "A", "size": 100, "mean_evidence_coverage": 0.9},
        {"doc_id": "B", "size": 50, "mean_evidence_coverage": 0.4},
        {"doc_id": "B", "size": 100, "mean_evidence_coverage": 0.8},
    ]
    _, bf = best_global_fixed(flat)
    oc, _ = oracle_per_doc(flat)
    assert oc >= bf
    # Here the oracle adds nothing (shared peak), so they are equal.
    assert math.isclose(oc, bf)


# --------------------------------------------------------------------------- #
# headroom
# --------------------------------------------------------------------------- #
def test_headroom_fields_and_nonnegative_gap():
    h = headroom(SYNTH)
    assert h["best_fixed_size"] == 100
    assert math.isclose(h["best_fixed_cov"], (0.90 + 0.50 + 0.60) / 3)
    assert math.isclose(h["oracle_cov"], (0.90 + 0.80 + 0.85) / 3)
    # abs_gap == oracle - best_fixed, and is non-negative.
    assert math.isclose(h["abs_gap"], h["oracle_cov"] - h["best_fixed_cov"])
    assert h["abs_gap"] >= 0.0
    # rel_gain_pct consistency.
    assert math.isclose(h["rel_gain_pct"], h["abs_gap"] / h["best_fixed_cov"] * 100.0)


def test_headroom_empty_is_safe():
    h = headroom([])
    assert h["abs_gap"] == 0.0
    assert h["rel_gain_pct"] == 0.0


# --------------------------------------------------------------------------- #
# coverage_under_sizes (exact + snapped)
# --------------------------------------------------------------------------- #
def test_coverage_under_sizes_exact_sizes():
    # Use each doc's exact oracle size -> equals the oracle mean coverage.
    _, oracle_sizes = oracle_per_doc(SYNTH)
    cov = coverage_under_sizes(SYNTH, oracle_sizes)
    assert math.isclose(cov, (0.90 + 0.80 + 0.85) / 3)


def test_coverage_under_sizes_snaps_to_nearest_available():
    # Requested sizes are NOT on the swept grid {50,100,200}; they must snap.
    #   A: 90  -> nearest 100 -> 0.90
    #   B: 175 -> nearest 200 -> 0.80
    #   C: 60  -> nearest 50  -> 0.85
    requested = {"A": 90, "B": 175, "C": 60}
    cov = coverage_under_sizes(SYNTH, requested)
    assert math.isclose(cov, (0.90 + 0.80 + 0.85) / 3)


def test_coverage_under_sizes_snap_ties_prefer_smaller():
    # 75 is equidistant from 50 and 100; tie -> snap to the smaller (50 -> 0.40).
    cov = coverage_under_sizes(SYNTH, {"A": 75})
    assert math.isclose(cov, 0.40)


def test_coverage_under_sizes_ignores_unknown_docs():
    # Doc 'Z' is not in the records and must not contribute.
    cov = coverage_under_sizes(SYNTH, {"A": 100, "Z": 100})
    assert math.isclose(cov, 0.90)


# --------------------------------------------------------------------------- #
# rank_corr
# --------------------------------------------------------------------------- #
def test_rank_corr_monotone_increasing_is_plus_one():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    ys = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert math.isclose(rank_corr(xs, ys), 1.0, abs_tol=1e-9)
    # Non-linear but still monotone-increasing -> still +1 (rank-based).
    ys2 = [1.0, 4.0, 9.0, 16.0, 25.0]
    assert math.isclose(rank_corr(xs, ys2), 1.0, abs_tol=1e-9)


def test_rank_corr_monotone_decreasing_is_minus_one():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    ys = [50.0, 40.0, 30.0, 20.0, 10.0]
    assert math.isclose(rank_corr(xs, ys), -1.0, abs_tol=1e-9)


def test_rank_corr_flat_is_near_zero():
    xs = [1.0, 2.0, 3.0, 4.0]
    ys = [7.0, 7.0, 7.0, 7.0]  # constant -> no ranking info -> 0.0
    assert math.isclose(rank_corr(xs, ys), 0.0, abs_tol=1e-9)


def test_rank_corr_no_association_small_magnitude():
    # A symmetric "tent" permutation: ys rises then falls, so the monotone
    # association nets to exactly zero.
    xs = [1.0, 2.0, 3.0, 4.0]
    ys = [1.0, 2.0, 2.0, 1.0]
    assert math.isclose(rank_corr(xs, ys), 0.0, abs_tol=1e-9)


def test_rank_corr_length_mismatch_raises():
    try:
        rank_corr([1.0, 2.0], [1.0])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on length mismatch")


# --------------------------------------------------------------------------- #
# fit_rho_to_size + predict_size
# --------------------------------------------------------------------------- #
def test_fit_recovers_positive_slope_on_clean_linear():
    # size = 100 + 200 * rho exactly -> b should be ~200, invert False.
    rho = {f"d{i}": i / 10.0 for i in range(11)}  # 0.0 .. 1.0
    opt = {k: int(round(100 + 200 * v)) for k, v in rho.items()}
    fit = fit_rho_to_size(rho, opt)
    assert fit["b"] > 0
    assert fit["invert"] is False
    assert math.isclose(fit["b"], 200.0, abs_tol=2.0)
    assert math.isclose(fit["a"], 100.0, abs_tol=2.0)
    assert fit["n"] == 11


def test_fit_recovers_negative_slope_sign():
    # Denser -> smaller leaves: size = 400 - 300 * rho.
    rho = {f"d{i}": i / 10.0 for i in range(11)}
    opt = {k: int(round(400 - 300 * v)) for k, v in rho.items()}
    fit = fit_rho_to_size(rho, opt)
    assert fit["b"] < 0
    assert fit["invert"] is True


def test_fit_constant_fallback_on_zero_variance():
    # All docs share one rho -> slope unidentifiable -> constant mean target.
    rho = {"a": 0.5, "b": 0.5, "c": 0.5}
    opt = {"a": 100, "b": 200, "c": 300}
    fit = fit_rho_to_size(rho, opt)
    assert fit["b"] == 0.0
    assert math.isclose(fit["a"], 200.0)  # mean of targets


def test_predict_size_clips_and_rounds():
    fit = {"a": 100.0, "b": 200.0}
    # rho=0.5 -> 200 (within bounds)
    assert predict_size(0.5, fit, l_min=50, l_max=400) == 200
    # rho large -> clipped to l_max
    assert predict_size(5.0, fit, l_min=50, l_max=400) == 400
    # rho negative-ish (a+b*rho below l_min) -> clipped to l_min
    assert predict_size(-5.0, fit, l_min=50, l_max=400) == 50
    # rounding: a=100, b=10, rho=0.05 -> 100.5 -> round to 100 (banker's) or 101;
    # assert it lands on one of the adjacent ints, proving int+round happened.
    assert predict_size(0.05, {"a": 100.0, "b": 10.0}) in (100, 101)


# --------------------------------------------------------------------------- #
# train_test_split_docs
# --------------------------------------------------------------------------- #
def test_train_test_split_deterministic():
    ids = [f"doc{i}" for i in range(10)]
    a = train_test_split_docs(ids, frac_train=0.7, seed=0)
    b = train_test_split_docs(ids, frac_train=0.7, seed=0)
    assert a == b
    # Independent of input order (sorted internally).
    c = train_test_split_docs(list(reversed(ids)), frac_train=0.7, seed=0)
    assert a == c


def test_train_test_split_disjoint_and_covers_all():
    ids = [f"doc{i}" for i in range(13)]
    train, test = train_test_split_docs(ids, frac_train=0.7, seed=42)
    assert set(train).isdisjoint(test)
    assert set(train) | set(test) == set(ids)
    # ~70% in train.
    assert len(train) == round(0.7 * 13)


def test_train_test_split_seed_changes_partition():
    ids = [f"doc{i}" for i in range(20)]
    t0, _ = train_test_split_docs(ids, seed=0)
    t1, _ = train_test_split_docs(ids, seed=1)
    # Different seeds should (very likely) give a different train set.
    assert t0 != t1


def test_train_test_split_edge_fractions():
    ids = [f"doc{i}" for i in range(5)]
    all_train, no_test = train_test_split_docs(ids, frac_train=1.0)
    assert set(all_train) == set(ids) and no_test == []
    no_train, all_test = train_test_split_docs(ids, frac_train=0.0)
    assert no_train == [] and set(all_test) == set(ids)


# --------------------------------------------------------------------------- #
# feature_target_correlations / select_and_fit
# --------------------------------------------------------------------------- #
# Feature "good" tracks the target perfectly (Spearman +1); "bad" tracks it
# perfectly inverted (Spearman -1); "noise" is constant (no ranking info, ~0).
# This mirrors the pilot finding: averaging good + bad cancels the signal, so a
# selector that picks the single strongest |corr| feature must recover it.
FEATS = {
    "A": {"good": 0.1, "bad": 0.9, "noise": 0.5},
    "B": {"good": 0.2, "bad": 0.8, "noise": 0.5},
    "C": {"good": 0.3, "bad": 0.7, "noise": 0.5},
    "D": {"good": 0.4, "bad": 0.6, "noise": 0.5},
}
TARGET = {"A": 50, "B": 100, "C": 200, "D": 400}


def test_feature_target_correlations_sorted_by_abs_corr():
    corrs = feature_target_correlations(FEATS, TARGET)
    names = [c[0] for c in corrs]
    # "good" and "bad" both have |corr| == 1; "noise" (constant) is last.
    assert names[-1] == "noise"
    assert set(names[:2]) == {"good", "bad"}
    by = dict(corrs)
    assert by["good"] == 1.0
    assert by["bad"] == -1.0
    assert abs(by["noise"]) < 1e-9


def test_feature_target_correlations_only_shared_docs():
    feats = dict(FEATS)
    feats["Z"] = {"good": 9.9, "bad": 0.0, "noise": 0.5}  # no target -> ignored
    corrs = feature_target_correlations(feats, TARGET)
    assert dict(corrs)["good"] == 1.0  # Z did not perturb the correlation


def test_feature_target_correlations_respects_feature_names_subset():
    corrs = feature_target_correlations(FEATS, TARGET, feature_names=["noise", "good"])
    assert {c[0] for c in corrs} == {"noise", "good"}


def test_select_and_fit_picks_strongest_feature_and_fits():
    res = select_and_fit(FEATS, TARGET, train_ids=["A", "B", "C", "D"])
    # Either perfectly-correlated feature is acceptable; both have |corr| == 1.
    assert res["feature"] in {"good", "bad"}
    assert abs(res["corr"]) == 1.0
    # The fitted line should predict the held-in targets monotonically and in range.
    p_small = predict_size(FEATS["A"][res["feature"]], res["fit"], l_min=50, l_max=400)
    p_large = predict_size(FEATS["D"][res["feature"]], res["fit"], l_min=50, l_max=400)
    if res["feature"] == "good":  # positive slope: A(0.1)->small, D(0.4)->large
        assert p_small < p_large
    else:  # "bad": negative slope
        assert p_small > p_large


def test_select_and_fit_uses_only_train_docs():
    # Selection must not peek at the held-out doc "D".
    res = select_and_fit(FEATS, TARGET, train_ids=["A", "B", "C"])
    assert res["feature"] in {"good", "bad"}


def test_select_and_fit_empty_train_is_safe():
    res = select_and_fit(FEATS, TARGET, train_ids=[])
    assert res["feature"] is None
    assert res["fit"]["n"] == 0
