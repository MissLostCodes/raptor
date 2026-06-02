import math

from experiments.metrics import evidence_f1, evidence_f1_max


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
