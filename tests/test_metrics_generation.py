import math

import pytest

from experiments.metrics import (
    rouge_l,
    bleu_1,
    bleu_4,
    meteor,
    narrativeqa_metrics,
    stage_meteor_data,
)


def test_rouge_l_identical():
    pred = "the quick brown fox jumps over the lazy dog"
    assert math.isclose(rouge_l(pred, [pred]), 1.0, rel_tol=1e-6)


def test_rouge_l_disjoint_non_crash():
    val = rouge_l("alpha beta gamma", ["delta epsilon zeta"])
    assert 0.0 <= val <= 1.0
    assert val == 0.0


def test_rouge_l_multiple_references_takes_best():
    pred = "the cat sat on the mat"
    refs = ["completely different text here", "the cat sat on the mat"]
    assert math.isclose(rouge_l(pred, refs), 1.0, rel_tol=1e-6)


def test_bleu_1_identical_high():
    pred = "the quick brown fox jumps"
    val = bleu_1(pred, [pred])
    assert val > 0.9


def test_bleu_4_identical_high():
    pred = "the quick brown fox jumps over the lazy dog today"
    val = bleu_4(pred, [pred])
    assert val > 0.5


def test_bleu_1_disjoint_non_crash():
    # smoothing => small positive or zero, never raises
    val = bleu_1("alpha beta gamma", ["delta epsilon zeta"])
    assert 0.0 <= val <= 1.0


def test_bleu_4_disjoint_non_crash():
    val = bleu_4("alpha beta gamma delta", ["one two three four"])
    assert 0.0 <= val <= 1.0


def test_bleu_multiple_references_best():
    pred = "the cat sat on the mat"
    refs = ["xxx yyy zzz", "the cat sat on the mat"]
    val = bleu_1(pred, refs)
    assert val > 0.9


def test_narrativeqa_metrics_keys():
    out = narrativeqa_metrics("the cat sat", ["the cat sat"])
    assert set(out.keys()) == {"rouge_l", "bleu_1", "bleu_4", "meteor"}
    assert math.isclose(out["rouge_l"], 1.0, rel_tol=1e-6)
    assert out["bleu_1"] > 0.9


def test_meteor_identical_or_skip():
    pred = "the quick brown fox"
    try:
        val = meteor(pred, [pred])
    except RuntimeError as e:
        pytest.skip(f"meteor data unavailable offline: {e}")
    assert 0.0 <= val <= 1.0
    assert val > 0.9


def test_stage_meteor_data_returns_bool_and_is_idempotent():
    first = stage_meteor_data()
    second = stage_meteor_data()
    assert isinstance(first, bool)
    assert first == second  # never raises; stable across calls
    # When staging succeeds, METEOR must actually compute (not return None).
    if first:
        assert meteor("the cat sat", ["the cat sat"]) > 0.9
        assert narrativeqa_metrics("the cat sat", ["the cat sat"])["meteor"] is not None
