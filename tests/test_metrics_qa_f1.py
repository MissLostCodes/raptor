import math

from experiments.metrics import normalize_answer, token_f1, qasper_answer_f1


def test_normalize_answer_removes_articles_punctuation_case():
    assert normalize_answer("The Quick, Brown Fox!") == "quick brown fox"
    assert normalize_answer("A an the") == ""
    assert normalize_answer("  Multiple   spaces  ") == "multiple spaces"


def test_token_f1_exact_match():
    assert token_f1("hello world", "hello world") == 1.0


def test_token_f1_partial_overlap():
    # prediction tokens: {the(removed), cat, sat} -> {cat, sat}
    # gold tokens: {the(removed), cat, ran} -> {cat, ran}
    # shared = {cat} count 1; precision = 1/2, recall = 1/2 => F1 = 0.5
    assert math.isclose(token_f1("the cat sat", "the cat ran"), 0.5, rel_tol=1e-9)


def test_token_f1_partial_overlap_uneven():
    # pred: x b c d (4 tokens), gold: x b (2 tokens) -- none are articles
    # shared = 2; precision = 2/4 = 0.5, recall = 2/2 = 1.0
    # F1 = 2*0.5*1.0/(0.5+1.0) = 1.0/1.5 = 0.6666...
    assert math.isclose(token_f1("x b c d", "x b"), 2 / 3, rel_tol=1e-9)


def test_token_f1_disjoint():
    assert token_f1("cat dog", "fish bird") == 0.0


def test_token_f1_both_empty():
    assert token_f1("the", "an") == 1.0  # both normalize to empty


def test_token_f1_one_empty():
    assert token_f1("hello", "the") == 0.0


def test_qasper_answer_f1_takes_max_over_golds():
    pred = "the cat sat"  # normalizes to "cat sat" (2 tokens)
    golds = ["dog ran far", "the cat sat on mat"]  # gold2 -> "cat sat on mat" (4)
    # gold2: shared {cat,sat}=2; p=2/2=1.0, r=2/4=0.5 => F1 = 2*1*0.5/1.5 = 2/3
    assert math.isclose(qasper_answer_f1(pred, golds), 2 / 3, rel_tol=1e-9)


def test_qasper_answer_f1_exact():
    assert qasper_answer_f1("paris", ["paris"]) == 1.0


def test_qasper_unanswerable_correct_empty_pred():
    assert qasper_answer_f1("", [], is_unanswerable=True) == 1.0


def test_qasper_unanswerable_correct_says_unanswerable():
    assert qasper_answer_f1("This is unanswerable", [], is_unanswerable=True) == 1.0


def test_qasper_unanswerable_wrong_gives_answer():
    assert qasper_answer_f1("the answer is paris", [], is_unanswerable=True) == 0.0


def test_qasper_answerable_but_pred_empty():
    # answerable gold present, prediction empty -> 0.0
    assert qasper_answer_f1("", ["paris"]) == 0.0
