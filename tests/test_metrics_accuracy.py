from experiments.metrics import extract_choice, accuracy, accuracy_on_subset

OPTIONS = ["The cat ran home", "A dog barked loudly", "Birds fly south", "Fish swim deep"]


def test_extract_choice_bare_letter():
    assert extract_choice("B", OPTIONS) == 1


def test_extract_choice_parenthesized_letter():
    assert extract_choice("(C)", OPTIONS) == 2


def test_extract_choice_answer_prefix_letter():
    assert extract_choice("Answer: D", OPTIONS) == 3
    assert extract_choice("The answer is A", OPTIONS) == 0


def test_extract_choice_numeric():
    assert extract_choice("2", OPTIONS) == 1
    assert extract_choice("Option 4", OPTIONS) == 3


def test_extract_choice_text_overlap_fallback():
    # No letter/number, must match by text overlap to option index 2
    out = "I think the birds fly south for winter"
    assert extract_choice(out, OPTIONS) == 2


def test_extract_choice_no_match():
    assert extract_choice("zzz qqq", OPTIONS) == -1


def test_extract_choice_empty():
    assert extract_choice("", OPTIONS) == -1


def test_accuracy_basic():
    preds = [0, 1, 2, 3]
    golds = [0, 1, 9, 3]
    assert accuracy(preds, golds) == 0.75


def test_accuracy_empty():
    assert accuracy([], []) == 0.0


def test_accuracy_on_subset():
    preds = [0, 1, 2, 3]
    golds = [0, 9, 2, 9]
    mask = [True, True, False, False]
    # over first two: pred0==gold0 (correct), pred1!=gold1 -> 0.5
    assert accuracy_on_subset(preds, golds, mask) == 0.5


def test_accuracy_on_subset_empty():
    assert accuracy_on_subset([0, 1], [0, 1], [False, False]) == 0.0
