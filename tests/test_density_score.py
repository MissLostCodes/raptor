from raptor.chunking.density_score import (
    DENSITY_WEIGHTS,
    density_score,
    density_to_leaf_tokens,
)


# A dense, technical paragraph: numerals, symbols, long/rare words, and high
# lexical diversity (chemistry/math flavored, little repetition).
DENSE_TECHNICAL = (
    "The activation enthalpy DeltaH^{++} = 73.4 kJ/mol was estimated from "
    "Arrhenius regression (R^2 = 0.997) across temperatures 298-373 K. "
    "Stoichiometric titration of 0.125 M H2SO4 against 0.0500 M NaOH yielded "
    "an equivalence point at pH 8.31 +/- 0.04. Crystallographic refinement "
    "converged to R1 = 0.0212, wR2 = 0.0589 for 4,217 unique reflections."
)

# Simple, repetitive, low-diversity prose: short words, no numbers/symbols,
# minimal unique vocabulary.
SIMPLE_REPETITIVE = "the cat sat on the mat. " * 12

# Symbol/numeral heavy text vs plain prose for the feature-sanity test.
NUMERIC_SYMBOLIC = (
    "x1=3.14, y2=2.72; z3=>1.41! a4&b5 c6%d7 e8@f9 (10-20) [30/40] {50*60}"
)
PLAIN_PROSE = (
    "the dog ran across the open field and then it lay down under a tree to rest"
)

# A bulleted list vs a flowing paragraph for the list-marker sanity test.
BULLETED_LIST = (
    "Shopping list for the week ahead:\n"
    "- apples\n"
    "- bananas\n"
    "* bread\n"
    "* milk\n"
    "1. eggs\n"
    "2. cheese\n"
    "a) butter\n"
)
PARAGRAPH = (
    "We went to the market in the morning and bought apples and bananas and "
    "some bread and milk before the rain started and we hurried back home."
)

DOCUMENTED_KEYS = {
    "mean_sentence_len_tokens",
    "mean_sentence_len_tokens_norm",
    "type_token_ratio",
    "numeral_symbol_density",
    "mean_word_len_chars",
    "mean_word_len_chars_norm",
    "list_marker_density",
    "nonstopword_ratio",
}


def test_empty_returns_zero_and_features():
    rho, feats = density_score("")
    assert rho == 0.0
    assert DOCUMENTED_KEYS.issubset(feats.keys())


def test_whitespace_returns_zero_and_features():
    rho, feats = density_score("   \n  \t\n")
    assert rho == 0.0
    assert DOCUMENTED_KEYS.issubset(feats.keys())


def test_deterministic():
    a = density_score(DENSE_TECHNICAL)
    b = density_score(DENSE_TECHNICAL)
    assert a == b


def test_rho_and_normalized_features_in_unit_interval():
    for doc in (
        DENSE_TECHNICAL,
        SIMPLE_REPETITIVE,
        NUMERIC_SYMBOLIC,
        PLAIN_PROSE,
        BULLETED_LIST,
        PARAGRAPH,
        "single",
        "",
        "   ",
    ):
        rho, feats = density_score(doc)
        assert 0.0 <= rho <= 1.0
        # every normalized feature must live in [0, 1]
        for key in (
            "mean_sentence_len_tokens_norm",
            "type_token_ratio",
            "numeral_symbol_density",
            "mean_word_len_chars_norm",
            "list_marker_density",
            "nonstopword_ratio",
        ):
            assert 0.0 <= feats[key] <= 1.0, (key, feats[key], doc[:20])


def test_dense_technical_scores_higher_than_simple_repetitive():
    dense_rho, _ = density_score(DENSE_TECHNICAL)
    simple_rho, _ = density_score(SIMPLE_REPETITIVE)
    assert dense_rho > simple_rho


def test_weights_override_changes_score():
    # Putting all weight on numeral_symbol_density should raise the dense doc's
    # rho relative to the plain prose doc (which has no numerals/symbols).
    only_numeral = {
        "mean_sentence_len_tokens_norm": 0.0,
        "type_token_ratio": 0.0,
        "numeral_symbol_density": 1.0,
        "mean_word_len_chars_norm": 0.0,
        "list_marker_density": 0.0,
        "nonstopword_ratio": 0.0,
    }
    dense_rho, _ = density_score(DENSE_TECHNICAL, weights=only_numeral)
    prose_rho, _ = density_score(PLAIN_PROSE, weights=only_numeral)
    assert dense_rho > prose_rho


# --- Feature-sanity tests -------------------------------------------------


def test_numeral_symbol_density_higher_for_numeric_text():
    _, numeric_feats = density_score(NUMERIC_SYMBOLIC)
    _, prose_feats = density_score(PLAIN_PROSE)
    assert (
        numeric_feats["numeral_symbol_density"]
        > prose_feats["numeral_symbol_density"]
    )


def test_list_marker_density_higher_for_bulleted_list():
    _, list_feats = density_score(BULLETED_LIST)
    _, para_feats = density_score(PARAGRAPH)
    assert list_feats["list_marker_density"] > para_feats["list_marker_density"]


# --- density_to_leaf_tokens ----------------------------------------------


def test_leaf_tokens_returns_int_within_bounds():
    for rho in (0.0, 0.25, 0.5, 0.75, 1.0):
        for invert in (True, False):
            n = density_to_leaf_tokens(rho, invert=invert)
            assert isinstance(n, int)
            assert 64 <= n <= 320


def test_leaf_tokens_invert_true_high_density_smaller_leaves():
    # invert=True: higher density -> smaller leaves.
    assert density_to_leaf_tokens(0.0, invert=True) == 320
    assert density_to_leaf_tokens(1.0, invert=True) == 64
    assert density_to_leaf_tokens(0.0, invert=True) > density_to_leaf_tokens(
        1.0, invert=True
    )


def test_leaf_tokens_invert_false_reversed():
    # invert=False: higher density -> larger leaves.
    assert density_to_leaf_tokens(0.0, invert=False) == 64
    assert density_to_leaf_tokens(1.0, invert=False) == 320
    assert density_to_leaf_tokens(0.0, invert=False) < density_to_leaf_tokens(
        1.0, invert=False
    )


def test_leaf_tokens_monotonic_invert_true():
    rhos = [i / 10 for i in range(11)]
    vals = [density_to_leaf_tokens(r, invert=True) for r in rhos]
    # non-increasing
    assert all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))


def test_leaf_tokens_monotonic_invert_false():
    rhos = [i / 10 for i in range(11)]
    vals = [density_to_leaf_tokens(r, invert=False) for r in rhos]
    # non-decreasing
    assert all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))


def test_leaf_tokens_respects_custom_bounds():
    for rho in (0.0, 0.5, 1.0):
        n = density_to_leaf_tokens(rho, l_min=100, l_max=200)
        assert 100 <= n <= 200
    assert density_to_leaf_tokens(0.0, l_min=100, l_max=200, invert=True) == 200
    assert density_to_leaf_tokens(1.0, l_min=100, l_max=200, invert=True) == 100


def test_leaf_tokens_clips_out_of_range_rho():
    # rho outside [0,1] is clipped, not extrapolated.
    assert density_to_leaf_tokens(-0.5, invert=True) == 320
    assert density_to_leaf_tokens(1.5, invert=True) == 64


def test_default_weights_are_equal_and_cover_features():
    keys = {
        "mean_sentence_len_tokens_norm",
        "type_token_ratio",
        "numeral_symbol_density",
        "mean_word_len_chars_norm",
        "list_marker_density",
        "nonstopword_ratio",
    }
    assert set(DENSITY_WEIGHTS.keys()) == keys
    # all equal placeholder weights
    assert len(set(DENSITY_WEIGHTS.values())) == 1
