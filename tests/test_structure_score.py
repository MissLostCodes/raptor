from raptor.chunking.structure_score import structure_score


STRUCTURED = (
    "# Introduction\n"
    "This document describes the system.\n"
    "1. Background\n"
    "Some background prose goes here for context.\n"
    "1.1 Prior Work\n"
    "We review the prior work in this subsection.\n"
    "## Methods\n"
    "The methods are described in detail below.\n"
    "2. Results\n"
    "The results are presented in this section.\n"
    "3. Conclusion\n"
    "We conclude with a short discussion of findings.\n"
)

FLAT = (
    "The quick brown fox jumps over the lazy dog. "
    "It was a bright cold day in April and the clocks were striking thirteen. "
    "She sells seashells by the seashore on a sunny afternoon. "
    "The rain in Spain falls mainly on the plain throughout the season. "
    "All happy families are alike but each unhappy family is unhappy in its own way."
)

DOCUMENTED_KEYS = {
    "numbered_heading_frac",
    "markdown_heading_frac",
    "allcaps_short_frac",
    "keyword_heading_frac",
    "heading_line_frac",
    "has_toc",
    "heading_spacing_regularity",
}


def test_structured_doc_scores_high():
    score, _ = structure_score(STRUCTURED)
    assert score > 0.6


def test_flat_prose_scores_low():
    score, _ = structure_score(FLAT)
    assert score < 0.2


def test_empty_returns_zero_and_features():
    score, feats = structure_score("")
    assert score == 0.0
    assert DOCUMENTED_KEYS.issubset(feats.keys())

    score2, feats2 = structure_score("   \n  \n\t")
    assert score2 == 0.0
    assert DOCUMENTED_KEYS.issubset(feats2.keys())


def test_deterministic():
    a = structure_score(STRUCTURED)
    b = structure_score(STRUCTURED)
    assert a == b


def test_features_have_all_documented_keys():
    _, feats = structure_score(STRUCTURED)
    assert DOCUMENTED_KEYS.issubset(feats.keys())


def test_score_always_in_unit_interval():
    for doc in (STRUCTURED, FLAT, "", "   ", "# Only one heading\nbody text here"):
        score, _ = structure_score(doc)
        assert 0.0 <= score <= 1.0
