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


# --- Generic Title Case heading detection (no numbering / markdown) ---

_BODY = (
    "We describe our approach in detail across a sentence that is "
    "clearly longer than any of the section headings above it."
)

# A scientific-paper-style doc: plain Title Case section headers, no numbering
# or markdown markers, each heading followed by a longer prose body line, with
# blank lines between sections.
TITLECASE_STRUCTURED = "\n\n".join(
    f"{heading}\n{_BODY}"
    for heading in (
        "Introduction",
        "Related Work",
        "Proposed Method",
        "Experiments",
        "Conclusion",
    )
)

# Plain prose: several long sentences separated by blank lines, no headings.
PLAIN_PROSE = "\n\n".join(
    [
        "The morning light slanted through the kitchen window and fell across "
        "the worn wooden table where the letters had been left unopened.",
        "She had walked the same path along the river every day for years, "
        "always pausing at the old stone bridge to watch the water move below.",
        "By the time the train arrived the platform was nearly empty and a cold "
        "wind pushed scraps of paper along the cracked concrete underfoot.",
        "Nobody in the village could quite remember when the clocktower had "
        "stopped, only that its hands had pointed at the same hour for as long "
        "as anyone cared to recall.",
    ]
)


def test_titlecase_structured_doc_scores_high():
    score, _ = structure_score(TITLECASE_STRUCTURED)
    assert score >= 0.5


def test_plain_prose_scores_low():
    score, _ = structure_score(PLAIN_PROSE)
    assert score < 0.5


def test_quality_sample_remains_unstructured():
    import os

    path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "demo",
        "quality_dataset_sample.txt",
    )
    with open(path, encoding="utf-8") as f:
        text = f.read()
    score, _ = structure_score(text)
    assert score < 0.5


# --- Predicate edge unit tests pinning the generic heading rule ---


def _generic_heading_frac(doc: str) -> float:
    return structure_score(doc)[1]["generic_heading_frac"]


def test_generic_heading_counts_when_blank_before_and_longer_body():
    # short Title Case line, preceded by blank line, followed by a longer body
    # line -> counts as a generic heading. Repeated so the doc routes structure.
    doc = TITLECASE_STRUCTURED
    assert _generic_heading_frac(doc) > 0.0
    assert structure_score(doc)[0] >= 0.5


def test_generic_heading_excluded_when_ends_with_period():
    # The same heading text but ending with a period does NOT count, so a
    # one-section variant stays below the routing threshold.
    doc = f"Introduction.\n{_BODY}"
    assert _generic_heading_frac(doc) == 0.0
    assert structure_score(doc)[0] < 0.5


def test_generic_heading_excluded_when_not_preceded_by_blank():
    # A short Title Case line immediately after body text (no blank line) does
    # NOT count as a heading.
    doc = (
        "This is a body sentence that runs on for a little while before it ends.\n"
        "Related Work\n"
        f"{_BODY}"
    )
    assert _generic_heading_frac(doc) == 0.0


def test_generic_heading_excluded_when_mostly_lowercase():
    # A short but mostly-lowercase line, even preceded by a blank and followed
    # by a longer body, is not "title-ish" and does NOT count.
    doc = f"the door was open\n{_BODY}"
    assert _generic_heading_frac(doc) == 0.0
