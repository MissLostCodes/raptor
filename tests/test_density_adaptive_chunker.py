"""Offline unit tests for the density-adaptive (two-axis) chunker.

No network / LLM: ``parse_fn``, ``score_fn``, and ``size_fn`` are all injectable, and
the structure/token sub-chunkers are deterministic. These exercise the mechanism only;
the calibrated coefficients come from the oracle sweep and are validated separately.
"""

import tiktoken

from raptor.chunking.base import Chunker
from raptor.chunking import get_chunker
from raptor.chunking.density_adaptive_chunker import (
    DensityAdaptiveChunker,
    make_calibrated_size_fn,
)


STRUCTURED = (
    "# Introduction\n"
    "This document describes the system in detail.\n"
    "1. Background\n"
    "Some background prose goes here for context.\n"
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
    "All happy families are alike but each unhappy family is unhappy in its own way. "
    "The quick brown fox jumps over the lazy dog once more to pad the token budget."
)


def _structured_parse_fn(text):
    return ["Introduction", "Background", "Methods", "Results", "Conclusion"]


def test_is_a_chunker():
    c = DensityAdaptiveChunker(parse_fn=_structured_parse_fn)
    assert isinstance(c, Chunker)


def test_empty_returns_empty():
    c = DensityAdaptiveChunker(parse_fn=_structured_parse_fn)
    assert c.chunk("") == []
    assert c.chunk("   \n  ") == []


def test_structure_axis_routes_like_ahc():
    # High structure score -> structure route; low -> token route.
    hi = DensityAdaptiveChunker(
        parse_fn=_structured_parse_fn, score_fn=lambda t: (0.6, {}), size_fn=lambda t: 1000
    )
    hi.chunk(STRUCTURED)
    assert hi.last_route == "structure"

    lo = DensityAdaptiveChunker(
        parse_fn=lambda t: [], score_fn=lambda t: (0.4, {}), size_fn=lambda t: 50
    )
    lo.chunk(FLAT)
    assert lo.last_route == "token"


def test_records_score_route_size_features():
    c = DensityAdaptiveChunker(
        parse_fn=lambda t: [],
        score_fn=lambda t: (0.1, {"f": 1}),
        size_fn=lambda t: 123,
    )
    chunks = c.chunk(FLAT)
    assert c.last_route == "token"
    assert c.last_score == 0.1
    assert c.last_size == 123
    assert c.last_features == {"f": 1}
    assert all(ch.strip() for ch in chunks)


def test_size_fn_drives_leaf_granularity():
    # Smaller leaf budget must yield at least as many token chunks as a larger one.
    enc = tiktoken.get_encoding("cl100k_base")
    small = DensityAdaptiveChunker(
        parse_fn=lambda t: [], score_fn=lambda t: (0.0, {}),
        size_fn=lambda t: 20, tokenizer=enc,
    )
    large = DensityAdaptiveChunker(
        parse_fn=lambda t: [], score_fn=lambda t: (0.0, {}),
        size_fn=lambda t: 400, tokenizer=enc,
    )
    n_small = len(small.chunk(FLAT))
    n_large = len(large.chunk(FLAT))
    assert n_small >= n_large
    assert n_small > 1  # a 20-token budget really did split the passage


def test_size_is_clipped_to_bounds():
    over = DensityAdaptiveChunker(
        parse_fn=lambda t: [], score_fn=lambda t: (0.0, {}),
        size_fn=lambda t: 99999, l_min=50, l_max=400,
    )
    over.chunk(FLAT)
    assert over.last_size == 400

    under = DensityAdaptiveChunker(
        parse_fn=lambda t: [], score_fn=lambda t: (0.0, {}),
        size_fn=lambda t: 1, l_min=50, l_max=400,
    )
    under.chunk(FLAT)
    assert under.last_size == 50


def test_make_calibrated_size_fn_positive_slope():
    # size = a + b*feature, with the feature read from density_score's feature dict.
    # Use mean_word_len_chars_norm (the pilot's strongest predictor, positive slope).
    size_fn = make_calibrated_size_fn(
        "mean_word_len_chars_norm", a=-2010.0, b=3385.0, l_min=50, l_max=400
    )
    short_words = "a it is on to be by of we an or"          # tiny mean word length
    long_words = "characterization optimization hierarchical representational"
    s_short = size_fn(short_words)
    s_long = size_fn(long_words)
    assert 50 <= s_short <= 400 and 50 <= s_long <= 400
    assert s_long > s_short  # longer words -> larger predicted leaf size


def test_make_calibrated_size_fn_unknown_feature_is_safe():
    size_fn = make_calibrated_size_fn("does_not_exist", a=100.0, b=5.0)
    # Missing feature -> 0.0 -> size = a = 100, clipped into range.
    assert size_fn(FLAT) == 100


def test_factory_builds_density_arm():
    c = get_chunker("density", parse_fn=lambda t: [], size_fn=lambda t: 80)
    assert isinstance(c, DensityAdaptiveChunker)
    c.chunk(FLAT)
    assert c.last_size == 80


def test_factory_density_default_placeholder_runs():
    # No size_fn -> uncalibrated composite placeholder; must still produce a valid size.
    c = get_chunker("density", parse_fn=lambda t: [], l_min=50, l_max=400)
    c.chunk(FLAT)
    assert 50 <= c.last_size <= 400
