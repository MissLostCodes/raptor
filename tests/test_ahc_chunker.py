from raptor.chunking.base import Chunker
from raptor.chunking.ahc_chunker import AHCChunker


STRUCTURED = (
    "# Introduction\n"
    "This document describes the system in detail.\n"
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


def _structured_parse_fn(text):
    # Real located titles present verbatim in STRUCTURED.
    return ["Introduction", "Background", "Methods", "Results", "Conclusion"]


def test_ahc_is_a_chunker():
    ahc = AHCChunker(parse_fn=_structured_parse_fn)
    assert isinstance(ahc, Chunker)


def test_high_structure_routes_to_structure():
    ahc = AHCChunker(parse_fn=_structured_parse_fn, max_tokens=1000, tau=0.5)
    chunks = ahc.chunk(STRUCTURED)
    assert ahc.last_route == "structure"
    assert ahc.last_score is not None
    assert ahc.last_features is not None
    assert all(c.strip() for c in chunks)


def test_flat_prose_routes_to_token():
    ahc = AHCChunker(parse_fn=lambda t: [], max_tokens=50, tau=0.5)
    chunks = ahc.chunk(FLAT)
    assert ahc.last_route == "token"
    assert all(c.strip() for c in chunks)


def test_threshold_behavior_with_injected_score_fn():
    feats = {"heading_line_frac": 0.0}

    above = AHCChunker(
        parse_fn=_structured_parse_fn,
        tau=0.5,
        max_tokens=1000,
        score_fn=lambda t: (0.6, feats),
    )
    above.chunk(STRUCTURED)
    assert above.last_route == "structure"
    assert above.last_score == 0.6

    below = AHCChunker(
        parse_fn=_structured_parse_fn,
        tau=0.5,
        max_tokens=1000,
        score_fn=lambda t: (0.4, feats),
    )
    below.chunk(STRUCTURED)
    assert below.last_route == "token"
    assert below.last_score == 0.4

    # exactly at tau routes to structure (>= tau)
    at = AHCChunker(
        parse_fn=_structured_parse_fn,
        tau=0.5,
        max_tokens=1000,
        score_fn=lambda t: (0.5, feats),
    )
    at.chunk(STRUCTURED)
    assert at.last_route == "structure"


def test_empty_returns_empty():
    ahc = AHCChunker(parse_fn=_structured_parse_fn)
    assert ahc.chunk("") == []
