"""Tests for evidence-locality / semantic-segment features (Phase 1).

These features replace the refuted cheap *surface* density features with ones
that measure a document's intrinsic semantic-segment size / evidence locality
(the H-new reframe). The module is split, like ``density_score`` and
``granularity_sweep``, into a PURE testable core (stdlib + math only) and a
guarded SBERT wrapper. Only the pure core is exercised here; the SBERT path is
tested through an injected ``embed_fn`` so no model/network is needed.
"""

import math

from raptor.chunking.segment_features import (
    paragraph_length_features,
    gzip_compressibility,
    semantic_segment_features,
    document_segment_features,
    segment_features,
    split_sentences,
)


# ---------------------------------------------------------------------------
# B1 — paragraph-length distribution (pure)
# ---------------------------------------------------------------------------
def test_paragraph_features_basic_counts():
    text = "a b c\n\nd e\n\nf g h i"  # 3 paragraphs: 3, 2, 4 tokens
    f = paragraph_length_features(text)
    assert f["n_paragraphs"] == 3
    assert f["paragraph_len_mean"] == 3.0  # (3+2+4)/3
    assert f["paragraph_len_median"] == 3.0


def test_paragraph_features_cv_zero_for_uniform():
    text = "a b\n\nc d\n\ne f"  # all paragraphs 2 tokens
    f = paragraph_length_features(text)
    assert f["paragraph_len_mean"] == 2.0
    assert f["paragraph_len_cv"] == 0.0  # no dispersion


def test_paragraph_features_empty():
    f = paragraph_length_features("")
    assert f["n_paragraphs"] == 0
    assert f["paragraph_len_mean"] == 0.0
    assert f["paragraph_len_cv"] == 0.0


def test_paragraph_features_single_block_no_blank_lines():
    # No blank-line separators -> treated as one paragraph.
    text = "one two three four five"
    f = paragraph_length_features(text)
    assert f["n_paragraphs"] == 1
    assert f["paragraph_len_mean"] == 5.0


# ---------------------------------------------------------------------------
# C4 — gzip compressibility (pure)
# ---------------------------------------------------------------------------
def test_gzip_ratio_lower_for_repetitive_text():
    repetitive = "ab " * 400  # highly redundant
    diverse = " ".join(f"w{i}x{i*7 % 31}" for i in range(400))  # little repetition
    r_rep = gzip_compressibility(repetitive)["gzip_ratio"]
    r_div = gzip_compressibility(diverse)["gzip_ratio"]
    assert 0.0 < r_rep < r_div  # redundant text compresses better (smaller ratio)


def test_gzip_ratio_empty_is_zero():
    assert gzip_compressibility("")["gzip_ratio"] == 0.0


# ---------------------------------------------------------------------------
# A1-A5 — semantic-segment features from adjacent cosines (pure)
# ---------------------------------------------------------------------------
def test_semantic_no_boundaries_when_flat():
    # Constant similarity -> zero variance -> no boundary -> one segment.
    cosines = [0.9, 0.9, 0.9]
    counts = [10, 10, 10, 10]  # 4 sentences, 40 tokens
    f = semantic_segment_features(cosines, counts)
    assert f["mean_adjacent_cosine"] == 0.9
    assert f["var_adjacent_cosine"] == 0.0
    assert f["n_segments"] == 1
    assert f["mean_segment_len_tokens"] == 40.0
    assert f["topic_shift_rate"] == 0.0
    assert f["segment_len_cv"] == 0.0


def test_semantic_single_boundary_splits_into_two_segments():
    # One sharp dip below the depth cutoff -> exactly one boundary.
    cosines = [0.9, 0.9, 0.1, 0.9, 0.9]  # boundary between sentence 2 and 3
    counts = [10, 10, 10, 10, 10, 10]    # 6 sentences, 60 tokens
    f = semantic_segment_features(cosines, counts, depth_k=1.0)
    assert f["n_segments"] == 2
    assert f["mean_segment_len_tokens"] == 30.0  # two 30-token segments
    # one boundary over 60 tokens -> 1000/60 per 1k tokens
    assert math.isclose(f["topic_shift_rate"], 1000.0 / 60.0, rel_tol=1e-9)


def test_semantic_handles_too_few_sentences():
    # 0 or 1 sentence -> no cosines; degrade gracefully.
    f0 = semantic_segment_features([], [])
    assert f0["n_segments"] == 0
    assert f0["mean_segment_len_tokens"] == 0.0
    f1 = semantic_segment_features([], [25])
    assert f1["n_segments"] == 1
    assert f1["mean_segment_len_tokens"] == 25.0


# ---------------------------------------------------------------------------
# split_sentences (pure)
# ---------------------------------------------------------------------------
def test_split_sentences_basic():
    sents = split_sentences("Hello world. This is a test! Is it? Yes.")
    assert len(sents) == 4


# ---------------------------------------------------------------------------
# Heavy wrapper via injected embed_fn (no SBERT needed)
# ---------------------------------------------------------------------------
def _fake_embed(sentences):
    """Deterministic 2-D embedder: vector points one of two ways by a keyword.

    Sentences containing 'alpha' map to ~(1,0); 'beta' to ~(0,1). Within a topic
    the cosine is ~1 (no boundary); across topics it is ~0 (a boundary).
    """
    vecs = []
    for s in sentences:
        if "beta" in s.lower():
            vecs.append([0.0, 1.0])
        else:
            vecs.append([1.0, 0.0])
    return vecs


def test_document_segment_features_detects_topic_change():
    text = (
        "alpha one is here. alpha two follows. alpha three again. "
        "beta one starts. beta two follows. beta three again."
    )
    f = document_segment_features(text, embed_fn=_fake_embed)
    # One clear alpha->beta switch -> two segments.
    assert f["n_segments"] == 2
    assert f["mean_segment_len_tokens"] > 0.0


def test_segment_features_merges_all_families_and_is_finite():
    text = (
        "alpha one is here.\n\n"
        "alpha two follows and continues for a while here.\n\n"
        "beta one starts a new topic. beta two follows it."
    )
    f = segment_features(text, embed_fn=_fake_embed)
    # B1 + C4 + A-tier keys all present.
    for key in (
        "paragraph_len_mean",
        "paragraph_len_cv",
        "gzip_ratio",
        "mean_adjacent_cosine",
        "var_adjacent_cosine",
        "topic_shift_rate",
        "mean_segment_len_tokens",
        "segment_len_cv",
        "n_segments",
    ):
        assert key in f
        assert isinstance(f[key], (int, float))
        assert not math.isnan(float(f[key]))


def test_segment_features_empty_text_safe():
    f = segment_features("", embed_fn=_fake_embed)
    assert f["n_segments"] == 0
    assert f["paragraph_len_mean"] == 0.0
    assert f["mean_segment_len_tokens"] == 0.0
