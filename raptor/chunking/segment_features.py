"""Evidence-locality / semantic-segment features for leaf-size prediction.

The H2 experiment refuted the six cheap *surface* density features
(:mod:`raptor.chunking.density_score`): they do not predict per-document optimal
leaf size well enough to beat a tuned fixed size. The reframe (see
``docs/specs/2026-06-18-document-features-brainstorm.md`` and
``docs/specs/2026-06-27-weighted-adaptive-chunking-plan.md``) is that optimal
leaf size is set by a document's intrinsic **semantic-segment size / evidence
locality**, not by surface lexical density. This module computes the
causally-on-target candidate features:

* **A1-A5 (semantic)** -- from adjacent-sentence cosine similarities:
  ``mean_adjacent_cosine`` (A1, local topical continuity),
  ``var_adjacent_cosine`` (A2, how lumpy the topic flow is),
  ``topic_shift_rate`` (A3, boundaries per 1k tokens),
  ``mean_segment_len_tokens`` (A4, the document's natural chunk size --
  **the headline candidate**), ``segment_len_cv`` (A5, segment heterogeneity).
* **B1 (structural)** -- ``paragraph_len_{mean,median,cv}`` and ``n_paragraphs``:
  paragraphs are author-intended units; their size is a cheap structural twin of
  A4.
* **C4 (linguistic)** -- ``gzip_ratio``: a model-free redundancy proxy (lower =
  more redundant => can tolerate bigger leaves).

Design mirrors :mod:`density_score` / :mod:`experiments.granularity_sweep`: a
PURE core (stdlib + ``math`` + ``gzip`` only, fully unit-testable, deterministic)
plus a guarded SBERT wrapper whose heavy import lives *inside* the function so
importing this module stays cheap and offline-safe. The semantic features take
an ``embed_fn`` so they are testable without a model; the default loads SBERT.

The output dict is shaped exactly like ``density_score``'s feature dict, so it
drops straight into the existing calibration harness
(:mod:`experiments.granularity_calibration`).
"""

from __future__ import annotations

import gzip as _gzip
import math
import re
from typing import Callable, Dict, List, Optional, Sequence

# Sentence splitting mirrors density_score's regex so the two modules agree on
# what a "sentence" is.
_SENTENCE_SPLIT_RE = re.compile(r"[.?!]+")
# Paragraphs are separated by one or more blank lines (optionally whitespace).
_PARAGRAPH_SPLIT_RE = re.compile(r"\n[ \t]*\n+")

EmbedFn = Callable[[List[str]], Sequence[Sequence[float]]]


# ---------------------------------------------------------------------------
# Small pure helpers.
# ---------------------------------------------------------------------------
def _mean(xs: Sequence[float]) -> float:
    return (sum(xs) / len(xs)) if xs else 0.0


def _variance(xs: Sequence[float]) -> float:
    """Population variance (0.0 for empty / length-1 input)."""
    n = len(xs)
    if n < 2:
        return 0.0
    m = sum(xs) / n
    return sum((x - m) ** 2 for x in xs) / n


def _std(xs: Sequence[float]) -> float:
    return math.sqrt(_variance(xs))


def _cv(xs: Sequence[float]) -> float:
    """Coefficient of variation (std / mean); 0.0 when mean is 0 or input short."""
    m = _mean(xs)
    if m == 0.0 or len(xs) < 2:
        return 0.0
    return _std(xs) / m


def _median(xs: Sequence[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return float(s[mid])
    return (s[mid - 1] + s[mid]) / 2.0


def _token_count(text: str) -> int:
    return len(text.split())


def split_sentences(text: str) -> List[str]:
    """Split ``text`` into non-empty sentences on ``[.?!]+`` (same rule as density_score)."""
    if not text or not text.strip():
        return []
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


def _cosine(u: Sequence[float], v: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(u, v))
    nu = math.sqrt(sum(a * a for a in u))
    nv = math.sqrt(sum(b * b for b in v))
    if nu == 0.0 or nv == 0.0:
        return 0.0
    return dot / (nu * nv)


# ---------------------------------------------------------------------------
# B1 -- paragraph-length distribution (PURE).
# ---------------------------------------------------------------------------
def paragraph_length_features(text: str) -> Dict[str, float]:
    """Distribution of tokens-per-paragraph (B1).

    Paragraphs split on blank-line boundaries; a document with no blank lines is
    one paragraph. Returns ``n_paragraphs`` and the mean / median / CV of the
    per-paragraph token counts. Empty input returns all zeros.
    """
    out = {
        "n_paragraphs": 0,
        "paragraph_len_mean": 0.0,
        "paragraph_len_median": 0.0,
        "paragraph_len_cv": 0.0,
    }
    if not text or not text.strip():
        return out
    paras = [p for p in _PARAGRAPH_SPLIT_RE.split(text) if p.strip()]
    if not paras:
        return out
    lens = [float(_token_count(p)) for p in paras]
    out["n_paragraphs"] = len(lens)
    out["paragraph_len_mean"] = _mean(lens)
    out["paragraph_len_median"] = _median(lens)
    out["paragraph_len_cv"] = _cv(lens)
    return out


# ---------------------------------------------------------------------------
# C4 -- gzip compressibility (PURE).
# ---------------------------------------------------------------------------
def gzip_compressibility(text: str) -> Dict[str, float]:
    """Model-free redundancy proxy (C4): compressed bytes / raw bytes.

    Lower ratio => more redundant text (compresses well) => tolerates bigger
    leaves. Uses a fixed compression level so the value is deterministic. Empty
    input returns ``0.0``.
    """
    if not text:
        return {"gzip_ratio": 0.0}
    raw = text.encode("utf-8")
    if not raw:
        return {"gzip_ratio": 0.0}
    # mtime=0 keeps the gzip header deterministic across runs.
    comp = _gzip.compress(raw, compresslevel=6, mtime=0)
    return {"gzip_ratio": len(comp) / len(raw)}


# ---------------------------------------------------------------------------
# A1-A5 -- semantic-segment features from adjacent cosines (PURE).
# ---------------------------------------------------------------------------
def semantic_segment_features(
    adjacent_cosines: Sequence[float],
    sentence_token_counts: Sequence[int],
    depth_k: float = 1.0,
) -> Dict[str, float]:
    """Semantic-coherence features (A1-A5) from precomputed adjacent cosines.

    ``adjacent_cosines[i]`` is the cosine between sentence ``i`` and ``i+1`` (so
    its length is ``len(sentence_token_counts) - 1`` for a real document).
    A *boundary* is placed after sentence ``i`` when ``cosines[i]`` falls a depth
    of ``depth_k`` standard deviations below the mean similarity -- a TextTiling-
    style depth cutoff. Segments are the runs of sentences between boundaries; a
    segment's length is the sum of its sentences' token counts.

    Returns:
        ``mean_adjacent_cosine`` (A1), ``var_adjacent_cosine`` (A2),
        ``topic_shift_rate`` (A3, boundaries per 1000 tokens),
        ``mean_segment_len_tokens`` (A4), ``segment_len_cv`` (A5),
        ``n_segments``.

    Degenerate cases: no sentences -> ``n_segments == 0`` and all zeros; one
    sentence (no cosines) -> a single segment equal to that sentence's tokens.
    """
    counts = [float(c) for c in sentence_token_counts]
    n_sent = len(counts)
    total_tokens = sum(counts)

    out = {
        "mean_adjacent_cosine": 0.0,
        "var_adjacent_cosine": 0.0,
        "topic_shift_rate": 0.0,
        "mean_segment_len_tokens": 0.0,
        "segment_len_cv": 0.0,
        "n_segments": 0,
    }
    if n_sent == 0:
        return out
    if n_sent == 1 or not adjacent_cosines:
        out["mean_segment_len_tokens"] = total_tokens
        out["n_segments"] = 1
        return out

    cos = [float(c) for c in adjacent_cosines]
    out["mean_adjacent_cosine"] = _mean(cos)
    out["var_adjacent_cosine"] = _variance(cos)

    # Depth cutoff: a boundary where similarity dips depth_k*std below the mean.
    cutoff = _mean(cos) - depth_k * _std(cos)
    boundaries = [i for i, c in enumerate(cos) if c < cutoff]

    # Build segment token lengths by cutting after each boundary sentence.
    seg_lens: List[float] = []
    start = 0
    for b in boundaries:
        seg_lens.append(sum(counts[start : b + 1]))
        start = b + 1
    seg_lens.append(sum(counts[start:]))  # trailing segment

    out["n_segments"] = len(seg_lens)
    out["mean_segment_len_tokens"] = _mean(seg_lens)
    out["segment_len_cv"] = _cv(seg_lens)
    out["topic_shift_rate"] = (
        (len(boundaries) / total_tokens * 1000.0) if total_tokens else 0.0
    )
    return out


# ---------------------------------------------------------------------------
# Heavy SBERT wrapper (guarded import) + top-level merge.
# ---------------------------------------------------------------------------
def _default_embed(sentences: List[str]) -> List[List[float]]:
    """Embed sentences with the project's SBERT model (HEAVY; guarded import)."""
    from raptor.EmbeddingModels import SBertEmbeddingModel

    model = SBertEmbeddingModel()
    return [list(model.create_embedding(s)) for s in sentences]


def document_segment_features(
    text: str,
    embed_fn: Optional[EmbedFn] = None,
    depth_k: float = 1.0,
) -> Dict[str, float]:
    """Compute A1-A5 for a document by embedding its sentences.

    ``embed_fn`` maps a list of sentence strings to a list of vectors; it
    defaults to SBERT (:func:`_default_embed`). Inject a fake in tests to avoid
    loading a model. Fewer than two sentences degrade gracefully (one or zero
    segments).
    """
    sentences = split_sentences(text)
    counts = [_token_count(s) for s in sentences]
    if len(sentences) < 2:
        return semantic_segment_features([], counts, depth_k=depth_k)

    embed = embed_fn or _default_embed
    vectors = embed(sentences)
    cosines = [
        _cosine(vectors[i], vectors[i + 1]) for i in range(len(vectors) - 1)
    ]
    return semantic_segment_features(cosines, counts, depth_k=depth_k)


def segment_features(
    text: str,
    embed_fn: Optional[EmbedFn] = None,
    depth_k: float = 1.0,
) -> Dict[str, float]:
    """Full locality-feature dict for a document: A1-A5 + B1 + C4 merged.

    Shaped like ``density_score``'s feature dict so it drops into the calibration
    harness. ``embed_fn`` is forwarded to the semantic features (defaults to
    SBERT). Empty input returns a safe all-zero dict.
    """
    feats: Dict[str, float] = {}
    feats.update(paragraph_length_features(text))
    feats.update(gzip_compressibility(text))
    feats.update(document_segment_features(text, embed_fn=embed_fn, depth_k=depth_k))
    return feats
