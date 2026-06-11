"""Deterministic, training-free "information density" scorer for AHC sizing.

`density_score(text)` inspects only cheap, regex/whitespace-based surface features of the
raw text (no LLM, no embeddings, no randomness, stdlib + `re` only) and returns a scalar
``rho`` in [0, 1] estimating how *information-dense* the text is, plus the feature dict
used to compute it. It is the second axis of a two-axis chunking controller: the existing
``structure_score`` decides chunk BOUNDARIES, while this density score is intended to
decide chunk SIZE (leaf granularity) via the ``density_to_leaf_tokens`` mapping helper.

Score formula (documented):

    rho = clip(
        sum_f w_f * feat_norm[f],         # weighted mean of the 6 normalized features
        0.0, 1.0,
    )

where ``w_f`` come from ``DENSITY_WEIGHTS`` (defaulting to EQUAL weights summing to 1.0)
and ``feat_norm[f]`` are the six normalized features below, each in [0, 1].

The six surface features and their bounded transforms:

1. ``mean_sentence_len_tokens`` -- text is split into sentences on ``[.?!]+`` and tokens
   by whitespace; this is the mean number of tokens per sentence. Longer sentences tend
   to pack more clauses/information per unit, so longer -> denser. Normalized as
   ``min(1.0, x / MEAN_SENTENCE_LEN_REF)`` with ``MEAN_SENTENCE_LEN_REF = 40``: ~40
   tokens/sentence is already a long, complex sentence; anything at or above that
   saturates the term. Stored as ``mean_sentence_len_tokens`` (raw) and
   ``mean_sentence_len_tokens_norm`` (in [0, 1]).

2. ``type_token_ratio`` -- unique lowercased alphabetic tokens / total alphabetic tokens.
   Already in [0, 1] by construction; higher means more lexically diverse (less
   repetition), which we treat as denser content. No reference constant needed.

3. ``numeral_symbol_density`` -- fraction of whitespace tokens that contain a digit or a
   non-alphanumeric symbol. A proxy for technical/tabular/mathematical content (numbers,
   operators, units, identifiers). Already in [0, 1]. No reference constant needed.

4. ``mean_word_len_chars`` -- mean characters per alphabetic word. Longer words skew
   toward rarer, more technical vocabulary, so longer -> denser. Normalized as
   ``min(1.0, x / MEAN_WORD_LEN_REF)`` with ``MEAN_WORD_LEN_REF = 8``: average English
   word length is ~4-5 chars, and an 8-char mean indicates consistently long/technical
   vocabulary, so 8 is a reasonable saturation point. Stored as ``mean_word_len_chars``
   (raw) and ``mean_word_len_chars_norm`` (in [0, 1]).

5. ``list_marker_density`` -- fraction of non-empty lines whose stripped start matches a
   bullet/enumeration marker (``-``, ``*``, ``•``, ``^\\d+[.)]\\s``, or
   ``^[A-Za-z][.)]\\s``). Already in [0, 1]. Dense, enumerated material (lists, tables of
   short facts) is information-rich per line. No reference constant needed.

6. ``nonstopword_ratio`` -- ``1 - (stopword tokens / total tokens)`` over whitespace
   tokens, using the small ~30-word ``_STOPWORDS`` set below. Higher means a larger share
   of content words (fewer function words), i.e. denser content. Already in [0, 1]. No
   reference constant needed.

All reference constants and the ``DENSITY_WEIGHTS`` are PLACEHOLDERS to be CALIBRATED
later from experiment data; they are deliberately interpretable, not tuned magic numbers.

Empty / whitespace-only input scores ``rho == 0.0`` with all features zeroed.
"""

import re
from typing import Dict, Optional, Tuple

# --- Normalization reference constants (documented above; calibration targets) ---
# Each maps a raw feature into [0, 1] via ``min(1.0, raw / REF)``.
MEAN_SENTENCE_LEN_REF = 40.0  # tokens/sentence at which the length term saturates
MEAN_WORD_LEN_REF = 8.0  # chars/word at which the word-length term saturates

# --- Default feature weights (EQUAL placeholders, summing to 1.0; to be calibrated) ---
# These are intentionally uniform; the real weights come from later experiments.
DENSITY_WEIGHTS: Dict[str, float] = {
    "mean_sentence_len_tokens_norm": 1.0 / 6.0,
    "type_token_ratio": 1.0 / 6.0,
    "numeral_symbol_density": 1.0 / 6.0,
    "mean_word_len_chars_norm": 1.0 / 6.0,
    "list_marker_density": 1.0 / 6.0,
    "nonstopword_ratio": 1.0 / 6.0,
}

# Small, documented set of ~30 common English stopwords (function words).
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "of", "at", "by",
    "for", "with", "to", "from", "in", "on", "is", "are", "was", "were",
    "be", "been", "it", "its", "this", "that", "as", "we", "our", "you",
}

_SENTENCE_SPLIT_RE = re.compile(r"[.?!]+")
_STOPWORD_EDGE_RE = re.compile(r"^[^a-z]+|[^a-z]+$")
_ALPHA_TOKEN_RE = re.compile(r"[A-Za-z]+")
_DIGIT_RE = re.compile(r"\d")
_HAS_NON_ALNUM_RE = re.compile(r"[^A-Za-z0-9]")

# List/enumeration markers, applied to the *stripped* start of a line.
_NUM_ENUM_RE = re.compile(r"^\d+[.)]\s")
_ALPHA_ENUM_RE = re.compile(r"^[A-Za-z][.)]\s")
_BULLET_CHARS = ("-", "*", "•")


def _zero_features() -> Dict:
    return {
        "mean_sentence_len_tokens": 0.0,
        "mean_sentence_len_tokens_norm": 0.0,
        "type_token_ratio": 0.0,
        "numeral_symbol_density": 0.0,
        "mean_word_len_chars": 0.0,
        "mean_word_len_chars_norm": 0.0,
        "list_marker_density": 0.0,
        "nonstopword_ratio": 0.0,
    }


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _is_list_marker(stripped: str) -> bool:
    """True if a stripped line starts with a bullet/enumeration marker."""
    if not stripped:
        return False
    if stripped[0] in _BULLET_CHARS:
        return True
    if _NUM_ENUM_RE.match(stripped):
        return True
    if _ALPHA_ENUM_RE.match(stripped):
        return True
    return False


def density_score(
    text: str, weights: Optional[Dict[str, float]] = None
) -> Tuple[float, Dict]:
    """Return (rho in [0, 1], features dict). Deterministic; no LLM/randomness.

    ``weights`` optionally overrides ``DENSITY_WEIGHTS`` (a dict keyed by the six
    normalized-feature names). Empty / whitespace-only input returns ``(0.0, features)``
    with every feature zeroed.
    """
    if weights is None:
        weights = DENSITY_WEIGHTS

    features = _zero_features()

    if not text or not text.strip():
        return 0.0, features

    # Whitespace tokens are the unit for sentence length, numeral/symbol density,
    # and the stopword ratio.
    tokens = text.split()
    n_tokens = len(tokens)
    if n_tokens == 0:
        return 0.0, features

    # 1. mean_sentence_len_tokens: tokens per sentence, sentences split on [.?!]+.
    sentences = [s for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    n_sentences = len(sentences) if sentences else 1
    mean_sentence_len = n_tokens / n_sentences
    features["mean_sentence_len_tokens"] = mean_sentence_len
    features["mean_sentence_len_tokens_norm"] = min(
        1.0, mean_sentence_len / MEAN_SENTENCE_LEN_REF
    )

    # 2. type_token_ratio: unique lowercased alpha tokens / total alpha tokens.
    alpha_tokens = [w.lower() for w in _ALPHA_TOKEN_RE.findall(text)]
    if alpha_tokens:
        features["type_token_ratio"] = len(set(alpha_tokens)) / len(alpha_tokens)
    else:
        features["type_token_ratio"] = 0.0

    # 3. numeral_symbol_density: fraction of tokens containing a digit or non-alnum.
    numeral_symbol = sum(
        1
        for tok in tokens
        if _DIGIT_RE.search(tok) or _HAS_NON_ALNUM_RE.search(tok)
    )
    features["numeral_symbol_density"] = numeral_symbol / n_tokens

    # 4. mean_word_len_chars: mean characters per alpha word.
    if alpha_tokens:
        mean_word_len = sum(len(w) for w in alpha_tokens) / len(alpha_tokens)
    else:
        mean_word_len = 0.0
    features["mean_word_len_chars"] = mean_word_len
    features["mean_word_len_chars_norm"] = min(1.0, mean_word_len / MEAN_WORD_LEN_REF)

    # 5. list_marker_density: fraction of non-empty lines that start with a marker.
    nonempty_lines = [ln for ln in text.splitlines() if ln.strip()]
    if nonempty_lines:
        marked = sum(1 for ln in nonempty_lines if _is_list_marker(ln.strip()))
        features["list_marker_density"] = marked / len(nonempty_lines)
    else:
        features["list_marker_density"] = 0.0

    # 6. nonstopword_ratio: 1 - (stopword tokens / total tokens).
    # Match stopwords by their lowercased alphabetic core, stripping any attached
    # edge punctuation (e.g. "the," or "it." still count as stopwords).
    stopword_tokens = 0
    for tok in tokens:
        core = _STOPWORD_EDGE_RE.sub("", tok.lower())
        if core in _STOPWORDS:
            stopword_tokens += 1
    features["nonstopword_ratio"] = 1.0 - (stopword_tokens / n_tokens)

    # rho = clipped weighted mean of the six normalized features.
    rho = sum(
        weights.get(name, 0.0) * features[name]
        for name in (
            "mean_sentence_len_tokens_norm",
            "type_token_ratio",
            "numeral_symbol_density",
            "mean_word_len_chars_norm",
            "list_marker_density",
            "nonstopword_ratio",
        )
    )
    return _clip(rho, 0.0, 1.0), features


def density_to_leaf_tokens(
    rho: float,
    l_min: int = 64,
    l_max: int = 320,
    invert: bool = True,
) -> int:
    """Map a density ``rho`` to a leaf-token budget, monotonically.

    With ``invert=True`` (default), higher density yields SMALLER leaves::

        l_min + (l_max - l_min) * (1 - rho)

    With ``invert=False`` the direction reverses (higher density -> larger leaves).
    The result is rounded to an int and clipped to ``[l_min, l_max]``; ``rho`` itself
    is clipped to ``[0, 1]`` first so out-of-range inputs are not extrapolated.

    The DIRECTION (``invert``) and the BOUNDS (``l_min``, ``l_max``) are calibration
    parameters, not fixed assumptions: later experiments choose them. ``invert=True`` is
    the working hypothesis that denser text wants finer-grained leaves.
    """
    rho = _clip(float(rho), 0.0, 1.0)
    factor = (1.0 - rho) if invert else rho
    value = l_min + (l_max - l_min) * factor
    return int(_clip(round(value), l_min, l_max))
