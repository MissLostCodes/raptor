"""Density-adaptive, two-axis chunker (the paper's proposed arm).

This is the integration of the two-axis chunking controller:

* **Axis 1 — structure (WHERE to cut).** A deterministic ``structure_score`` routes the
  document to a structure-aware chunker (explicit headings) or a token chunker (prose),
  exactly as :class:`~raptor.chunking.ahc_chunker.AHCChunker` does.
* **Axis 2 — density (HOW BIG each leaf is).** A ``size_fn`` maps the document's surface
  text to a per-document leaf-token budget, which becomes the ``max_tokens`` of whichever
  chunker the structure axis selected. Fixed-size RAPTOR uses one global leaf size; this
  arm sizes leaves per document.

The size policy is **injected**, not hard-coded, because the 8-doc QASPER pilot showed the
equal-weight composite density score ``rho`` is near-zero predictive of the oracle-optimal
leaf size (its features carry strong but oppositely-signed signal that the uniform mean
cancels — see ``docs/results/2026-06-11-h2-density-feature-preview.md``). The calibrated
policy — a single density feature selected against the oracle sweep, fit as
``size = a + b * feature`` — is built with :func:`make_calibrated_size_fn` and passed in.
The default ``size_fn`` is the **uncalibrated composite placeholder** and exists only so
the mechanism is runnable before calibration; do not report numbers with it.

After each ``chunk`` call the chunker records ``last_score``, ``last_route``, ``last_size``,
and ``last_features`` (mirroring ``AHCChunker``'s routing-capture contract so the pipeline's
``routing_info`` keeps working), plus ``last_size`` for granularity logging.
"""

from typing import Callable, Dict, List, Optional, Tuple

import tiktoken

from .base import Chunker
from .density_score import density_score, density_to_leaf_tokens
from .structure_chunker import StructureChunker
from .structure_score import structure_score
from .token_chunker import TokenChunker


def _clip_int(x: float, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, round(x))))


def make_calibrated_size_fn(
    feature_name: str,
    a: float,
    b: float,
    l_min: int = 50,
    l_max: int = 400,
    density_fn: Callable[[str], Tuple[float, Dict]] = density_score,
) -> Callable[[str], int]:
    """Build the calibrated density->size policy ``size = clip(a + b * feature)``.

    ``feature_name`` is one of the ``density_score`` feature keys (e.g.
    ``"mean_word_len_chars_norm"``), and ``a`` / ``b`` are the two parameters fit by
    :func:`experiments.granularity_calibration.select_and_fit` (its returned
    ``fit["a"]`` / ``fit["b"]``). The returned callable extracts that one feature from a
    document's density features and maps it to a leaf-token budget, clipped to
    ``[l_min, l_max]`` so predictions never leave the swept range. This is the bridge
    from the (offline, oracle-calibrated) fit to the (online, training-free) chunker — it
    needs only the feature value at inference, no SBERT, no LLM, no labels.
    """

    def size_fn(text: str) -> int:
        _, feats = density_fn(text)
        feat_value = float(feats.get(feature_name, 0.0))
        return _clip_int(a + b * feat_value, l_min, l_max)

    return size_fn


def _placeholder_composite_size_fn(
    l_min: int, l_max: int
) -> Callable[[str], int]:
    """UNCALIBRATED fallback: composite ``rho`` -> size via ``density_to_leaf_tokens``.

    Only so the mechanism runs before calibration. The pilot showed the composite is a
    poor sizing signal; never report results with this policy.
    """

    def size_fn(text: str) -> int:
        rho, _ = density_score(text)
        return density_to_leaf_tokens(rho, l_min=l_min, l_max=l_max)

    return size_fn


class DensityAdaptiveChunker(Chunker):
    """Two-axis chunker: structure score routes, density-derived ``size_fn`` sets leaf size.

    Stage 1 (size): ``size = clip(size_fn(text), l_min, l_max)`` — the per-document leaf
        token budget from the density axis.
    Stage 2 (route): ``structure_score(text) >= tau`` -> structure chunker, else token
        chunker — the structure axis (identical routing to ``AHCChunker``).
    Stage 3 (chunk/repair): the routed chunker is constructed with ``max_tokens = size``
        and run; the structure chunker does its own per-section token repair at that size.

    The routed sub-chunkers are rebuilt per ``chunk`` call because the size is per
    document; this is cheap (``TokenChunker`` is trivial and ``StructureChunker`` just
    holds a reusable ``parse_fn``).
    """

    def __init__(
        self,
        parse_fn: Callable[[str], List[str]],
        size_fn: Optional[Callable[[str], int]] = None,
        tau: float = 0.5,
        l_min: int = 50,
        l_max: int = 400,
        tokenizer=None,
        score_fn: Optional[Callable[[str], Tuple[float, Dict]]] = None,
    ):
        self.parse_fn = parse_fn
        self.tau = tau
        self.l_min = l_min
        self.l_max = l_max
        self.tokenizer = tokenizer or tiktoken.get_encoding("cl100k_base")
        self.score_fn = score_fn or structure_score
        # Injected calibrated policy, or the uncalibrated composite placeholder.
        self.size_fn = size_fn or _placeholder_composite_size_fn(l_min, l_max)

        self.last_score: Optional[float] = None
        self.last_route: Optional[str] = None
        self.last_size: Optional[int] = None
        self.last_features: Optional[Dict] = None

    def _leaf_size(self, text: str) -> int:
        return _clip_int(self.size_fn(text), self.l_min, self.l_max)

    def chunk(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []

        size = self._leaf_size(text)
        score, features = self.score_fn(text)

        if score >= self.tau:
            route = "structure"
            chunker: Chunker = StructureChunker(
                parse_fn=self.parse_fn, max_tokens=size, tokenizer=self.tokenizer
            )
        else:
            route = "token"
            chunker = TokenChunker(max_tokens=size, tokenizer=self.tokenizer)
        chunks = chunker.chunk(text)

        self.last_score = score
        self.last_route = route
        self.last_size = size
        self.last_features = features
        return [c for c in chunks if c.strip()]
