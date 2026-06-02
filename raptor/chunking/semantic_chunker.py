import re
from typing import Callable, List, Optional, Sequence

import numpy as np
import tiktoken

from .base import Chunker
from .token_chunker import TokenChunker

_SENTENCE_SPLIT_RE = re.compile(r"[^.!?\n]+[.!?\n]?")


def _split_sentences(text: str) -> List[str]:
    """Split text into sentence-ish units, keeping the sentence text (and its
    terminator). Mirrors the spirit of utils.split_text's delimiter handling."""
    pieces = _SENTENCE_SPLIT_RE.findall(text)
    return [p.strip() for p in pieces if p.strip()]


def _cosine_distance(a: Sequence[float], b: Sequence[float]) -> float:
    va = np.asarray(a, dtype=float)
    vb = np.asarray(b, dtype=float)
    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return 1.0 - float(np.dot(va, vb) / (na * nb))


class SemanticChunker(Chunker):
    """Embedding-similarity boundary chunker.

    Splits text into sentences, embeds each sentence (via injectable `embed_fn`),
    measures cosine distance between consecutive sentences, and inserts a breakpoint
    wherever the distance reaches the `breakpoint_percentile`-th percentile. Consecutive
    sentences between breakpoints form a chunk; any chunk exceeding `max_tokens` is
    repaired by the token `fallback` chunker.
    """

    def __init__(
        self,
        embed_fn: Optional[Callable[[List[str]], List[Sequence[float]]]] = None,
        max_tokens: int = 100,
        tokenizer=None,
        breakpoint_percentile: int = 85,
        fallback: Optional[Chunker] = None,
    ):
        self.max_tokens = max_tokens
        self.tokenizer = tokenizer or tiktoken.get_encoding("cl100k_base")
        self.breakpoint_percentile = breakpoint_percentile
        self.fallback = fallback or TokenChunker(
            max_tokens=max_tokens, tokenizer=self.tokenizer
        )
        self.embed_fn = embed_fn or self._default_embed_fn

    @staticmethod
    def _default_embed_fn(sentences: List[str]) -> List[Sequence[float]]:
        # Lazy: only construct/load the SBERT model on first real call (never at
        # import or construction time, so the offline test suite stays network-free).
        from ..EmbeddingModels import SBertEmbeddingModel

        model = SBertEmbeddingModel()
        return [list(model.create_embedding(s)) for s in sentences]

    def chunk(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []

        sentences = _split_sentences(text)
        if len(sentences) <= 1:
            return [c for c in self.fallback.chunk(text) if c.strip()]

        embeddings = self.embed_fn(sentences)
        distances = [
            _cosine_distance(embeddings[i], embeddings[i + 1])
            for i in range(len(sentences) - 1)
        ]

        # A breakpoint after sentence i ends the current group.
        breakpoints = set()
        if distances:
            threshold = float(np.percentile(distances, self.breakpoint_percentile))
            for i, d in enumerate(distances):
                if d >= threshold and d > 0.0:
                    breakpoints.add(i)

        groups: List[List[str]] = []
        current: List[str] = []
        for i, sentence in enumerate(sentences):
            current.append(sentence)
            if i in breakpoints:
                groups.append(current)
                current = []
        if current:
            groups.append(current)

        chunks: List[str] = []
        for group in groups:
            group_text = " ".join(group).strip()
            if not group_text:
                continue
            if len(self.tokenizer.encode(group_text)) > self.max_tokens:
                chunks.extend(c for c in self.fallback.chunk(group_text) if c.strip())
            else:
                chunks.append(group_text)
        return [c for c in chunks if c.strip()]
