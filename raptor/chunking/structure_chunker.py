from typing import Callable, List, Optional

import tiktoken

from .base import Chunker
from .token_chunker import TokenChunker


def locate_sections(text: str, titles: List[str]) -> List[int]:
    """Return the start offset of each title in `text`, searching forward to preserve
    reading order. Returns -1 for titles that cannot be located."""
    lowered = text.lower()
    offsets: List[int] = []
    search_from = 0
    for title in titles:
        t = title.strip().lower()
        if not t:
            offsets.append(-1)
            continue
        idx = lowered.find(t, search_from)
        offsets.append(idx)
        if idx >= 0:
            search_from = idx + len(t)
    return offsets


class StructureChunker(Chunker):
    """Structure-aware chunker.

    Pipeline: parse section titles (injectable `parse_fn`) -> locate each title in the
    raw text by character offset -> slice the document at those boundaries -> token-split
    any section that exceeds `max_tokens` (per-section repair). Falls back to token
    chunking when fewer than two titles can be located.
    """

    def __init__(
        self,
        parse_fn: Callable[[str], List[str]],
        max_tokens: int = 100,
        tokenizer=None,
        fallback: Optional[Chunker] = None,
    ):
        self.parse_fn = parse_fn
        self.max_tokens = max_tokens
        self.tokenizer = tokenizer or tiktoken.get_encoding("cl100k_base")
        self.fallback = fallback or TokenChunker(
            max_tokens=max_tokens, tokenizer=self.tokenizer
        )

    def chunk(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []

        titles = self.parse_fn(text)
        located = sorted({o for o in locate_sections(text, titles) if o >= 0})

        if len(located) < 2:
            return self.fallback.chunk(text)

        bounds = located + [len(text)]
        sections: List[str] = []
        if located[0] > 0:
            sections.append(text[: located[0]])  # preamble before first heading
        for i in range(len(located)):
            sections.append(text[bounds[i] : bounds[i + 1]])

        chunks: List[str] = []
        for sec in sections:
            if not sec.strip():
                continue
            if len(self.tokenizer.encode(sec)) > self.max_tokens:
                chunks.extend(self.fallback.chunk(sec))  # repair oversized section
            else:
                chunks.append(sec)
        return chunks
